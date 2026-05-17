"""Tier 4 LLM judge unit tests.

These tests inject a fake ``azure.ai.evaluation`` module into
``sys.modules`` so the real Azure OpenAI / azure-ai-evaluation package
isn't needed at test time. The wrapper's lazy-import path is exercised
end-to-end, including the JudgeScore projection from the evaluator's
1-5 Likert-scale return value into ``[0, 1]``.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Dict, List

import pytest

from agent_learning.config import JudgeConfig, JudgeRuntimeConfig
from agent_learning.judges import build_judges
from agent_learning.judges.llm import (
    LlmAdherenceJudge,
    LlmCompletionJudge,
    LlmIntentJudge,
)
from agent_learning.judges.llm import _base as llm_base


# --------------------------------------------------------------------- fakes


class FakeEvaluator:
    """Mimics an ``azure.ai.evaluation`` evaluator class instance."""

    # Class-level default; tests override via monkeypatch.setattr.
    score: float = 5.0

    def __init__(self, model_config: Dict[str, Any]) -> None:
        self.model_config = model_config
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(dict(kwargs))
        return {"score": type(self).score}


def _install_fake_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Stand up a fake azure.ai.evaluation package in sys.modules."""
    pkg_azure = types.ModuleType("azure")
    pkg_azure_ai = types.ModuleType("azure.ai")
    pkg_eval = types.ModuleType("azure.ai.evaluation")

    pkg_eval.IntentResolutionEvaluator = FakeEvaluator
    pkg_eval.TaskAdherenceEvaluator = FakeEvaluator
    pkg_eval.TaskCompletionEvaluator = FakeEvaluator

    monkeypatch.setitem(sys.modules, "azure", pkg_azure)
    monkeypatch.setitem(sys.modules, "azure.ai", pkg_azure_ai)
    monkeypatch.setitem(sys.modules, "azure.ai.evaluation", pkg_eval)
    return pkg_eval


# --------------------------------------------------------------- score projection


def test_normalize_passes_through_zero_one_range() -> None:
    assert llm_base._normalize(0.0) == 0.0
    assert llm_base._normalize(0.5) == 0.5
    assert llm_base._normalize(1.0) == 1.0


def test_normalize_maps_likert_scale_to_zero_one() -> None:
    # 1.0 sits at the boundary; the [0,1] passthrough branch wins.
    assert llm_base._normalize(1.0) == pytest.approx(1.0)
    assert llm_base._normalize(3.0) == pytest.approx(0.5)
    assert llm_base._normalize(5.0) == pytest.approx(1.0)


def test_normalize_clamps_out_of_range() -> None:
    assert llm_base._normalize(-2.0) == 0.0
    assert llm_base._normalize(7.0) == 1.0


def test_project_to_score_uses_score_key() -> None:
    score = llm_base._project_to_score({"score": 4.0}, threshold=0.5, name="intent")
    assert score.label == "pass"
    assert score.normalized == pytest.approx(0.75)


def test_project_to_score_uses_named_score_key() -> None:
    score = llm_base._project_to_score(
        {"intent_score": 2.0}, threshold=0.5, name="intent"
    )
    assert score.label == "fail"
    assert score.normalized == pytest.approx(0.25)


def test_project_to_score_falls_back_to_first_numeric() -> None:
    score = llm_base._project_to_score(
        {"reason": "looks great", "value": 0.9}, threshold=0.5, name="adherence"
    )
    assert score.label == "pass"
    assert score.normalized == pytest.approx(0.9)


def test_project_to_score_rejects_non_dict() -> None:
    with pytest.raises(TypeError, match="expected mapping"):
        llm_base._project_to_score("not-a-dict", threshold=0.5, name="x")


def test_project_to_score_rejects_no_numeric() -> None:
    with pytest.raises(ValueError, match="no numeric score"):
        llm_base._project_to_score({"reason": "ok"}, threshold=0.5, name="x")


# ----------------------------------------------------------------- lazy import


def test_lazy_import_failure_message() -> None:
    """If azure.ai.evaluation isn't installed, surface an actionable error."""
    judge = LlmIntentJudge(cfg=JudgeConfig())
    # Force the lazy import to fail by short-circuiting __import__.
    original_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__  # type: ignore[index]

    def broken_import(name, *args, **kwargs):
        if name.startswith("azure.ai.evaluation"):
            raise ImportError("simulated missing optional")
        return original_import(name, *args, **kwargs)

    import builtins as _builtins

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_builtins, "__import__", broken_import)
        with pytest.raises(ImportError, match="azure-ai-evaluation"):
            judge._evaluator()


# ------------------------------------------------------------- intent judge


def test_intent_judge_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_eval = _install_fake_module(monkeypatch)
    # JudgeConfig.threshold defaults to 3 (Likert scale); pass a [0,1]
    # threshold so the projected score's label matches our expectation.
    judge = LlmIntentJudge(cfg=JudgeConfig(threshold=0.5))  # type: ignore[arg-type]
    score = judge.score(query="What is 2+2?", response="four")
    assert score.label == "pass"
    assert score.normalized == pytest.approx(1.0)
    evaluator = judge._cached_evaluator
    assert evaluator is not None
    assert evaluator.calls[0]["query"] == "What is 2+2?"
    assert evaluator.calls[0]["response"] == "four"
    # phi / action_id should be stripped before reaching the evaluator.
    assert "phi" not in evaluator.calls[0]
    assert "action_id" not in evaluator.calls[0]


def test_intent_judge_uses_request_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_module(monkeypatch)
    judge = LlmIntentJudge(cfg=JudgeConfig(threshold=0.5))  # type: ignore[arg-type]
    score = judge.score(request="hello", response="world")
    assert score.label == "pass"
    assert judge._cached_evaluator.calls[0]["query"] == "hello"  # type: ignore[union-attr]


def test_intent_judge_requires_query_and_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_module(monkeypatch)
    judge = LlmIntentJudge(cfg=JudgeConfig())
    with pytest.raises(ValueError, match="requires query"):
        judge.score(query=None, response="r")
    with pytest.raises(ValueError, match="requires query"):
        judge.score(query="q", response=None)


def test_intent_judge_strips_nlp_only_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_module(monkeypatch)
    judge = LlmIntentJudge(cfg=JudgeConfig())
    judge.score(query="q", response="r", phi=[0.1, 0.2], action_id=2)
    call = judge._cached_evaluator.calls[0]  # type: ignore[union-attr]
    assert "phi" not in call
    assert "action_id" not in call


# ----------------------------------------------------------- adherence + completion


def test_adherence_judge_routes_to_task_adherence_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_module(monkeypatch)
    judge = LlmAdherenceJudge(cfg=JudgeConfig())
    judge.score(query="q", response="r")
    assert isinstance(judge._cached_evaluator, FakeEvaluator)
    # The evaluator_attr resolves to TaskAdherenceEvaluator.
    assert judge.evaluator_attr == "TaskAdherenceEvaluator"


def test_completion_judge_routes_to_task_completion_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_module(monkeypatch)
    judge = LlmCompletionJudge(cfg=JudgeConfig())
    judge.score(query="q", response="r")
    assert judge.evaluator_attr == "TaskCompletionEvaluator"


def test_failing_score_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_module(monkeypatch)
    # Drop the evaluator's canned score below the threshold.
    monkeypatch.setattr(FakeEvaluator, "score", 1.5)
    judge = LlmIntentJudge(cfg=JudgeConfig(threshold=0.5))  # type: ignore[arg-type]
    score = judge.score(query="q", response="r")
    # 1.5 on a 1-5 scale -> normalized = 0.125
    assert score.label == "fail"
    assert score.normalized == pytest.approx(0.125)
    assert score.confidence == pytest.approx(0.875)


# ------------------------------------------------------------------- factory


def test_build_judges_tier_llm_returns_llm_judges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_module(monkeypatch)
    cfg = JudgeRuntimeConfig(tier="llm")
    intent, adherence, completion = build_judges(cfg)
    assert isinstance(intent, LlmIntentJudge)
    assert isinstance(adherence, LlmAdherenceJudge)
    assert isinstance(completion, LlmCompletionJudge)


def test_build_judges_legacy_mode_llm_returns_llm_judges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_module(monkeypatch)
    cfg = JudgeRuntimeConfig()
    cfg.tier = None
    cfg.mode = "llm"
    intent, _, _ = build_judges(cfg)
    assert isinstance(intent, LlmIntentJudge)
