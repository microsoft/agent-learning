"""Tier 3 SLM judge unit tests.

These tests use a ``FakeSlmRunner`` substituted via the ``runner=``
constructor argument, so the real Phi-4-mini-instruct ONNX model is
never loaded. A separate integration-only test in
``test_judges_slm_integration.py`` (skipped unless the model is on disk)
exercises the real ``SlmRunner``.
"""

from __future__ import annotations

import importlib
import sys
from typing import List

import pytest

from agent_learning.config import JudgeRuntimeConfig, SlmJudgeConfig
from agent_learning.judges import build_judges
from agent_learning.judges.slm import (
    SlmAdherenceJudge,
    SlmCompletionJudge,
    SlmIntentJudge,
)
from agent_learning.judges.slm import _base as slm_base


# --------------------------------------------------------------------- helpers


class FakeSlmRunner:
    """Capture every prompt and return scripted replies."""

    def __init__(self, replies: List[str]) -> None:
        self.replies = list(replies)
        self.prompts: List[str] = []
        self.kwargs: List[dict] = []

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
    ) -> str:
        self.prompts.append(prompt)
        self.kwargs.append(
            {"max_new_tokens": max_new_tokens, "temperature": temperature}
        )
        if not self.replies:
            return ""
        # Repeat the last reply forever once the script is exhausted, to
        # keep the cascade tests tolerant.
        if len(self.replies) == 1:
            return self.replies[0]
        return self.replies.pop(0)


@pytest.fixture
def runner_pass() -> FakeSlmRunner:
    return FakeSlmRunner(['{"verdict": "pass", "confidence": 0.92}'])


@pytest.fixture
def runner_fail() -> FakeSlmRunner:
    return FakeSlmRunner(['{"verdict": "fail", "confidence": 0.81}'])


# ---------------------------------------------------------------- lazy import


def test_friendly_error_when_genai_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """If onnxruntime_genai is not importable, _require_genai raises the friendly message."""
    monkeypatch.setitem(sys.modules, "onnxruntime_genai", None)
    importlib.reload(slm_base)
    try:
        with pytest.raises(ImportError, match=r"\[slm\] extra"):
            slm_base._require_genai()
    finally:
        monkeypatch.delitem(sys.modules, "onnxruntime_genai", raising=False)
        importlib.reload(slm_base)


# ------------------------------------------------------------- verdict parser


def test_parse_verdict_pass_json() -> None:
    probability, raw = slm_base.parse_verdict(
        '{"verdict": "pass", "confidence": 0.87}'
    )
    assert probability == pytest.approx(0.87, abs=1e-9)
    assert raw["verdict"] == "pass"


def test_parse_verdict_fail_json() -> None:
    probability, raw = slm_base.parse_verdict(
        '{"verdict": "fail", "confidence": 0.7}'
    )
    assert probability == pytest.approx(0.3, abs=1e-9)
    assert raw["verdict"] == "fail"


def test_parse_verdict_extracts_embedded_json() -> None:
    text = (
        "Sure! Here's my evaluation:\n"
        '{"verdict": "pass", "confidence": 0.6}\nDone.'
    )
    probability, raw = slm_base.parse_verdict(text)
    assert probability == pytest.approx(0.6, abs=1e-9)
    assert raw["verdict"] == "pass"


def test_parse_verdict_falls_back_to_pass_keyword() -> None:
    probability, raw = slm_base.parse_verdict("Yes, this response would pass.")
    assert probability == pytest.approx(0.75, abs=1e-9)
    assert raw["heuristic"] == "pass"


def test_parse_verdict_falls_back_to_fail_keyword() -> None:
    probability, raw = slm_base.parse_verdict("No, this would fail the test.")
    assert probability == pytest.approx(0.25, abs=1e-9)
    assert raw["heuristic"] == "fail"


def test_parse_verdict_returns_neutral_on_garbage() -> None:
    probability, raw = slm_base.parse_verdict("xyzzy lorem ipsum")
    assert probability == pytest.approx(0.5, abs=1e-9)
    assert raw["heuristic"] == "neutral"


def test_parse_verdict_clamps_confidence() -> None:
    probability, _ = slm_base.parse_verdict(
        '{"verdict": "pass", "confidence": 1.7}'
    )
    assert probability == 1.0
    probability, _ = slm_base.parse_verdict(
        '{"verdict": "pass", "confidence": -0.3}'
    )
    assert probability == 0.0


def test_parse_verdict_handles_empty() -> None:
    probability, _ = slm_base.parse_verdict("")
    assert probability == 0.5


# --------------------------------------------------------------- prompt shape


def test_render_chat_prompt_uses_phi_template() -> None:
    prompt = slm_base.render_chat_prompt(system="sys", user="user")
    assert "<|system|>" in prompt
    assert "<|user|>" in prompt
    assert "<|assistant|>" in prompt
    assert "sys" in prompt
    assert "user" in prompt


# ------------------------------------------------------------- intent judge


def test_intent_pass(runner_pass: FakeSlmRunner) -> None:
    judge = SlmIntentJudge.load_or_default(
        model_dir="(mocked)", runner=runner_pass
    )
    score = judge.score(query="what is 2+2", response="four")
    assert score.label == "pass"
    assert score.normalized == pytest.approx(0.92, abs=1e-9)
    # The prompt should embed both query and response.
    assert "what is 2+2" in runner_pass.prompts[0]
    assert "four" in runner_pass.prompts[0]


def test_intent_fail(runner_fail: FakeSlmRunner) -> None:
    judge = SlmIntentJudge.load_or_default(
        model_dir="(mocked)", runner=runner_fail
    )
    score = judge.score(query="what is 2+2", response="i prefer tea")
    assert score.label == "fail"
    assert score.normalized == pytest.approx(0.19, abs=1e-9)


def test_intent_requires_both_inputs() -> None:
    judge = SlmIntentJudge.load_or_default(
        model_dir="(mocked)", runner=FakeSlmRunner([""])
    )
    with pytest.raises(ValueError, match="both query and response"):
        judge.score(query="", response="something")
    with pytest.raises(ValueError, match="both query and response"):
        judge.score(query="something", response="")


def test_intent_handles_malformed_reply() -> None:
    judge = SlmIntentJudge.load_or_default(
        model_dir="(mocked)", runner=FakeSlmRunner(["completely broken"])
    )
    score = judge.score(query="q", response="r")
    # Neutral fallback (~0.5) means fail-by-default since 0.5 >= pass_threshold.
    assert score.normalized == pytest.approx(0.5, abs=1e-9)


# ------------------------------------------------------------ adherence judge


def test_adherence_pass(runner_pass: FakeSlmRunner) -> None:
    judge = SlmAdherenceJudge.load_or_default(
        model_dir="(mocked)", runner=runner_pass
    )
    score = judge.score(
        response="the answer mentions alpha",
        contract={"required_substrings": ["alpha"]},
    )
    assert score.label == "pass"
    assert "alpha" in runner_pass.prompts[0]


def test_adherence_no_contract_uses_default_text() -> None:
    runner = FakeSlmRunner(['{"verdict": "pass", "confidence": 0.7}'])
    judge = SlmAdherenceJudge.load_or_default(
        model_dir="(mocked)", runner=runner
    )
    score = judge.score(response="r", contract=None)
    assert score.label == "pass"
    assert "no explicit contract" in runner.prompts[0]


def test_adherence_requires_response() -> None:
    judge = SlmAdherenceJudge.load_or_default(
        model_dir="(mocked)", runner=FakeSlmRunner([""])
    )
    with pytest.raises(ValueError, match="requires a response"):
        judge.score(response=None)


# ----------------------------------------------------------- completion judge


def test_completion_pass_with_expected_tokens(runner_pass: FakeSlmRunner) -> None:
    judge = SlmCompletionJudge.load_or_default(
        model_dir="(mocked)", runner=runner_pass
    )
    score = judge.score(
        response="alpha and beta are present",
        expected_tokens=["alpha", "beta"],
    )
    assert score.label == "pass"
    prompt = runner_pass.prompts[0]
    assert "- alpha" in prompt
    assert "- beta" in prompt


def test_completion_falls_back_to_contract_tokens() -> None:
    runner = FakeSlmRunner(['{"verdict": "pass", "confidence": 0.6}'])
    judge = SlmCompletionJudge.load_or_default(
        model_dir="(mocked)", runner=runner
    )
    judge.score(
        response="alpha",
        contract={"completion_tokens": ["alpha", "gamma"]},
    )
    assert "- alpha" in runner.prompts[0]
    assert "- gamma" in runner.prompts[0]


def test_completion_no_targets_uses_default_text() -> None:
    runner = FakeSlmRunner(['{"verdict": "pass", "confidence": 0.55}'])
    judge = SlmCompletionJudge.load_or_default(
        model_dir="(mocked)", runner=runner
    )
    judge.score(response="r")
    assert "no explicit checklist" in runner.prompts[0]


def test_completion_requires_response() -> None:
    judge = SlmCompletionJudge.load_or_default(
        model_dir="(mocked)", runner=FakeSlmRunner([""])
    )
    with pytest.raises(ValueError, match="requires a response"):
        judge.score(response=None, expected_tokens=["x"])


# ------------------------------------------------- generation parameter wiring


def test_judge_passes_max_new_tokens_and_temperature() -> None:
    runner = FakeSlmRunner(['{"verdict": "pass", "confidence": 0.9}'])
    judge = SlmIntentJudge.load_or_default(
        model_dir="(mocked)",
        max_new_tokens=128,
        temperature=0.3,
        runner=runner,
    )
    judge.score(query="q", response="r")
    assert runner.kwargs[0]["max_new_tokens"] == 128
    assert runner.kwargs[0]["temperature"] == pytest.approx(0.3)


# ----------------------------------------------------------- factory routing


def test_build_judges_tier_slm_returns_slm_judges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Avoid loading the real model by intercepting SlmRunner.get_shared.
    fake = FakeSlmRunner(['{"verdict": "pass", "confidence": 0.5}'])
    monkeypatch.setattr(
        slm_base.SlmRunner,
        "get_shared",
        classmethod(lambda cls, model_dir: fake),
    )
    cfg = JudgeRuntimeConfig(
        tier="slm",
        slm=SlmJudgeConfig(model_dir="(mocked)"),
    )
    intent, adherence, completion = build_judges(cfg)
    assert isinstance(intent, SlmIntentJudge)
    assert isinstance(adherence, SlmAdherenceJudge)
    assert isinstance(completion, SlmCompletionJudge)


# --------------------------------------------------------- config env vars


def test_slm_config_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_LEARNING_SLM_MODEL_DIR", "/tmp/model")
    monkeypatch.setenv("AGENT_LEARNING_SLM_PASS_THRESHOLD", "0.7")
    monkeypatch.setenv("AGENT_LEARNING_SLM_MAX_NEW_TOKENS", "32")
    monkeypatch.setenv("AGENT_LEARNING_SLM_TEMPERATURE", "0.1")
    cfg = SlmJudgeConfig()
    assert cfg.model_dir == "/tmp/model"
    assert cfg.pass_threshold == pytest.approx(0.7)
    assert cfg.max_new_tokens == 32
    assert cfg.temperature == pytest.approx(0.1)


# ------------------------------------------------------------- runner errors


def test_runner_rejects_empty_model_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing SlmRunner('') should fail before we touch onnxruntime."""
    with pytest.raises(ValueError, match="non-empty model_dir"):
        slm_base.SlmRunner("")


def test_runner_rejects_missing_directory(tmp_path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError, match="SLM model directory not found"):
        slm_base.SlmRunner(str(missing))
