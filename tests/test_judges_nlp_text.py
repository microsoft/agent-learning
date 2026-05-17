"""Tier 2 NLP text judge unit tests."""

from __future__ import annotations

import importlib
import os
import sys
from typing import List

import pytest

# Skip the entire module when the [nlp] extra is unavailable.
pytest.importorskip("sklearn")
pytest.importorskip("joblib")

from agent_learning.config import JudgeRuntimeConfig, NlpTextJudgeConfig
from agent_learning.judges import build_judges
from agent_learning.judges.nlp_text import (  # noqa: E402
    NlpTextAdherenceJudge,
    NlpTextCompletionJudge,
    NlpTextIntentJudge,
)
from agent_learning.judges.nlp_text import _base as nlp_text_base  # noqa: E402


# --------------------------------------------------------------------- helpers


def _intent_training_rows() -> List[dict]:
    positives = [
        {"query": "what is the capital of france", "response": "paris is the capital of france", "label": 1},
        {"query": "how do i sort a list", "response": "use sorted or list.sort for in-place", "label": 1},
        {"query": "explain dependency injection", "response": "inversion of control passes dependencies in", "label": 1},
        {"query": "weather tomorrow", "response": "tomorrow will be sunny and 72 degrees", "label": 1},
        {"query": "convert miles to kilometers", "response": "multiply miles by 1.609 to get kilometers", "label": 1},
        {"query": "write a haiku about coffee", "response": "dark roast morning brew steam rising", "label": 1},
    ]
    negatives = [
        {"query": "what is the capital of france", "response": "i prefer tea over coffee in the morning", "label": 0},
        {"query": "how do i sort a list", "response": "the weather is sunny today", "label": 0},
        {"query": "explain dependency injection", "response": "the price of bananas is rising", "label": 0},
        {"query": "weather tomorrow", "response": "use a hashmap for constant time lookup", "label": 0},
        {"query": "convert miles to kilometers", "response": "the lion is the king of the jungle", "label": 0},
        {"query": "write a haiku about coffee", "response": "stocks closed mixed on tuesday", "label": 0},
    ]
    return positives + negatives


def _adherence_training_rows() -> List[dict]:
    return [
        {"response": "the result is positive and complete", "label": 1},
        {"response": "all checks passed successfully", "label": 1},
        {"response": "verified valid response confirmed", "label": 1},
        {"response": "request fulfilled correct output", "label": 1},
        {"response": "invalid request denied", "label": 0},
        {"response": "error failed missing data", "label": 0},
        {"response": "incorrect output rejected", "label": 0},
        {"response": "violation forbidden refused", "label": 0},
    ]


# ----------------------------------------------------------- lazy import error


def test_friendly_error_when_sklearn_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """If sklearn is not importable, _require_sklearn raises the friendly message."""
    # Hide the real sklearn from importlib for the duration of the test.
    saved = {name: sys.modules[name] for name in list(sys.modules) if name == "sklearn" or name.startswith("sklearn.")}
    try:
        for name in list(saved.keys()):
            monkeypatch.setitem(sys.modules, name, None)  # type: ignore[arg-type]
        monkeypatch.setitem(sys.modules, "sklearn", None)  # type: ignore[arg-type]
        # Reload our helper to clear any cached references.
        importlib.reload(nlp_text_base)
        with pytest.raises(ImportError, match=r"\[nlp\] extra"):
            nlp_text_base._require_sklearn()
    finally:
        # Restore real modules before reloading so subsequent tests work.
        for name, mod in saved.items():
            monkeypatch.setitem(sys.modules, name, mod)
        importlib.reload(nlp_text_base)


# ------------------------------------------------------------- intent judge


def test_intent_unfitted_returns_pass_threshold() -> None:
    judge = NlpTextIntentJudge.load_or_default(None, pass_threshold=0.5)
    score = judge.score(query="hello", response="hi there")
    assert score.normalized == pytest.approx(0.5, abs=1e-9)
    assert score.features["fitted"] is False


def test_intent_requires_both_query_and_response() -> None:
    judge = NlpTextIntentJudge.load_or_default(None)
    with pytest.raises(ValueError, match="both query and response"):
        judge.score(query="", response="something")
    with pytest.raises(ValueError, match="both query and response"):
        judge.score(query="something", response="")


def test_intent_fit_separates_positive_and_negative() -> None:
    judge = NlpTextIntentJudge(pass_threshold=0.5)
    judge.fit(_intent_training_rows())
    pos = judge.score(query="what is the capital of france", response="paris is the capital")
    neg = judge.score(query="what is the capital of france", response="i prefer tea over coffee")
    assert pos.normalized > neg.normalized
    assert pos.features["fitted"] is True


def test_intent_fit_with_one_label_stays_unfitted() -> None:
    judge = NlpTextIntentJudge(pass_threshold=0.5)
    judge.fit(
        [
            {"query": "q", "response": "r1", "label": 1},
            {"query": "q", "response": "r2", "label": 1},
        ]
    )
    assert judge.fitted is False


# ------------------------------------------------------------ adherence judge


def test_adherence_unfitted_falls_back_to_rule_engine() -> None:
    judge = NlpTextAdherenceJudge.load_or_default(None, pass_threshold=0.5)
    score = judge.score(response="some response", contract=None)
    # No contract → rule engine returns 1.0.
    assert score.normalized == pytest.approx(1.0, abs=1e-9)
    assert score.features["learned_probability"] is None
    assert score.features["fitted"] is False


def test_adherence_combined_when_fitted() -> None:
    judge = NlpTextAdherenceJudge(pass_threshold=0.5)
    judge.fit(_adherence_training_rows())
    assert judge.fitted is True
    score = judge.score(response="the result is positive and complete", contract=None)
    learned = score.features["learned_probability"]
    rule = score.features["rule_probability"]
    assert learned is not None
    assert score.normalized == pytest.approx(0.5 * (rule + learned), abs=1e-9)


def test_adherence_requires_response() -> None:
    judge = NlpTextAdherenceJudge.load_or_default(None)
    with pytest.raises(ValueError, match="requires a response"):
        judge.score(response=None)


def test_adherence_contract_drives_rule_engine() -> None:
    judge = NlpTextAdherenceJudge.load_or_default(None, pass_threshold=0.5)
    contract = {
        "required_substrings": ["alpha"],
        "forbidden_substrings": ["beta"],
    }
    bad = judge.score(response="this mentions beta only", contract=contract)
    good = judge.score(response="this mentions alpha clearly", contract=contract)
    assert good.normalized > bad.normalized


# ----------------------------------------------------------- completion judge


def test_completion_unfitted_uses_rule_signal_only() -> None:
    judge = NlpTextCompletionJudge.load_or_default(None, pass_threshold=0.5)
    score = judge.score(
        response="alpha beta gamma",
        expected_tokens=["alpha", "beta", "gamma"],
    )
    assert score.normalized == pytest.approx(1.0, abs=1e-9)
    assert score.features["learned_probability"] is None


def test_completion_partial_coverage() -> None:
    judge = NlpTextCompletionJudge.load_or_default(None, pass_threshold=0.5)
    score = judge.score(
        response="alpha only",
        expected_tokens=["alpha", "beta", "gamma", "delta"],
    )
    # 1 hit out of 4 targets.
    assert score.normalized == pytest.approx(0.25, abs=1e-9)
    assert score.features["hits"] == 1


def test_completion_expected_tokens_overrides_contract() -> None:
    judge = NlpTextCompletionJudge.load_or_default(None, pass_threshold=0.5)
    contract = {"completion_tokens": ["x", "y"]}
    score = judge.score(
        response="alpha",
        expected_tokens=["alpha"],
        contract=contract,
    )
    assert score.features["hits"] == 1
    assert score.features["total_targets"] == 1


def test_completion_requires_response() -> None:
    judge = NlpTextCompletionJudge.load_or_default(None)
    with pytest.raises(ValueError, match="requires a response"):
        judge.score(response=None, expected_tokens=["x"])


# ------------------------------------------------------------- snapshot I/O


def test_intent_snapshot_roundtrip(tmp_path) -> None:
    judge = NlpTextIntentJudge(pass_threshold=0.5)
    judge.fit(_intent_training_rows())
    judge.save(str(tmp_path))
    header_path = tmp_path / "intent.nlp_text.json"
    blob_path = tmp_path / "intent.nlp_text.joblib"
    assert header_path.is_file()
    assert blob_path.is_file()

    loaded = NlpTextIntentJudge.load_or_default(str(tmp_path), pass_threshold=0.5)
    assert loaded.fitted is True
    score_orig = judge.score(query="weather tomorrow", response="sunny and 72 degrees")
    score_loaded = loaded.score(query="weather tomorrow", response="sunny and 72 degrees")
    assert score_orig.normalized == pytest.approx(score_loaded.normalized, abs=1e-9)


def test_unfitted_snapshot_has_no_blob(tmp_path) -> None:
    judge = NlpTextIntentJudge(pass_threshold=0.5)
    judge.save(str(tmp_path))
    assert (tmp_path / "intent.nlp_text.json").is_file()
    assert not (tmp_path / "intent.nlp_text.joblib").exists()


# ----------------------------------------------------------- factory routing


def test_build_judges_tier_nlp_returns_text_judges() -> None:
    cfg = JudgeRuntimeConfig(tier="nlp", nlp_text=NlpTextJudgeConfig(snapshot_dir=""))
    intent, adherence, completion = build_judges(cfg)
    assert isinstance(intent, NlpTextIntentJudge)
    assert isinstance(adherence, NlpTextAdherenceJudge)
    assert isinstance(completion, NlpTextCompletionJudge)


def test_build_judges_legacy_mode_nlp_uses_binary_judge() -> None:
    from agent_learning.judges.nlp import (
        NlpAdherenceJudge,
        NlpCompletionJudge,
        NlpIntentJudge,
    )

    cfg = JudgeRuntimeConfig(tier=None, mode="nlp")
    intent, adherence, completion = build_judges(cfg)
    assert isinstance(intent, NlpIntentJudge)
    assert isinstance(adherence, NlpAdherenceJudge)
    assert isinstance(completion, NlpCompletionJudge)


# --------------------------------------------------------- config env vars


def test_nlp_text_config_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_LEARNING_NLP_TEXT_JUDGE_DIR", "/tmp/foo")
    monkeypatch.setenv("AGENT_LEARNING_NLP_TEXT_PASS_THRESHOLD", "0.7")
    monkeypatch.setenv("AGENT_LEARNING_NLP_TEXT_MAX_FEATURES", "5000")
    monkeypatch.setenv("AGENT_LEARNING_NLP_TEXT_NGRAM_MIN", "1")
    monkeypatch.setenv("AGENT_LEARNING_NLP_TEXT_NGRAM_MAX", "3")
    cfg = NlpTextJudgeConfig()
    assert cfg.snapshot_dir == "/tmp/foo"
    assert cfg.pass_threshold == pytest.approx(0.7)
    assert cfg.max_features == 5000
    assert cfg.ngram_min == 1
    assert cfg.ngram_max == 3
