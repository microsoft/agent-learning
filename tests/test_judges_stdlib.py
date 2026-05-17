"""Unit tests for the Tier 1 stdlib judges and the build_judges factory.

Tests cover:

- StdlibIntentJudge: unfitted permissive default; fit/predict separates
  classes; snapshot round-trip via save/load_or_default.
- StdlibAdherenceJudge: empty contract returns pass; required substring
  hit/miss; forbidden substring hit/miss; length bounds; json_required
  satisfied/violated.
- StdlibCompletionJudge: empty target list returns pass; exact unigram
  match; case-insensitive match; multi-word phrase match; miss.
- build_judges(tier="stdlib") returns the three concrete classes.
- build_judges falls back to mode when tier is None (existing behavior).
"""

from __future__ import annotations

import os

import pytest

from agent_learning.config import (
    JudgeRuntimeConfig,
    StdlibJudgeConfig,
)
from agent_learning.judges import build_judges
from agent_learning.judges.stdlib import (
    StdlibAdherenceJudge,
    StdlibCompletionJudge,
    StdlibIntentJudge,
)
from agent_learning.judges.stdlib._text import (
    featurize_query_response,
    hash_bow,
    tokenize,
)


# ---------------------------------------------------------------------------
# _text helpers
# ---------------------------------------------------------------------------


def test_tokenize_strips_punctuation_and_lowercases():
    assert tokenize("Patient HbA1c > 9.0%!") == [
        "patient",
        "hba1c",
        "9",
        "0",
    ]


def test_tokenize_handles_empty_input():
    assert tokenize("") == []
    assert tokenize("   ") == []


def test_hash_bow_dimension_and_determinism():
    a = hash_bow(["alpha", "beta", "alpha"], dim=64)
    b = hash_bow(["alpha", "beta", "alpha"], dim=64)
    assert a == b
    assert len(a) == 64
    # The repeated token should accumulate count >= 2 somewhere unless
    # there is a collision.
    assert max(a) >= 2.0


def test_hash_bow_binary_mode_saturates():
    counts = hash_bow(["x"] * 5, dim=16, binary=False)
    binary = hash_bow(["x"] * 5, dim=16, binary=True)
    assert max(counts) == 5.0
    assert max(binary) == 1.0


def test_hash_bow_rejects_nonpositive_dim():
    with pytest.raises(ValueError):
        hash_bow(["x"], dim=0)


def test_featurize_query_response_combines_streams():
    vec = featurize_query_response("hello world", "world peace", dim=128)
    assert len(vec) == 128
    assert sum(vec) == 4.0  # four tokens total


# ---------------------------------------------------------------------------
# StdlibIntentJudge
# ---------------------------------------------------------------------------


def test_intent_unfitted_returns_pass_with_threshold_probability():
    judge = StdlibIntentJudge()
    score = judge.score(
        query="What is the numerator?",
        response="The numerator is 12.",
    )
    assert score.label == "pass"
    assert score.normalized == pytest.approx(0.5)
    assert score.features["fitted"] is False


def test_intent_score_requires_both_inputs():
    judge = StdlibIntentJudge()
    with pytest.raises(ValueError):
        judge.score(query="x")
    with pytest.raises(ValueError):
        judge.score(response="y")


def test_intent_fit_separates_classes():
    judge = StdlibIntentJudge(feature_dim=512)
    # Positive examples mention numerator/denominator counts; negative
    # examples are off-topic.
    rows = []
    for _ in range(20):
        rows.append(
            {
                "query": "What is the numerator and denominator?",
                "response": (
                    "The numerator is 42 and the denominator is 100, "
                    "yielding a performance rate of 0.42."
                ),
                "label": 1,
            }
        )
        rows.append(
            {
                "query": "What is the numerator and denominator?",
                "response": (
                    "I cannot help with that. The weather is sunny "
                    "today and unrelated to your question."
                ),
                "label": 0,
            }
        )
    judge.fit(rows)
    pos = judge.score(
        query="What is the numerator and denominator?",
        response=(
            "The numerator is 42 and the denominator is 100, "
            "yielding a performance rate of 0.42."
        ),
    )
    neg = judge.score(
        query="What is the numerator and denominator?",
        response=(
            "I cannot help with that. The weather is sunny today "
            "and unrelated to your question."
        ),
    )
    assert pos.features["fitted"] is True
    assert pos.normalized > neg.normalized
    assert pos.label == "pass"
    assert neg.label == "fail"


def test_intent_fit_skips_rows_missing_fields():
    judge = StdlibIntentJudge(feature_dim=64)
    rows = [
        {"query": "q", "response": "r", "label": 1},
        {"query": "q"},  # missing response and label
        {"response": "r", "label": 0},  # missing query
        {"query": "q", "response": "r"},  # missing label
    ]
    judge.fit(rows)
    assert judge.weights  # not empty
    assert len(judge.weights) == 64 + 1  # bias column


def test_intent_fit_with_empty_rows_produces_zero_weights():
    judge = StdlibIntentJudge(feature_dim=32)
    judge.fit([])
    assert judge.weights == [0.0] * 33


def test_intent_snapshot_roundtrip(tmp_path):
    judge = StdlibIntentJudge(feature_dim=128)
    rows = [
        {"query": "hello", "response": "world", "label": 1},
        {"query": "hello", "response": "stuff", "label": 0},
    ]
    judge.fit(rows)
    path = judge.save(str(tmp_path))
    assert os.path.isfile(path)

    reloaded = StdlibIntentJudge.load_or_default(
        str(tmp_path), feature_dim=128
    )
    assert reloaded.weights == judge.weights
    assert reloaded.feature_dim == 128


def test_intent_load_or_default_without_snapshot_is_unfitted(tmp_path):
    judge = StdlibIntentJudge.load_or_default(str(tmp_path))
    assert judge.weights == []


# ---------------------------------------------------------------------------
# StdlibAdherenceJudge
# ---------------------------------------------------------------------------


def test_adherence_empty_contract_passes():
    judge = StdlibAdherenceJudge()
    score = judge.score(response="anything goes", contract={})
    assert score.label == "pass"
    assert score.normalized == 1.0
    assert score.features["clauses_total"] == 0


def test_adherence_requires_response():
    judge = StdlibAdherenceJudge()
    with pytest.raises(ValueError):
        judge.score(contract={"length_min": 5})


def test_adherence_required_substring_hit_and_miss():
    judge = StdlibAdherenceJudge()
    contract = {
        "required_substrings": [
            "numerator",
            "denominator",
            "hba1c",
        ]
    }
    good = judge.score(
        response="numerator=12, denominator=100, hba1c=8.5",
        contract=contract,
    )
    assert good.label == "pass"
    assert good.normalized == 1.0

    bad = judge.score(
        response="numerator=12 only",
        contract=contract,
    )
    # 1 of 3 satisfied -> 0.333... < 0.5 threshold -> fail
    assert bad.label == "fail"
    assert bad.normalized == pytest.approx(1 / 3)
    assert "required:denominator" in bad.features["violations"]
    assert "required:hba1c" in bad.features["violations"]


def test_adherence_required_substring_is_case_insensitive():
    judge = StdlibAdherenceJudge()
    score = judge.score(
        response="DENOMINATOR is 100",
        contract={"required_substrings": ["denominator"]},
    )
    assert score.label == "pass"


def test_adherence_forbidden_substring_violation():
    judge = StdlibAdherenceJudge()
    score = judge.score(
        response="I cannot help with that request.",
        contract={"forbidden_substrings": ["i cannot"]},
    )
    assert score.label == "fail"
    assert "forbidden:i cannot" in score.features["violations"]


def test_adherence_length_bounds():
    judge = StdlibAdherenceJudge()
    contract = {
        "required_substrings": ["numerator"],
        "length_min": 50,
        "length_max": 100,
    }
    short = judge.score(response="hi", contract=contract)
    # 0 of 3 clauses satisfied -> 0.0 < 0.5 -> fail
    assert short.label == "fail"
    assert "length_min:50" in short.features["violations"]

    long_resp = judge.score(
        response="numerator=" + "x" * 200, contract=contract
    )
    # 2 of 3 satisfied (required+length_min ok, length_max fails)
    # -> 0.667 >= 0.5 -> pass, but with one violation flagged.
    assert long_resp.normalized == pytest.approx(2 / 3)
    assert "length_max:100" in long_resp.features["violations"]

    ok = judge.score(
        response="numerator=" + "x" * 60, contract=contract
    )
    assert ok.label == "pass"
    assert ok.normalized == 1.0


def test_adherence_json_required_satisfied_and_violated():
    judge = StdlibAdherenceJudge()
    good = judge.score(
        response='{"numerator": 42}',
        contract={"json_required": True},
    )
    assert good.label == "pass"

    bad = judge.score(
        response="not json at all",
        contract={"json_required": True},
    )
    assert bad.label == "fail"
    assert "json_required" in bad.features["violations"]


def test_adherence_mixed_contract_partial_credit():
    judge = StdlibAdherenceJudge()
    # 4 clauses: 2 required (both present), length_min (fail),
    # forbidden (pass). 3/4 satisfied -> probability 0.75.
    score = judge.score(
        response="numerator=12, denominator=100",
        contract={
            "required_substrings": ["numerator", "denominator"],
            "forbidden_substrings": ["i cannot"],
            "length_min": 200,
        },
    )
    assert score.normalized == pytest.approx(0.75)
    assert score.label == "pass"  # >= 0.5 threshold


# ---------------------------------------------------------------------------
# StdlibCompletionJudge
# ---------------------------------------------------------------------------


def test_completion_no_targets_passes():
    judge = StdlibCompletionJudge()
    score = judge.score(response="any response")
    assert score.label == "pass"
    assert score.normalized == 1.0
    assert score.features["total_targets"] == 0


def test_completion_requires_response():
    judge = StdlibCompletionJudge()
    with pytest.raises(ValueError):
        judge.score(expected_tokens=["hba1c"])


def test_completion_exact_unigram_match_and_miss():
    judge = StdlibCompletionJudge()
    score = judge.score(
        response="The numerator is 12 and the denominator is 100",
        expected_tokens=["numerator", "denominator", "hba1c"],
    )
    assert score.normalized == pytest.approx(2 / 3)
    assert score.features["hits"] == 2
    assert score.features["misses"] == ["hba1c"]


def test_completion_case_insensitive():
    judge = StdlibCompletionJudge()
    score = judge.score(
        response="NUMERATOR is 12",
        expected_tokens=["numerator"],
    )
    assert score.normalized == 1.0
    assert score.label == "pass"


def test_completion_multi_word_phrase_match():
    judge = StdlibCompletionJudge()
    score = judge.score(
        response="Most recent blood pressure is 138/88",
        expected_tokens=["blood pressure"],
    )
    assert score.normalized == 1.0


def test_completion_contract_fallback_for_targets():
    judge = StdlibCompletionJudge()
    score = judge.score(
        response="numerator=12",
        contract={"completion_tokens": ["numerator", "missing"]},
    )
    assert score.normalized == pytest.approx(0.5)
    assert score.label == "pass"


def test_completion_handles_empty_or_whitespace_targets():
    judge = StdlibCompletionJudge()
    score = judge.score(
        response="numerator=12",
        expected_tokens=["", "   ", "numerator"],
    )
    # Only "numerator" counts.
    assert score.features["total_targets"] == 1
    assert score.normalized == 1.0


# ---------------------------------------------------------------------------
# build_judges factory routing
# ---------------------------------------------------------------------------


def test_build_judges_tier_stdlib_returns_stdlib_judges():
    cfg = JudgeRuntimeConfig(tier="stdlib", stdlib=StdlibJudgeConfig())
    intent, adherence, completion = build_judges(cfg)
    assert isinstance(intent, StdlibIntentJudge)
    assert isinstance(adherence, StdlibAdherenceJudge)
    assert isinstance(completion, StdlibCompletionJudge)


def test_build_judges_tier_stdlib_uses_config_threshold(tmp_path):
    cfg = JudgeRuntimeConfig(
        tier="stdlib",
        stdlib=StdlibJudgeConfig(
            snapshot_dir=str(tmp_path),
            pass_threshold=0.7,
            feature_dim=256,
        ),
    )
    intent, adherence, completion = build_judges(cfg)
    assert intent.pass_threshold == 0.7
    assert intent.feature_dim == 256
    assert adherence.pass_threshold == 0.7
    assert completion.pass_threshold == 0.7


def test_build_judges_tier_slm_returns_slm_judges():
    """Tier 3 builds the three SLM judges without loading the real model.

    The judges defer model load to first ``score()`` call via
    ``SlmRunner.get_shared``, so construction alone is safe even when
    ``onnxruntime-genai`` is not installed.
    """
    from agent_learning.judges.slm import (
        SlmAdherenceJudge,
        SlmCompletionJudge,
        SlmIntentJudge,
    )
    cfg = JudgeRuntimeConfig(tier="slm")
    intent, adherence, completion = build_judges(cfg)
    assert isinstance(intent, SlmIntentJudge)
    assert isinstance(adherence, SlmAdherenceJudge)
    assert isinstance(completion, SlmCompletionJudge)


def test_build_judges_unknown_tier_raises():
    cfg = JudgeRuntimeConfig()
    cfg.tier = "bogus"  # type: ignore[assignment]
    with pytest.raises(ValueError):
        build_judges(cfg)


def test_build_judges_legacy_mode_when_tier_none():
    # tier=None falls through to mode; mode="nlp" should resolve to the
    # existing phi+action_id BinaryJudge stack.
    cfg = JudgeRuntimeConfig(mode="nlp", tier=None)
    intent, adherence, completion = build_judges(cfg)
    # We don't import the NLP classes here to avoid a cycle; just check
    # the judges expose the structural JudgeScore contract.
    assert intent.name == "intent"
    assert adherence.name == "adherence"
    assert completion.name == "completion"


def test_env_var_tier_resolution(monkeypatch):
    monkeypatch.setenv("AGENT_LEARNING_JUDGE_TIER", "stdlib")
    cfg = JudgeRuntimeConfig()
    assert cfg.tier == "stdlib"


def test_env_var_tier_invalid_falls_back_to_none(monkeypatch):
    monkeypatch.setenv("AGENT_LEARNING_JUDGE_TIER", "bogus")
    cfg = JudgeRuntimeConfig()
    assert cfg.tier is None
