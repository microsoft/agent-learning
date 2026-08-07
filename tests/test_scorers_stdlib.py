"""Unit tests for the Tier 1 stdlib scorers and the build_scorers factory.

Tests cover:

- StdlibIntentScorer: unfitted permissive default; fit/predict separates
  classes; snapshot round-trip via save/load_or_default.
- StdlibAdherenceScorer: empty contract returns pass; required substring
  hit/miss; forbidden substring hit/miss; length bounds; json_required
  satisfied/violated.
- StdlibCompletionScorer: empty target list returns pass; exact unigram
  match; case-insensitive match; multi-word phrase match; miss.
- build_scorers(tier="stdlib") returns the three concrete classes.
- build_scorers falls back to mode when tier is None (existing behavior).
"""

from __future__ import annotations

import os

import pytest

from agent_learning.config import (
    ScoreRuntimeConfig,
    StdlibScoreConfig,
)
from agent_learning.scorers import build_scorers
from agent_learning.scorers.stdlib import (
    StdlibAdherenceScorer,
    StdlibCompletionScorer,
    StdlibIntentScorer,
)
from agent_learning.scorers.stdlib._text import (
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
# StdlibIntentScorer
# ---------------------------------------------------------------------------


def test_intent_unfitted_returns_pass_with_threshold_probability():
    scorer = StdlibIntentScorer()
    score = scorer.score(
        query="What is the numerator?",
        response="The numerator is 12.",
    )
    assert score.label == "pass"
    assert score.normalized == pytest.approx(0.5)
    assert score.features["fitted"] is False


def test_intent_score_requires_both_inputs():
    scorer = StdlibIntentScorer()
    with pytest.raises(ValueError):
        scorer.score(query="x")
    with pytest.raises(ValueError):
        scorer.score(response="y")


def test_intent_fit_separates_classes():
    scorer = StdlibIntentScorer(feature_dim=512)
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
    scorer.fit(rows)
    pos = scorer.score(
        query="What is the numerator and denominator?",
        response=(
            "The numerator is 42 and the denominator is 100, "
            "yielding a performance rate of 0.42."
        ),
    )
    neg = scorer.score(
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
    scorer = StdlibIntentScorer(feature_dim=64)
    rows = [
        {"query": "q", "response": "r", "label": 1},
        {"query": "q"},  # missing response and label
        {"response": "r", "label": 0},  # missing query
        {"query": "q", "response": "r"},  # missing label
    ]
    scorer.fit(rows)
    assert scorer.weights  # not empty
    assert len(scorer.weights) == 64 + 1  # bias column


def test_intent_fit_with_empty_rows_produces_zero_weights():
    scorer = StdlibIntentScorer(feature_dim=32)
    scorer.fit([])
    assert scorer.weights == [0.0] * 33


def test_intent_snapshot_roundtrip(tmp_path):
    scorer = StdlibIntentScorer(feature_dim=128)
    rows = [
        {"query": "hello", "response": "world", "label": 1},
        {"query": "hello", "response": "stuff", "label": 0},
    ]
    scorer.fit(rows)
    path = scorer.save(str(tmp_path))
    assert os.path.isfile(path)

    reloaded = StdlibIntentScorer.load_or_default(
        str(tmp_path), feature_dim=128
    )
    assert reloaded.weights == scorer.weights
    assert reloaded.feature_dim == 128


def test_intent_load_or_default_without_snapshot_is_unfitted(tmp_path):
    scorer = StdlibIntentScorer.load_or_default(str(tmp_path))
    assert scorer.weights == []


# ---------------------------------------------------------------------------
# StdlibAdherenceScorer
# ---------------------------------------------------------------------------


def test_adherence_empty_contract_passes():
    scorer = StdlibAdherenceScorer()
    score = scorer.score(response="anything goes", contract={})
    assert score.label == "pass"
    assert score.normalized == 1.0
    assert score.features["clauses_total"] == 0


def test_adherence_requires_response():
    scorer = StdlibAdherenceScorer()
    with pytest.raises(ValueError):
        scorer.score(contract={"length_min": 5})


def test_adherence_required_substring_hit_and_miss():
    scorer = StdlibAdherenceScorer()
    contract = {
        "required_substrings": [
            "numerator",
            "denominator",
            "hba1c",
        ]
    }
    good = scorer.score(
        response="numerator=12, denominator=100, hba1c=8.5",
        contract=contract,
    )
    assert good.label == "pass"
    assert good.normalized == 1.0

    bad = scorer.score(
        response="numerator=12 only",
        contract=contract,
    )
    # 1 of 3 satisfied -> 0.333... < 0.5 threshold -> fail
    assert bad.label == "fail"
    assert bad.normalized == pytest.approx(1 / 3)
    assert "required:denominator" in bad.features["violations"]
    assert "required:hba1c" in bad.features["violations"]


def test_adherence_required_substring_is_case_insensitive():
    scorer = StdlibAdherenceScorer()
    score = scorer.score(
        response="DENOMINATOR is 100",
        contract={"required_substrings": ["denominator"]},
    )
    assert score.label == "pass"


def test_adherence_forbidden_substring_violation():
    scorer = StdlibAdherenceScorer()
    score = scorer.score(
        response="I cannot help with that request.",
        contract={"forbidden_substrings": ["i cannot"]},
    )
    assert score.label == "fail"
    assert "forbidden:i cannot" in score.features["violations"]


def test_adherence_length_bounds():
    scorer = StdlibAdherenceScorer()
    contract = {
        "required_substrings": ["numerator"],
        "length_min": 50,
        "length_max": 100,
    }
    short = scorer.score(response="hi", contract=contract)
    # 0 of 3 clauses satisfied -> 0.0 < 0.5 -> fail
    assert short.label == "fail"
    assert "length_min:50" in short.features["violations"]

    long_resp = scorer.score(
        response="numerator=" + "x" * 200, contract=contract
    )
    # 2 of 3 satisfied (required+length_min ok, length_max fails)
    # -> 0.667 >= 0.5 -> pass, but with one violation flagged.
    assert long_resp.normalized == pytest.approx(2 / 3)
    assert "length_max:100" in long_resp.features["violations"]

    ok = scorer.score(
        response="numerator=" + "x" * 60, contract=contract
    )
    assert ok.label == "pass"
    assert ok.normalized == 1.0


def test_adherence_json_required_satisfied_and_violated():
    scorer = StdlibAdherenceScorer()
    good = scorer.score(
        response='{"numerator": 42}',
        contract={"json_required": True},
    )
    assert good.label == "pass"

    bad = scorer.score(
        response="not json at all",
        contract={"json_required": True},
    )
    assert bad.label == "fail"
    assert "json_required" in bad.features["violations"]


def test_adherence_mixed_contract_partial_credit():
    scorer = StdlibAdherenceScorer()
    # 4 clauses: 2 required (both present), length_min (fail),
    # forbidden (pass). 3/4 satisfied -> probability 0.75.
    score = scorer.score(
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
# StdlibCompletionScorer
# ---------------------------------------------------------------------------


def test_completion_no_targets_passes():
    scorer = StdlibCompletionScorer()
    score = scorer.score(response="any response")
    assert score.label == "pass"
    assert score.normalized == 1.0
    assert score.features["total_targets"] == 0


def test_completion_requires_response():
    scorer = StdlibCompletionScorer()
    with pytest.raises(ValueError):
        scorer.score(expected_tokens=["hba1c"])


def test_completion_exact_unigram_match_and_miss():
    scorer = StdlibCompletionScorer()
    score = scorer.score(
        response="The numerator is 12 and the denominator is 100",
        expected_tokens=["numerator", "denominator", "hba1c"],
    )
    assert score.normalized == pytest.approx(2 / 3)
    assert score.features["hits"] == 2
    assert score.features["misses"] == ["hba1c"]


def test_completion_case_insensitive():
    scorer = StdlibCompletionScorer()
    score = scorer.score(
        response="NUMERATOR is 12",
        expected_tokens=["numerator"],
    )
    assert score.normalized == 1.0
    assert score.label == "pass"


def test_completion_multi_word_phrase_match():
    scorer = StdlibCompletionScorer()
    score = scorer.score(
        response="Most recent blood pressure is 138/88",
        expected_tokens=["blood pressure"],
    )
    assert score.normalized == 1.0


def test_completion_contract_fallback_for_targets():
    scorer = StdlibCompletionScorer()
    score = scorer.score(
        response="numerator=12",
        contract={"completion_tokens": ["numerator", "missing"]},
    )
    assert score.normalized == pytest.approx(0.5)
    assert score.label == "pass"


def test_completion_handles_empty_or_whitespace_targets():
    scorer = StdlibCompletionScorer()
    score = scorer.score(
        response="numerator=12",
        expected_tokens=["", "   ", "numerator"],
    )
    # Only "numerator" counts.
    assert score.features["total_targets"] == 1
    assert score.normalized == 1.0


# ---------------------------------------------------------------------------
# build_scorers factory routing
# ---------------------------------------------------------------------------


def test_build_scorers_tier_stdlib_returns_stdlib_scorers():
    cfg = ScoreRuntimeConfig(tier="stdlib", stdlib=StdlibScoreConfig())
    intent, adherence, completion = build_scorers(cfg)
    assert isinstance(intent, StdlibIntentScorer)
    assert isinstance(adherence, StdlibAdherenceScorer)
    assert isinstance(completion, StdlibCompletionScorer)


def test_build_scorers_tier_stdlib_uses_config_threshold(tmp_path):
    cfg = ScoreRuntimeConfig(
        tier="stdlib",
        stdlib=StdlibScoreConfig(
            snapshot_dir=str(tmp_path),
            pass_threshold=0.7,
            feature_dim=256,
        ),
    )
    intent, adherence, completion = build_scorers(cfg)
    assert intent.pass_threshold == 0.7
    assert intent.feature_dim == 256
    assert adherence.pass_threshold == 0.7
    assert completion.pass_threshold == 0.7


def test_build_scorers_tier_slm_returns_slm_scorers():
    """Tier 3 builds the three SLM scorers without loading the real model.

    The scorers defer model load to first ``score()`` call via
    ``SlmRunner.get_shared``, so construction alone is safe even when
    ``onnxruntime-genai`` is not installed.
    """
    from agent_learning.scorers.slm import (
        SlmAdherenceScorer,
        SlmCompletionScorer,
        SlmIntentScorer,
    )
    cfg = ScoreRuntimeConfig(tier="slm")
    intent, adherence, completion = build_scorers(cfg)
    assert isinstance(intent, SlmIntentScorer)
    assert isinstance(adherence, SlmAdherenceScorer)
    assert isinstance(completion, SlmCompletionScorer)


def test_build_scorers_unknown_tier_raises():
    cfg = ScoreRuntimeConfig()
    cfg.tier = "bogus"  # type: ignore[assignment]
    with pytest.raises(ValueError):
        build_scorers(cfg)


def test_build_scorers_legacy_mode_when_tier_none():
    # tier=None falls through to mode; mode="nlp" should resolve to the
    # existing phi+action_id BinaryScorer stack.
    cfg = ScoreRuntimeConfig(mode="nlp", tier=None)
    intent, adherence, completion = build_scorers(cfg)
    # We don't import the NLP classes here to avoid a cycle; just check
    # the scorers expose the structural ScoreResult contract.
    assert intent.name == "intent"
    assert adherence.name == "adherence"
    assert completion.name == "completion"


def test_env_var_tier_resolution(monkeypatch):
    monkeypatch.setenv("AGENT_LEARNING_SCORE_TIER", "stdlib")
    cfg = ScoreRuntimeConfig()
    assert cfg.tier == "stdlib"


def test_env_var_tier_invalid_falls_back_to_none(monkeypatch):
    monkeypatch.setenv("AGENT_LEARNING_SCORE_TIER", "bogus")
    cfg = ScoreRuntimeConfig()
    assert cfg.tier is None
