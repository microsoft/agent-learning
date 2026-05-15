"""Tests for the reward shaper."""

from __future__ import annotations

from agent_learning.config import ShapingConfig
from agent_learning.rewards.shaping import RewardShaper
from agent_learning.types import MetricName, MetricResult


def _result(metric: MetricName, normalized: float) -> MetricResult:
    return MetricResult(
        metric=metric,
        score=normalized,
        normalized=normalized,
        status="completed",
    )


def test_all_perfect_scores_yield_max_reward() -> None:
    cfg = ShapingConfig(
        intent_resolution_weight=0.5,
        task_adherence_weight=0.3,
        task_completion_weight=0.2,
    )
    shaper = RewardShaper(cfg)
    shaped = shaper.shape(
        [
            _result(MetricName.INTENT_RESOLUTION, 1.0),
            _result(MetricName.TASK_ADHERENCE, 1.0),
            _result(MetricName.TASK_COMPLETION, 1.0),
        ]
    )
    # Each metric maps 1.0 -> +1 signed; weighted sum = 0.5+0.3+0.2 = 1.0
    assert abs(shaped.value - 1.0) < 1e-9
    assert len(shaped.metric_contributions) == 3


def test_all_zero_scores_yield_minimum_reward() -> None:
    cfg = ShapingConfig(
        intent_resolution_weight=0.5,
        task_adherence_weight=0.3,
        task_completion_weight=0.2,
    )
    shaper = RewardShaper(cfg)
    shaped = shaper.shape(
        [
            _result(MetricName.INTENT_RESOLUTION, 0.0),
            _result(MetricName.TASK_ADHERENCE, 0.0),
            _result(MetricName.TASK_COMPLETION, 0.0),
        ]
    )
    assert abs(shaped.value + 1.0) < 1e-9


def test_skipped_metrics_are_ignored() -> None:
    cfg = ShapingConfig(
        intent_resolution_weight=0.5,
        task_adherence_weight=0.3,
        task_completion_weight=0.2,
    )
    shaper = RewardShaper(cfg)
    skipped = MetricResult(metric=MetricName.TASK_ADHERENCE, score=None, normalized=None, status="skipped")
    shaped = shaper.shape(
        [
            _result(MetricName.INTENT_RESOLUTION, 1.0),
            skipped,
            _result(MetricName.TASK_COMPLETION, 1.0),
        ]
    )
    # Only intent (0.5) + completion (0.2) contribute. signed = 2*1-1 = 1
    assert abs(shaped.value - 0.7) < 1e-9


def test_latency_penalty_applied() -> None:
    cfg = ShapingConfig(
        intent_resolution_weight=0.0,
        task_adherence_weight=0.0,
        task_completion_weight=0.0,
        latency_penalty_threshold_ms=1000,
        latency_penalty_value=-0.2,
    )
    shaper = RewardShaper(cfg)
    shaped = shaper.shape([], latency_ms=2000)
    assert abs(shaped.value + 0.2) < 1e-9
    assert shaped.penalties[0][0] == "latency"
