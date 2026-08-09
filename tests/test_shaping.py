"""Tests for the reward shaper."""

from __future__ import annotations

from agent_learning.config import ShapingConfig
from agent_learning.rewards import RewardWriter
from agent_learning.rewards.shaping import RewardShaper
from agent_learning.storage import InMemoryStore
from agent_learning.types import Episode, MetricName, MetricResult, RewardSource


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


def test_all_skipped_metrics_do_not_persist_aggregate() -> None:
    store = InMemoryStore()
    episode = Episode(agent_id="scout")
    results = [
        MetricResult(
            metric=metric,
            score=None,
            normalized=None,
            status="skipped",
        )
        for metric in MetricName
    ]
    shaped = RewardShaper().shape(results)

    rewards = RewardWriter(store).write(episode, results, shaped)

    assert not any(reward.source == RewardSource.AGGREGATE for reward in rewards)
    assert store.get_rewards_for_episode(episode.id, episode.agent_id) == []


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


# ---------------------------------------------------------------------------
# Measure-routing reward + hallucinated-member penalty (\u00a75.2 of design doc)
# ---------------------------------------------------------------------------


def _zero_weighted_cfg(**overrides) -> ShapingConfig:
    return ShapingConfig(
        intent_resolution_weight=0.0,
        task_adherence_weight=0.0,
        task_completion_weight=0.0,
        latency_penalty_threshold_ms=10_000_000,
        latency_penalty_value=0.0,
        **overrides,
    )


def test_routing_correct_adds_positive_reward() -> None:
    cfg = _zero_weighted_cfg(route_correct_reward=0.2, route_wrong_penalty=-0.3)
    shaper = RewardShaper(cfg)
    shaped = shaper.shape([], routing_correct=True)
    assert abs(shaped.value - 0.2) < 1e-9
    assert any(p[0] == "route_correct" for p in shaped.penalties)


def test_routing_wrong_adds_negative_penalty() -> None:
    cfg = _zero_weighted_cfg(route_correct_reward=0.2, route_wrong_penalty=-0.3)
    shaper = RewardShaper(cfg)
    shaped = shaper.shape([], routing_correct=False)
    assert abs(shaped.value + 0.3) < 1e-9
    assert any(p[0] == "route_wrong" for p in shaped.penalties)


def test_routing_none_is_neutral() -> None:
    cfg = _zero_weighted_cfg(route_correct_reward=0.2, route_wrong_penalty=-0.3)
    shaper = RewardShaper(cfg)
    shaped = shaper.shape([], routing_correct=None)
    assert abs(shaped.value) < 1e-9
    assert all(p[0] not in ("route_correct", "route_wrong") for p in shaped.penalties)


def test_hallucinated_class_adds_penalty() -> None:
    cfg = _zero_weighted_cfg(hallucinated_class_penalty=-0.25)
    shaper = RewardShaper(cfg)
    shaped = shaper.shape([], hallucinated_class=True)
    assert abs(shaped.value + 0.25) < 1e-9
    assert any(p[0] == "hallucinated_class" for p in shaped.penalties)


def test_reward_is_clamped_to_unit_range() -> None:
    cfg = ShapingConfig(
        intent_resolution_weight=0.10,
        task_adherence_weight=0.20,
        task_completion_weight=0.50,
        route_correct_reward=0.20,
        route_wrong_penalty=-0.30,
        hallucinated_class_penalty=-0.25,
        latency_penalty_threshold_ms=10_000_000,
        latency_penalty_value=0.0,
    )
    shaper = RewardShaper(cfg)
    # Push everything negative simultaneously \u2192 raw sum < -1, must clamp to -1.
    shaped = shaper.shape(
        [
            _result(MetricName.INTENT_RESOLUTION, 0.0),
            _result(MetricName.TASK_ADHERENCE, 0.0),
            _result(MetricName.TASK_COMPLETION, 0.0),
        ],
        routing_correct=False,
        hallucinated_class=True,
    )
    assert shaped.value == -1.0
