"""Reward shaping: combine multiple metric results into one scalar reward.

The shaper turns each metric's normalised score (in ``[0, 1]``) into
a signed contribution (in ``[-1, 1]``) and sums them with the
configured weights. Additional behavioural penalties (latency, cost)
can be applied after the metric sum. The final reward is clamped to
``[-1, 1]`` so it is always safe to feed into a policy-gradient
update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from ..config import ShapingConfig
from ..types import Episode, MetricName, MetricResult


@dataclass
class ShapedReward:
    """Result of shaping a metric collection into a scalar reward.

    ``metric_contributions`` exposes the signed contribution of every
    metric individually so callers can persist per-metric rewards
    alongside the aggregate.
    """

    value: float
    metric_contributions: List[tuple]
    penalties: List[tuple]


class RewardShaper:
    """Combine multiple :class:`MetricResult` into a scalar in ``[-1, 1]``."""

    def __init__(self, config: Optional[ShapingConfig] = None) -> None:
        self._config = config or ShapingConfig()

    @property
    def config(self) -> ShapingConfig:
        return self._config

    def shape(
        self,
        results: Iterable[MetricResult],
        *,
        latency_ms: Optional[int] = None,
        routing_correct: Optional[bool] = None,
        hallucinated_class: bool = False,
    ) -> ShapedReward:
        """Combine scores + behavioural signals into a single scalar.

        Args:
            results: Per-score metric outputs.
            latency_ms: Episode latency in milliseconds; if above the
                configured threshold the latency penalty is applied.
            routing_correct: ``True`` if the calling system routed the
                request to a class id in the allowed set
                (→ add ``route_correct_reward``); ``False`` if it
                picked a class outside the allowed set (→ add
                ``route_wrong_penalty``, which is negative). ``None``
                disables the term entirely (no signal either way).
            hallucinated_class: ``True`` if the rendered output
                references an entity/class id not in the allowed set
                (→ add ``hallucinated_class_penalty``).
        """
        weights = {
            MetricName.INTENT_RESOLUTION: self._config.intent_resolution_weight,
            MetricName.TASK_ADHERENCE: self._config.task_adherence_weight,
            MetricName.TASK_COMPLETION: self._config.task_completion_weight,
        }

        contributions: List[tuple] = []
        total = 0.0
        for r in results:
            if r.normalized is None or r.status != "completed":
                continue
            weight = weights.get(r.metric, 0.0)
            # Map [0, 1] -> [-1, 1]
            signed = 2.0 * r.normalized - 1.0
            contribution = weight * signed
            total += contribution
            contributions.append((r.metric, weight, signed, contribution))

        penalties: List[tuple] = []
        if (
            latency_ms is not None
            and latency_ms > self._config.latency_penalty_threshold_ms
        ):
            total += self._config.latency_penalty_value
            penalties.append(("latency", latency_ms, self._config.latency_penalty_value))

        if routing_correct is True:
            total += self._config.route_correct_reward
            penalties.append(("route_correct", True, self._config.route_correct_reward))
        elif routing_correct is False:
            total += self._config.route_wrong_penalty
            penalties.append(("route_wrong", False, self._config.route_wrong_penalty))

        if hallucinated_class:
            total += self._config.hallucinated_class_penalty
            penalties.append(
                ("hallucinated_class", True, self._config.hallucinated_class_penalty)
            )

        clamped = max(-1.0, min(1.0, total))
        return ShapedReward(value=clamped, metric_contributions=contributions, penalties=penalties)


def shape_episode_reward(
    episode: Episode,
    results: Iterable[MetricResult],
    config: Optional[ShapingConfig] = None,
) -> ShapedReward:
    """Convenience wrapper that pulls latency from the episode automatically."""
    shaper = RewardShaper(config)
    return shaper.shape(results, latency_ms=episode.request_latency_ms)


__all__ = ["RewardShaper", "ShapedReward", "shape_episode_reward"]
