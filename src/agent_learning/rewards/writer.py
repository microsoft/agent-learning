"""Persist metric results and aggregate rewards for an episode."""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional

from ..storage.base import LearningStore
from ..storage.cosmos import get_default_store
from ..types import Episode, MetricName, MetricResult, Reward, RewardSource
from .shaping import ShapedReward

logger = logging.getLogger(__name__)


class RewardWriter:
    """Translate metric results into stored :class:`Reward` records.

    The writer stores one ``Reward`` per metric (``source=METRIC``)
    plus a single aggregate reward (``source=AGGREGATE``) that the
    learner consumes. Storing per-metric values keeps the audit trail
    complete and lets downstream analytics decompose the aggregate.
    """

    def __init__(self, store: Optional[LearningStore] = None) -> None:
        self._store = store

    @property
    def store(self) -> LearningStore:
        if self._store is None:
            self._store = get_default_store()
        return self._store

    def write(
        self,
        episode: Episode,
        results: Iterable[MetricResult],
        shaped: ShapedReward,
        *,
        rubric: Optional[str] = None,
    ) -> List[Reward]:
        """Persist metric and aggregate rewards. Returns the stored rows."""
        stored: List[Reward] = []
        results_list = list(results)

        # 1) Persist the raw metric results themselves (richer than rewards)
        try:
            self.store.store_metric_results(episode.id, episode.agent_id, results_list)
        except Exception as exc:  # pragma: no cover - persistence error path
            logger.warning("Failed to persist metric results for %s: %s", episode.id, exc)

        # 2) Per-metric reward records (signed contributions)
        contributions = {m: signed for m, _w, signed, _c in shaped.metric_contributions}
        for result in results_list:
            if result.status != "completed" or result.normalized is None:
                continue
            signed_value = contributions.get(
                result.metric,
                2.0 * result.normalized - 1.0,
            )
            reward = Reward(
                episode_id=episode.id,
                agent_id=episode.agent_id,
                source=RewardSource.METRIC,
                value=max(-1.0, min(1.0, signed_value)),
                raw_value=result.score,
                metric=result.metric,
                rubric=rubric or result.metric.value,
                evaluator=result.evaluator,
                metadata={"reason": result.reason} if result.reason else {},
            )
            try:
                self.store.store_reward(reward)
                stored.append(reward)
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "Failed to persist %s reward for %s: %s",
                    result.metric.value,
                    episode.id,
                    exc,
                )

        # 3) Latency / cost penalty rewards (kept separate for analytics)
        for kind, raw_value, value in shaped.penalties:
            reward = Reward(
                episode_id=episode.id,
                agent_id=episode.agent_id,
                source=RewardSource.LATENCY_PENALTY
                if kind == "latency"
                else RewardSource.COST_PENALTY,
                value=max(-1.0, min(1.0, value)),
                raw_value=raw_value,
                rubric=kind,
            )
            try:
                self.store.store_reward(reward)
                stored.append(reward)
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to persist %s penalty for %s: %s", kind, episode.id, exc)

        # 4) Aggregate reward consumed by the learner
        aggregate = Reward(
            episode_id=episode.id,
            agent_id=episode.agent_id,
            source=RewardSource.AGGREGATE,
            value=shaped.value,
            rubric=rubric or "aggregate",
            metadata={
                "metric_contributions": [
                    {
                        "metric": m.value if isinstance(m, MetricName) else str(m),
                        "weight": w,
                        "signed": signed,
                        "contribution": c,
                    }
                    for m, w, signed, c in shaped.metric_contributions
                ],
                "penalties": [{"kind": k, "raw": r, "value": v} for k, r, v in shaped.penalties],
            },
        )
        try:
            self.store.store_reward(aggregate)
            stored.append(aggregate)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to persist aggregate reward for %s: %s", episode.id, exc)

        return stored


__all__ = ["RewardWriter"]
