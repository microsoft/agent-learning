"""Convenience helpers to evaluate an episode across all default metrics."""

from __future__ import annotations

from typing import Iterable, List, Optional

from ..config import ScoreConfig
from ..types import Episode, MetricResult
from .base import MetricEvaluator, MetricRequest
from .intent_resolution import IntentResolutionMetric
from .task_adherence import TaskAdherenceMetric
from .task_completion import TaskCompletionMetric


def default_metrics(score_config: Optional[ScoreConfig] = None) -> List[MetricEvaluator]:
    """Return the three native metrics wired with the same score config."""
    cfg = score_config or ScoreConfig()
    return [
        IntentResolutionMetric(cfg),
        TaskAdherenceMetric(cfg),
        TaskCompletionMetric(cfg),
    ]


def evaluate_all(
    episode: Episode,
    metrics: Optional[Iterable[MetricEvaluator]] = None,
    *,
    score_config: Optional[ScoreConfig] = None,
) -> List[MetricResult]:
    """Evaluate one episode against every supplied metric.

    Each metric is given its own try/except inside :meth:`evaluate`,
    so a single failing scorer will not prevent the others from
    producing scores.
    """
    metric_list = list(metrics) if metrics is not None else default_metrics(score_config)
    request = MetricRequest.from_episode(episode)
    return [m.evaluate(request) for m in metric_list]


__all__ = ["default_metrics", "evaluate_all"]
