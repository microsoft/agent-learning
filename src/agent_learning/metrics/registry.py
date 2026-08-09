"""Convenience helpers to evaluate an episode across all default metrics."""

from __future__ import annotations

from typing import Iterable, List, Optional

from ..config import ScoreConfig, ScoreRuntimeConfig
from ..scorers import build_scorers
from ..types import Episode, MetricResult
from .base import MetricEvaluator, MetricRequest
from .intent_resolution import IntentResolutionMetric
from .local import local_metrics
from .task_adherence import TaskAdherenceMetric
from .task_completion import TaskCompletionMetric


def default_metrics(
    score_config: Optional[ScoreConfig] = None,
    score_runtime_config: Optional[ScoreRuntimeConfig] = None,
) -> List[MetricEvaluator]:
    """Return local metrics by default, or Azure metrics when configured."""
    runtime = score_runtime_config or ScoreRuntimeConfig()
    llm = score_config or runtime.llm
    if score_config is not None or runtime.tier == "llm" or (
        runtime.tier is None and llm.enabled
    ):
        return [
            IntentResolutionMetric(llm),
            TaskAdherenceMetric(llm),
            TaskCompletionMetric(llm),
        ]
    if runtime.tier is None:
        runtime.tier = "stdlib"
    return local_metrics(build_scorers(runtime))


def evaluate_all(
    episode: Episode,
    metrics: Optional[Iterable[MetricEvaluator]] = None,
    *,
    score_config: Optional[ScoreConfig] = None,
    score_runtime_config: Optional[ScoreRuntimeConfig] = None,
) -> List[MetricResult]:
    """Evaluate one episode against every supplied metric.

    Each metric is given its own try/except inside :meth:`evaluate`,
    so a single failing scorer will not prevent the others from
    producing scores.
    """
    metric_list = (
        list(metrics)
        if metrics is not None
        else default_metrics(score_config, score_runtime_config)
    )
    request = MetricRequest.from_episode(episode)
    return [m.evaluate(request) for m in metric_list]


__all__ = ["default_metrics", "evaluate_all"]
