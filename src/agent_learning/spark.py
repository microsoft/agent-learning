"""PySpark adapters for distributed episode scoring.

Scoring is distributed across Spark partitions. Policy updates and persistence
remain driver responsibilities so executors do not depend on process-local
stores or SDK singletons.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .config import ShapingConfig
from .metrics.base import MetricEvaluator
from .metrics.registry import default_metrics, evaluate_all
from .rewards.shaping import shape_episode_reward
from .types import Episode

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

MetricFactory = Callable[[], Iterable[MetricEvaluator]]


def _default_metric_factory() -> Iterable[MetricEvaluator]:
    return default_metrics()


@dataclass(frozen=True)
class SparkEpisodeScorer:
    """Score episode-shaped Spark rows in parallel.

    Each input row must contain an ``episode_json`` column with an
    :meth:`Episode.to_dict` payload. JSON preserves arbitrary nested episode
    metadata that Spark SQL cannot represent with a stable inferred schema.
    Evaluators are created once per partition.
    The reward is null when no metric contributes and no penalty applies,
    distinguishing missing evidence from a valid zero reward.
    """

    metric_factory: MetricFactory = _default_metric_factory
    shaping_config: ShapingConfig | None = None
    episode_column: str = "episode_json"

    def transform(self, episodes: DataFrame) -> DataFrame:
        """Return one scored row per input episode without persisting results."""
        try:
            from pyspark.sql.types import (
                DoubleType,
                StringType,
                StructField,
                StructType,
            )
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "PySpark support requires the 'spark' extra: "
                "pip install 'agent-learning[spark]'"
            ) from exc

        schema = StructType(
            [
                StructField("episode_id", StringType(), nullable=False),
                StructField("agent_id", StringType(), nullable=False),
                StructField("reward", DoubleType(), nullable=True),
                StructField("metrics_json", StringType(), nullable=False),
            ]
        )
        scored = episodes.rdd.mapPartitions(self._score_partition)
        return episodes.sparkSession.createDataFrame(scored, schema=schema)

    def _score_partition(
        self, rows: Iterator[Any]
    ) -> Iterator[tuple[str, str, float | None, str]]:
        metrics = list(self.metric_factory())
        for row in rows:
            data = _episode_from_row(row, self.episode_column)
            episode = Episode.from_dict(data)
            results = evaluate_all(episode, metrics)
            shaped = shape_episode_reward(episode, results, self.shaping_config)
            yield (
                episode.id,
                episode.agent_id,
                shaped.value if shaped.metric_contributions or shaped.penalties else None,
                json.dumps(
                    [result.to_dict() for result in results],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )


def score_episode_dataframe(
    episodes: DataFrame,
    *,
    metric_factory: MetricFactory = _default_metric_factory,
    shaping_config: ShapingConfig | None = None,
    episode_column: str = "episode_json",
) -> DataFrame:
    """Score an episode DataFrame using one evaluator set per partition."""
    return SparkEpisodeScorer(metric_factory, shaping_config, episode_column).transform(episodes)


def _episode_from_row(row: Any, episode_column: str) -> dict[str, Any]:
    if hasattr(row, "asDict"):
        row = row.asDict(recursive=True)
    if not isinstance(row, Mapping):
        raise TypeError("Spark episode rows must be Row or mapping instances")
    payload = row.get(episode_column)
    if isinstance(payload, str):
        data = json.loads(payload)
        if isinstance(data, dict):
            return data
    if isinstance(payload, Mapping):
        return dict(payload)
    raise ValueError(
        f"Spark row column {episode_column!r} must contain episode JSON or a mapping"
    )


__all__ = ["MetricFactory", "SparkEpisodeScorer", "score_episode_dataframe"]
