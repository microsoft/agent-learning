"""PySpark integration tests."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

import pytest

pyspark = pytest.importorskip("pyspark")

from pyspark.sql import SparkSession

from agent_learning import Episode, score_episode_dataframe


@pytest.fixture(scope="module")
def spark() -> Iterator[SparkSession]:
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")
    session = (
        SparkSession.builder.master("local[1]")
        .appName("agent-learning-tests")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def test_score_episode_dataframe(spark: SparkSession) -> None:
    episode = Episode(
        id="spark-episode",
        agent_id="spark-agent",
        user_input="Answer the task",
        assistant_output="Task completed",
        action_id="answer",
        execution_status="completed",
        metadata={"correct_action_id": "answer", "task_completed": True},
    )
    episodes = spark.createDataFrame(
        [(json.dumps(episode.to_dict()),)],
        ["episode_json"],
    )

    scored = score_episode_dataframe(episodes).collect()

    assert len(scored) == 1
    assert scored[0].episode_id == episode.id


@pytest.mark.parametrize("latency, expected", [(None, None), (101, -0.25)])
def test_skipped_metrics_preserve_missing_rewards_and_penalties(spark, latency, expected):
    from agent_learning import ShapingConfig

    episode = Episode(id="skipped", request_latency_ms=latency)
    source = spark.createDataFrame([(json.dumps(episode.to_dict()),)], ["episode_json"])
    scored = score_episode_dataframe(
        source,
        shaping_config=ShapingConfig(
            latency_penalty_threshold_ms=100, latency_penalty_value=-0.25
        ),
    )

    assert scored.schema["reward"].nullable
    result = scored.collect()[0]
    assert result.reward == expected
    assert all(metric["status"] == "skipped" for metric in json.loads(result.metrics_json))


def test_failed_evaluator_does_not_emit_neutral_reward(spark):
    def metric_factory():
        from agent_learning import IntentResolutionMetric

        def fail(**kwargs):
            raise RuntimeError("Test evaluator failure")

        return [IntentResolutionMetric(evaluator=fail)]

    episode = Episode(id="failed", user_input="Question", assistant_output="Answer")
    source = spark.createDataFrame([(json.dumps(episode.to_dict()),)], ["episode_json"])
    result = score_episode_dataframe(source, metric_factory=metric_factory).collect()[0]

    assert result.reward is None
    assert json.loads(result.metrics_json)[0]["status"] == "skipped"


def test_completed_metrics_can_emit_valid_zero_reward(spark):
    from agent_learning import ShapingConfig

    episode = Episode(
        id="zero", user_input="Question", assistant_output="Answer",
        action_id="answer", metadata={"correct_action_id": "answer"},
    )
    source = spark.createDataFrame([(json.dumps(episode.to_dict()),)], ["episode_json"])
    result = score_episode_dataframe(
        source,
        shaping_config=ShapingConfig(
            intent_resolution_weight=0.0,
            task_adherence_weight=0.0,
            task_completion_weight=0.0,
        ),
    ).collect()[0]

    assert result.reward == 0.0
    assert any(metric["status"] == "completed" for metric in json.loads(result.metrics_json))
