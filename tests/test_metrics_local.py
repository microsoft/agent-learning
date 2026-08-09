"""Tests for default on-device metric routing."""

from __future__ import annotations

from agent_learning.config import ScoreConfig
from agent_learning.metrics import (
    IntentResolutionMetric,
    LocalScorerMetric,
    default_metrics,
    evaluate_all,
)
from agent_learning.types import Episode, MetricName


def test_default_metrics_are_local_without_azure_configuration(monkeypatch) -> None:
    for name in (
        "AGENT_LEARNING_SCORE_ENDPOINT",
        "AGENT_LEARNING_SCORE_DEPLOYMENT",
        "AGENT_LEARNING_SCORE_TIER",
    ):
        monkeypatch.delenv(name, raising=False)

    metrics = default_metrics()

    assert len(metrics) == 3
    assert all(isinstance(metric, LocalScorerMetric) for metric in metrics)


def test_explicit_score_config_preserves_azure_metrics() -> None:
    metrics = default_metrics(
        ScoreConfig(
            azure_endpoint="https://example.openai.azure.com",
            azure_deployment="grader",
            credential_mode="none",
        )
    )

    assert isinstance(metrics[0], IntentResolutionMetric)


def test_correct_action_id_overrides_contradictory_completion_flag(
    monkeypatch,
) -> None:
    monkeypatch.delenv("AGENT_LEARNING_SCORE_ENDPOINT", raising=False)
    monkeypatch.delenv("AGENT_LEARNING_SCORE_DEPLOYMENT", raising=False)
    episode = Episode(
        user_input="Answer the task",
        assistant_output="Used the wrong action",
        action_id="wrong",
        execution_status="completed",
        metadata={"correct_action_id": "right", "task_completed": True},
    )

    results = evaluate_all(episode)

    completion = next(
        result for result in results if result.metric == MetricName.TASK_COMPLETION
    )
    assert completion.normalized == 0.0
    assert completion.reason == "derived from action_id and metadata.correct_action_id"