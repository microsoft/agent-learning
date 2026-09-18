"""Round-trip tests for the durable record types."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from agent_learning import ConsumedInput
from agent_learning.learners import LearnerResult
from agent_learning.types import (
    Action,
    AgentTaskSummary,
    Episode,
    MetricName,
    MetricResult,
    PolicySnapshot,
    Reward,
    RewardSource,
    ToolCall,
    TrainingRun,
    TrainingStatus,
)


def test_episode_roundtrip() -> None:
    ep = Episode(
        agent_id="dq",
        task_id="summary",
        user_input="hello",
        assistant_output="hi there",
        tool_calls=[ToolCall(name="lookup", arguments={"q": "x"}, result="ok")],
        action_id="prompt_A",
        action_logprob=-0.5,
        context_features={"intent": "greet"},
    )
    other = Episode.from_dict(ep.to_dict())
    assert other.id == ep.id
    assert other.agent_id == "dq"
    assert other.task_id == "summary"
    assert other.tool_calls[0].name == "lookup"
    assert other.action_id == "prompt_A"
    assert other.action_logprob == -0.5


def test_episode_full_state() -> None:
    full = Episode(
        intent_summary="Complete the task",
        action_id="respond",
        expected_outcome="The task is complete",
        execution_status="completed",
        result_summary="Completed successfully",
    )
    assert full.is_full is True
    full.result_summary = None
    assert full.is_full is False


def test_reward_roundtrip() -> None:
    r = Reward(
        episode_id="ep1",
        agent_id="dq",
        source=RewardSource.METRIC,
        value=0.5,
        metric=MetricName.INTENT_RESOLUTION,
    )
    other = Reward.from_dict(r.to_dict())
    assert other.source == RewardSource.METRIC
    assert other.metric == MetricName.INTENT_RESOLUTION
    assert other.value == 0.5


def test_metric_result_roundtrip() -> None:
    m = MetricResult(
        metric=MetricName.TASK_ADHERENCE,
        score=1.0,
        normalized=1.0,
        status="completed",
        reason="all good",
    )
    other = MetricResult.from_dict(m.to_dict())
    assert other.metric == MetricName.TASK_ADHERENCE
    assert other.normalized == 1.0


def test_policy_snapshot_roundtrip() -> None:
    snap = PolicySnapshot(
        agent_id="dq",
        task_id="summary",
        version=2,
        actions=[Action(id="a"), Action(id="b")],
        logits={"a": 0.5, "b": -0.5},
        baseline=0.1,
        episodes_seen=10,
        updates_applied=1,
    )
    other = PolicySnapshot.from_dict(snap.to_dict())
    assert other.task_id == "summary"
    assert other.version == 2
    assert other.logits == {"a": 0.5, "b": -0.5}
    assert other.actions[0].id == "a"


def test_training_run_roundtrip() -> None:
    run = TrainingRun(
        agent_id="dq",
        task_id="summary",
        policy_id="p1",
        algorithm="reinforce",
        status=TrainingStatus.RUNNING,
        episode_ids=["ep1", "ep2"],
        hyperparameters={"lr": 0.1},
    )
    other = TrainingRun.from_dict(run.to_dict())
    assert other.task_id == "summary"
    assert other.status == TrainingStatus.RUNNING
    assert other.hyperparameters["lr"] == 0.1
    assert other.episode_ids == ["ep1", "ep2"]


def test_agent_task_summary_uses_task_name() -> None:
    task = AgentTaskSummary(id="weekly-summary", name="Weekly summary")
    assert task.id == "weekly-summary"
    assert task.name == "Weekly summary"


@pytest.mark.parametrize("field_name", ["episode_id", "reward_id"])
def test_consumed_input_roundtrip_and_immutability(field_name: str) -> None:
    item = ConsumedInput("ep1", "reward1")
    payload = json.loads(json.dumps(item.to_dict()))

    assert payload == {"episode_id": "ep1", "reward_id": "reward1"}
    assert ConsumedInput.from_dict(payload) == item
    with pytest.raises(FrozenInstanceError):
        setattr(item, field_name, "replacement")


@pytest.mark.parametrize(
    "receipt",
    [None, [], [ConsumedInput("ep2", "r2"), ConsumedInput("ep1", "r1"), ConsumedInput("ep2", "r2")]],
)
def test_training_run_lineage_roundtrip(receipt: list[ConsumedInput] | None) -> None:
    run = TrainingRun(
        policy_id="before",
        output_policy_id="after",
        episode_ids=["ep2", "skipped", "ep1", "ep2"],
        consumed_inputs=receipt,
        hyperparameters={"learner": {"learning_rate": 0.4}, "policy": {"max_logit_abs": 3.0}},
        metadata={"lineage_version": 1, "replay_implementation": "reinforce_softmax_v1"},
    )
    reloaded = TrainingRun.from_dict(json.loads(json.dumps(run.to_dict())))

    assert reloaded == run
    assert reloaded.consumed_inputs == receipt


def test_legacy_training_run_has_unavailable_lineage() -> None:
    payload = {
        "id": "old-run",
        "agent_id": "agent",
        "policy_id": "input",
        "status": "succeeded",
        "episode_ids": ["ep1"],
        "metadata": {"policy_version": 1},
    }
    run = TrainingRun.from_dict(payload)

    assert run.policy_id == "input"
    assert run.episode_ids == ["ep1"]
    assert run.output_policy_id is None
    assert run.consumed_inputs is None
    assert run.hyperparameters == {}
    assert "lineage_version" not in run.metadata


@pytest.mark.parametrize("receipt", [{}, "", False, 0])
def test_invalid_receipt_shape_is_not_treated_as_empty(receipt: object) -> None:
    with pytest.raises(TypeError, match="consumed_inputs must be a JSON array or null"):
        TrainingRun.from_dict({"id": "invalid-run", "consumed_inputs": receipt})


def test_lineage_fields_preserve_existing_positional_arguments() -> None:
    run = TrainingRun(
        "run", "agent", "input", "custom", TrainingStatus.SUCCEEDED,
        ["ep1"], {"lr": 0.1}, {"episodes_used": 1}, None,
        "started", "completed", {"key": "value"}, "created", "task",
    )
    result = LearnerResult(1, 0.5, 0.1, 0.2, {"a": 0.01}, {"custom": True})

    assert run.task_id == "task"
    assert run.created_at == "created"
    assert run.policy_id == "input"
    assert run.metadata == {"key": "value"}
    assert run.output_policy_id is None
    assert run.consumed_inputs is None
    assert result.extra == {"custom": True}
    assert result.consumed_inputs is None
