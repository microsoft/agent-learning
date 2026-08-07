"""Round-trip tests for the durable record types."""

from __future__ import annotations

from agent_learning.types import (
    Action,
    AgentTaskInfo,
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


def test_episode_completion_state() -> None:
    assert Episode(metadata={"status": "completed"}).is_complete is True
    assert (
        Episode(
            user_input="task",
            assistant_output="partial",
            metadata={"status": "in_progress"},
        ).is_complete
        is False
    )
    assert Episode(user_input="task", assistant_output="done").is_complete is True
    assert Episode(user_input="task").is_complete is False


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


def test_agent_task_info_uses_task_name() -> None:
    task = AgentTaskInfo.from_metadata("weekly-summary", {"task_name": "Weekly summary"})
    assert task.to_dict() == {"id": "weekly-summary", "name": "Weekly summary"}
