"""Tests for the in-memory store - exercises the LearningStore contract."""

from __future__ import annotations

from agent_learning.storage import InMemoryStore
from agent_learning.types import (
    Action,
    Episode,
    MetricName,
    MetricResult,
    PolicySnapshot,
    Reward,
    RewardSource,
)


def test_episode_crud() -> None:
    store = InMemoryStore()
    ep = Episode(agent_id="dq", user_input="hi", assistant_output="hello")
    store.store_episode(ep)
    assert store.get_episode(ep.id, "dq").user_input == "hi"
    assert store.query_episodes("dq")[0].id == ep.id


def test_metrics_and_rewards() -> None:
    store = InMemoryStore()
    ep_id = "ep-1"
    store.store_metric_results(
        ep_id,
        "dq",
        [
            MetricResult(
                metric=MetricName.INTENT_RESOLUTION,
                score=4.0,
                normalized=0.75,
                status="completed",
            )
        ],
    )
    assert store.get_metric_results(ep_id, "dq")[0].score == 4.0

    reward = Reward(
        episode_id=ep_id,
        agent_id="dq",
        source=RewardSource.AGGREGATE,
        value=0.4,
    )
    store.store_reward(reward)
    rewards = store.get_rewards_for_episode(ep_id, "dq")
    assert len(rewards) == 1
    assert rewards[0].value == 0.4


def test_policy_versioning() -> None:
    store = InMemoryStore()
    p1 = PolicySnapshot(agent_id="dq", version=0, actions=[Action(id="a")], logits={"a": 0.0})
    p2 = PolicySnapshot(agent_id="dq", version=1, actions=[Action(id="a")], logits={"a": 1.0})
    store.store_policy(p1)
    store.store_policy(p2)
    latest = store.get_latest_policy("dq")
    assert latest is not None
    assert latest.version == 1


def test_task_policy_history_and_active_pointer() -> None:
    store = InMemoryStore()
    first = PolicySnapshot(agent_id="dq", task_id="chat", version=0)
    second = PolicySnapshot(agent_id="dq", task_id="chat", version=1)
    other = PolicySnapshot(agent_id="dq", task_id="animation", version=5)
    store.store_policy(first)
    store.store_policy(second)
    store.store_policy(other)

    assert [policy.id for policy in store.list_policies("dq", "chat")] == [
        second.id,
        first.id,
    ]
    assert store.get_active_policy("dq", "chat") == second


def test_agent_task_discovery_and_full_episode_count() -> None:
    store = InMemoryStore()
    store.store_episode(
        Episode(
            agent_id="dq",
            agent_name="Demo Agent",
            task_id="chat",
            task_name="Chat",
            intent_summary="answer a question",
            action_id="answer",
            expected_outcome="an answer",
            execution_status="completed",
            result_summary="answered",
            created_at="2026-08-09T10:00:00+00:00",
        )
    )
    store.store_episode(
        Episode(
            agent_id="dq",
            task_id="animation",
            created_at="2026-08-09T11:00:00+00:00",
        )
    )

    assert store.list_agents()[0].name == "Demo Agent"
    assert [task.id for task in store.list_agent_tasks("dq")] == ["animation", "chat"]
    assert store.count_episodes("dq") == 2
    assert store.count_episodes("dq", full_only=True) == 1
    assert len(store.query_episodes("dq", task_id="chat")) == 1
    assert store.query_episodes("dq", full_only=True, limit=1)[0].task_id == "chat"
