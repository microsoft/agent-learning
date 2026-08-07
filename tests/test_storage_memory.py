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


def test_agent_listing_and_completed_episode_queries() -> None:
    store = InMemoryStore()
    store.store_policy(
        PolicySnapshot(
            agent_id="assistant",
            task_id="summary",
            actions=[Action(id="a")],
            metadata={
                "agent_name": "Assistant",
                "task_name": "Weekly summary",
            },
        )
    )
    store.store_episode(
        Episode(
            id="completed",
            agent_id="assistant",
            task_id="summary",
            metadata={"status": "completed"},
        )
    )
    store.store_episode(
        Episode(
            id="in-progress",
            agent_id="assistant",
            task_id="summary",
            user_input="Task",
            assistant_output="Partial output",
            metadata={"status": "in_progress"},
        )
    )
    store.store_episode(
        Episode(
            id="captured",
            agent_id="assistant",
            task_id="summary",
            user_input="Task",
            assistant_output="Finished output",
        )
    )

    assert [agent.to_dict() for agent in store.list_agents()] == [
        {"id": "assistant", "name": "Assistant"}
    ]
    assert [
        task.to_dict() for task in store.list_agent_tasks("assistant")
    ] == [{"id": "summary", "name": "Weekly summary"}]
    assert store.count_completed_episodes("assistant") == 2
    assert store.count_completed_episodes("assistant", task_id="summary") == 2
    assert {
        episode.id
        for episode in store.query_episodes(
            "assistant",
            task_id="summary",
            completed_only=True,
        )
    } == {"completed", "captured"}


def test_task_policy_history_is_independent() -> None:
    store = InMemoryStore()
    store.store_policy(
        PolicySnapshot(
            id="summary-v0",
            agent_id="assistant",
            task_id="summary",
            version=0,
        )
    )
    store.store_policy(
        PolicySnapshot(
            id="summary-v1",
            agent_id="assistant",
            task_id="summary",
            version=1,
        )
    )
    store.store_policy(
        PolicySnapshot(
            id="translate-v3",
            agent_id="assistant",
            task_id="translate",
            version=3,
        )
    )

    assert [
        policy.id
        for policy in store.list_policies(
            "assistant",
            task_id="summary",
        )
    ] == ["summary-v1", "summary-v0"]
    assert store.get_latest_policy(
        "assistant",
        task_id="summary",
    ).id == "summary-v1"
