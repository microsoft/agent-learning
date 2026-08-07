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
            agent_id="scout",
            actions=[Action(id="a")],
            metadata={"name": "Scout"},
        )
    )
    store.store_episode(
        Episode(
            id="completed",
            agent_id="scout",
            metadata={"status": "completed"},
        )
    )
    store.store_episode(
        Episode(
            id="in-progress",
            agent_id="scout",
            user_input="Task",
            assistant_output="Partial output",
            metadata={"status": "in_progress"},
        )
    )
    store.store_episode(
        Episode(
            id="captured",
            agent_id="scout",
            user_input="Task",
            assistant_output="Finished output",
        )
    )

    assert [agent.to_dict() for agent in store.list_agents()] == [
        {"id": "scout", "name": "Scout"}
    ]
    assert store.count_completed_episodes("scout") == 2
    assert {
        episode.id
        for episode in store.query_episodes("scout", completed_only=True)
    } == {"completed", "captured"}
