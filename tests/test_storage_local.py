"""Tests for the local file store - LearningStore contract + persistence."""

from __future__ import annotations

from pathlib import Path

from agent_learning.storage import LocalFileStore
from agent_learning.types import (
    Action,
    Episode,
    MetricName,
    MetricResult,
    PolicySnapshot,
    Reward,
    RewardSource,
)


def test_episode_crud(tmp_path: Path) -> None:
    store = LocalFileStore(tmp_path)
    ep = Episode(agent_id="dq", user_input="hi", assistant_output="hello")
    store.store_episode(ep)
    got = store.get_episode(ep.id, "dq")
    assert got is not None
    assert got.user_input == "hi"
    assert store.query_episodes("dq")[0].id == ep.id


def test_metrics_and_rewards(tmp_path: Path) -> None:
    store = LocalFileStore(tmp_path)
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


def test_reward_upsert_is_idempotent(tmp_path: Path) -> None:
    store = LocalFileStore(tmp_path)
    reward = Reward(episode_id="ep-1", agent_id="dq", value=0.5)
    store.store_reward(reward)
    reward.value = 0.9
    store.store_reward(reward)
    rewards = store.get_rewards_for_episode("ep-1", "dq")
    assert len(rewards) == 1
    assert rewards[0].value == 0.9


def test_policy_versioning(tmp_path: Path) -> None:
    store = LocalFileStore(tmp_path)
    p1 = PolicySnapshot(agent_id="dq", version=0, actions=[Action(id="a")], logits={"a": 0.0})
    p2 = PolicySnapshot(agent_id="dq", version=1, actions=[Action(id="a")], logits={"a": 1.0})
    store.store_policy(p1)
    store.store_policy(p2)
    latest = store.get_latest_policy("dq")
    assert latest is not None
    assert latest.version == 1


def test_agent_listing_and_completed_episode_count(tmp_path: Path) -> None:
    store = LocalFileStore(tmp_path)
    store.store_policy(
        PolicySnapshot(
            agent_id="scout",
            actions=[Action(id="a")],
            metadata={"name": "Scout"},
        )
    )
    store.store_policy(
        PolicySnapshot(agent_id="writer", actions=[Action(id="a")])
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
            metadata={"status": "in_progress"},
        )
    )

    assert [agent.to_dict() for agent in store.list_agents()] == [
        {"id": "scout", "name": "Scout"},
        {"id": "writer", "name": "writer"},
    ]
    assert store.count_completed_episodes("scout") == 1
    assert [
        episode.id
        for episode in store.query_episodes("scout", completed_only=True)
    ] == ["completed"]


def test_query_filters_and_limit(tmp_path: Path) -> None:
    store = LocalFileStore(tmp_path)
    store.store_episode(Episode(agent_id="dq", created_at="2024-01-01T00:00:00+00:00", policy_id="p1"))
    store.store_episode(Episode(agent_id="dq", created_at="2024-06-01T00:00:00+00:00", policy_id="p2"))
    store.store_episode(Episode(agent_id="dq", created_at="2024-12-01T00:00:00+00:00", policy_id="p1"))

    # Newest first
    ordered = store.query_episodes("dq")
    assert [e.created_at for e in ordered] == [
        "2024-12-01T00:00:00+00:00",
        "2024-06-01T00:00:00+00:00",
        "2024-01-01T00:00:00+00:00",
    ]
    # Filter by policy_id
    assert len(store.query_episodes("dq", policy_id="p1")) == 2
    # Filter by window
    windowed = store.query_episodes("dq", start_date="2024-05-01", end_date="2024-07-01")
    assert len(windowed) == 1
    # Limit
    assert len(store.query_episodes("dq", limit=1)) == 1


def test_persists_across_instances(tmp_path: Path) -> None:
    store = LocalFileStore(tmp_path)
    ep = Episode(agent_id="dq", user_input="persist me")
    store.store_episode(ep)

    reloaded = LocalFileStore(tmp_path)
    got = reloaded.get_episode(ep.id, "dq")
    assert got is not None
    assert got.user_input == "persist me"


def test_agent_partitions_are_isolated(tmp_path: Path) -> None:
    store = LocalFileStore(tmp_path)
    store.store_episode(Episode(id="e1", agent_id="a"))
    store.store_episode(Episode(id="e1", agent_id="b"))
    assert store.get_episode("e1", "a") is not None
    assert store.get_episode("e1", "b") is not None
    assert len(store.query_episodes("a")) == 1


def test_unsafe_ids_cannot_escape_root(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = LocalFileStore(root)
    # agent_id crafted to attempt path traversal must stay contained.
    ep = Episode(agent_id="../../evil", user_input="x")
    store.store_episode(ep)

    got = store.get_episode(ep.id, "../../evil")
    assert got is not None
    assert got.user_input == "x"
    # No directory named "evil" leaked outside the store root.
    assert not (tmp_path / "evil").exists()
    # Every persisted file lives under the configured root.
    for path in root.rglob("*.json"):
        assert root.resolve() in path.resolve().parents


def test_missing_records_return_none(tmp_path: Path) -> None:
    store = LocalFileStore(tmp_path)
    assert store.get_episode("nope", "dq") is None
    assert store.get_policy("nope", "dq") is None
    assert store.get_run("nope", "dq") is None
    assert store.get_latest_policy("dq") is None
    assert store.get_metric_results("nope", "dq") == []
    assert store.get_rewards_for_episode("nope", "dq") == []
