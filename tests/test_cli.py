"""Tests for task-aware CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_learning import cli
from agent_learning.policy import SoftmaxPolicy
from agent_learning.storage import InMemoryStore
from agent_learning.types import (
    Action,
    Episode,
    MetricName,
    MetricResult,
    Reward,
    RewardSource,
)


def _full_episode() -> Episode:
    return Episode(
        agent_id="agent-1",
        agent_name="Agent One",
        task_id="chat",
        task_name="Chat",
        intent_summary="answer the user",
        action_id="respond",
        action_name="Respond",
        expected_outcome="a correct answer",
        execution_status="completed",
        result_summary="answered correctly",
    )


def test_discovery_and_full_episode_count(monkeypatch, capsys) -> None:
    store = InMemoryStore()
    store.store_episode(_full_episode())
    store.store_episode(Episode(agent_id="agent-1", task_id="animation"))
    monkeypatch.setattr(cli, "get_default_store", lambda: store)

    assert cli.main(["list"]) == 0
    assert json.loads(capsys.readouterr().out) == [
        {"id": "agent-1", "name": "Agent One"}
    ]
    assert cli.main(["tasks-list", "agent-1"]) == 0
    assert [task["id"] for task in json.loads(capsys.readouterr().out)] == [
        "animation",
        "chat",
    ]
    assert cli.main(["task-episodes-count", "agent-1"]) == 0
    assert capsys.readouterr().out.strip() == "1"


def test_episode_inspection_includes_scores_and_final_reward(monkeypatch, capsys) -> None:
    store = InMemoryStore()
    episode = _full_episode()
    store.store_episode(episode)
    store.store_metric_results(
        episode.id,
        episode.agent_id,
        [
            MetricResult(
                metric=MetricName.TASK_COMPLETION,
                score=5.0,
                normalized=1.0,
                status="completed",
            )
        ],
    )
    store.store_reward(
        Reward(
            episode_id=episode.id,
            agent_id=episode.agent_id,
            source=RewardSource.AGGREGATE,
            value=0.9,
        )
    )
    monkeypatch.setattr(cli, "get_default_store", lambda: store)

    assert cli.main(["task-episodes-list", "agent-1"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["episode"]["intent_summary"] == "answer the user"
    assert payload[0]["final_reward"] == 0.9
    assert payload[0]["task_completion_quality"]["metric"] == "task_completion"


def test_task_policy_init_and_inspection(monkeypatch, capsys, tmp_path: Path) -> None:
    store = InMemoryStore()
    monkeypatch.setattr(cli, "get_default_store", lambda: store)
    actions_path = tmp_path / "actions.json"
    actions_path.write_text(
        json.dumps([{"id": "respond", "description": "Respond directly"}]),
        encoding="utf-8",
    )

    assert (
        cli.main(
            [
                "task-policy-init",
                "--agent-id",
                "agent-1",
                "--task-id",
                "chat",
                "--actions",
                str(actions_path),
            ]
        )
        == 0
    )
    initialized = json.loads(capsys.readouterr().out)
    assert initialized["task_id"] == "chat"

    assert (
        cli.main(
            ["task-policy", "--agent-id", "agent-1", "--task-id", "chat"]
        )
        == 0
    )
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["current_policy"]["task_id"] == "chat"
    assert inspected["previous_policy"] is None


def test_task_episode_register_persists_full_episode(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    store = InMemoryStore()
    monkeypatch.setattr(cli, "get_default_store", lambda: store)
    episode_path = tmp_path / "episode.json"
    episode_path.write_text(
        json.dumps(
            {
                "intent_summary": "assess a patient with a sore throat",
                "action_id": "order_strep_test",
                "expected_outcome": "order a strep throat test",
                "execution_status": "completed",
                "result_summary": "ordered the strep throat test",
            }
        ),
        encoding="utf-8",
    )

    assert (
        cli.main(
            [
                "task-episode-register",
                "--agent-id",
                "triage-nurse",
                "--task-id",
                "sore-throat-triage",
                "--episode",
                str(episode_path),
            ]
        )
        == 0
    )
    registered = json.loads(capsys.readouterr().out)

    episode = store.get_episode(registered["id"], "triage-nurse")
    assert episode is not None
    assert episode.task_id == "sore-throat-triage"
    assert episode.intent_summary == "assess a patient with a sore throat"
    assert episode.execution_status == "completed"
    assert episode.is_full


def test_score_uses_local_stdlib_without_configuration(
    monkeypatch, capsys
) -> None:
    for name in (
        "AGENT_LEARNING_SCORE_ENDPOINT",
        "AGENT_LEARNING_SCORE_DEPLOYMENT",
        "AGENT_LEARNING_SCORE_TIER",
    ):
        monkeypatch.delenv(name, raising=False)
    store = InMemoryStore()
    monkeypatch.setattr(cli, "get_default_store", lambda: store)
    episode = Episode(
        agent_id="scout",
        task_id="context-window",
        user_input="What is the context window?",
        assistant_output="The context window is 922,000 tokens.",
        intent_summary="Report the context window",
        action_id="inspect_context",
        expected_outcome="Return the live context limit",
        execution_status="completed",
        result_summary="Returned the live limit",
        metadata={
            "correct_action_id": "inspect_context",
            "task_completed": True,
        },
    )
    store.store_episode(episode)

    assert cli.main(["score", "--agent-id", "scout"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == {"episodes_seen": 1, "newly_scored": 1}
    metrics = store.get_metric_results(episode.id, episode.agent_id)
    assert len(metrics) == 3
    assert all(metric.status == "completed" for metric in metrics)
    assert all((metric.evaluator or "").startswith("local:") for metric in metrics)
    rewards = store.get_rewards_for_episode(episode.id, episode.agent_id)
    aggregate = next(reward for reward in rewards if reward.source == RewardSource.AGGREGATE)
    assert aggregate.value > 0.0


def test_score_replaces_skipped_only_evaluation(monkeypatch, capsys) -> None:
    store = InMemoryStore()
    monkeypatch.setattr(cli, "get_default_store", lambda: store)
    episode = Episode(
        agent_id="scout",
        task_id="context-window",
        user_input="What is the context window?",
        assistant_output="The context window is 922,000 tokens.",
        action_id="inspect_context",
        execution_status="completed",
        metadata={"task_completed": True},
    )
    store.store_episode(episode)
    store.store_metric_results(
        episode.id,
        episode.agent_id,
        [
            MetricResult(
                metric=metric,
                score=None,
                normalized=None,
                status="skipped",
                reason="remote scorer was not configured",
            )
            for metric in MetricName
        ],
    )
    store.store_reward(
        Reward(
            episode_id=episode.id,
            agent_id=episode.agent_id,
            source=RewardSource.AGGREGATE,
            value=0.0,
            created_at="2026-08-09T00:00:00+00:00",
        )
    )

    assert cli.main(["score", "--agent-id", "scout"]) == 0
    assert json.loads(capsys.readouterr().out)["newly_scored"] == 1
    metrics = store.get_metric_results(episode.id, episode.agent_id)
    assert sum(metric.status == "completed" for metric in metrics) == 3
    aggregates = [
        reward
        for reward in store.get_rewards_for_episode(episode.id, episode.agent_id)
        if reward.source == RewardSource.AGGREGATE
    ]
    assert len(aggregates) == 2
    assert max(aggregates, key=lambda reward: reward.created_at).value > 0.0


def test_agent_training_uses_one_limit_and_preserves_task_policy_history(
    monkeypatch, capsys
) -> None:
    store = InMemoryStore()
    monkeypatch.setattr(cli, "get_default_store", lambda: store)
    for task_id in ("chat", "animation"):
        policy = SoftmaxPolicy.from_actions(
            [Action(id="respond")],
            agent_id="agent-1",
            task_id=task_id,
        )
        snapshot = policy.snapshot()
        store.store_policy(snapshot)
        for index in range(2):
            episode = Episode(
                agent_id="agent-1",
                task_id=task_id,
                action_id="respond",
                policy_id=snapshot.id,
                policy_version=snapshot.version,
                created_at=f"2026-08-07T00:00:0{index + (2 if task_id == 'animation' else 0)}+00:00",
            )
            store.store_episode(episode)
            store.store_reward(
                Reward(
                    episode_id=episode.id,
                    agent_id=episode.agent_id,
                    source=RewardSource.AGGREGATE,
                    value=0.8,
                )
            )

    assert (
        cli.main(
            [
                "train",
                "--agent-id",
                "agent-1",
                "--limit",
                "3",
                "--skip-scoring",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)

    assert len(result["runs"]) == 2
    assert sum(len(run["episode_ids"]) for run in result["runs"]) == 3
    assert len(store.list_policies("agent-1", "chat")) == 2
    assert len(store.list_policies("agent-1", "animation")) == 2
    for task_id in ("chat", "animation"):
        policies = store.list_policies("agent-1", task_id)
        assert policies[0].version == 1
        assert policies[1].version == 0
        assert policies[0].id != policies[1].id


def test_episode_limit_is_capped_at_500() -> None:
    with pytest.raises(SystemExit):
        cli.main(["train", "--agent-id", "agent-1", "--limit", "501"])