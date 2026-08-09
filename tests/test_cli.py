"""Tests for task-aware CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_learning import __version__, cli
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


def test_help_and_version_print_sdk_version(capsys) -> None:
    assert f"SDK version {__version__}" in cli._build_arg_parser().format_help()
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"agent-learn {__version__}"


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
    assert (
        cli.main(["task-episodes-count", "agent-1", "--include-incomplete"])
        == 0
    )
    assert capsys.readouterr().out.strip() == "2"


def test_episode_count_and_list_use_the_training_date_window(
    monkeypatch, capsys
) -> None:
    store = InMemoryStore()
    monkeypatch.setattr(cli, "get_default_store", lambda: store)
    for index, created_at in enumerate(
        (
            "2026-08-09T05:40:00+00:00",
            "2026-08-09T06:07:25+00:00",
            "2026-08-09T06:12:23+00:00",
            "2026-08-09T06:13:10+00:00",
            "2026-08-09T06:13:55+00:00",
            "2026-08-09T06:14:37+00:00",
            "2026-08-09T06:20:00+00:00",
        )
    ):
        episode = _full_episode()
        episode.id = f"episode-{index}"
        episode.created_at = created_at
        store.store_episode(episode)

    window = [
        "--task-id",
        "chat",
        "--start-date",
        "2026-08-09T05:42:45.258Z",
        "--end-date",
        "2026-08-09T06:16:07.333Z",
    ]
    assert cli.main(["task-episodes-count", "agent-1", *window]) == 0
    assert capsys.readouterr().out.strip() == "5"
    assert cli.main(["task-episodes-list", "agent-1", *window]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert len(listed) == 5
    assert all(
        "2026-08-09T05:42:45.258+00:00"
        <= item["episode"]["created_at"]
        <= "2026-08-09T06:16:07.333+00:00"
        for item in listed
    )


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
        json.dumps(
            [
                {"id": "respond", "description": "Respond directly"},
                {"id": "delegate", "description": "Delegate the response"},
            ]
        ),
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
                "--decision-context",
                "Choose how the agent should respond to a chat request",
                "--actions",
                str(actions_path),
            ]
        )
        == 0
    )
    initialized = json.loads(capsys.readouterr().out)
    assert initialized["task_id"] == "chat"
    assert initialized["metadata"] == {
        "policy_scope": "delegated_decision",
        "decision_context": "Choose how the agent should respond to a chat request",
    }

    assert (
        cli.main(
            ["task-policy", "--agent-id", "agent-1", "--task-id", "chat"]
        )
        == 0
    )
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["current_policy"]["task_id"] == "chat"
    assert inspected["previous_policy"] is None


def test_task_policy_init_requires_two_unique_decision_actions(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    store = InMemoryStore()
    monkeypatch.setattr(cli, "get_default_store", lambda: store)
    actions_path = tmp_path / "actions.json"
    actions_path.write_text(json.dumps([{"id": "only"}]), encoding="utf-8")

    result = cli.main(
        [
            "task-policy-init",
            "--agent-id",
            "scout",
            "--task-id",
            "not-a-decision",
            "--decision-context",
            "There is only one action",
            "--actions",
            str(actions_path),
        ]
    )

    assert result == 2
    assert "at least two" in capsys.readouterr().err
    assert store.get_active_policy("scout", "not-a-decision") is None


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


def test_task_policy_decide_returns_learned_feedback(monkeypatch, capsys) -> None:
    store = InMemoryStore()
    monkeypatch.setattr(cli, "get_default_store", lambda: store)
    policy = SoftmaxPolicy.from_actions(
        [Action(id="use_skill"), Action(id="use_model")],
        agent_id="scout",
        task_id="choose-delegation",
        initial_logits={"use_skill": 1.0, "use_model": 0.0},
    )
    snapshot = policy.snapshot()
    snapshot.metadata = {
        "policy_scope": "delegated_decision",
        "decision_context": "Choose whether to delegate to a skill or language model",
    }
    store.store_policy(snapshot)
    outcomes = [
        ("correct", "use_skill", 0.8),
        ("incorrect", "use_model", -0.3),
    ]
    for label, correct_action_id, reward_value in outcomes:
        episode = Episode(
            id=label,
            agent_id="scout",
            task_id="choose-delegation",
            policy_id=snapshot.id,
            action_id="use_skill",
            execution_status="completed",
            result_summary=label,
            metadata={"correct_action_id": correct_action_id},
        )
        store.store_episode(episode)
        store.store_metric_results(
            episode.id,
            episode.agent_id,
            [
                MetricResult(
                    metric=MetricName.TASK_COMPLETION,
                    score=1.0 if label == "correct" else 0.0,
                    normalized=1.0 if label == "correct" else 0.0,
                    status="completed",
                    reason=label,
                )
            ],
        )
        store.store_reward(
            Reward(
                episode_id=episode.id,
                agent_id=episode.agent_id,
                source=RewardSource.AGGREGATE,
                value=reward_value,
            )
        )

    assert (
        cli.main(
            [
                "task-policy-decide",
                "--agent-id",
                "scout",
                "--task-id",
                "choose-delegation",
                "--greedy",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)

    assert result["selected_action"]["id"] == "use_skill"
    assert result["selected_action"]["probability"] > 0.5
    assert result["recommended_action"]["id"] == "use_skill"
    feedback = result["selected_action_feedback"]
    assert feedback["attempts"] == 2
    assert feedback["correctness_rate"] == 0.5
    assert feedback["mean_reward"] == pytest.approx(0.25)
    assert {item["was_correct"] for item in feedback["recent_outcomes"]} == {
        True,
        False,
    }
    assert {
        item["score_breakdown"]["task_completion"]["normalized"]
        for item in feedback["recent_outcomes"]
    } == {0.0, 1.0}


def test_decision_only_excludes_unmarked_question_policy(monkeypatch, capsys) -> None:
    store = InMemoryStore()
    monkeypatch.setattr(cli, "get_default_store", lambda: store)
    question = SoftmaxPolicy.from_actions(
        [Action(id="answer")], agent_id="scout", task_id="answer-question"
    ).snapshot()
    decision = SoftmaxPolicy.from_actions(
        [Action(id="delegate")], agent_id="scout", task_id="choose-delegation"
    ).snapshot()
    decision.metadata = {
        "policy_scope": "delegated_decision",
        "decision_context": "Choose a delegate",
    }
    store.store_policy(question)
    store.store_policy(decision)

    assert cli.main(["tasks-list", "scout", "--decision-only"]) == 0
    assert json.loads(capsys.readouterr().out) == [
        {"id": "choose-delegation", "name": "choose-delegation"}
    ]
    assert (
        cli.main(
            [
                "train",
                "--agent-id",
                "scout",
                "--task-id",
                "answer-question",
                "--decision-only",
            ]
        )
        == 2
    )
    result = json.loads(capsys.readouterr().out)
    assert result["skipped"] == [
        {"task_id": "answer-question", "reason": "not a delegated decision policy"}
    ]


def test_decision_episode_registration_requires_marked_policy(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    store = InMemoryStore()
    monkeypatch.setattr(cli, "get_default_store", lambda: store)
    policy = SoftmaxPolicy.from_actions(
        [Action(id="delegate")], agent_id="scout", task_id="choose-delegation"
    ).snapshot()
    policy.metadata = {
        "policy_scope": "delegated_decision",
        "decision_context": "Choose a delegate",
    }
    store.store_policy(policy)
    episode_path = tmp_path / "decision.json"
    episode_path.write_text(
        json.dumps(
            {
                "policy_id": policy.id,
                "policy_version": policy.version,
                "action_id": "delegate",
                "intent_summary": "Choose a delegate",
                "expected_outcome": "Use the best delegate",
                "execution_status": "completed",
                "result_summary": "Delegation completed",
            }
        ),
        encoding="utf-8",
    )

    assert (
        cli.main(
            [
                "task-episode-register",
                "--agent-id",
                "scout",
                "--task-id",
                "choose-delegation",
                "--episode",
                str(episode_path),
                "--require-decision-policy",
            ]
        )
        == 0
    )
    registered = json.loads(capsys.readouterr().out)
    assert store.get_episode(registered["id"], "scout") is not None

    invalid_path = tmp_path / "invalid-decision.json"
    invalid_payload = json.loads(episode_path.read_text(encoding="utf-8"))
    invalid_payload["metadata"] = {"correct_action_id": "outside-policy"}
    invalid_path.write_text(json.dumps(invalid_payload), encoding="utf-8")
    assert (
        cli.main(
            [
                "task-episode-register",
                "--agent-id",
                "scout",
                "--task-id",
                "choose-delegation",
                "--episode",
                str(invalid_path),
                "--require-decision-policy",
            ]
        )
        == 2
    )
    assert "correct_action_id" in capsys.readouterr().err


def test_pending_decision_episode_can_be_completed_in_place(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    store = InMemoryStore()
    monkeypatch.setattr(cli, "get_default_store", lambda: store)
    policy = SoftmaxPolicy.from_actions(
        [Action(id="use_functions"), Action(id="use_container_apps")],
        agent_id="scout",
        task_id="choose-message-processor",
    ).snapshot()
    policy.metadata = {
        "policy_scope": "delegated_decision",
        "decision_context": "Choose a message-processing workload",
    }
    store.store_policy(policy)
    episode_path = tmp_path / "pending-decision.json"
    payload = {
        "id": "pending-decision",
        "policy_id": policy.id,
        "policy_version": policy.version,
        "action_id": "use_functions",
        "intent_summary": "Choose a message-processing workload",
        "expected_outcome": "The user accepts the recommendation or reports its result",
        "metadata": {"feedback_status": "pending"},
    }
    episode_path.write_text(json.dumps(payload), encoding="utf-8")

    register = [
        "task-episode-register",
        "--agent-id",
        "scout",
        "--task-id",
        "choose-message-processor",
        "--episode",
        str(episode_path),
        "--require-decision-policy",
    ]
    assert cli.main(register) == 0
    capsys.readouterr()
    assert not store.get_episode("pending-decision", "scout").is_full
    assert cli.main(["task-episodes-count", "scout"]) == 0
    assert capsys.readouterr().out.strip() == "0"
    assert (
        cli.main(["task-episodes-count", "scout", "--include-incomplete"])
        == 0
    )
    assert capsys.readouterr().out.strip() == "1"
    assert (
        cli.main(
            [
                "task-episodes-list",
                "scout",
                "--task-id",
                "choose-message-processor",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == []
    assert (
        cli.main(
            [
                "task-episodes-list",
                "scout",
                "--task-id",
                "choose-message-processor",
                "--include-incomplete",
            ]
        )
        == 0
    )
    assert len(json.loads(capsys.readouterr().out)) == 1
    assert (
        cli.main(
            [
                "score",
                "--agent-id",
                "scout",
                "--task-id",
                "choose-message-processor",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {
        "episodes_seen": 0,
        "newly_scored": 0,
    }
    assert store.get_metric_results("pending-decision", "scout") == []
    assert store.get_rewards_for_episode("pending-decision", "scout") == []

    payload.update(
        {
            "execution_status": "completed",
            "result_summary": "The user accepted the Azure Functions recommendation",
            "metadata": {
                "feedback_status": "accepted",
                "correct_action_id": "use_functions",
                "task_completed": True,
            },
        }
    )
    episode_path.write_text(json.dumps(payload), encoding="utf-8")

    assert cli.main(register) == 0
    capsys.readouterr()
    assert store.get_episode("pending-decision", "scout").is_full
    assert cli.main(["task-episodes-count", "scout"]) == 0
    assert capsys.readouterr().out.strip() == "1"
    assert (
        cli.main(["task-episodes-count", "scout", "--include-incomplete"])
        == 0
    )
    assert capsys.readouterr().out.strip() == "1"


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
        intent_summary="Report the context window",
        action_id="inspect_context",
        expected_outcome="Return the live context limit",
        execution_status="completed",
        result_summary="Returned the live context limit",
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
                    intent_summary="complete the selected task",
                action_id="respond",
                    expected_outcome="return a completed result",
                    execution_status="completed",
                    result_summary="completed the selected task",
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


def test_train_enforces_minimum_selected_episode_count(
    monkeypatch, capsys
) -> None:
    store = InMemoryStore()
    monkeypatch.setattr(cli, "get_default_store", lambda: store)
    policy = SoftmaxPolicy.from_actions(
        [Action(id="respond")], agent_id="agent-1", task_id="chat"
    )
    store.store_policy(policy.snapshot())
    for index in range(3):
        episode = _full_episode()
        episode.id = f"episode-{index}"
        store.store_episode(episode)
        store.store_reward(
            Reward(
                episode_id=episode.id,
                agent_id=episode.agent_id,
                source=RewardSource.AGGREGATE,
                value=0.8,
            )
        )
    for index in range(2):
        store.store_episode(
            Episode(
                id=f"pending-{index}",
                agent_id="agent-1",
                task_id="chat",
                action_id="respond",
                intent_summary="answer the user",
                expected_outcome="await user feedback",
                metadata={"feedback_status": "pending"},
            )
        )

    result = cli.main(
        [
            "train",
            "--agent-id",
            "agent-1",
            "--task-id",
            "chat",
            "--limit",
            "3",
            "--min-episodes",
            "5",
            "--skip-scoring",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert result == 2
    assert output["runs"] == []
    assert output["skipped"] == [
        {
            "task_id": "chat",
            "reason": "selected batch has 3 episodes; minimum is 5",
        }
    ]
    assert store.get_active_policy("agent-1", "chat").version == 0