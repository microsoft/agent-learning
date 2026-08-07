"""Tests for the durable task-oriented CLI workflow."""

from __future__ import annotations

import json
from pathlib import Path

from agent_learning.cli import main
from agent_learning.storage import LocalFileStore
from agent_learning.types import (
    Episode,
    MetricName,
    MetricResult,
    Reward,
    RewardSource,
)


def _actions_file(tmp_path: Path) -> Path:
    path = tmp_path / "actions.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "direct",
                    "description": "Use a direct response",
                    "parameters": {"style": "direct"},
                }
            ]
        ),
        encoding="utf-8",
    )
    return path


def _stdout_json(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


def test_task_commands_persist_a_complete_episode(tmp_path: Path, capsys) -> None:
    store_dir = tmp_path / "store"
    actions = _actions_file(tmp_path)

    assert main(
        [
            "task-policy-init",
            "--agent-id",
            "assistant",
            "--task-id",
            "summary",
            "--actions",
            str(actions),
            "--store-dir",
            str(store_dir),
        ]
    ) == 0
    policy = _stdout_json(capsys)

    assert main(
        [
            "task-intent",
            "--agent-id",
            "assistant",
            "--task-id",
            "summary",
            "--intent",
            "Summarise the open issues",
            "--context",
            '{"repository": "microsoft/agents-learning-sdk"}',
            "--episode-id",
            "task-1",
            "--store-dir",
            str(store_dir),
        ]
    ) == 0
    decision = _stdout_json(capsys)

    assert decision["episode_id"] == "task-1"
    assert decision["action_id"] == "direct"
    assert decision["action"]["parameters"] == {"style": "direct"}
    assert decision["policy_id"] == policy["id"]
    assert decision["policy_version"] == policy["version"]
    assert decision["probabilities"] == {"direct": 1.0}

    store = LocalFileStore(store_dir)
    started = store.get_episode("task-1", "assistant")
    assert started is not None
    assert started.task_id == "summary"
    assert started.user_input == "Summarise the open issues"
    assert started.assistant_output == ""
    assert started.action_id == "direct"
    assert started.context_features == {
        "repository": "microsoft/agents-learning-sdk"
    }
    assert started.metadata["status"] == "in_progress"
    assert started.metadata["decision"]["action"]["id"] == "direct"

    assert main(
        [
            "task-complete",
            "--agent-id",
            "assistant",
            "--episode-id",
            "task-1",
            "--output",
            "There are no open issues.",
            "--store-dir",
            str(store_dir),
        ]
    ) == 0
    completion = _stdout_json(capsys)

    assert completion["episode_id"] == "task-1"
    assert completion["status"] == "completed"
    completed = store.get_episode("task-1", "assistant")
    assert completed is not None
    assert completed.assistant_output == "There are no open issues."
    assert completed.metadata["status"] == "completed"
    assert completed.metadata["completed_at"]
    assert completed.request_latency_ms is not None
    assert completed.request_latency_ms >= 0

    assert main(
        [
            "task-policy",
            "--agent-id",
            "assistant",
            "--task-id",
            "summary",
            "--store-dir",
            str(store_dir),
        ]
    ) == 0
    assert _stdout_json(capsys)[0]["id"] == policy["id"]


def test_agent_listing_and_completed_episode_count(tmp_path: Path, capsys) -> None:
    store_dir = tmp_path / "store"
    assert main(
        [
            "task-policy-init",
            "--agent-id",
            "assistant",
            "--task-id",
            "summary",
            "--agent-name",
            "Assistant Agent",
            "--task-name",
            "Weekly summary",
            "--actions",
            str(_actions_file(tmp_path)),
            "--store-dir",
            str(store_dir),
        ]
    ) == 0
    capsys.readouterr()

    store = LocalFileStore(store_dir)
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
            metadata={"status": "in_progress"},
        )
    )

    assert main(["agents-list", "--store-dir", str(store_dir)]) == 0
    assert json.loads(capsys.readouterr().out) == [
        {"id": "assistant", "name": "Assistant Agent"}
    ]

    assert main(
        ["agents-episodes-count", "assistant", "--store-dir", str(store_dir)]
    ) == 0
    assert json.loads(capsys.readouterr().out) == 1

    assert main(
        ["agent-tasks-list", "assistant", "--store-dir", str(store_dir)]
    ) == 0
    assert json.loads(capsys.readouterr().out) == [
        {"id": "summary", "name": "Weekly summary"}
    ]


def test_task_policy_history_preserves_replaced_snapshots(
    tmp_path: Path,
    capsys,
) -> None:
    store_dir = tmp_path / "store"
    command = [
        "task-policy-init",
        "--agent-id",
        "assistant",
        "--task-id",
        "summary",
        "--actions",
        str(_actions_file(tmp_path)),
        "--store-dir",
        str(store_dir),
    ]
    assert main(command) == 0
    first = _stdout_json(capsys)
    assert main(command) == 0
    second = _stdout_json(capsys)

    assert first["id"] != second["id"]
    assert [first["version"], second["version"]] == [0, 1]
    assert main(
        [
            "task-policy",
            "--agent-id",
            "assistant",
            "--task-id",
            "summary",
            "--history",
            "2",
            "--store-dir",
            str(store_dir),
        ]
    ) == 0
    assert [
        snapshot["id"] for snapshot in json.loads(capsys.readouterr().out)
    ] == [second["id"], first["id"]]


def test_completed_episode_review_includes_learning_signals(
    tmp_path: Path,
    capsys,
) -> None:
    store_dir = tmp_path / "store"
    store = LocalFileStore(store_dir)
    episode = Episode(
        id="episode-1",
        agent_id="assistant",
        task_id="summary",
        user_input="Summarise the open issues",
        assistant_output="There are no open issues.",
        action_id="direct",
        policy_id="policy-1",
        policy_version=2,
        metadata={"status": "completed"},
    )
    store.store_episode(episode)
    store.store_metric_results(
        episode.id,
        episode.agent_id,
        [
            MetricResult(
                metric=MetricName.TASK_COMPLETION,
                score=5,
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

    assert main(
        [
            "agents-episodes-list",
            "assistant",
            "--task-id",
            "summary",
            "--store-dir",
            str(store_dir),
        ]
    ) == 0
    review = json.loads(capsys.readouterr().out)
    assert review[0]["intent_summary"] == "Summarise the open issues"
    assert review[0]["chosen_action"] == "direct"
    assert review[0]["score_breakdown"][0]["metric"] == "task_completion"
    assert review[0]["final_reward"] == 0.9
    assert review[0]["execution_result"] == "There are no open issues."


def test_train_requires_five_completed_episodes_by_default(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    monkeypatch.delenv("AGENT_LEARNING_MIN_TRAIN_EPISODES", raising=False)
    store_dir = tmp_path / "store"
    assert main(
        [
            "task-policy-init",
            "--agent-id",
            "assistant",
            "--task-id",
            "summary",
            "--actions",
            str(_actions_file(tmp_path)),
            "--store-dir",
            str(store_dir),
        ]
    ) == 0
    capsys.readouterr()

    store = LocalFileStore(store_dir)
    for index in range(4):
        store.store_episode(
            Episode(
                id=f"completed-{index}",
                agent_id="assistant",
                task_id="summary",
                metadata={"status": "completed"},
            )
        )

    command = [
        "train",
        "--agent-id",
        "assistant",
        "--task-id",
        "summary",
        "--limit",
        "3",
        "--store-dir",
        str(store_dir),
    ]
    assert main(command) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "at least 5" in captured.err

    store.store_episode(
        Episode(
            id="completed-4",
            agent_id="assistant",
            task_id="summary",
            metadata={"status": "completed"},
        )
    )
    call: dict = {}

    class _Run:
        def to_dict(self) -> dict:
            return {"status": "succeeded"}

    def run_offline_batch(self, agent_id: str, **kwargs):
        call.update({"agent_id": agent_id, **kwargs})
        return _Run()

    monkeypatch.setattr(
        "agent_learning.cli.LearningRunner.run_offline_batch",
        run_offline_batch,
    )

    assert main(command) == 0
    assert _stdout_json(capsys) == {"status": "succeeded"}
    assert call["agent_id"] == "assistant"
    assert call["task_id"] == "summary"
    assert call["episode_limit"] == 3
    assert call["completed_only"] is True


def test_train_minimum_is_configurable(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    store_dir = tmp_path / "store"
    assert main(
        [
            "task-policy-init",
            "--agent-id",
            "assistant",
            "--task-id",
            "summary",
            "--actions",
            str(_actions_file(tmp_path)),
            "--store-dir",
            str(store_dir),
        ]
    ) == 0
    capsys.readouterr()
    LocalFileStore(store_dir).store_episode(
        Episode(
            id="completed",
            agent_id="assistant",
            task_id="summary",
            metadata={"status": "completed"},
        )
    )
    monkeypatch.setenv("AGENT_LEARNING_MIN_TRAIN_EPISODES", "1")
    monkeypatch.setattr(
        "agent_learning.cli.LearningRunner.run_offline_batch",
        lambda self, agent_id, **kwargs: type(
            "Run",
            (),
            {"to_dict": lambda self: {"status": "succeeded"}},
        )(),
    )

    assert main(
        [
            "train",
            "--agent-id",
            "assistant",
            "--task-id",
            "summary",
            "--limit",
            "1",
            "--store-dir",
            str(store_dir),
        ]
    ) == 0
    assert _stdout_json(capsys) == {"status": "succeeded"}


def test_task_intent_requires_an_existing_policy(tmp_path: Path, capsys) -> None:
    assert main(
        [
            "task-intent",
            "--agent-id",
            "assistant",
            "--task-id",
            "summary",
            "--intent",
            "Do the task",
            "--store-dir",
            str(tmp_path / "store"),
        ]
    ) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "task-policy-init" in captured.err


def test_task_complete_requires_an_existing_episode(tmp_path: Path, capsys) -> None:
    assert main(
        [
            "task-complete",
            "--agent-id",
            "assistant",
            "--episode-id",
            "missing",
            "--output",
            "Done",
            "--store-dir",
            str(tmp_path / "store"),
        ]
    ) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "No episode found" in captured.err


def test_task_intent_rejects_invalid_context(tmp_path: Path, capsys) -> None:
    store_dir = tmp_path / "store"
    assert main(
        [
            "task-policy-init",
            "--agent-id",
            "assistant",
            "--task-id",
            "summary",
            "--actions",
            str(_actions_file(tmp_path)),
            "--store-dir",
            str(store_dir),
        ]
    ) == 0
    capsys.readouterr()

    assert main(
        [
            "task-intent",
            "--agent-id",
            "assistant",
            "--task-id",
            "summary",
            "--intent",
            "Do the task",
            "--context",
            "not-json",
            "--store-dir",
            str(store_dir),
        ]
    ) == 2

    assert "--context must be valid JSON" in capsys.readouterr().err


def test_task_fields_are_redacted_before_persistence(tmp_path: Path, capsys) -> None:
    store_dir = tmp_path / "store"
    assert main(
        [
            "task-policy-init",
            "--agent-id",
            "assistant",
            "--task-id",
            "summary",
            "--actions",
            str(_actions_file(tmp_path)),
            "--store-dir",
            str(store_dir),
        ]
    ) == 0
    capsys.readouterr()

    assert main(
        [
            "task-intent",
            "--agent-id",
            "assistant",
            "--task-id",
            "summary",
            "--intent",
            "Use token=private-value",
            "--episode-id",
            "task-secret",
            "--store-dir",
            str(store_dir),
        ]
    ) == 0
    capsys.readouterr()
    assert main(
        [
            "task-complete",
            "--agent-id",
            "assistant",
            "--episode-id",
            "task-secret",
            "--result",
            "******",
            "--store-dir",
            str(store_dir),
        ]
    ) == 0
    capsys.readouterr()

    content = (store_dir / "episodes" / "assistant" / "task-secret.json").read_text(
        encoding="utf-8"
    )
    assert "private-value" not in content
    assert "[REDACTED]" in content
