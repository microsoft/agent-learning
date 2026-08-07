"""Tests for the durable task-oriented CLI workflow."""

from __future__ import annotations

import json
from pathlib import Path

from agent_learning.cli import main
from agent_learning.storage import LocalFileStore
from agent_learning.types import Episode


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
            "policy-init",
            "--agent-id",
            "scout",
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
            "scout",
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
    started = store.get_episode("task-1", "scout")
    assert started is not None
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
            "scout",
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
    completed = store.get_episode("task-1", "scout")
    assert completed is not None
    assert completed.assistant_output == "There are no open issues."
    assert completed.metadata["status"] == "completed"
    assert completed.metadata["completed_at"]
    assert completed.request_latency_ms is not None
    assert completed.request_latency_ms >= 0

    assert main(
        [
            "policy",
            "--agent-id",
            "scout",
            "--store-dir",
            str(store_dir),
        ]
    ) == 0
    assert _stdout_json(capsys)["id"] == policy["id"]


def test_agent_listing_and_completed_episode_count(tmp_path: Path, capsys) -> None:
    store_dir = tmp_path / "store"
    assert main(
        [
            "policy-init",
            "--agent-id",
            "scout",
            "--agent-name",
            "Scout Agent",
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

    assert main(["agents-list", "--store-dir", str(store_dir)]) == 0
    assert json.loads(capsys.readouterr().out) == [
        {"id": "scout", "name": "Scout Agent"}
    ]

    assert main(
        ["agents-episodes-count", "scout", "--store-dir", str(store_dir)]
    ) == 0
    assert json.loads(capsys.readouterr().out) == 1


def test_train_requires_five_completed_episodes_by_default(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    monkeypatch.delenv("AGENT_LEARNING_MIN_TRAIN_EPISODES", raising=False)
    store_dir = tmp_path / "store"
    assert main(
        [
            "policy-init",
            "--agent-id",
            "scout",
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
                agent_id="scout",
                metadata={"status": "completed"},
            )
        )

    command = [
        "train",
        "--agent-id",
        "scout",
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
            agent_id="scout",
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
    assert call["agent_id"] == "scout"
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
            "policy-init",
            "--agent-id",
            "scout",
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
            agent_id="scout",
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
            "scout",
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
            "scout",
            "--intent",
            "Do the task",
            "--store-dir",
            str(tmp_path / "store"),
        ]
    ) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "policy-init" in captured.err


def test_task_complete_requires_an_existing_episode(tmp_path: Path, capsys) -> None:
    assert main(
        [
            "task-complete",
            "--agent-id",
            "scout",
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
            "policy-init",
            "--agent-id",
            "scout",
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
            "scout",
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
            "policy-init",
            "--agent-id",
            "scout",
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
            "scout",
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
            "scout",
            "--episode-id",
            "task-secret",
            "--result",
            "******",
            "--store-dir",
            str(store_dir),
        ]
    ) == 0
    capsys.readouterr()

    content = (store_dir / "episodes" / "scout" / "task-secret.json").read_text(
        encoding="utf-8"
    )
    assert "private-value" not in content
    assert "[REDACTED]" in content
