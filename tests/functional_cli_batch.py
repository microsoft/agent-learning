"""Exercise the batch agent-learning CLI workflow over captured episodes.

Run ``functional_cli_interactive.py`` first to create the isolated local store
and its scored patient-triage episodes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
DEFAULT_ACTIONS_PATH = REPO_ROOT / "examples" / "next_best_action_patient_care_actions.json"
DEFAULT_STORE_DIR = REPO_ROOT / "data" / "functional-tests" / "patient-care-cli"
AGENT_ID = "triage-nurse"
TASK_ID = "sore-throat-triage"


def _cli_environment(store_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["AGENT_LEARNING_STORE_BACKEND"] = "local"
    env["AGENT_LEARNING_LOCAL_STORE_DIR"] = str(store_dir)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(SRC_DIR), existing_pythonpath) if part
    )
    return env


def _run_cli(arguments: list[str], env: dict[str, str]) -> Any:
    print(f"+ agent-learn {' '.join(arguments)}")
    completed = subprocess.run(
        [sys.executable, "-m", "agent_learning.cli", *arguments],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"CLI command failed with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    output = completed.stdout.strip()
    return json.loads(output) if output else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--actions", type=Path, default=DEFAULT_ACTIONS_PATH)
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    args = parser.parse_args()
    if not 1 <= args.limit <= 500:
        parser.error("--limit must be between 1 and 500")
    return args


def _correct_action_id(actions_path: Path) -> str:
    actions = json.loads(actions_path.read_text(encoding="utf-8"))
    correct = [
        action["id"]
        for action in actions
        if action.get("parameters", {}).get("is_correct_for_sore_throat_case") is True
    ]
    if len(correct) != 1:
        raise ValueError("The actions fixture must identify exactly one correct action")
    return correct[0]


def main() -> int:
    args = _parse_args()
    store_dir = args.store_dir.resolve()
    if not store_dir.is_dir():
        raise SystemExit(
            f"Functional store not found at {store_dir}. "
            "Run tests/functional_cli_interactive.py first."
        )
    env = _cli_environment(store_dir)
    correct_action_id = _correct_action_id(args.actions.resolve())

    agents = _run_cli(["list"], env)
    if AGENT_ID not in {agent["id"] for agent in agents}:
        raise AssertionError(f"Agent {AGENT_ID!r} was not discovered")

    tasks = _run_cli(["tasks-list", AGENT_ID], env)
    if TASK_ID not in {task["id"] for task in tasks}:
        raise AssertionError(f"Task {TASK_ID!r} was not discovered")

    episode_count = _run_cli(
        ["task-episodes-count", AGENT_ID, "--task-id", TASK_ID],
        env,
    )
    if episode_count < 2:
        raise AssertionError("The batch workflow requires multiple captured episodes")

    episodes = _run_cli(
        [
            "task-episodes-list",
            AGENT_ID,
            "--task-id",
            TASK_ID,
            "--limit",
            str(args.limit),
            "--include-incomplete",
        ],
        env,
    )
    expected_list_count = min(episode_count, args.limit)
    if len(episodes) != expected_list_count:
        raise AssertionError(
            f"Expected {expected_list_count} listed episodes, found {len(episodes)}"
        )
    if any(item["final_reward"] is None for item in episodes):
        raise AssertionError("Every functional episode must have a final reward")
    if any(not item["episode"]["intent_summary"] for item in episodes):
        raise AssertionError("Every functional episode must store its intent")
    if any(not item["episode"]["result_summary"] for item in episodes):
        raise AssertionError("Every functional episode must store its completion result")

    training = _run_cli(
        [
            "train",
            "--agent-id",
            AGENT_ID,
            "--task-id",
            TASK_ID,
            "--limit",
            str(args.limit),
            "--skip-scoring",
        ],
        env,
    )
    if len(training["runs"]) != 1:
        raise AssertionError(f"Expected one training run, found {len(training['runs'])}")
    episodes_used = training["runs"][0]["metrics"]["episodes_used"]
    if episodes_used != expected_list_count:
        raise AssertionError(
            f"Expected training to use {expected_list_count} episodes, used {episodes_used}"
        )

    policy = _run_cli(
        ["task-policy", "--agent-id", AGENT_ID, "--task-id", TASK_ID],
        env,
    )
    current = policy["current_policy"]
    previous = policy["previous_policy"]
    if previous is None or current["version"] != previous["version"] + 1:
        raise AssertionError("Training did not create the next policy version")

    before = previous["action_probabilities"][correct_action_id]
    after = current["action_probabilities"][correct_action_id]
    if after <= before:
        raise AssertionError(
            f"Expected {correct_action_id!r} probability to increase, got {before} -> {after}"
        )
    preferred_action = max(
        current["action_probabilities"],
        key=current["action_probabilities"].get,
    )
    if preferred_action != correct_action_id:
        raise AssertionError(f"The trained policy did not prefer {correct_action_id!r}")

    print("\nBatch functional test passed")
    print(f"Store: {store_dir}")
    print(f"Episodes trained: {episodes_used}")
    print(f"Policy version: {previous['version']} -> {current['version']}")
    print(f"Correct-action probability: {before:.4f} -> {after:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())