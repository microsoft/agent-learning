"""Exercise the interactive agent-learning CLI and SDK workflow.

Run this script before ``functional_cli_batch.py``. It uses an isolated local
store under ``data/functional-tests`` and replaces that store on every run.
The clinical scenario is illustrative test data, not medical guidance.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
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


def _load_actions(actions_path: Path) -> tuple[list[dict[str, Any]], str]:
    actions = json.loads(actions_path.read_text(encoding="utf-8"))
    correct = [
        action["id"]
        for action in actions
        if action.get("parameters", {}).get("is_correct_for_sore_throat_case") is True
    ]
    if len(correct) != 1:
        raise ValueError("The actions fixture must identify exactly one correct action")
    return actions, correct[0]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--actions", type=Path, default=DEFAULT_ACTIONS_PATH)
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 500:
        parser.error("--episodes must be between 1 and 500")
    return args


def _reset_store(store_dir: Path) -> None:
    if store_dir.exists():
        if not store_dir.is_dir():
            raise ValueError(f"Store path is not a directory: {store_dir}")
        if store_dir != DEFAULT_STORE_DIR.resolve() and any(store_dir.iterdir()):
            raise ValueError(
                f"Refusing to replace nonempty custom store directory: {store_dir}"
            )
        shutil.rmtree(store_dir)
    store_dir.mkdir(parents=True)


def main() -> int:
    args = _parse_args()
    actions_path = args.actions.resolve()
    store_dir = args.store_dir.resolve()
    _, correct_action_id = _load_actions(actions_path)

    _reset_store(store_dir)
    env = _cli_environment(store_dir)

    initialized = _run_cli(
        [
            "task-policy-init",
            "--agent-id",
            AGENT_ID,
            "--task-id",
            TASK_ID,
            "--actions",
            str(actions_path),
        ],
        env,
    )

    sys.path.insert(0, str(SRC_DIR))
    from agent_learning import (
        LocalFileStore,
        MetricName,
        MetricResult,
        PolicySnapshot,
        RewardShaper,
        RewardWriter,
        SoftmaxPolicy,
    )

    policy = SoftmaxPolicy.from_snapshot(
        PolicySnapshot.from_dict(initialized),
        rng=random.Random(args.seed),
    )
    store = LocalFileStore(store_dir)
    reward_shaper = RewardShaper()
    reward_writer = RewardWriter(store)
    successful = 0

    with tempfile.TemporaryDirectory(prefix="agent-learning-episodes-") as temp_dir:
        episode_dir = Path(temp_dir)
        for index in range(args.episodes):
            decision = policy.choose()
            task_completed = decision.action.id == correct_action_id
            successful += int(task_completed)
            episode_path = episode_dir / f"episode-{index + 1}.json"
            episode_path.write_text(
                json.dumps(
                    {
                        "agent_name": "Triage Nurse Agent",
                        "task_name": "Triage a patient with a sore throat",
                        "user_input": (
                            "A patient arrives with a sore throat and pain when swallowing, "
                            "without breathing difficulty. Choose the next triage action."
                        ),
                        "assistant_output": (
                            f"Selected action: {decision.action.id}. "
                            f"{decision.action.description or ''}"
                        ).strip(),
                        "intent_summary": "Select the next triage action for a sore throat",
                        "action_type": "clinical_triage_simulation",
                        "action_id": decision.action.id,
                        "action_name": decision.action.description,
                        "target": "patient with a sore throat",
                        "input_summary": "Sore throat with painful swallowing and no breathing difficulty",
                        "expected_outcome": "Order a diagnostic test for strep throat",
                        "execution_status": "completed" if task_completed else "failed",
                        "result_summary": (
                            "Ordered the strep throat test"
                            if task_completed
                            else f"Selected {decision.action.id} instead of ordering the strep throat test"
                        ),
                        "policy_id": initialized["id"],
                        "policy_version": initialized["version"],
                        "action_logprob": decision.logprob,
                        "metadata": {
                            "functional_test": True,
                            "correct_action_id": correct_action_id,
                            "task_completed": task_completed,
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            registered = _run_cli(
                [
                    "task-episode-register",
                    "--agent-id",
                    AGENT_ID,
                    "--task-id",
                    TASK_ID,
                    "--episode",
                    str(episode_path),
                ],
                env,
            )

            episode = store.get_episode(registered["id"], AGENT_ID)
            if episode is None or not episode.is_full:
                raise AssertionError("The CLI did not persist a full episode")
            quality = 1.0 if task_completed else 0.0
            metrics = [
                MetricResult(
                    metric=metric,
                    score=quality,
                    normalized=quality,
                    status="completed",
                    evaluator="functional-simulator",
                )
                for metric in (
                    MetricName.INTENT_RESOLUTION,
                    MetricName.TASK_ADHERENCE,
                    MetricName.TASK_COMPLETION,
                )
            ]
            reward_writer.write(episode, metrics, reward_shaper.shape(metrics))

    count = _run_cli(
        ["task-episodes-count", AGENT_ID, "--task-id", TASK_ID],
        env,
    )
    if count != args.episodes:
        raise AssertionError(f"Expected {args.episodes} episodes, found {count}")

    print("\nInteractive functional test passed")
    print(f"Store: {store_dir}")
    print(f"Episodes: {count} ({successful} successful, {count - successful} unsuccessful)")
    print(f"Correct action: {correct_action_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())