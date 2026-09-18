"""Subprocess operator workflow without cloud dependencies or model calls."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _run(script: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["AGENT_LEARNING_STORE_BACKEND"] = "cosmos"
    env["AGENT_LEARNING_SCORE_TIER"] = "llm"
    env["AGENT_LEARNING_LR"] = "0.00001"
    return subprocess.run(
        [sys.executable, "-B", str(_ROOT / "examples" / script), *arguments],
        env=env, capture_output=True, text=True, check=False,
    )


def test_demo_and_operator_commands(tmp_path: Path) -> None:
    root = tmp_path / "demo"
    demo = _run("audit_replay_demo.py", "--store-dir", str(root))
    assert demo.returncode == 0, demo.stderr
    manifest = json.loads(demo.stdout)
    assert manifest["before_rescore"] == manifest["after_rescore"] == "verified"
    assert manifest["output_policy"]["probabilities"]["catalog"] > manifest["input_policy"]["probabilities"]["catalog"]
    initial_files = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    arguments = ["--store-dir", str(root), "--agent-id", "audit-demo", "--run-id", manifest["run_id"]]

    inspected = _run("audit_run.py", *arguments, "--format", "json")
    assert inspected.returncode == 0, inspected.stderr
    report = json.loads(inspected.stdout)
    assert report["status"] == "verified"
    assert len(report["contributions"]) == 40
    assert all(item["reward_decomposition"] for item in report["contributions"])
    assert {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()} == initial_files

    text = _run("audit_run.py", *arguments)
    assert text.returncode == 0, text.stderr
    assert "VERIFIED [reconciled]" in text.stdout
    assert "Entropy contribution" in text.stdout
    assert "Recorded reward composition" in text.stdout
    assert "Scope: numerical consistency, not immutable provenance" in text.stdout

    repeat = _run("audit_replay_demo.py", "--store-dir", str(root))
    assert repeat.returncode == 2
    assert "Existing data will not be replaced" in repeat.stderr
    assert {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()} == initial_files

    run_path = root / "runs" / "audit-demo" / f"{manifest['run_id']}.json"
    run = json.loads(run_path.read_text())
    run["metrics"]["mean_reward"] += 0.3
    run_path.write_text(json.dumps(run))
    mismatch = _run("audit_run.py", *arguments, "--format", "json")
    assert mismatch.returncode == 1
    assert json.loads(mismatch.stdout)["status"] == "mismatch"

    run["metadata"].pop("lineage_version")
    run_path.write_text(json.dumps(run))
    legacy = _run("audit_run.py", *arguments, "--format", "json")
    assert legacy.returncode == 2
    assert json.loads(legacy.stdout)["status"] == "insufficient_lineage"

    missing = _run(
        "audit_run.py", "--store-dir", str(root), "--agent-id", "audit-demo",
        "--run-id", "absent", "--format", "json",
    )
    assert missing.returncode == 3
    assert json.loads(missing.stdout)["status"] == "missing_record"


def test_operator_refuses_nonexistent_store_without_creating_it(tmp_path: Path) -> None:
    root = tmp_path / "missing"
    result = _run("audit_run.py", "--store-dir", str(root), "--agent-id", "agent", "--run-id", "run")

    assert result.returncode == 2
    assert not root.exists()
