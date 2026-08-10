"""Tests for the Linux install helper script."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_install_script_reports_the_release_artifact(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "install-linux.sh"

    completed = subprocess.run(
        [
            "bash",
            str(script_path),
            "--version",
            "0.7.0",
            "--install-dir",
            str(tmp_path),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Dry run: would install agent-learn" in completed.stdout
    assert "https://github.com/microsoft/agent-learning/releases/download/v0.7.0/agent-learning-cli-0.7.0-linux-x64.tar.gz" in completed.stdout
    assert str(tmp_path) in completed.stdout
