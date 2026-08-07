"""Tests for the local Scout audit adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_learning.scout import ScoutAuditAdapter


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_execute_records_outcome_and_local_judge_signals(tmp_path: Path) -> None:
    path = tmp_path / "scout-audit.jsonl"
    adapter = ScoutAuditAdapter(path)

    result = adapter.execute(
        intent="Create the weekly summary",
        action_path=["automation", "weekly-summary"],
        action=lambda: {"message": "Summary created"},
        contract={"required_substrings": ["summary"]},
        expected_tokens=["created"],
    )

    assert result == {"message": "Summary created"}
    records = _records(path)
    assert len(records) == 1
    record = records[0]
    assert record["intent"] == "Create the weekly summary"
    assert record["action_path"] == ["automation", "weekly-summary"]
    assert record["outcome"]["status"] == "succeeded"
    assert record["outcome"]["result"] == {"message": "Summary created"}
    assert set(record["judge_signals"]) == {"intent", "adherence", "completion"}
    assert record["judge_signals"]["adherence"]["normalized"] == 1.0
    assert record["judge_signals"]["completion"]["normalized"] == 1.0


def test_execute_records_failure_then_reraises(tmp_path: Path) -> None:
    path = tmp_path / "scout-audit.jsonl"
    adapter = ScoutAuditAdapter(path)

    def fail() -> None:
        raise RuntimeError("skill failed")

    with pytest.raises(RuntimeError, match="skill failed"):
        adapter.execute(
            intent="Run the formatter skill",
            action_path=["skill", "formatter"],
            action=fail,
        )

    record = _records(path)[0]
    assert record["outcome"]["status"] == "failed"
    assert record["outcome"]["result"] is None
    assert record["outcome"]["error"] == "skill failed"


def test_execute_rejects_invalid_action_path_before_running(tmp_path: Path) -> None:
    called = False

    def action() -> None:
        nonlocal called
        called = True

    with pytest.raises(ValueError, match="action_path"):
        ScoutAuditAdapter(tmp_path / "audit.jsonl").execute(
            intent="Do something",
            action_path=[],
            action=action,
        )

    assert called is False


def test_adapter_requires_all_three_judges(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="intent, adherence, and completion"):
        ScoutAuditAdapter(tmp_path / "audit.jsonl", judges=())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_execute_async_appends_one_record_per_action(tmp_path: Path) -> None:
    path = tmp_path / "scout-audit.jsonl"
    adapter = ScoutAuditAdapter(path)

    async def call_mcp(value: int) -> str:
        return f"Found {value} issues"

    assert await adapter.execute_async(
        intent="List open issues",
        action_path=["mcp", "github", "list_issues"],
        action=call_mcp,
        args=(2,),
        expected_tokens=["issues"],
    ) == "Found 2 issues"

    records = _records(path)
    assert len(records) == 1
    assert records[0]["action_path"] == ["mcp", "github", "list_issues"]
    assert records[0]["outcome"]["status"] == "succeeded"


def test_sensitive_values_are_redacted(tmp_path: Path) -> None:
    path = tmp_path / "scout-audit.jsonl"
    adapter = ScoutAuditAdapter(path)

    adapter.execute(
        intent="Use token=private-value",
        action_path=["skill", "lookup"],
        action=lambda: {"note": "done"},
    )

    content = path.read_text(encoding="utf-8")
    assert "private-value" not in content
    assert "[REDACTED]" in content
