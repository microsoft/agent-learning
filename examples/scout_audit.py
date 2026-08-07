"""Audit Scout automation, skill, and MCP executions to a local JSONL file."""

from __future__ import annotations

import asyncio

from agent_learning import ScoutAuditAdapter

audit = ScoutAuditAdapter("scout-audit.jsonl")


def run_automation() -> str:
    return "Weekly summary created"


def run_skill(text: str) -> str:
    return text.upper()


async def run_mcp(owner: str, repo: str) -> dict:
    return {"owner": owner, "repo": repo, "open_issues": 2}


async def main() -> None:
    audit.execute(
        intent="Create the weekly summary",
        action_path=["automation", "weekly-summary"],
        action=run_automation,
        expected_tokens=["summary", "created"],
    )
    audit.execute(
        intent="Format the summary as uppercase",
        action_path=["skill", "uppercase"],
        action=run_skill,
        args=("weekly summary",),
        contract={"required_substrings": ["SUMMARY"]},
    )
    await audit.execute_async(
        intent="List open issues in microsoft/agents-learning-sdk",
        action_path=["mcp", "github", "list_issues"],
        action=run_mcp,
        action_kwargs={"owner": "microsoft", "repo": "agents-learning-sdk"},
        expected_tokens=["open_issues"],
    )
    print(f"Wrote three audit records to {audit.audit_path}")


if __name__ == "__main__":
    asyncio.run(main())
