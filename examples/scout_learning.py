"""Record Scout automation, skill, and MCP executions for local learning."""

from __future__ import annotations

import asyncio

from agent_learning import ScoutLearningAdapter

learning = ScoutLearningAdapter("scout-learning.jsonl")


def run_automation() -> str:
    return "Weekly summary created"


def run_skill(text: str) -> str:
    return text.upper()


async def run_mcp(owner: str, repo: str) -> dict:
    return {"owner": owner, "repo": repo, "open_issues": 2}


async def main() -> None:
    learning.execute(
        intent="Create the weekly summary",
        action_path=["automation", "weekly-summary"],
        action=run_automation,
        expected_tokens=["summary", "created"],
    )
    learning.execute(
        intent="Format the summary as uppercase",
        action_path=["skill", "uppercase"],
        action=run_skill,
        args=("weekly summary",),
        contract={"required_substrings": ["SUMMARY"]},
    )
    await learning.execute_async(
        intent="List open issues in microsoft/agents-learning-sdk",
        action_path=["mcp", "github", "list_issues"],
        action=run_mcp,
        action_kwargs={"owner": "microsoft", "repo": "agents-learning-sdk"},
        expected_tokens=["open_issues"],
    )
    print(f"Wrote three learning records to {learning.learning_path}")


if __name__ == "__main__":
    asyncio.run(main())
