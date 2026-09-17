"""Map Microsoft Agent Framework tools to agent-learning actions."""

from __future__ import annotations

from collections.abc import Sequence

from agent_framework import FunctionTool

from ...types import Action


def map_action_tools(
    action_tools: Sequence[FunctionTool],
) -> tuple[dict[str, FunctionTool], list[Action]]:
    """Validate MAF registrations and derive the native action space."""
    tools_by_name: dict[str, FunctionTool] = {}
    for function_tool in action_tools:
        if function_tool.name in tools_by_name:
            raise ValueError(f"duplicate action tool name: {function_tool.name!r}")
        if function_tool.approval_mode != "never_require":
            raise ValueError(
                f"action tool {function_tool.name!r} must use "
                "approval_mode='never_require'; agent-learning controls approval "
                "for wrapped actions"
            )
        tools_by_name[function_tool.name] = function_tool

    actions = [
        Action(
            id=function_tool.name,
            description=function_tool.description,
            parameters={"input_schema": function_tool.parameters()},
        )
        for function_tool in tools_by_name.values()
    ]
    return tools_by_name, actions


__all__ = ["map_action_tools"]
