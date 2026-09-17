"""Map Microsoft Agent Framework tools to agent-learning actions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from agent_framework import FunctionTool

from ...types import Action

_LEARNING_METADATA_KEY = "agent_learning"


def learning_action(*, task_id: str) -> Callable[[FunctionTool], FunctionTool]:
    """Mark a MAF function tool as an action for one learning task."""
    if not task_id:
        raise ValueError("task_id must not be empty")

    def decorate(function_tool: FunctionTool) -> FunctionTool:
        properties = dict(function_tool.additional_properties or {})
        learning_metadata = dict(properties.get(_LEARNING_METADATA_KEY) or {})
        task_ids = set(learning_metadata.get("task_ids") or [])
        task_ids.add(task_id)
        learning_metadata["task_ids"] = sorted(task_ids)
        properties[_LEARNING_METADATA_KEY] = learning_metadata
        function_tool.additional_properties = properties
        return function_tool

    return decorate


def belongs_to_task(function_tool: FunctionTool, task_id: str) -> bool:
    """Return whether a tool is marked as an action for the task."""
    properties = function_tool.additional_properties or {}
    learning_metadata = properties.get(_LEARNING_METADATA_KEY)
    return isinstance(learning_metadata, Mapping) and task_id in (
        learning_metadata.get("task_ids") or []
    )


def map_action_tools(
    action_tools: Sequence[FunctionTool],
    *,
    task_id: str,
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
        if not belongs_to_task(function_tool, task_id):
            raise ValueError(
                f"action tool {function_tool.name!r} is not marked for task {task_id!r}"
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


__all__ = ["belongs_to_task", "learning_action", "map_action_tools"]
