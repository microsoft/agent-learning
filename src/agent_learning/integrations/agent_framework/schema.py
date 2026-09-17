"""Build the composite input schema exposed to Microsoft Agent Framework."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from agent_framework import FunctionTool


def _rewrite_refs(value: Any, *, tool_name: str) -> Any:
    if isinstance(value, dict):
        rewritten = {}
        for key, item in value.items():
            if key == "$ref":
                if not isinstance(item, str) or not item.startswith("#/$defs/"):
                    raise ValueError(
                        f"action tool {tool_name!r} contains an unsupported $ref: {item!r}"
                    )
                item = f"#/$defs/{tool_name}__{item.removeprefix('#/$defs/')}"
            rewritten[key] = _rewrite_refs(item, tool_name=tool_name)
        return rewritten
    if isinstance(value, list):
        return [_rewrite_refs(item, tool_name=tool_name) for item in value]
    return value


def build_composite_schema(
    action_tools: Mapping[str, FunctionTool],
) -> dict[str, Any]:
    """Nest each child schema under its exact tool name."""
    action_properties: dict[str, Any] = {}
    definitions: dict[str, Any] = {}

    for tool_name, function_tool in action_tools.items():
        child_schema = deepcopy(function_tool.parameters())
        child_definitions = child_schema.pop("$defs", {})
        if not isinstance(child_definitions, dict):
            raise TypeError(f"action tool {tool_name!r} has invalid $defs")

        action_properties[tool_name] = _rewrite_refs(
            child_schema,
            tool_name=tool_name,
        )
        for definition_name, definition in child_definitions.items():
            definitions[f"{tool_name}__{definition_name}"] = _rewrite_refs(
                definition,
                tool_name=tool_name,
            )

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action_inputs": {
                "type": "object",
                "properties": action_properties,
                "required": list(action_properties),
                "additionalProperties": False,
            },
        },
        "required": ["action_inputs"],
        "additionalProperties": False,
    }
    if definitions:
        schema["$defs"] = definitions
    return schema


__all__ = ["build_composite_schema"]
