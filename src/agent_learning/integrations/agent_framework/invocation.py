"""Invoke the policy-selected MAF child tool and capture its episode."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from typing import Any

from agent_framework import FunctionInvocationContext, FunctionTool

from ...capture import CaptureContext, EpisodeCapture
from ...decision import DecisionResult
from ...types import Action

EpisodeMetadataResolver = Callable[
    [str, Mapping[str, Any], Any],
    Mapping[str, Any],
]


def serialize_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, default=str)


async def invoke_selected_action(
    *,
    decision: DecisionResult,
    action: Action,
    action_tools: Mapping[str, FunctionTool],
    action_inputs: Mapping[str, Mapping[str, Any]],
    capture: EpisodeCapture,
    capture_context: CaptureContext,
    context: FunctionInvocationContext,
    episode_metadata_resolver: EpisodeMetadataResolver | None = None,
) -> dict[str, Any]:
    """Invoke a resolved policy action and close its capture context."""
    selected_tool = action_tools.get(action.id)
    if selected_tool is None:
        raise ValueError(f"policy action {action.id!r} has no registered MAF tool")
    if selected_tool.name not in action_inputs:
        raise ValueError(f"missing arguments for selected action {selected_tool.name!r}")

    selected_arguments = dict(action_inputs[selected_tool.name])
    child_context = FunctionInvocationContext(
        function=selected_tool,
        arguments=selected_arguments,
        session=context.session,
        metadata=context.metadata,
        kwargs=context.kwargs,
        tools=context.tools,
    )
    started = time.perf_counter()
    try:
        result = await selected_tool.invoke(
            arguments=selected_arguments,
            context=child_context,
            skip_parsing=True,  # TODO review
        )
    except BaseException as exc:
        capture.record_tool_call(
            capture_context,
            selected_tool.name,
            selected_arguments,
            duration_ms=int((time.perf_counter() - started) * 1000),
            error=f"{type(exc).__name__}: {exc}",
        )
        capture.end(
            capture_context,
            "",
            execution_status="failed",
            result_summary=str(exc),
        )
        raise

    serialized_result = serialize_result(result)
    capture.record_tool_call(
        capture_context,
        selected_tool.name,
        selected_arguments,
        result=serialized_result,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )
    extra_metadata = (
        dict(episode_metadata_resolver(action.id, selected_arguments, result))
        if episode_metadata_resolver is not None
        else None
    )
    capture.end(
        capture_context,
        serialized_result,
        execution_status="completed",
        result_summary=f"Executed {selected_tool.name}",
        extra_metadata=extra_metadata,
    )
    return {
        "decision": decision.to_dict(),
        "selected_action": action.id,
        "result": result,
    }


__all__ = ["EpisodeMetadataResolver", "invoke_selected_action", "serialize_result"]
