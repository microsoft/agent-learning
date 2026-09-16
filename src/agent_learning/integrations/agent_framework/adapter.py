"""Microsoft Agent Framework integration."""

from __future__ import annotations

import random
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, cast

from agent_framework import (
    AgentContext,
    FunctionInvocationContext,
    FunctionTool,
    agent_middleware,
)

from ...capture import EpisodeCapture
from ...config import CaptureConfig
from ...decision import DecisionAuthority, DecisionFrame, DecisionStatus, TaskPolicy
from ...policy import SoftmaxPolicy
from ...storage import LearningStore, get_default_store
from ...types import Action, PolicySnapshot
from .actions import map_action_tools
from .invocation import EpisodeMetadataResolver, invoke_selected_action
from .schema import build_composite_schema
from .state import USER_INPUT_KEY

StateEncoder = Callable[[dict[str, Any]], Mapping[str, Any]]
TargetResolver = Callable[[dict[str, Any]], str | None]
DecisionFrameResolver = Callable[[Mapping[str, Any], Sequence[Action]], DecisionFrame]
AgentMiddlewareMethod = Callable[
    [Any, AgentContext, Callable[[], Awaitable[None]]],
    Awaitable[None],
]
agent_middleware_method = cast(
    Callable[[AgentMiddlewareMethod], AgentMiddlewareMethod],
    agent_middleware,
)


class AgentFrameworkLearningAdapter:
    """Expose an agent-learning policy as one composite MAF tool."""

    def __init__(
        self,
        *,
        task_policy: TaskPolicy,
        action_tools: Mapping[str, FunctionTool],
        capture: EpisodeCapture,
        intent_summary: str,
        expected_outcome: str,
        state_encoder: StateEncoder | None,
        target_resolver: TargetResolver | None,
        decision_frame_resolver: DecisionFrameResolver | None,
        episode_metadata_resolver: EpisodeMetadataResolver | None,
        agent_name: str | None,
        rng: random.Random | None,
    ) -> None:
        policy_kind = task_policy.snapshot().metadata.get("policy_kind")
        if state_encoder is not None:
            raise NotImplementedError(
                "state_encoder is reserved for future contextual policy support"
            )
        if policy_kind not in (None, "softmax"):
            raise NotImplementedError(
                f"policy kind {policy_kind!r} is not supported by the MAF integration yet"
            )
        if (
            task_policy.authority is DecisionAuthority.FULL
            and decision_frame_resolver is None
        ):
            raise ValueError(
                "full decision authority requires a decision_frame_resolver"
            )
        if (
            task_policy.authority is DecisionAuthority.LOW
            and decision_frame_resolver is not None
        ):
            raise ValueError(
                "decision_frame_resolver is only valid for full decision authority"
            )
        self.task_policy = task_policy
        self.action_tools = dict(action_tools)
        self.capture = capture
        self.intent_summary = intent_summary
        self.expected_outcome = expected_outcome
        self.state_encoder = state_encoder
        self.target_resolver = target_resolver
        self.decision_frame_resolver = decision_frame_resolver
        self.episode_metadata_resolver = episode_metadata_resolver
        self.agent_name = agent_name
        self._rng = rng
        # TODO give this a better agent-facing description:
        self._tools = (
            FunctionTool(
                name="execute_learning_action",
                description=(
                    "Resolve a policy-controlled decision and execute its authorized "
                    "action when ready. Supply valid arguments for every candidate action."
                ),
                input_model=build_composite_schema(self.action_tools),
                func=self._execute_learning_action,
            ),
        )

    @classmethod
    def from_tools(
        cls,
        *,
        agent_id: str,
        task_id: str,
        action_tools: Sequence[FunctionTool],
        intent_summary: str,
        expected_outcome: str,
        state_encoder: StateEncoder | None = None,
        target_resolver: TargetResolver | None = None,
        decision_authority: DecisionAuthority | str | None = None,
        decision_frame_resolver: DecisionFrameResolver | None = None,
        episode_metadata_resolver: EpisodeMetadataResolver | None = None,
        store: LearningStore | None = None,
        agent_name: str | None = None,
        rng: random.Random | None = None,
    ) -> AgentFrameworkLearningAdapter:
        """Build an integration from tagged MAF action tools."""
        if state_encoder is not None:
            raise NotImplementedError(
                "state_encoder is reserved for future contextual policy support"
            )
        learning_store = store or get_default_store()
        tools_by_name, actions = map_action_tools(action_tools, task_id=task_id)
        active_snapshot = learning_store.get_active_policy(agent_id, task_id)
        requested_authority = (
            DecisionAuthority(decision_authority)
            if decision_authority is not None
            else None
        )
        if active_snapshot is None:
            initial_authority = requested_authority or DecisionAuthority.LOW
            if (
                initial_authority is DecisionAuthority.FULL
                and decision_frame_resolver is None
            ):
                raise ValueError(
                    "full decision authority requires a decision_frame_resolver"
                )
            if (
                initial_authority is DecisionAuthority.LOW
                and decision_frame_resolver is not None
            ):
                raise ValueError(
                    "decision_frame_resolver is only valid for full decision authority"
                )
            policy = SoftmaxPolicy.from_actions(
                actions,
                agent_id=agent_id,
                task_id=task_id,
                rng=rng,
            )
            active_snapshot = policy.snapshot()
            active_snapshot.metadata["decision_authority"] = initial_authority.value
            learning_store.store_policy(active_snapshot)
        elif active_snapshot.actions != actions:
            raise ValueError("registered MAF action tools do not match the active policy actions")

        task_policy = TaskPolicy(active_snapshot, rng=rng)
        if requested_authority is not None and requested_authority is not task_policy.authority:
            raise ValueError(
                "requested decision authority does not match the active policy"
            )

        capture = EpisodeCapture(
            CaptureConfig(
                enabled=True,
                agent_id=agent_id,
                agent_name=agent_name,
                task_id=task_id,
            ),
            store=learning_store,
        )
        return cls(
            task_policy=task_policy,
            action_tools=tools_by_name,
            capture=capture,
            intent_summary=intent_summary,
            expected_outcome=expected_outcome,
            state_encoder=state_encoder,
            target_resolver=target_resolver,
            decision_frame_resolver=decision_frame_resolver,
            episode_metadata_resolver=episode_metadata_resolver,
            agent_name=agent_name,
            rng=rng,
        )

    def update_policy(self, snapshot: PolicySnapshot) -> None:
        """Replace the decision policy with a validated updated snapshot."""
        current_snapshot = self.task_policy.snapshot()
        if snapshot.agent_id != current_snapshot.agent_id:
            raise ValueError("updated policy belongs to a different agent")
        if snapshot.task_id != current_snapshot.task_id:
            raise ValueError("updated policy belongs to a different task")
        if snapshot.actions != current_snapshot.actions:
            raise ValueError("updated policy actions do not match the integration tools")
        policy_kind = snapshot.metadata.get("policy_kind")
        if policy_kind not in (None, "softmax"):
            raise NotImplementedError(
                f"policy kind {policy_kind!r} is not supported by the MAF integration yet"
            )

        task_policy = TaskPolicy(snapshot, rng=self._rng)
        if (
            task_policy.authority is DecisionAuthority.FULL
            and self.decision_frame_resolver is None
        ):
            raise ValueError(
                "full decision authority requires a decision_frame_resolver"
            )
        if (
            task_policy.authority is DecisionAuthority.LOW
            and self.decision_frame_resolver is not None
        ):
            raise ValueError(
                "decision_frame_resolver is only valid for full decision authority"
            )
        self.task_policy = task_policy

    @property
    def tools(self) -> tuple[FunctionTool, ...]:
        return self._tools

    @property
    def middleware(self) -> list[Callable[..., Any]]:
        return [self.process_agent]

    @agent_middleware_method
    async def process_agent(
        self,
        context: AgentContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        # pass the latest user message to _execute_learning_action() through the function invocation context
        # so it can pass it to episode capture
        user_input = next(
            (
                str(getattr(message, "text", ""))
                for message in reversed(context.messages)
                if getattr(message, "role", None) == "user"
            ),
            "",
        )
        context.function_invocation_kwargs[USER_INPUT_KEY] = user_input
        await call_next()

    async def _execute_learning_action(
        self,
        decision_context: dict[str, Any],
        action_inputs: dict[str, dict[str, Any]],
        context: FunctionInvocationContext,
    ) -> dict[str, Any]:
        user_input = context.kwargs.get(USER_INPUT_KEY)
        if not isinstance(user_input, str):
            raise TypeError("learning meta-tool invoked outside configured agent middleware")
        if self.task_policy.authority is DecisionAuthority.FULL:
            if self.decision_frame_resolver is None:  # pragma: no cover - constructor invariant
                raise RuntimeError("full decision authority has no frame resolver")
            decision = self.task_policy.decide(
                self.decision_frame_resolver(
                    decision_context,
                    self.task_policy.snapshot().actions,
                )
            )
            if decision.status is not DecisionStatus.RESOLVED:
                return {
                    "decision": decision.to_dict(),
                    "selected_action": None,
                    "result": None,
                }
            action = decision.selected_action
        else:
            decision = self.task_policy.decide()
            action = decision.proposed_action
        if action is None:
            raise ValueError("policy decision did not select an executable action")

        capture_context = self.capture.start(
            user_input,
            task_id=decision.task_id,
            intent_summary=self.intent_summary,
            action_type="tool",
            action_name=action.id,
            target=(
                self.target_resolver(decision_context)
                if self.target_resolver
                else None
            ),
            input_summary=str(decision_context),
            expected_outcome=self.expected_outcome,
            policy_id=decision.policy_id,
            policy_version=decision.policy_version,
            action_id=action.id,
            action_logprob=decision.action_logprob,
            context_features={},
            metadata={"decision": decision.to_dict()},
        )
        return await invoke_selected_action(
            decision=decision,
            action=action,
            action_tools=self.action_tools,
            action_inputs=action_inputs,
            decision_context=decision_context,
            capture=self.capture,
            capture_context=capture_context,
            context=context,
            episode_metadata_resolver=self.episode_metadata_resolver,
        )


__all__ = [
    "AgentFrameworkLearningAdapter",
    "DecisionFrameResolver",
    "EpisodeMetadataResolver",
    "StateEncoder",
    "TargetResolver",
]
