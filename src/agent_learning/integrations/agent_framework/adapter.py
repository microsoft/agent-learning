"""Microsoft Agent Framework integration."""

from __future__ import annotations

import random
import uuid
from collections.abc import Awaitable, Callable, Mapping, MutableMapping, Sequence
from typing import Any, Literal, cast

from agent_framework import (
    SKIP_PARSING,
    AgentContext,
    AgentResponse,
    Content,
    FunctionInvocationContext,
    FunctionTool,
    Message,
    agent_middleware,
    function_middleware,
)

from ...autonomy import assess_autonomy
from ...capture import EpisodeCapture
from ...config import AutonomyConfig, CaptureConfig
from ...decision import (
    DecisionAuthority,
    DecisionFrame,
    DecisionResult,
    DecisionStatus,
    TaskPolicy,
    TieBreakDisposition,
)
from ...policy import SoftmaxPolicy
from ...storage import LearningStore, get_default_store
from ...types import Action, PolicySnapshot
from .actions import map_action_tools
from .invocation import EpisodeMetadataResolver, invoke_selected_action
from .schema import build_composite_schema
from .state import (
    PENDING_DECISIONS_KEY,
    USER_INPUT_KEY,
    PendingDecision,
)

StateEncoder = Callable[[dict[str, Any]], Mapping[str, Any]]
TargetResolver = Callable[[Mapping[str, Any]], str | None]
DecisionFrameResolver = Callable[[Mapping[str, Any], Sequence[Action]], DecisionFrame]
AgentMiddlewareMethod = Callable[
    [Any, AgentContext, Callable[[], Awaitable[None]]],
    Awaitable[None],
]
FunctionMiddlewareMethod = Callable[
    [Any, FunctionInvocationContext, Callable[[], Awaitable[None]]],
    Awaitable[None],
]
agent_middleware_method = cast(
    Callable[[AgentMiddlewareMethod], AgentMiddlewareMethod],
    agent_middleware,
)
function_middleware_method = cast(
    Callable[[FunctionMiddlewareMethod], FunctionMiddlewareMethod],
    function_middleware,
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
        autonomy_config: AutonomyConfig | None,
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
        self.autonomy_config = autonomy_config
        self._rng = rng
        self._audit_rng = rng or random.Random()
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
                result_parser=SKIP_PARSING,
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
        autonomy_config: AutonomyConfig | None = None,
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
            autonomy_config=autonomy_config,
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
        return [self.process_agent, self.process_function]

    @function_middleware_method
    async def process_function(
        self,
        context: FunctionInvocationContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        """Passthrough function middleware that makes call_id available downstream in invocation context.

        Just passing through this middleware makes it possible for _execute_learning_action to access call_id
        through context.metadata.get("call_id"), which it needs to resolve approvals ."""
        await call_next()

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
        # unlike approvals which are dispatched by agent-framework to the corresponding tool,
        # user rejections never reach the tool and need to be handled by this middleware instead:
        next_approval = self._handle_rejected_approval_responses(context)
        if next_approval is not None:
            # request new approval from user before doing anything else
            context.result = AgentResponse(
                messages=[Message(role="assistant", contents=[next_approval])]
            )
            return
        await call_next()

    async def _execute_learning_action(
        self,
        decision_context: dict[str, Any],
        action_inputs: dict[str, dict[str, Any]],
        context: FunctionInvocationContext,
    ) -> Any:
        """Drive integration of agent-learning with agent-framework through a meta-tool.

        This tool wraps self._tools and appears as one tool to agents. When agents call this meta-tool,
        it handles decision making, approvals, and execution of the selected actions.
        This tool takes as inputs the union of inputs of all wrapped tools, so that once the decision is made,
        it can proceed with execution within one call_id without requiring an additional round trip to the
        agent.
        """
        user_input = context.kwargs.get(USER_INPUT_KEY)
        if not isinstance(user_input, str):
            raise TypeError("learning meta-tool invoked outside configured agent middleware")

        # was the current call_id we are processing triggered by a user approval?
        # if yes, find the decision details from session state instead of re-running decision making
        pending = self._restore_approved_decision(context)
        if pending is not None:
            return await self._execute_pending_decision(pending, context)

        # not triggered by a user approval, proceed with normal decision making
        autonomy: dict[str, Any] | None = None
        approval_required = False
        if self.task_policy.authority is DecisionAuthority.FULL:
            if self.decision_frame_resolver is None:  # pragma: no cover - constructor invariant
                raise RuntimeError("full decision authority has no frame resolver")
            decision = self.task_policy.decide(
                self.decision_frame_resolver(
                    decision_context,
                    self.task_policy.snapshot().actions,
                )
            )
            if decision.status in {
                DecisionStatus.NEEDS_USER_FEEDBACK,
                DecisionStatus.NEEDS_USER_TIE_BREAK,
            }:
                action = decision.proposed_action
                approval_required = True
            elif decision.status is DecisionStatus.RESOLVED:
                action = decision.selected_action
            elif decision.status in {DecisionStatus.NEEDS_EVIDENCE, DecisionStatus.NO_VIABLE_OPTION}:
                # let agent handle the situation
                return {
                    "decision": decision.to_dict(),
                    "selected_action": None,
                    "result": None,
                }
        else:
            snapshot = self.task_policy.snapshot()
            assessment = assess_autonomy(
                self.capture.store,
                snapshot,
                self.autonomy_config,
            )
            audit_sampled = (
                assessment.eligible
                and self._audit_rng.random() < assessment.audit_rate
            )
            decision = self.task_policy.decide(
                selected_action_id=(
                    assessment.recommended_action_id
                    if assessment.eligible
                    else None
                )
            )
            action = decision.proposed_action
            autonomy = {
                **assessment.to_dict(),
                "audit_sampled": audit_sampled,
            }
            if not assessment.eligible or audit_sampled:
                approval_required = True
        if action is None:
            raise ValueError("policy decision did not select an executable action")

        if approval_required:
            pending = PendingDecision(
                decision=decision,
                action_id=action.id,
                decision_context=dict(decision_context),
                action_inputs={
                    name: dict(arguments)
                    for name, arguments in action_inputs.items()
                },
                user_input=user_input,
                autonomy=autonomy,
            )
            return self._request_approval(context, pending)

        return await self._execute_decision(
            decision=decision,
            action=action,
            decision_context=decision_context,
            action_inputs=action_inputs,
            user_input=user_input,
            context=context,
            autonomy=autonomy,
        )

    async def _execute_pending_decision(
        self,
        pending: PendingDecision,
        context: FunctionInvocationContext,
    ) -> dict[str, Any]:
        decision = self.task_policy.adjudicate(
            pending.decision,
            TieBreakDisposition.ACCEPT,
        )
        action = decision.selected_action or next(
            action
            for action in self.task_policy.snapshot().actions
            if action.id == pending.action_id
        )
        return await self._execute_decision(
            decision=decision,
            action=action,
            decision_context=pending.decision_context,
            action_inputs=pending.action_inputs,
            user_input=pending.user_input,
            context=context,
            autonomy=pending.autonomy,
            feedback_status="accepted",
        )

    async def _execute_decision(
        self,
        *,
        decision: DecisionResult,
        action: Action,
        decision_context: Mapping[str, Any],
        action_inputs: Mapping[str, Mapping[str, Any]],
        user_input: str,
        context: FunctionInvocationContext,
        autonomy: Mapping[str, Any] | None,
        feedback_status: Literal["accepted"] | None = None,
    ) -> dict[str, Any]:

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
            metadata={
                "decision": decision.to_dict(),
                **({"autonomy": dict(autonomy)} if autonomy else {}),
                **(
                    {
                        "feedback_status": feedback_status,
                        "correct_action_id": action.id,
                    }
                    if feedback_status
                    else {}
                ),
            },
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

    def _request_approval(
        self,
        context: FunctionInvocationContext,
        pending: PendingDecision,
    ) -> Content:
        """Save the decision details for the current call_id and prepare the approval request.

        By storing the decision details in the session state, this method lets the adapter retrieve all
        details post-approval since agent-framework will call the tool with the same call_id.
        """
        if context.session is None:
            raise RuntimeError(
                "agent-learning decision approval requires an AgentSession"
            )
        call_id = context.metadata.get("call_id")
        if not isinstance(call_id, str):
            raise TypeError("MAF invocation did not provide a call ID")
        pending_decisions = self._pending_decisions(context.session.state)
        pending_decisions[call_id] = pending.to_dict()
        function_call = Content.from_function_call(
            call_id=call_id,
            id=call_id,
            name=context.function.name,
            arguments={
                "decision_context": pending.decision_context,
                "action_inputs": pending.action_inputs,
            },
        )
        return Content.from_function_approval_request(
            id=call_id,
            function_call=function_call,
            additional_properties={
                "agent_learning": {
                    "action_id": pending.action_id,
                    "decision": pending.decision.to_dict(),
                }
            },
        )

    def _request_follow_up_approval(
        self,
        context: AgentContext,
        response: Content,
        pending: PendingDecision,
    ) -> Content:
        if (
            context.session is None
            or response.function_call is None
            or response.function_call.name is None
        ):
            raise TypeError("MAF approval response is missing session call state")
        call_id = f"agent-learning-{uuid.uuid4().hex}"
        self._pending_decisions(context.session.state)[call_id] = pending.to_dict()
        function_call = Content.from_function_call(
            call_id=call_id,
            id=call_id,
            name=response.function_call.name,
            arguments={
                "decision_context": pending.decision_context,
                "action_inputs": pending.action_inputs,
            },
        )
        return Content.from_function_approval_request(
            id=call_id,
            function_call=function_call,
            additional_properties={
                "agent_learning": {
                    "action_id": pending.action_id,
                    "decision": pending.decision.to_dict(),
                }
            },
        )

    def _restore_approved_decision(
        self,
        context: FunctionInvocationContext,
    ) -> PendingDecision | None:
        """Restore the decision details of a pending action once it's been approved, if it exists.

        Given a function call's context, if that call contains an approval, this method attempts to restore
        the corresponding pending decision details needed for execution.
        """
        approval_response = context.metadata.get("approval_response")
        if not isinstance(approval_response, Content):
            return None
        if approval_response.approved is not True or context.session is None:
            return None
        if not isinstance(approval_response.id, str):
            raise TypeError("MAF approval response has no correlation ID")
        pending = self._pending_decisions(context.session.state).pop(
            approval_response.id,
            None,
        )
        if not isinstance(pending, dict):
            raise TypeError("approved agent-learning decision is no longer pending")
        pending_decision = PendingDecision.from_dict(pending)
        self._validate_approval_response(
            approval_response,
            pending_decision,
            call_id=context.metadata.get("call_id"),
        )
        return pending_decision

    def _handle_rejected_approval_responses(self, context: AgentContext) -> Content | None:
        """Record rejections and return the next tie-break approval, if needed."""
        if context.session is None:
            return None
        pending_decisions = self._pending_decisions(context.session.state)
        for message in context.messages:
            for content in message.contents:
                if (
                    content.type != "function_approval_response"
                    or content.approved is not False
                    or not isinstance(content.id, str)
                    or content.id not in pending_decisions
                ):
                    continue
                pending = PendingDecision.from_dict(
                    pending_decisions.pop(content.id)
                )
                self._validate_approval_response(content, pending)
                decision = self.task_policy.adjudicate(
                    pending.decision,
                    TieBreakDisposition.REJECT,
                )
                if decision.status is DecisionStatus.NEEDS_USER_TIE_BREAK:
                    if decision.proposed_action is None:
                        raise RuntimeError("tie-break decision has no proposed action")
                    return self._request_follow_up_approval(
                        context,
                        content,
                        PendingDecision(
                            decision=decision,
                            action_id=decision.proposed_action.id,
                            decision_context=pending.decision_context,
                            action_inputs=pending.action_inputs,
                            user_input=pending.user_input,
                            autonomy=pending.autonomy,
                        ),
                    )
                capture_context = self.capture.start(
                    pending.user_input,
                    task_id=decision.task_id,
                    intent_summary=self.intent_summary,
                    action_type="tool",
                    action_name=pending.action_id,
                    target=(
                        self.target_resolver(pending.decision_context)
                        if self.target_resolver
                        else None
                    ),
                    input_summary=str(pending.decision_context),
                    expected_outcome=self.expected_outcome,
                    policy_id=decision.policy_id,
                    policy_version=decision.policy_version,
                    action_id=pending.action_id,
                    action_logprob=decision.action_logprob,
                    metadata={
                        "decision": decision.to_dict(),
                        "feedback_status": "rejected",
                        **(
                            {"autonomy": pending.autonomy}
                            if pending.autonomy
                            else {}
                        ),
                    },
                )
                self.capture.end(
                    capture_context,
                    "",
                    execution_status="rejected",
                    result_summary="User rejected execution",
                )
        return None

    def _validate_approval_response(
        self,
        response: Content,
        pending: PendingDecision,
        *,
        call_id: Any = None,
    ) -> None:
        function_call = response.function_call
        if (
            not isinstance(response.id, str)
            or function_call is None
            or function_call.name != self._tools[0].name
            or response.id != function_call.call_id
            or (call_id is not None and function_call.call_id != call_id)
        ):
            raise ValueError("approval response does not match the pending tool call")
        arguments = function_call.parse_arguments()
        expected_arguments = {
            "decision_context": pending.decision_context,
            "action_inputs": pending.action_inputs,
        }
        if not isinstance(arguments, Mapping) or dict(arguments) != expected_arguments:
            raise ValueError("approval response arguments do not match the pending decision")

    @staticmethod
    def _pending_decisions(
        session_state: MutableMapping[str, Any],
    ) -> MutableMapping[str, Any]:
        value = session_state.setdefault(PENDING_DECISIONS_KEY, {})
        if not isinstance(value, MutableMapping):
            raise TypeError("pending agent-learning session state must be a mapping")
        return value


__all__ = [
    "AgentFrameworkLearningAdapter",
    "DecisionFrameResolver",
    "EpisodeMetadataResolver",
    "StateEncoder",
    "TargetResolver",
]
