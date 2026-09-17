"""Microsoft Agent Framework integration with full decision authority.

The model calls one composite tool with an incident ID and arguments for every
candidate recovery action. Application-owned code builds a trusted
``DecisionFrame`` from current operational signals and historical outcomes.
Agent-learning applies safety constraints, resolves the evidence, and executes
a child tool only when the result is authorized. Completed episodes are scored
for audit, but full-authority decisions are not trained with REINFORCE.

The incident data and recovery tools are deterministic stand-ins for monitoring,
change-management, incident-history, and orchestration services.

Prerequisites:

* Sign in locally with ``az login`` or run with a managed identity.
* Grant the identity the ``Cognitive Services OpenAI User`` role.
* Set ``AZURE_OPENAI_ENDPOINT`` and ``AZURE_OPENAI_CHAT_DEPLOYMENT_NAME``.
* ``pip install -e '.[agent-framework,examples]'`` from the repository root.

Run with ``python examples/agent_framework_full_authority.py``.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agent_framework import (  # type: ignore[attr-defined]
    Agent,
    AgentSession,
    FunctionTool,
    tool,
)
from agent_framework.openai import OpenAIChatClient
from azure.identity import DefaultAzureCredential

from agent_learning.config import ScoreRuntimeConfig
from agent_learning.decision import (
    DecisionAuthority,
    DecisionCriterion,
    DecisionFrame,
    DecisionOption,
    EvidencePoint,
)
from agent_learning.integrations.agent_framework import (
    AgentFrameworkLearningAdapter,
    learning_action,
)
from agent_learning.storage import InMemoryStore
from agent_learning.training import LearningRunner
from agent_learning.types import Action, RewardSource

AGENT_ID = "maf-full-authority"
TASK_ID = "choose-recovery-action"


@dataclass(frozen=True)
class Incident:
    incident_id: str
    diagnosis: str
    change_approved: bool
    primary_control_plane_available: bool
    secondary_health_support: float
    replication_lag_seconds: int
    recovery_point_objective_seconds: int
    restart_diagnostic_support: float
    restart_estimated_seconds: int
    failover_estimated_seconds: int
    restart_will_restore: bool
    failover_will_restore: bool


@dataclass(frozen=True)
class RecoveryHistory:
    restart_success_rate: float
    restart_samples: int
    failover_success_rate: float
    failover_samples: int


class IncidentCatalog:
    def __init__(self) -> None:
        self.incidents = [
            Incident(
                incident_id="INC-001",
                diagnosis="single unhealthy service process",
                change_approved=True,
                primary_control_plane_available=True,
                secondary_health_support=0.94,
                replication_lag_seconds=8,
                recovery_point_objective_seconds=60,
                restart_diagnostic_support=0.96,
                restart_estimated_seconds=30,
                failover_estimated_seconds=210,
                restart_will_restore=True,
                failover_will_restore=True,
            ),
            Incident(
                incident_id="INC-002",
                diagnosis="primary region network isolation",
                change_approved=True,
                primary_control_plane_available=True,
                secondary_health_support=0.98,
                replication_lag_seconds=12,
                recovery_point_objective_seconds=60,
                restart_diagnostic_support=0.12,
                restart_estimated_seconds=45,
                failover_estimated_seconds=120,
                restart_will_restore=False,
                failover_will_restore=True,
            ),
        ]
        self.recovery_history = RecoveryHistory(
            restart_success_rate=0.78,
            restart_samples=48,
            failover_success_rate=0.91,
            failover_samples=22,
        )
        self._incidents_by_id = {
            incident.incident_id: incident for incident in self.incidents
        }

    def get(self, incident_id: str) -> Incident:
        try:
            return self._incidents_by_id[incident_id]
        except KeyError as exc:
            raise ValueError(f"unknown incident: {incident_id}") from exc


def build_action_tools(catalog: IncidentCatalog) -> list[FunctionTool]:
    @learning_action(task_id=TASK_ID)
    @tool(
        name="restart_service",
        description="Restart the unhealthy service in its current region.",
        approval_mode="never_require",
    )
    async def restart_service(incident_id: str) -> dict[str, Any]:
        await asyncio.sleep(0.2)
        incident = catalog.get(incident_id)
        return {
            "incident_id": incident_id,
            "action": "restart_service",
            "restored": incident.restart_will_restore,
        }

    @learning_action(task_id=TASK_ID)
    @tool(
        name="failover_to_secondary",
        description="Fail over the service to its configured secondary region.",
        approval_mode="never_require",
    )
    async def failover_to_secondary(incident_id: str) -> dict[str, Any]:
        await asyncio.sleep(1.0)
        incident = catalog.get(incident_id)
        return {
            "incident_id": incident_id,
            "action": "failover_to_secondary",
            "restored": incident.failover_will_restore,
        }

    return [restart_service, failover_to_secondary]


@tool(
    name="incident_runbook",
    description="Return the standard response runbook for an incident.",
    approval_mode="never_require",
)
def incident_runbook(incident_id: str) -> str:
    return f"Runbook for {incident_id}: verify impact, recover service, then notify the owner."


def speed_support(estimated_seconds: int) -> float:
    """Normalize a recovery estimate against a five-minute response objective."""
    return max(0.0, 1.0 - estimated_seconds / 300)


def history_confidence(sample_count: int) -> float:
    """Increase confidence with observations while retaining residual uncertainty."""
    return min(0.95, 0.5 + sample_count / 100)


def build_decision_frame(
    catalog: IncidentCatalog,
    decision_context: Mapping[str, Any],
    actions: Sequence[Action],
) -> DecisionFrame:
    """Build a request-scoped frame from trusted current and historical evidence."""
    incident = catalog.get(str(decision_context["incident_id"]))
    history = catalog.recovery_history
    actions_by_id = {action.id: action for action in actions}
    return DecisionFrame(
        task=f"Recover service for {incident.incident_id}: {incident.diagnosis}",
        criteria=[
            DecisionCriterion(
                id="restoration_likelihood",
                weight=0.75,
                minimum_sources=2,
            ),
            DecisionCriterion(id="recovery_speed", weight=0.25),
        ],
        constraints=[
            "change_approved",
            "target_available",
            "within_data_loss_budget",
        ],
        options=[
            DecisionOption(
                action=actions_by_id["restart_service"],
                constraint_results={
                    "change_approved": incident.change_approved,
                    "target_available": incident.primary_control_plane_available,
                    "within_data_loss_budget": True,
                },
                evidence=[
                    EvidencePoint(
                        criterion_id="restoration_likelihood",
                        source="diagnostic_classifier",
                        support=incident.restart_diagnostic_support,
                        confidence=0.90,
                    ),
                    EvidencePoint(
                        criterion_id="restoration_likelihood",
                        source="historical_restart_outcomes",
                        support=history.restart_success_rate,
                        confidence=history_confidence(history.restart_samples),
                    ),
                    EvidencePoint(
                        criterion_id="recovery_speed",
                        source="orchestrator_estimate",
                        support=speed_support(incident.restart_estimated_seconds),
                        confidence=0.90,
                    ),
                ],
            ),
            DecisionOption(
                action=actions_by_id["failover_to_secondary"],
                constraint_results={
                    "change_approved": incident.change_approved,
                    "target_available": incident.secondary_health_support >= 0.80,
                    "within_data_loss_budget": (
                        incident.replication_lag_seconds
                        <= incident.recovery_point_objective_seconds
                    ),
                },
                evidence=[
                    EvidencePoint(
                        criterion_id="restoration_likelihood",
                        source="secondary_region_health_probe",
                        support=incident.secondary_health_support,
                        confidence=0.95,
                    ),
                    EvidencePoint(
                        criterion_id="restoration_likelihood",
                        source="historical_failover_outcomes",
                        support=history.failover_success_rate,
                        confidence=history_confidence(history.failover_samples),
                    ),
                    EvidencePoint(
                        criterion_id="recovery_speed",
                        source="orchestrator_estimate",
                        support=speed_support(incident.failover_estimated_seconds),
                        confidence=0.85,
                    ),
                ],
            ),
        ],
        minimum_margin=0.05,
        uncertainty_penalty=0.10,
    )


def outcome_metadata(
    catalog: IncidentCatalog,
    action_id: str,
    decision_context: Mapping[str, Any],
    result: Any,
) -> Mapping[str, Any]:
    incident = catalog.get(str(decision_context["incident_id"]))
    restored = result.get("restored") if isinstance(result, dict) else False
    return {
        "task_completed": restored is True,
        "evidence_snapshot": {
            "diagnosis": incident.diagnosis,
            "replication_lag_seconds": incident.replication_lag_seconds,
            "recovery_point_objective_seconds": (
                incident.recovery_point_objective_seconds
            ),
            "restart_history_samples": catalog.recovery_history.restart_samples,
            "failover_history_samples": catalog.recovery_history.failover_samples,
        },
        "score_contract": {
            "required_substrings": [incident.incident_id, action_id, "restored"],
            "json_required": True,
        },
        "expected_tokens": [incident.incident_id, action_id, "restored"],
    }


def required_setting(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Set {name} before running this example")
    return value


async def run_with_approvals(
    agent: Agent,
    prompt: str,
    session: AgentSession,
) -> None:
    """Resume one MAF run until every agent-learning approval is answered."""
    response = await agent.run(prompt, session=session)
    while response.user_input_requests:
        request = response.user_input_requests[0]
        approval = request.additional_properties.get("agent_learning", {})
        action_id = approval.get("action_id", "unknown action")
        answer = await asyncio.to_thread(
            input,
            f"Approve agent-learning action {action_id!r}? [y/N] ",
        )
        response = await agent.run(
            request.to_function_approval_response(
                approved=answer.strip().lower() in {"y", "yes"}
            ),
            session=session,
        )


async def main() -> None:
    catalog = IncidentCatalog()
    store = InMemoryStore()
    credential = DefaultAzureCredential()
    learning = AgentFrameworkLearningAdapter.from_tools(
        agent_id=AGENT_ID,
        task_id=TASK_ID,
        action_tools=build_action_tools(catalog),
        intent_summary="Choose an authorized service recovery action",
        expected_outcome="Restore service without violating recovery constraints",
        decision_authority=DecisionAuthority.FULL,
        decision_frame_resolver=lambda context, actions: build_decision_frame(
            catalog, context, actions
        ),
        target_resolver=lambda context: str(context["incident_id"]),
        episode_metadata_resolver=lambda action_id, context, result: outcome_metadata(
            catalog, action_id, context, result
        ),
        store=store,
        agent_name="ReasonedIncidentAgent",
    )
    agent = Agent(
        client=OpenAIChatClient(
            model=required_setting("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
            azure_endpoint=required_setting("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "latest"),
            credential=credential,
        ),
        name="ReasonedIncidentAgent",
        instructions=(
            "Recover each incident by calling execute_learning_action once. Include the "
            "incident_id in decision_context and provide that incident_id in the inputs "
            "for both restart_service and failover_to_secondary. Application code owns "
            "the evidence and safety checks. Follow the returned decision status and do "
            "not claim an action ran unless selected_action is set."
        ),
        tools=[incident_runbook, *learning.tools],
        middleware=learning.middleware,
    )

    try:
        async with agent:
            for incident in catalog.incidents:
                await run_with_approvals(
                    agent,
                    f"Recover {incident.incident_id}: {incident.diagnosis}.",
                    AgentSession(),
                )
    finally:
        credential.close()

    runner = LearningRunner(
        store=store,
        score_runtime_config=ScoreRuntimeConfig(tier="stdlib"),
    )
    episodes = list(
        reversed(store.query_episodes(AGENT_ID, task_id=TASK_ID, full_only=True))
    )
    for episode in episodes:
        rewards = runner.score_and_record(episode)
        aggregate = next(
            reward.value for reward in rewards if reward.source is RewardSource.AGGREGATE
        )
        decision = episode.metadata["decision"]
        print(
            f"{episode.target}: action={episode.action_id} "
            f"status={decision['status']} "
            f"authorization={decision['authorization_basis']} "
            f"score={aggregate:+.3f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
