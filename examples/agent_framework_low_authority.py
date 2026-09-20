"""Microsoft Agent Framework learning loop with a low-authority marginal policy.

The agent can consult an ordinary runbook, but agent-learning chooses between a
fast cached assessment and a slower verified assessment. The selected child
tool executes inside the generated composite tool and produces one episode.
After each round, Tier 1 stdlib scorers evaluate the batch and REINFORCE updates
the policy used by the next round. No LLM is used for evaluation.

Prerequisites:

* Sign in locally with ``az login`` or run with a managed identity.
* Grant the identity the ``Cognitive Services OpenAI User`` role.
* Set ``AZURE_OPENAI_ENDPOINT`` and ``AZURE_OPENAI_CHAT_DEPLOYMENT_NAME``.
* ``pip install -e '.[agent-framework,examples]'`` from the repository root.

Run with ``python examples/agent_framework_low_authority.py --rounds 3 --episodes 6``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agent_framework import Agent, AgentSession, tool  # type: ignore[attr-defined]
from agent_framework.openai import OpenAIChatClient
from azure.identity import DefaultAzureCredential

from agent_learning.config import LearnerConfig, ScoreRuntimeConfig, ShapingConfig
from agent_learning.integrations.agent_framework import (
    AgentFrameworkLearningAdapter,
)
from agent_learning.learners import ReinforceLearner
from agent_learning.policy import SoftmaxPolicy
from agent_learning.storage import InMemoryStore
from agent_learning.training import LearningRunner
from agent_learning.types import TrainingRun

AGENT_ID = "maf-basic"
TASK_ID = "choose-response-tool"


@dataclass(frozen=True)
class Incident:
    incident_id: str
    current_error_rate: float
    cached_error_rate: float
    cache_age_seconds: int

    @property
    def expected_recommendation(self) -> str:
        return "escalate" if self.current_error_rate >= 0.05 else "monitor"


class IncidentCatalog:
    def __init__(self, count: int, rng: random.Random) -> None:
        incidents: list[Incident] = []
        for index in range(1, count + 1):
            current_error_rate = rng.choice((0.02, 0.08))
            cache_is_fresh = rng.random() < 0.70
            cached_error_rate = (
                current_error_rate
                if cache_is_fresh
                else (0.08 if current_error_rate < 0.05 else 0.02)
            )
            incidents.append(
                Incident(
                    incident_id=f"INC-{index:03d}",
                    current_error_rate=current_error_rate,
                    cached_error_rate=cached_error_rate,
                    cache_age_seconds=30 if cache_is_fresh else 600,
                )
            )
        self.incidents = incidents
        self._incidents_by_id = {
            incident.incident_id: incident for incident in incidents
        }

    def get(self, incident_id: str) -> Incident:
        try:
            return self._incidents_by_id[incident_id]
        except KeyError as exc:
            raise ValueError(f"unknown incident: {incident_id}") from exc


def build_action_tools(catalog: IncidentCatalog) -> list[Any]:
    @tool(
        name="cached_assessment",
        description="Assess an incident quickly using potentially stale cached telemetry.",
        approval_mode="never_require",
    )
    async def cached_assessment(incident_id: str) -> dict[str, Any]:
        """Return a fast assessment from cached telemetry."""
        await asyncio.sleep(0.2)
        incident = catalog.get(incident_id)
        return {
            "incident_id": incident_id,
            "recommendation": (
                "escalate" if incident.cached_error_rate >= 0.05 else "monitor"
            ),
            "source": "telemetry cache",
            "error_rate": incident.cached_error_rate,
            "cache_age_seconds": incident.cache_age_seconds,
        }

    @tool(
        name="verified_assessment",
        description="Assess an incident using current verified telemetry.",
        approval_mode="never_require",
    )
    async def verified_assessment(incident_id: str) -> dict[str, Any]:
        """Return an accurate assessment from current telemetry."""
        await asyncio.sleep(1.0)
        incident = catalog.get(incident_id)
        return {
            "incident_id": incident_id,
            "recommendation": incident.expected_recommendation,
            "source": "verified telemetry",
            "error_rate": incident.current_error_rate,
            "cache_age_seconds": 0,
        }

    return [cached_assessment, verified_assessment]


@tool(
    name="incident_runbook",
    description="Return the standard response runbook for an incident.",
    approval_mode="never_require",
)
def incident_runbook(incident_id: str) -> str:
    """An ordinary MAF tool that is not controlled by agent-learning."""
    return f"Runbook for {incident_id}: assess impact, then notify the service owner."


def required_setting(name: str) -> str:
    """Read a required non-secret Azure OpenAI setting."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Set {name} before running this example")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=6, help="Episodes per round")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.rounds < 1 or args.episodes < 1:
        parser.error("--rounds and --episodes must be at least 1")
    return args


def outcome_metadata(
    catalog: IncidentCatalog,
    _action_id: str,
    action_arguments: Mapping[str, Any],
    result: Any,
) -> Mapping[str, Any]:
    """Prepare ground truth consumed by the in-process scorer adapters."""
    incident = catalog.get(str(action_arguments["incident_id"]))
    recommendation = result.get("recommendation") if isinstance(result, dict) else None
    completed = recommendation == incident.expected_recommendation
    return {
        "task_completed": completed,
        "score_contract": {
            "required_substrings": [incident.incident_id, "recommendation", "source"],
            "json_required": True,
        },
        "expected_tokens": [incident.incident_id, incident.expected_recommendation],
    }


def probabilities(policy: SoftmaxPolicy) -> dict[str, float]:
    return {
        action.id: round(probability, 3)
        for action, probability in zip(policy.actions(), policy.probabilities())
    }


def _format_distribution(distribution: Mapping[str, float]) -> str:
    return "  ".join(
        f"{action_id}={probability:.3f}"
        for action_id, probability in distribution.items()
    )


def _report_policy(policy: SoftmaxPolicy, label: str) -> None:
    distribution = probabilities(policy)
    best = max(distribution, key=distribution.__getitem__)
    print(
        f"  {label:12s} -> best={best:20s} | "
        f"{_format_distribution(distribution)}"
    )


def print_summary(
    round_number: int,
    training_run: TrainingRun,
    store: InMemoryStore,
    before: dict[str, float],
    policy: SoftmaxPolicy,
) -> None:
    round_episodes = [
        episode
        for episode_id in training_run.episode_ids
        if (episode := store.get_episode(episode_id, AGENT_ID)) is not None
    ]
    mistakes = sum(
        episode.metadata.get("task_completed") is False
        for episode in round_episodes
    )
    executed_actions = [
        f"{episode.target}={episode.action_id}"
        for episode in reversed(round_episodes)
    ]
    print(
        f"  round {round_number:3d}: "
        f"mean_reward={training_run.metrics['mean_reward']:+.3f}  "
        f"mistakes={mistakes}/{len(round_episodes)} "
    )
    print(f"    executed actions:       {'  '.join(executed_actions)}")
    print(
        "    candidate probabilities: "
        f"{_format_distribution(before)} -> "
        f"{_format_distribution(probabilities(policy))}"
    )


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
    args = parse_args()
    rng = random.Random(args.seed)
    incident_catalog = IncidentCatalog(
        args.rounds * args.episodes,
        random.Random(args.seed + 1),
    )
    action_tools = build_action_tools(incident_catalog)
    store = InMemoryStore()
    credential = DefaultAzureCredential()

    learning = AgentFrameworkLearningAdapter.from_tools(
        agent_id=AGENT_ID,
        task_id=TASK_ID,
        action_tools=action_tools,
        intent_summary="Recommend whether to monitor or escalate a service incident",
        expected_outcome="Return the correct incident recommendation",
        target_resolver=lambda context: str(context["incident_id"]),
        episode_metadata_resolver=lambda action_id, context, result: outcome_metadata(
            incident_catalog, action_id, context, result
        ),
        store=store,
        agent_name="ToolDecisionAgent",
        rng=rng,
        tool_name="recommend_incident_response",
    )
    policy = SoftmaxPolicy.from_snapshot(learning.task_policy.snapshot(), rng=rng)
    runner = LearningRunner(
        store=store,
        policy=policy,
        score_runtime_config=ScoreRuntimeConfig(tier="stdlib"),
        shaping_config=ShapingConfig(
            latency_penalty_threshold_ms=500,
            latency_penalty_value=-0.15,
        ),
        learner=ReinforceLearner(
            LearnerConfig(learning_rate=0.3, entropy_bonus=0.01)
        ),
    )
    agent = Agent(
        client=OpenAIChatClient(
            model=required_setting("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
            azure_endpoint=required_setting("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "latest"),
            credential=credential,
        ),
        name="ToolDecisionAgent",
        instructions=(
            "Assess the incident and return a brief recommendation. You may consult "
            "incident_runbook normally. To perform an assessment, call "
            "recommend_incident_response once. Include action_inputs for both "
            "cached_assessment and verified_assessment, "
            "each with the incident_id."
        ),
        tools=[incident_runbook],
    )
    learning.register(agent)

    prompt = (
        "Assess incident {incident_id}. Current error rate is {error_rate:.0%}; "
        "cached telemetry is {cache_age} seconds old."
    )
    try:
        async with agent:
            print("=== Policy BEFORE training ===")
            _report_policy(policy, "uniform prior")
            print(
                f"\nTraining for {args.rounds} rounds x {args.episodes} episodes "
                "(stdlib scorers) ..."
            )
            for round_index in range(args.rounds):
                round_incidents = incident_catalog.incidents[
                    round_index * args.episodes: (round_index + 1) * args.episodes
                ]
                before = probabilities(policy)

                for incident in round_incidents:
                    await run_with_approvals(
                        agent,
                        prompt.format(
                            incident_id=incident.incident_id,
                            error_rate=incident.current_error_rate,
                            cache_age=incident.cache_age_seconds,
                        ),
                        AgentSession(),
                    )

                training_run = runner.run_offline_batch(
                    AGENT_ID,
                    task_id=TASK_ID,
                    episode_limit=args.episodes,
                )
                learning.update_policy(policy.snapshot())
                print_summary(
                    round_index + 1,
                    training_run,
                    store,
                    before,
                    policy,
                )
            print("\n=== Policy AFTER training ===")
            _report_policy(policy, "learned")
    finally:
        credential.close()


if __name__ == "__main__":
    asyncio.run(main())
