"""Judged optimization example: learn from the SDK's own tiered judges.

``quickstart.py`` stubs the reward and ``next_best_action.py`` simulates
an outcome. This example closes the loop with the SDK's *real* judge
layer: every episode is scored by the Tier 1 (pure standard library,
zero-dependency) judges returned by
:func:`agent_learning.judges.build_judges`, and those scores -- not a
hand-written oracle -- drive the policy update.

The scenario is a support agent that must answer a password-reset
ticket in a required format. The policy chooses one of three response
templates; the judges decide how good each answer is:

    * ``template_rich``   -- names the ticket, resets the password, and
      walks the user through verification. Passes every judge.
    * ``template_terse``  -- "All set." Adds nothing the contract asks
      for, so task-completion collapses.
    * ``template_offtopic`` -- refuses, then pitches an unrelated
      ``PLAN-999`` upsell. Trips a forbidden phrase, a wrong upstream
      route, and a hallucinated-class penalty.

This single example exercises all six SDK building blocks:

    Judges    build_judges(tier="stdlib") scores each episode.
    Metrics   JudgeScore -> MetricResult bridges judges into the reward.
    Shaping   RewardShaper folds in the routing + hallucination penalties
              that quickstart never triggers.
    Policy    a marginal SoftmaxPolicy over the three templates.
    Learning  the built-in ReinforceLearner (not a re-implementation)
              consumes the shaped reward.
    Episodes  full Episode records carry the contract, expected tokens,
              tool calls, and conversation history the judges read.

Run with ``python examples/judged_optimization.py``. It uses the
in-memory store and the stdlib judges, so it needs no Azure credentials
and no optional extras.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Tuple

from agent_learning.config import JudgeRuntimeConfig, LearnerConfig, ShapingConfig
from agent_learning.judges import build_judges
from agent_learning.judges.base import Judge, JudgeScore
from agent_learning.learners import ReinforceLearner
from agent_learning.policy import SoftmaxPolicy
from agent_learning.rewards import RewardShaper, RewardWriter
from agent_learning.storage import InMemoryStore
from agent_learning.training import LearningRunner
from agent_learning.types import (
    Action,
    Episode,
    MetricName,
    MetricResult,
    ToolCall,
)

# ---------------------------------------------------------------------------
# 1. The task -- one fixed contract the response must satisfy.
#    ``allowed_classes`` is the set of intent classes an upstream router
#    is permitted to send here; ``expected_tokens`` are the artifacts a
#    complete answer must mention.
# ---------------------------------------------------------------------------
SYSTEM_MESSAGE = "You are a support agent. Resolve the ticket in the required format."
ALLOWED_CLASSES = ["account_access"]
EXPECTED_TOKENS = ["reset", "verify", "identity", "link"]
TICKETS = ["TICKET-4471", "TICKET-5582", "TICKET-6693"]


def render_query(ticket: str) -> str:
    """The user's message for a given ticket."""
    return f"Ticket {ticket}: I cannot log in. Please reset my password."


def contract_for(ticket: str) -> Dict[str, object]:
    """The per-episode adherence contract the judge enforces."""
    return {
        "required_substrings": [ticket, "password"],
        "forbidden_substrings": ["I don't know", "cannot help"],
        "length_min": 30,
        "length_max": 400,
    }


# ---------------------------------------------------------------------------
# 2. Action space -- three response templates plus the upstream-routing and
#    entity-grounding truth each one implies. In a real system these two
#    flags come from your router and an entity checker; here they are fixed
#    per template so the shaping penalties are easy to see firing.
# ---------------------------------------------------------------------------
def _rich(ticket: str) -> str:
    return (
        f"For {ticket}: I have started a password reset. Please verify your "
        "identity, then open the secure link we emailed to finish the reset."
    )


def _terse(_ticket: str) -> str:
    return "All set."


def _offtopic(_ticket: str) -> str:
    return "I don't know. Cannot help. Try our PLAN-999 premium upgrade instead."


class Template:
    """A candidate response strategy the policy can pick."""

    def __init__(
        self,
        action_id: str,
        render: Callable[[str], str],
        *,
        routing_correct: bool,
        hallucinated: bool,
        description: str,
    ) -> None:
        self.action_id = action_id
        self.render = render
        self.routing_correct = routing_correct
        self.hallucinated = hallucinated
        self.description = description


TEMPLATES: Dict[str, Template] = {
    t.action_id: t
    for t in (
        Template(
            "template_rich",
            _rich,
            routing_correct=True,
            hallucinated=False,
            description="Names the ticket, resets the password, guides verification.",
        ),
        Template(
            "template_terse",
            _terse,
            routing_correct=True,
            hallucinated=False,
            description="Acknowledges but resolves nothing the contract asks for.",
        ),
        Template(
            "template_offtopic",
            _offtopic,
            routing_correct=False,
            hallucinated=True,
            description="Refuses and pitches an unrelated, hallucinated plan.",
        ),
    )
}
ACTIONS: List[Action] = [
    Action(id=t.action_id, description=t.description) for t in TEMPLATES.values()
]
ACTION_IDS = [a.id for a in ACTIONS]


# ---------------------------------------------------------------------------
# 3. Judges -- the SDK's Tier 1 stdlib trio. Adherence and completion are
#    deterministic rule engines (no training). Intent is a bag-of-words
#    classifier we fit in-process on a handful of labelled responses so it
#    contributes a real, calibrated signal rather than its permissive
#    unfitted default.
# ---------------------------------------------------------------------------
def build_scored_judges() -> Tuple[Judge, Judge, Judge]:
    """Build the stdlib judges and fit the intent head on a tiny corpus."""
    intent, adherence, completion = build_judges(JudgeRuntimeConfig(tier="stdlib"))

    corpus: List[dict] = []
    for ticket in TICKETS:
        query = render_query(ticket)
        corpus.append({"query": query, "response": _rich(ticket), "label": 1})
        corpus.append({"query": query, "response": _terse(ticket), "label": 0})
        corpus.append({"query": query, "response": _offtopic(ticket), "label": 0})
    intent.fit(corpus)  # type: ignore[attr-defined]  # StdlibIntentJudge.fit

    return intent, adherence, completion


# ---------------------------------------------------------------------------
# 4. Metrics bridge -- map each JudgeScore onto the MetricResult shape the
#    reward shaper already understands. This is the one adapter that lets
#    any judge tier feed the existing reward pipeline unchanged.
# ---------------------------------------------------------------------------
def judge_episode(
    judges: Tuple[Judge, Judge, Judge],
    episode: Episode,
) -> List[MetricResult]:
    """Score one episode with the three judges -> three MetricResults."""
    intent, adherence, completion = judges
    query = episode.user_input
    response = episode.assistant_output
    contract = episode.metadata.get("contract", {})
    expected = episode.metadata.get("expected_tokens", [])

    scored: List[Tuple[MetricName, JudgeScore, str]] = [
        (MetricName.INTENT_RESOLUTION, intent.score(query=query, response=response), "intent"),
        (MetricName.TASK_ADHERENCE, adherence.score(response=response, contract=contract), "adherence"),
        (
            MetricName.TASK_COMPLETION,
            completion.score(response=response, expected_tokens=expected),
            "completion",
        ),
    ]
    return [
        MetricResult(
            metric=name,
            score=js.confidence,
            normalized=js.normalized,
            status="completed",
            reason=js.label,
            properties=js.features,
            evaluator=f"tier1-stdlib:{judge_name}",
        )
        for name, js, judge_name in scored
    ]


# ---------------------------------------------------------------------------
# 5. Runner -- wire the judges and the dormant shaping penalties into the
#    existing pipeline WITHOUT touching the SDK. We override two hooks:
#      * evaluate_episode  -> score with the judges (instead of Azure).
#      * score_and_record  -> shape with routing + hallucination signals
#                             the default shape_episode_reward never passes.
# ---------------------------------------------------------------------------
class JudgedRunner(LearningRunner):
    """A LearningRunner whose reward comes from the stdlib judges."""

    def __init__(self, *, judges: Tuple[Judge, Judge, Judge], **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._judges = judges

    def evaluate_episode(self, episode: Episode) -> List[MetricResult]:
        return judge_episode(self._judges, episode)

    def score_and_record(self, episode: Episode):
        results = self.evaluate_episode(episode)
        # The base runner only passes latency to the shaper; here we also
        # feed the routing and hallucination signals so those penalties in
        # ShapingConfig actually contribute to the reward.
        shaped = self._shaper.shape(
            results,
            latency_ms=episode.request_latency_ms,
            routing_correct=episode.metadata.get("routing_correct"),
            hallucinated_class=bool(episode.metadata.get("hallucinated_class", False)),
        )
        return self._writer.write(episode, results, shaped)


# ---------------------------------------------------------------------------
# 6. Reporting helpers.
# ---------------------------------------------------------------------------
def _report(policy: SoftmaxPolicy, label: str) -> None:
    probs = policy.probabilities()
    best = ACTION_IDS[max(range(len(probs)), key=probs.__getitem__)]
    dist = "  ".join(f"{aid}={p:.2f}" for aid, p in zip(ACTION_IDS, probs))
    print(f"  {label:24s} -> best={best:18s} | {dist}")


# ---------------------------------------------------------------------------
# 7. Main loop.
# ---------------------------------------------------------------------------
def main() -> None:
    rng = random.Random(11)
    store = InMemoryStore()

    judges = build_scored_judges()

    policy = SoftmaxPolicy.from_actions(ACTIONS, agent_id="support", rng=rng)
    store.store_policy(policy.snapshot())

    runner = JudgedRunner(
        judges=judges,
        store=store,
        policy=policy,
        metrics=[],  # judges are supplied via evaluate_episode instead
        shaper=RewardShaper(ShapingConfig()),
        writer=RewardWriter(store),
        learner=ReinforceLearner(LearnerConfig(learning_rate=0.5, entropy_bonus=0.02)),
    )

    print("=== Policy BEFORE training ===")
    _report(policy, "uniform prior")

    rounds = 30
    per_round = 30
    print(f"\nTraining for {rounds} rounds x {per_round} episodes (stdlib judges) ...")
    for r in range(rounds):
        agent_id = f"support-r{r}"
        for i in range(per_round):
            decision = policy.choose()
            template = TEMPLATES[decision.action.id]
            ticket = TICKETS[(r * per_round + i) % len(TICKETS)]
            query = render_query(ticket)
            response = template.render(ticket)
            episode = Episode(
                id=f"r{r}-ep{i}",
                agent_id=agent_id,
                user_input=query,
                assistant_output=response,
                system_message=SYSTEM_MESSAGE,
                conversation_history=[{"role": "user", "content": query}],
                tool_calls=[
                    ToolCall(name="lookup_ticket", arguments={"id": ticket}, result="status=open")
                ],
                policy_id=policy.snapshot().id,
                policy_version=policy.snapshot().version,
                action_id=decision.action.id,
                action_logprob=decision.logprob,
                request_latency_ms=1200,
                context_features={"segment": "support", "ticket": ticket},
                metadata={
                    "contract": contract_for(ticket),
                    "expected_tokens": EXPECTED_TOKENS,
                    "allowed_classes": ALLOWED_CLASSES,
                    "routing_correct": template.routing_correct,
                    "hallucinated_class": template.hallucinated,
                },
            )
            store.store_episode(episode)

        run = runner.run_offline_batch(agent_id, episode_limit=per_round)
        if r % 5 == 0 or r == rounds - 1:
            probs = policy.probabilities()
            p_rich = probs[ACTION_IDS.index("template_rich")]
            print(
                f"  round {r:3d}: mean_reward={run.metrics['mean_reward']:+.3f}  "
                f"P(template_rich)={p_rich:.2f}"
            )

    print("\n=== Policy AFTER training ===")
    _report(policy, "learned")

    # Show the judge decomposition for one episode of each template so the
    # reward gradient is reproducible.
    print("\n=== Per-template judge scores (single ticket) ===")
    ticket = TICKETS[0]
    for action_id, template in TEMPLATES.items():
        probe = Episode(
            id=f"probe-{action_id}",
            agent_id="probe",
            user_input=render_query(ticket),
            assistant_output=template.render(ticket),
            metadata={
                "contract": contract_for(ticket),
                "expected_tokens": EXPECTED_TOKENS,
            },
        )
        results = judge_episode(judges, probe)
        shaped = RewardShaper(ShapingConfig()).shape(
            results,
            routing_correct=template.routing_correct,
            hallucinated_class=template.hallucinated,
        )
        breakdown = "  ".join(f"{m.metric.value}={m.normalized:.2f}" for m in results)
        print(f"  {action_id:18s} reward={shaped.value:+.2f} | {breakdown}")


if __name__ == "__main__":
    main()
