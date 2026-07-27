"""Next Best Action example: a contextual bandit for customer retention.

Inspired by the "Next Best Action" agent pattern in
`azure-agents-control-plane`, this example shows how to use the SDK's
*contextual* softmax policy to recommend the single best action for a
given customer state (churn risk and account segment).

Unlike ``quickstart.py`` -- which uses the marginal ``SoftmaxPolicy``
where one action is globally best -- here the best action DEPENDS ON THE
CONTEXT:

    * a high-risk ENTERPRISE account warrants a human retention call,
    * a high-risk SMB is better served by a targeted discount,
    * a low-risk account should simply be left alone.

The agent starts from a uniform policy and learns this mapping from
reward feedback alone, using REINFORCE-with-baseline over a linear
contextual softmax.

Run with ``python examples/next_best_action.py``. It uses the in-memory
store and a stubbed outcome simulator, so it needs no Azure credentials.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from agent_learning.config import LearnerConfig, ShapingConfig
from agent_learning.learners import Learner, LearnerResult
from agent_learning.policy import ContextualSoftmaxPolicy
from agent_learning.policy.base import Policy
from agent_learning.rewards import RewardShaper, RewardWriter
from agent_learning.storage import InMemoryStore
from agent_learning.training import LearningRunner
from agent_learning.types import (
    Action,
    Episode,
    MetricName,
    MetricResult,
    Reward,
    RewardSource,
)

# ---------------------------------------------------------------------------
# 1. Action space -- the discrete "next best actions" the agent can take.
# ---------------------------------------------------------------------------
ACTIONS: List[Action] = [
    Action(id="no_action", description="Leave the customer alone (no outreach)"),
    Action(id="email_nudge", description="Send an automated re-engagement email"),
    Action(id="discount_offer", description="Offer a targeted retention discount"),
    Action(id="retention_call", description="Schedule a human retention call"),
]
ACTION_IDS = [a.id for a in ACTIONS]


# ---------------------------------------------------------------------------
# 2. Context -- a customer state and its fixed-length feature encoding phi.
# ---------------------------------------------------------------------------
SEGMENTS = ("enterprise", "smb", "consumer")
FEATURE_DIM = 5


@dataclass
class Customer:
    """A single account the agent must decide a next best action for."""

    churn_risk: float  # 0..1 (probability the account churns soon)
    segment: str  # one of SEGMENTS

    def features(self) -> np.ndarray:
        """Encode the customer as the length-``FEATURE_DIM`` vector phi."""
        return np.array(
            [
                1.0,  # bias term (lets the policy learn base rates)
                self.churn_risk,  # 0..1
                1.0 if self.segment == "enterprise" else 0.0,
                1.0 if self.segment == "smb" else 0.0,
                1.0 if self.segment == "consumer" else 0.0,
            ],
            dtype=np.float64,
        )


def sample_customer(rng: random.Random) -> Customer:
    """Draw a random customer to simulate an incoming decision request."""
    return Customer(churn_risk=round(rng.random(), 3), segment=rng.choice(SEGMENTS))


# ---------------------------------------------------------------------------
# 3. Hidden environment -- how good each action is for a given customer.
#    This is the "ground truth" the agent must discover from reward alone;
#    the policy never sees this table.
# ---------------------------------------------------------------------------
def _base_success(customer: Customer) -> Dict[str, float]:
    """Noise-free success probability in [0, 1] for every action.

    The optimal action is context-dependent: leave healthy accounts
    alone, and escalate at-risk accounts according to their segment
    (enterprise -> human call, SMB -> discount, consumer -> email).
    No single action is globally best, so the policy must condition on
    the context to do well.
    """
    if customer.churn_risk < 0.5:  # healthy account -> outreach wastes budget
        return {"no_action": 0.90, "email_nudge": 0.60, "discount_offer": 0.30, "retention_call": 0.20}
    # at-risk account -> escalate based on segment value
    if customer.segment == "enterprise":
        return {"no_action": 0.05, "email_nudge": 0.50, "discount_offer": 0.70, "retention_call": 0.95}
    if customer.segment == "smb":
        return {"no_action": 0.10, "email_nudge": 0.60, "discount_offer": 0.90, "retention_call": 0.60}
    return {"no_action": 0.15, "email_nudge": 0.85, "discount_offer": 0.70, "retention_call": 0.30}


def outcome_success(customer: Customer, action_id: str, rng: random.Random) -> float:
    """Sample a noisy success score for taking ``action_id``."""
    base = _base_success(customer)[action_id]
    return max(0.0, min(1.0, base + rng.gauss(0.0, 0.05)))


def optimal_action(customer: Customer) -> str:
    """The best action under the noise-free table (used only for scoring)."""
    scores = _base_success(customer)
    return max(scores, key=scores.__getitem__)


# ---------------------------------------------------------------------------
# 4. A contextual REINFORCE learner.
#    The built-in ``ReinforceLearner`` only supports the marginal
#    ``SoftmaxPolicy``, so we implement the linear-softmax policy gradient
#    here. For a customer with features phi, the action probabilities are
#    ``pi = softmax(W @ phi)`` and the REINFORCE-with-baseline gradient for
#    action-row ``W_k`` is:
#
#        dW_k = lr * (R - b) * (1[k == chosen] - pi_k) * phi
# ---------------------------------------------------------------------------
class ContextualReinforceLearner(Learner):
    """REINFORCE-with-baseline for a :class:`ContextualSoftmaxPolicy`."""

    def __init__(self, config: Optional[LearnerConfig] = None) -> None:
        self._config = config or LearnerConfig()

    def update(
        self,
        policy: Policy,
        episodes: Iterable[Episode],
        rewards: Iterable[Reward],
    ) -> LearnerResult:
        if not isinstance(policy, ContextualSoftmaxPolicy):
            raise TypeError("ContextualReinforceLearner requires a ContextualSoftmaxPolicy")

        # Only the AGGREGATE reward per episode feeds the gradient.
        aggregate: Dict[str, float] = {}
        for r in rewards:
            if r.source == RewardSource.AGGREGATE and r.episode_id not in aggregate:
                aggregate[r.episode_id] = r.value

        snapshot = policy.snapshot()
        action_index = {a.id: i for i, a in enumerate(snapshot.actions)}

        # Pass 1: gather the usable (features, chosen action, reward) triples.
        samples: List[Tuple[np.ndarray, int, float, Episode]] = []
        for episode in episodes:
            if episode.action_id is None or episode.action_id not in action_index:
                continue
            reward = aggregate.get(episode.id)
            if reward is None:
                continue
            phi = np.asarray(episode.context_features.get("phi", []), dtype=np.float64)
            if phi.shape != (FEATURE_DIM,):
                continue
            samples.append((phi, action_index[episode.action_id], reward, episode))

        if not samples:
            return LearnerResult(
                episodes_used=0,
                mean_reward=0.0,
                baseline_before=snapshot.baseline,
                baseline_after=snapshot.baseline,
                logit_deltas={aid: 0.0 for aid in action_index},
                extra={"reason": "no actionable episodes"},
            )

        # Batch-mean baseline centres the advantages each round -- a much
        # lower-variance estimator than a lagging EMA, which makes the
        # contextual weights converge faster and more monotonically.
        used = len(samples)
        mean_reward = sum(reward for _phi, _k, reward, _ep in samples) / used

        # Pass 2: accumulate the policy-gradient step under the pre-update
        # policy (probabilities are held fixed across the batch).
        deltas: Dict[str, np.ndarray] = {aid: np.zeros(FEATURE_DIM) for aid in action_index}
        for phi, chosen, reward, episode in samples:
            probs = np.asarray(policy.probabilities(phi), dtype=np.float64)
            advantage = reward - mean_reward
            # Importance weight keeps offline updates unbiased when the
            # episode was logged under an older policy version.
            advantage *= self._importance_weight(episode, probs, action_index)

            entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, None))))
            for aid, k in action_index.items():
                indicator = 1.0 if k == chosen else 0.0
                # Policy gradient: move the chosen action's row toward higher
                # reward, weighted by how surprising the choice was.
                grad = advantage * (indicator - probs[k])
                # Entropy bonus keeps exploration alive so the policy does not
                # collapse onto a single globally-safe action before it has
                # learned the context-specific winners.
                grad += self._config.entropy_bonus * (-math.log(max(probs[k], 1e-12)) - entropy)
                deltas[aid] += grad * phi

        # Average the batch gradient and scale by the learning rate.
        lr = self._config.learning_rate
        vector_deltas = {aid: lr * grad / used for aid, grad in deltas.items()}

        # Track a slow EMA baseline in the snapshot for continuity / reporting.
        decay = self._config.baseline_decay
        baseline_after = decay * snapshot.baseline + (1.0 - decay) * mean_reward

        policy.apply_update(
            {aid: grad.tolist() for aid, grad in vector_deltas.items()},
            baseline=baseline_after,
            episodes_seen=snapshot.episodes_seen + used,
        )

        return LearnerResult(
            episodes_used=used,
            mean_reward=mean_reward,
            baseline_before=snapshot.baseline,
            baseline_after=baseline_after,
            # LearnerResult tracks scalars; report each row's step size.
            logit_deltas={aid: float(np.linalg.norm(grad)) for aid, grad in vector_deltas.items()},
            extra={"batch_baseline": mean_reward},
        )

    def _importance_weight(
        self,
        episode: Episode,
        probs: np.ndarray,
        action_index: Dict[str, int],
    ) -> float:
        """Return the clipped pi_target / pi_behaviour ratio for an episode."""
        if episode.action_logprob is None or episode.action_id is None:
            return 1.0
        k = action_index[episode.action_id]
        p_target = max(float(probs[k]), 1e-12)
        p_behaviour = max(math.exp(episode.action_logprob), 1e-12)
        return min(p_target / p_behaviour, self._config.importance_clip)


# ---------------------------------------------------------------------------
# 5. Runner that scores each episode from the simulated outcome (no Azure).
# ---------------------------------------------------------------------------
class _RetentionRunner(LearningRunner):
    """Turns the stored simulated success into the three judge metrics."""

    def evaluate_episode(self, episode: Episode) -> List[MetricResult]:
        success = float(episode.metadata.get("_success", 0.0))
        return [
            MetricResult(metric=metric, score=success, normalized=success, status="completed")
            for metric in (
                MetricName.INTENT_RESOLUTION,
                MetricName.TASK_ADHERENCE,
                MetricName.TASK_COMPLETION,
            )
        ]


# ---------------------------------------------------------------------------
# 6. Helpers for reporting.
# ---------------------------------------------------------------------------
def recommend(policy: ContextualSoftmaxPolicy, customer: Customer) -> Tuple[str, List[float]]:
    """Return the greedy next best action and the full action distribution."""
    probs = policy.probabilities(customer.features())
    return ACTION_IDS[int(np.argmax(probs))], probs


def _print_recommendations(
    policy: ContextualSoftmaxPolicy,
    probes: List[Tuple[str, Customer]],
) -> None:
    for label, customer in probes:
        best, probs = recommend(policy, customer)
        dist = "  ".join(f"{aid}={p:.2f}" for aid, p in zip(ACTION_IDS, probs))
        print(f"  {label:22s} -> {best:15s} | {dist}")


def _optimality_rate(policy: ContextualSoftmaxPolicy, n: int = 4000) -> float:
    """Fraction of random customers where the greedy pick is optimal."""
    rng = random.Random(123)
    hits = sum(
        1
        for _ in range(n)
        if recommend(policy, (c := sample_customer(rng)))[0] == optimal_action(c)
    )
    return hits / n


# ---------------------------------------------------------------------------
# 7. Main loop.
# ---------------------------------------------------------------------------
def main() -> None:
    rng = random.Random(7)
    store = InMemoryStore()

    policy = ContextualSoftmaxPolicy.from_actions(
        ACTIONS, feature_dim=FEATURE_DIM, agent_id="retention", rng=rng
    )
    store.store_policy(policy.snapshot())

    runner = _RetentionRunner(
        store=store,
        policy=policy,
        metrics=[],  # no Azure judges; evaluate_episode is overridden above
        shaper=RewardShaper(ShapingConfig()),
        writer=RewardWriter(store),
        learner=ContextualReinforceLearner(
            LearnerConfig(learning_rate=0.8, entropy_bonus=0.03)
        ),
    )

    # A handful of representative accounts we inspect before and after.
    probes: List[Tuple[str, Customer]] = [
        ("High-risk ENTERPRISE", Customer(0.90, "enterprise")),
        ("High-risk SMB", Customer(0.85, "smb")),
        ("High-risk CONSUMER", Customer(0.80, "consumer")),
        ("Low-risk ENTERPRISE", Customer(0.10, "enterprise")),
    ]

    print("=== Recommended next best action BEFORE training ===")
    _print_recommendations(policy, probes)
    print(f"  optimal-action rate: {_optimality_rate(policy):.0%}")

    # Online loop: each round samples fresh episodes under the current policy,
    # then applies one batched contextual REINFORCE update.
    rounds = 200
    episodes_per_round = 150
    print(f"\nTraining for {rounds} rounds x {episodes_per_round} episodes ...")
    for r in range(rounds):
        agent_id = f"retention-r{r}"
        for i in range(episodes_per_round):
            customer = sample_customer(rng)
            phi = customer.features()
            decision = policy.choose(phi)
            success = outcome_success(customer, decision.action.id, rng)
            episode = Episode(
                id=f"r{r}-ep{i}",
                agent_id=agent_id,
                user_input=f"{customer.segment} account, churn_risk={customer.churn_risk}",
                assistant_output=f"next_best_action={decision.action.id}",
                policy_id=policy.snapshot().id,
                policy_version=policy.snapshot().version,
                action_id=decision.action.id,
                action_logprob=decision.logprob,
                context_features={"phi": phi.tolist()},
                metadata={"_success": success},
            )
            store.store_episode(episode)

        run = runner.run_offline_batch(agent_id, episode_limit=episodes_per_round)
        if r % 20 == 0 or r == rounds - 1:
            print(
                f"  round {r:3d}: mean_reward={run.metrics['mean_reward']:+.3f}  "
                f"optimal-action rate={_optimality_rate(policy):.0%}"
            )

    print("\n=== Recommended next best action AFTER training ===")
    _print_recommendations(policy, probes)
    print(f"  optimal-action rate: {_optimality_rate(policy):.0%}")


if __name__ == "__main__":
    main()
