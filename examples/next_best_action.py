"""Next Best Action: a config-driven contextual bandit.

This example learns to recommend the single best action for a given
context (a "next best action" agent) using the SDK's *contextual*
softmax policy trained with REINFORCE-with-baseline.

Unlike ``quickstart.py`` -- which uses the marginal ``SoftmaxPolicy``
where one action is globally best -- here the best action DEPENDS ON THE
CONTEXT, so the agent must condition on the context to do well. It starts
from a uniform policy and learns the mapping from reward feedback alone.

The use case is **externalised to a YAML file** so the same engine can
drive very different domains without code changes. Each YAML file defines
the action space, the context schema (how a state is encoded into the
feature vector ``phi``), a hidden reward ``environment`` the policy must
discover from reward alone, a set of probe contexts, and the training
hyper-parameters.

Bundled use cases (in this folder):

    * ``next_best_action_retention_risk.yaml`` -- customer retention:
      leave healthy accounts alone; escalate at-risk accounts by segment
      (enterprise -> human call, SMB -> discount, consumer -> email).
    * ``next_best_action_patient_care.yaml`` -- care-management outreach:
      monitor stable patients; escalate at-risk patients by care pathway
      (post-op -> physician, chronic -> nurse, behavioral -> education).
    * ``next_best_action_game_play.yaml`` -- a game "director": hold the
      line for engaged players; help frustrated players by playstyle
      (explorer -> hint, achiever -> loot, socializer -> co-op).

Run (defaults to the retention-risk use case)::

    python examples/next_best_action.py
    python examples/next_best_action.py examples/next_best_action_patient_care.yaml
    python examples/next_best_action.py examples/next_best_action_game_play.yaml

It uses the in-memory store and a stubbed outcome simulator, so it needs
no Azure credentials.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency hint
    raise SystemExit("This example needs PyYAML. Install it with: pip install pyyaml") from exc

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

_HERE = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _HERE / "next_best_action_retention_risk.yaml"


# ---------------------------------------------------------------------------
# 1. Use-case model -- the parsed contents of a use-case YAML file.
# ---------------------------------------------------------------------------
@dataclass
class ContextVariable:
    """One dimension of the context and how it is sampled + encoded."""

    name: str
    kind: str  # "continuous" | "categorical"
    low: float = 0.0
    high: float = 1.0
    ndigits: Optional[int] = None
    categories: Tuple[str, ...] = ()


@dataclass
class Rule:
    """One row of the hidden reward table, guarded by a ``when`` condition."""

    name: str
    when: Dict[str, Any]
    success: Dict[str, float]


@dataclass
class Probe:
    """A representative context inspected before and after training."""

    label: str
    context: Dict[str, Any]


@dataclass
class TrainingConfig:
    """REINFORCE-with-baseline hyper-parameters and loop sizing."""

    seed: int = 7
    rounds: int = 200
    episodes_per_round: int = 150
    learning_rate: float = 0.5
    entropy_bonus: float = 0.03


# Comparison operators allowed inside a rule's ``when`` condition.
_OPS: Dict[str, Callable[[Any, Any], bool]] = {
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
}


def _rule_matches(context: Dict[str, Any], when: Dict[str, Any]) -> bool:
    """Return True when every clause in ``when`` holds for ``context``."""
    for var, cond in when.items():
        value = context.get(var)
        if isinstance(cond, dict):
            for op, target in cond.items():
                fn = _OPS.get(op)
                if fn is None:
                    raise ValueError(f"Unknown operator {op!r} in rule condition.")
                if not fn(value, target):
                    return False
        elif value != cond:
            return False
    return True


@dataclass
class UseCase:
    """A fully-parsed next-best-action problem definition."""

    name: str
    description: str
    agent_id: str
    entity: str
    decision_label: str
    actions: List[Action]
    variables: List[ContextVariable]
    noise_std: float
    rules: List[Rule]
    probes: List[Probe]
    training: TrainingConfig

    @property
    def action_ids(self) -> List[str]:
        return [a.id for a in self.actions]

    @property
    def feature_dim(self) -> int:
        """Length of ``phi``: bias + continuous values + one-hot categoricals."""
        dim = 1  # bias term
        for var in self.variables:
            dim += 1 if var.kind == "continuous" else len(var.categories)
        return dim

    def encode(self, context: Dict[str, Any]) -> np.ndarray:
        """Encode a context as the fixed-length feature vector ``phi``."""
        feats: List[float] = [1.0]  # bias lets the policy learn base rates
        for var in self.variables:
            if var.kind == "continuous":
                feats.append(float(context[var.name]))
            else:  # one-hot the categorical value
                value = context[var.name]
                feats.extend(1.0 if value == c else 0.0 for c in var.categories)
        return np.array(feats, dtype=np.float64)

    def sample_context(self, rng: random.Random) -> Dict[str, Any]:
        """Draw a random context to simulate an incoming decision request."""
        context: Dict[str, Any] = {}
        for var in self.variables:
            if var.kind == "continuous":
                value = rng.uniform(var.low, var.high)
                if var.ndigits is not None:
                    value = round(value, var.ndigits)
                context[var.name] = value
            else:
                context[var.name] = rng.choice(var.categories)
        return context

    def base_success(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Noise-free success probability per action (the hidden ground truth)."""
        for rule in self.rules:
            if _rule_matches(context, rule.when):
                return rule.success
        raise ValueError(
            "No environment rule matched the context; add a default rule "
            "(one with no `when:`) as the last rule."
        )

    def outcome_success(self, context: Dict[str, Any], action_id: str, rng: random.Random) -> float:
        """Sample a noisy success score in [0, 1] for taking ``action_id``."""
        base = self.base_success(context)[action_id]
        return max(0.0, min(1.0, base + rng.gauss(0.0, self.noise_std)))

    def optimal_action(self, context: Dict[str, Any]) -> str:
        """The best action under the noise-free table (used only for scoring)."""
        scores = self.base_success(context)
        return max(scores, key=scores.__getitem__)

    def describe(self, context: Dict[str, Any]) -> str:
        """Human-readable one-line summary of a context for the episode log."""
        return ", ".join(f"{var.name}={context[var.name]}" for var in self.variables)


# ---------------------------------------------------------------------------
# 2. YAML loading + validation.
# ---------------------------------------------------------------------------
def _parse_variable(data: Dict[str, Any]) -> ContextVariable:
    name = str(data["name"])
    kind = str(data.get("type", "continuous"))
    if kind == "continuous":
        raw_round = data.get("round")
        return ContextVariable(
            name=name,
            kind="continuous",
            low=float(data.get("low", 0.0)),
            high=float(data.get("high", 1.0)),
            ndigits=int(raw_round) if raw_round is not None else None,
        )
    if kind == "categorical":
        categories = tuple(str(c) for c in data["categories"])
        if not categories:
            raise ValueError(f"Categorical variable {name!r} needs non-empty `categories`.")
        return ContextVariable(name=name, kind="categorical", categories=categories)
    raise ValueError(f"Variable {name!r} has unknown type {kind!r}.")


def _parse_rule(data: Dict[str, Any], action_ids: Sequence[str]) -> Rule:
    success = {str(k): float(v) for k, v in data["success"].items()}
    missing = [aid for aid in action_ids if aid not in success]
    if missing:
        raise ValueError(f"Rule {data.get('name', '?')!r} is missing success values for: {missing}")
    return Rule(
        name=str(data.get("name", "rule")),
        when=dict(data.get("when") or {}),
        success=success,
    )


def _parse_use_case(data: Dict[str, Any]) -> UseCase:
    actions = [Action(id=str(a["id"]), description=a.get("description")) for a in data["actions"]]
    if not actions:
        raise ValueError("`actions` must list at least one action.")
    action_ids = [a.id for a in actions]

    variables = [_parse_variable(v) for v in data["context"]["variables"]]
    if not variables:
        raise ValueError("`context.variables` must list at least one variable.")

    env = data["environment"]
    rules = [_parse_rule(r, action_ids) for r in env["rules"]]
    if not rules:
        raise ValueError("`environment.rules` must list at least one rule.")

    probes = [Probe(label=str(p["label"]), context=dict(p["context"])) for p in data.get("probes", [])]

    t = data.get("training") or {}
    training = TrainingConfig(
        seed=int(t.get("seed", 7)),
        rounds=int(t.get("rounds", 200)),
        episodes_per_round=int(t.get("episodes_per_round", 150)),
        learning_rate=float(t.get("learning_rate", 0.5)),
        entropy_bonus=float(t.get("entropy_bonus", 0.03)),
    )

    return UseCase(
        name=str(data.get("name", "Next Best Action")),
        description=str(data.get("description", "")),
        agent_id=str(data.get("agent_id", "agent")),
        entity=str(data.get("entity", "context")),
        decision_label=str(data.get("decision_label", "next_best_action")),
        actions=actions,
        variables=variables,
        noise_std=float(env.get("noise_std", 0.0)),
        rules=rules,
        probes=probes,
        training=training,
    )


def load_use_case(path: Union[str, Path]) -> UseCase:
    """Load and validate a use-case definition from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"Config file {path} must contain a YAML mapping.")
    try:
        return _parse_use_case(data)
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid use-case config {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# 3. A contextual REINFORCE learner.
#    The built-in ``ReinforceLearner`` only supports the marginal
#    ``SoftmaxPolicy``, so we implement the linear-softmax policy gradient
#    here. For a context with features phi, the action probabilities are
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
        feature_dim = policy.feature_dim

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
            if phi.shape != (feature_dim,):
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
        deltas: Dict[str, np.ndarray] = {aid: np.zeros(feature_dim) for aid in action_index}
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
# 4. Runner that scores each episode from the simulated outcome (no Azure).
# ---------------------------------------------------------------------------
class _SimulatedOutcomeRunner(LearningRunner):
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
# 5. Helpers for reporting.
# ---------------------------------------------------------------------------
def recommend(
    uc: UseCase, policy: ContextualSoftmaxPolicy, context: Dict[str, Any]
) -> Tuple[str, List[float]]:
    """Return the greedy next best action and the full action distribution."""
    probs = policy.probabilities(uc.encode(context))
    return uc.action_ids[int(np.argmax(probs))], probs


def _print_recommendations(uc: UseCase, policy: ContextualSoftmaxPolicy) -> None:
    label_w = max((len(p.label) for p in uc.probes), default=0)
    action_w = max(len(a) for a in uc.action_ids)
    for probe in uc.probes:
        best, probs = recommend(uc, policy, probe.context)
        dist = "  ".join(f"{aid}={p:.2f}" for aid, p in zip(uc.action_ids, probs))
        print(f"  {probe.label:{label_w}s} -> {best:{action_w}s} | {dist}")


def _optimality_rate(uc: UseCase, policy: ContextualSoftmaxPolicy, n: int = 4000) -> float:
    """Fraction of random contexts where the greedy pick is optimal."""
    rng = random.Random(123)
    hits = sum(
        1
        for _ in range(n)
        if recommend(uc, policy, (c := uc.sample_context(rng)))[0] == uc.optimal_action(c)
    )
    return hits / n


# ---------------------------------------------------------------------------
# 6. Training loop.
# ---------------------------------------------------------------------------
def run_use_case(uc: UseCase) -> ContextualSoftmaxPolicy:
    """Train a contextual policy for one use case and report before/after."""
    rng = random.Random(uc.training.seed)
    store = InMemoryStore()

    policy = ContextualSoftmaxPolicy.from_actions(
        uc.actions, feature_dim=uc.feature_dim, agent_id=uc.agent_id, rng=rng
    )
    store.store_policy(policy.snapshot())

    runner = _SimulatedOutcomeRunner(
        store=store,
        policy=policy,
        metrics=[],  # no Azure judges; evaluate_episode is overridden above
        shaper=RewardShaper(ShapingConfig()),
        writer=RewardWriter(store),
        learner=ContextualReinforceLearner(
            LearnerConfig(
                learning_rate=uc.training.learning_rate,
                entropy_bonus=uc.training.entropy_bonus,
            )
        ),
    )

    print(f"=== {uc.name}: next best action per {uc.entity} ===")
    if uc.description:
        print(uc.description)

    print("\n=== Recommended next best action BEFORE training ===")
    _print_recommendations(uc, policy)
    print(f"  optimal-action rate: {_optimality_rate(uc, policy):.0%}")

    # Online loop: each round samples fresh episodes under the current policy,
    # then applies one batched contextual REINFORCE update.
    rounds = uc.training.rounds
    episodes_per_round = uc.training.episodes_per_round
    print(f"\nTraining for {rounds} rounds x {episodes_per_round} episodes ...")
    for r in range(rounds):
        agent_id = f"{uc.agent_id}-r{r}"
        for i in range(episodes_per_round):
            context = uc.sample_context(rng)
            phi = uc.encode(context)
            decision = policy.choose(phi)
            success = uc.outcome_success(context, decision.action.id, rng)
            snapshot = policy.snapshot()
            episode = Episode(
                id=f"r{r}-ep{i}",
                agent_id=agent_id,
                user_input=uc.describe(context),
                assistant_output=f"{uc.decision_label}={decision.action.id}",
                policy_id=snapshot.id,
                policy_version=snapshot.version,
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
                f"optimal-action rate={_optimality_rate(uc, policy):.0%}"
            )

    print("\n=== Recommended next best action AFTER training ===")
    _print_recommendations(uc, policy)
    print(f"  optimal-action rate: {_optimality_rate(uc, policy):.0%}")
    return policy


# ---------------------------------------------------------------------------
# 7. Entry point.
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Config-driven Next Best Action contextual bandit.")
    parser.add_argument(
        "config",
        nargs="?",
        default=str(_DEFAULT_CONFIG),
        help="Path to a use-case YAML file (defaults to the retention-risk use case).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    use_case = load_use_case(args.config)
    run_use_case(use_case)


if __name__ == "__main__":
    main()
