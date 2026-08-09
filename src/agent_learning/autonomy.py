"""Evidence gates for autonomous task-policy execution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .config import AutonomyConfig
from .policy.softmax_bandit import SoftmaxPolicy
from .storage.base import LearningStore
from .types import Episode, PolicySnapshot, RewardSource


def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    """Return the Wilson-score lower bound for a Bernoulli proportion."""
    if total <= 0:
        return 0.0
    proportion = successes / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    center = proportion + z_squared / (2.0 * total)
    spread = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z_squared / (4.0 * total * total)
    )
    return max(0.0, (center - spread) / denominator)


@dataclass(frozen=True)
class AutonomyAssessment:
    """Inspectable evidence supporting or blocking autonomous execution."""

    eligible: bool
    recommended_action_id: str
    scored_outcomes: int
    correctness_evaluated: int
    correct_outcomes: int
    correctness_rate: float | None
    correctness_lower_bound: float
    mean_reward: float | None
    action_probability: float
    probability_margin: float
    stable_snapshots: int
    criteria: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "recommended_action_id": self.recommended_action_id,
            "scored_outcomes": self.scored_outcomes,
            "correctness_evaluated": self.correctness_evaluated,
            "correct_outcomes": self.correct_outcomes,
            "correctness_rate": self.correctness_rate,
            "correctness_lower_bound": self.correctness_lower_bound,
            "mean_reward": self.mean_reward,
            "action_probability": self.action_probability,
            "probability_margin": self.probability_margin,
            "stable_snapshots": self.stable_snapshots,
            "criteria": self.criteria,
        }


def _latest_aggregate(store: LearningStore, episode: Episode) -> float | None:
    rewards = [
        reward
        for reward in store.get_rewards_for_episode(episode.id, episode.agent_id)
        if reward.source == RewardSource.AGGREGATE
    ]
    if not rewards:
        return None
    return max(rewards, key=lambda reward: reward.created_at).value


def _recommended_action(snapshot: PolicySnapshot) -> tuple[str, float, float]:
    probabilities = SoftmaxPolicy.from_snapshot(snapshot).probabilities()
    winner_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    winner_probability = probabilities[winner_index]
    runner_up = max(
        (probability for index, probability in enumerate(probabilities) if index != winner_index),
        default=0.0,
    )
    return (
        snapshot.actions[winner_index].id,
        winner_probability,
        winner_probability - runner_up,
    )


def _stable_snapshot_count(
    store: LearningStore,
    snapshot: PolicySnapshot,
    recommended_action_id: str,
    required: int,
) -> int:
    history = [
        snapshot,
        *[
            candidate
            for candidate in store.list_policies(
                snapshot.agent_id,
                snapshot.task_id,
                limit=max(required * 2, required),
            )
            if candidate.id != snapshot.id and candidate.version < snapshot.version
        ],
    ]
    history.sort(key=lambda candidate: candidate.version, reverse=True)
    stable = 0
    for candidate in history:
        if candidate.updates_applied < 1:
            continue
        candidate_action_id, _, _ = _recommended_action(candidate)
        if candidate_action_id != recommended_action_id:
            break
        stable += 1
        if stable >= required:
            break
    return stable


def assess_autonomy(
    store: LearningStore,
    snapshot: PolicySnapshot,
    config: AutonomyConfig | None = None,
) -> AutonomyAssessment:
    """Assess whether the active policy has enough evidence for autonomy."""
    config = config or AutonomyConfig()
    recommended_action_id, action_probability, probability_margin = (
        _recommended_action(snapshot)
    )
    action_ids = {action.id for action in snapshot.actions}
    scored_outcomes = 0
    correctness_evaluated = 0
    correct_outcomes = 0
    reward_total = 0.0
    episodes = store.query_episodes(
        snapshot.agent_id,
        task_id=snapshot.task_id,
        limit=500,
        full_only=True,
    )
    for episode in episodes:
        if episode.action_id != recommended_action_id:
            continue
        reward = _latest_aggregate(store, episode)
        if reward is None:
            continue
        scored_outcomes += 1
        reward_total += reward
        correct_action_id = episode.metadata.get("correct_action_id")
        if correct_action_id in action_ids:
            correctness_evaluated += 1
            correct_outcomes += int(correct_action_id == recommended_action_id)

    correctness_rate = (
        correct_outcomes / correctness_evaluated if correctness_evaluated else None
    )
    lower_bound = wilson_lower_bound(
        correct_outcomes,
        correctness_evaluated,
        config.wilson_z,
    )
    mean_reward = reward_total / scored_outcomes if scored_outcomes else None
    stable_snapshots = _stable_snapshot_count(
        store,
        snapshot,
        recommended_action_id,
        config.stable_snapshots,
    )
    criteria: dict[str, dict[str, Any]] = {
        "minimum_outcomes": {
            "actual": scored_outcomes,
            "required": config.min_outcomes,
            "met": scored_outcomes >= config.min_outcomes,
        },
        "correctness_lower_bound": {
            "actual": lower_bound,
            "required": config.min_correctness_lower_bound,
            "wilson_z": config.wilson_z,
            "met": lower_bound >= config.min_correctness_lower_bound,
        },
        "positive_mean_reward": {
            "actual": mean_reward,
            "required_greater_than": config.min_mean_reward,
            "met": mean_reward is not None and mean_reward > config.min_mean_reward,
        },
        "action_probability": {
            "actual": action_probability,
            "required": config.min_action_probability,
            "met": action_probability >= config.min_action_probability,
        },
        "probability_margin": {
            "actual": probability_margin,
            "required": config.min_probability_margin,
            "met": probability_margin >= config.min_probability_margin,
        },
        "stable_snapshots": {
            "actual": stable_snapshots,
            "required": config.stable_snapshots,
            "met": stable_snapshots >= config.stable_snapshots,
        },
    }
    eligible = all(criterion["met"] for criterion in criteria.values())
    return AutonomyAssessment(
        eligible=eligible,
        recommended_action_id=recommended_action_id,
        scored_outcomes=scored_outcomes,
        correctness_evaluated=correctness_evaluated,
        correct_outcomes=correct_outcomes,
        correctness_rate=correctness_rate,
        correctness_lower_bound=lower_bound,
        mean_reward=mean_reward,
        action_probability=action_probability,
        probability_margin=probability_margin,
        stable_snapshots=stable_snapshots,
        criteria=criteria,
    )


__all__ = ["AutonomyAssessment", "assess_autonomy", "wilson_lower_bound"]