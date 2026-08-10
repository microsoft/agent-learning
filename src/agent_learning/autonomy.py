"""Evidence gates for autonomous task-policy execution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .config import AutonomyConfig, AutonomyTier
from .policy.softmax_bandit import SoftmaxPolicy
from .storage.base import LearningStore
from .types import Episode, PolicySnapshot, RewardSource

_AMBIGUITY_POINTS = {"low": 0, "medium": 1, "high": 2}
_VARIABILITY_POINTS = {"stable": 0, "variable": 1, "dynamic": 2}
_OBSERVABILITY_POINTS = {"direct": 0, "delayed": 1, "subjective": 2}
_IMPACT_POINTS = {"low": 0, "medium": 1, "high": 2, "critical": 3}
_REVERSIBILITY_POINTS = {"reversible": 0, "costly": 1, "irreversible": 2}
_TIER_ORDER: tuple[AutonomyTier, ...] = ("low", "standard", "high", "critical")


@dataclass(frozen=True)
class ComplexityProfile:
    """Declared complexity inputs for one reusable decision policy."""

    intent_ambiguity: str = "medium"
    context_variability: str = "variable"
    outcome_observability: str = "delayed"
    decision_impact: str = "medium"
    reversibility: str = "costly"
    requires_human_approval: bool = False
    rationale: str = ""

    def __post_init__(self) -> None:
        choices = {
            "intent_ambiguity": _AMBIGUITY_POINTS,
            "context_variability": _VARIABILITY_POINTS,
            "outcome_observability": _OBSERVABILITY_POINTS,
            "decision_impact": _IMPACT_POINTS,
            "reversibility": _REVERSIBILITY_POINTS,
        }
        for name, valid in choices.items():
            value = getattr(self, name)
            if value not in valid:
                raise ValueError(
                    f"{name} must be one of {', '.join(sorted(valid))}; got {value!r}"
                )
        if not isinstance(self.requires_human_approval, bool):
            raise TypeError("requires_human_approval must be a boolean")
        if not isinstance(self.rationale, str):
            raise TypeError("rationale must be a string")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ComplexityProfile:
        """Parse a strict profile and reject misspelled or unknown fields."""
        if not isinstance(payload, dict):
            raise TypeError("complexity profile must be a JSON object")
        valid_fields = {
            "intent_ambiguity",
            "context_variability",
            "outcome_observability",
            "decision_impact",
            "reversibility",
            "requires_human_approval",
            "rationale",
        }
        unknown = sorted(set(payload) - valid_fields)
        if unknown:
            raise ValueError(f"unknown complexity profile fields: {', '.join(unknown)}")
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_ambiguity": self.intent_ambiguity,
            "context_variability": self.context_variability,
            "outcome_observability": self.outcome_observability,
            "decision_impact": self.decision_impact,
            "reversibility": self.reversibility,
            "requires_human_approval": self.requires_human_approval,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class ComplexityAssessment:
    """Deterministic complexity score and tier for one policy."""

    tier: AutonomyTier
    score: int
    intent_score: int
    decision_score: int
    action_count: int
    action_space_points: int
    profile_source: str
    profile: ComplexityProfile
    risk_floors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "score": self.score,
            "intent_score": self.intent_score,
            "decision_score": self.decision_score,
            "action_count": self.action_count,
            "action_space_points": self.action_space_points,
            "profile_source": self.profile_source,
            "profile": self.profile.to_dict(),
            "risk_floors": list(self.risk_floors),
        }


def _at_least(tier: AutonomyTier, floor: AutonomyTier) -> AutonomyTier:
    return _TIER_ORDER[max(_TIER_ORDER.index(tier), _TIER_ORDER.index(floor))]


def assess_complexity(
    profile: ComplexityProfile,
    action_count: int,
    *,
    profile_source: str = "configured",
) -> ComplexityAssessment:
    """Map declared intent/decision complexity to a proportional tier."""
    if action_count < 2:
        raise ValueError("decision complexity requires at least two actions")
    action_space_points = 0 if action_count == 2 else 1 if action_count <= 4 else 2
    intent_score = (
        _AMBIGUITY_POINTS[profile.intent_ambiguity]
        + _VARIABILITY_POINTS[profile.context_variability]
        + _OBSERVABILITY_POINTS[profile.outcome_observability]
    )
    decision_score = (
        _IMPACT_POINTS[profile.decision_impact]
        + _REVERSIBILITY_POINTS[profile.reversibility]
        + action_space_points
    )
    score = intent_score + decision_score
    if score <= 2:
        tier: AutonomyTier = "low"
    elif score <= 6:
        tier = "standard"
    elif score <= 9:
        tier = "high"
    else:
        tier = "critical"

    risk_floors: list[str] = []
    if profile.decision_impact == "critical":
        tier = "critical"
        risk_floors.append("critical_impact")
    if profile.decision_impact == "high" and profile.reversibility == "irreversible":
        tier = "critical"
        risk_floors.append("high_impact_irreversible")
    if (
        profile.intent_ambiguity == "high"
        and profile.outcome_observability == "subjective"
    ):
        tier = _at_least(tier, "high")
        risk_floors.append("ambiguous_subjective_outcome")

    return ComplexityAssessment(
        tier=tier,
        score=score,
        intent_score=intent_score,
        decision_score=decision_score,
        action_count=action_count,
        action_space_points=action_space_points,
        profile_source=profile_source,
        profile=profile,
        risk_floors=tuple(risk_floors),
    )


def complexity_for_snapshot(snapshot: PolicySnapshot) -> ComplexityAssessment:
    """Resolve persisted complexity or a conservative standard default."""
    payload = snapshot.metadata.get("complexity_profile")
    if payload is None:
        profile = ComplexityProfile()
        source = "default"
    else:
        profile = ComplexityProfile.from_dict(payload)
        source = str(snapshot.metadata.get("complexity_profile_source") or "configured")
    return assess_complexity(profile, len(snapshot.actions), profile_source=source)


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
    authorization_basis: str
    user_feedback_status: str | None
    user_feedback_episode_id: str | None
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
    audit_rate: float
    complexity: ComplexityAssessment
    criteria: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "authorization_basis": self.authorization_basis,
            "user_feedback_status": self.user_feedback_status,
            "user_feedback_episode_id": self.user_feedback_episode_id,
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
            "audit_rate": self.audit_rate,
            "complexity": self.complexity.to_dict(),
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


def _action_confidence(
    snapshot: PolicySnapshot, action_id: str
) -> tuple[float, float]:
    probabilities = SoftmaxPolicy.from_snapshot(snapshot).probabilities()
    action_index = next(
        index for index, action in enumerate(snapshot.actions) if action.id == action_id
    )
    action_probability = probabilities[action_index]
    runner_up = max(
        (
            probability
            for index, probability in enumerate(probabilities)
            if index != action_index
        ),
        default=0.0,
    )
    return action_probability, action_probability - runner_up


def _latest_explicit_user_feedback(
    snapshot: PolicySnapshot,
    episodes: list[Episode],
    action_ids: set[str],
) -> tuple[str | None, str | None, str | None]:
    candidates: list[tuple[str, str, str | None, str | None]] = []
    persisted = snapshot.metadata.get("explicit_user_feedback")
    if isinstance(persisted, dict):
        status = str(persisted.get("status") or "").strip().lower()
        action_id = persisted.get("action_id")
        episode_id = persisted.get("episode_id")
        if status in {"accepted", "rejected"}:
            if status == "accepted" and action_id not in action_ids:
                status = "rejected"
            candidates.append(
                (
                    str(persisted.get("recorded_at") or ""),
                    status,
                    str(action_id) if action_id in action_ids else None,
                    str(episode_id or "") or None,
                )
            )
    for episode in episodes:
        status = str(episode.metadata.get("feedback_status") or "").strip().lower()
        if status not in {"accepted", "rejected"}:
            continue
        action_id = episode.action_id
        if episode.action_id not in action_ids:
            status = "rejected"
            action_id = None
        correct_action_id = episode.metadata.get("correct_action_id")
        if status == "accepted" and correct_action_id not in {
            None,
            episode.action_id,
        }:
            status = "rejected"
            action_id = None
        candidates.append((episode.created_at, status, action_id, episode.id))
        break
    if candidates:
        _, status, action_id, episode_id = max(candidates, key=lambda item: item[0])
        return status, action_id, episode_id
    return None, None, None


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
    complexity = complexity_for_snapshot(snapshot)
    config = config or AutonomyConfig.for_tier(complexity.tier)
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
    (
        user_feedback_status,
        user_feedback_action_id,
        user_feedback_episode_id,
    ) = _latest_explicit_user_feedback(
        snapshot, episodes, action_ids
    )
    if user_feedback_status == "accepted" and user_feedback_action_id is not None:
        recommended_action_id = user_feedback_action_id
        action_probability, probability_margin = _action_confidence(
            snapshot, recommended_action_id
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
        "human_approval_not_required": {
            "actual": complexity.profile.requires_human_approval,
            "required": False,
            "met": not complexity.profile.requires_human_approval,
        },
        "latest_user_feedback_not_rejected": {
            "actual": user_feedback_status,
            "required_not": "rejected",
            "met": user_feedback_status != "rejected",
        },
    }
    statistically_eligible = all(criterion["met"] for criterion in criteria.values())
    explicitly_approved = (
        user_feedback_status == "accepted"
        and not complexity.profile.requires_human_approval
    )
    eligible = explicitly_approved or statistically_eligible
    authorization_basis = (
        "user_acceptance"
        if explicitly_approved
        else "statistical_evidence"
        if statistically_eligible
        else "none"
    )
    return AutonomyAssessment(
        eligible=eligible,
        authorization_basis=authorization_basis,
        user_feedback_status=user_feedback_status,
        user_feedback_episode_id=user_feedback_episode_id,
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
        audit_rate=0.0 if explicitly_approved else config.audit_rate,
        complexity=complexity,
        criteria=criteria,
    )


__all__ = [
    "AutonomyAssessment",
    "ComplexityAssessment",
    "ComplexityProfile",
    "assess_autonomy",
    "assess_complexity",
    "complexity_for_snapshot",
    "wilson_lower_bound",
]