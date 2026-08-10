"""Tests for evidence-gated autonomous decisions."""

from __future__ import annotations

import pytest

from agent_learning.autonomy import (
    ComplexityProfile,
    assess_autonomy,
    assess_complexity,
    wilson_lower_bound,
)
from agent_learning.config import AutonomyConfig
from agent_learning.policy import SoftmaxPolicy
from agent_learning.storage import InMemoryStore
from agent_learning.training import LearningRunner
from agent_learning.types import Action, Episode, PolicySnapshot, Reward, RewardSource


def _policy(version: int, *, winner_logit: float = 2.0) -> PolicySnapshot:
    return PolicySnapshot(
        id=f"policy-{version}",
        agent_id="scout",
        task_id="choose-delegate",
        version=version,
        actions=[Action(id="winner"), Action(id="alternative")],
        logits={"winner": winner_logit, "alternative": 0.0},
        episodes_seen=version * 20,
        updates_applied=version,
        metadata={"policy_scope": "delegated_decision"},
    )


def _store_outcomes(store: InMemoryStore, count: int, *, reward: float = 0.7) -> None:
    for index in range(count):
        episode = Episode(
            id=f"episode-{index}",
            agent_id="scout",
            task_id="choose-delegate",
            intent_summary="Choose a delegate",
            action_id="winner",
            expected_outcome="Use the correct delegate",
            execution_status="completed",
            result_summary="The observable outcome supported the winner",
            metadata={"correct_action_id": "winner", "task_completed": True},
        )
        store.store_episode(episode)
        store.store_reward(
            Reward(
                episode_id=episode.id,
                agent_id=episode.agent_id,
                source=RewardSource.AGGREGATE,
                value=reward,
            )
        )


def test_wilson_lower_bound_requires_more_than_twenty_perfect_outcomes() -> None:
    assert wilson_lower_bound(20, 20) < 0.90
    assert wilson_lower_bound(40, 40) > 0.90


def test_autonomy_requires_all_evidence_gates() -> None:
    store = InMemoryStore()
    snapshot = _policy(1)
    store.store_policy(snapshot)
    _store_outcomes(store, 20)

    assessment = assess_autonomy(store, snapshot)

    assert not assessment.eligible
    assert assessment.criteria["minimum_outcomes"]["met"]
    assert not assessment.criteria["correctness_lower_bound"]["met"]
    assert not assessment.criteria["stable_snapshots"]["met"]


def test_autonomy_accepts_strong_stable_policy() -> None:
    store = InMemoryStore()
    for version in range(1, 4):
        store.store_policy(_policy(version))
    snapshot = store.get_active_policy("scout", "choose-delegate")
    assert snapshot is not None
    _store_outcomes(store, 40)

    assessment = assess_autonomy(store, snapshot)

    assert assessment.eligible
    assert assessment.recommended_action_id == "winner"
    assert assessment.scored_outcomes == 40
    assert assessment.correctness_evaluated == 40
    assert assessment.correctness_rate == 1.0
    assert assessment.correctness_lower_bound > 0.90
    assert assessment.mean_reward == pytest.approx(0.7)
    assert assessment.action_probability > 0.80
    assert assessment.probability_margin > 0.70
    assert assessment.stable_snapshots == 3
    assert all(item["met"] for item in assessment.criteria.values())


def test_observable_reward_drift_revokes_autonomy_without_correctness_labels() -> None:
    store = InMemoryStore()
    for version in range(1, 4):
        store.store_policy(_policy(version))
    snapshot = store.get_active_policy("scout", "choose-delegate")
    assert snapshot is not None
    _store_outcomes(store, 40)
    assert assess_autonomy(store, snapshot).eligible

    for index in range(50):
        episode = Episode(
            id=f"drift-{index}",
            agent_id="scout",
            task_id="choose-delegate",
            intent_summary="Choose a delegate",
            action_id="winner",
            expected_outcome="Complete the delegated task",
            execution_status="failed",
            result_summary="The automatic execution failed",
            metadata={"task_completed": False, "outcome_source": "observable"},
        )
        store.store_episode(episode)
        store.store_reward(
            Reward(
                episode_id=episode.id,
                agent_id=episode.agent_id,
                source=RewardSource.AGGREGATE,
                value=-0.8,
            )
        )

    assessment = assess_autonomy(store, snapshot)

    assert not assessment.eligible
    assert assessment.scored_outcomes == 90
    assert assessment.correctness_evaluated == 40
    assert assessment.correctness_lower_bound > 0.90
    assert assessment.mean_reward is not None and assessment.mean_reward < 0.0
    assert not assessment.criteria["positive_mean_reward"]["met"]


def test_autonomy_config_validates_thresholds() -> None:
    with pytest.raises(ValueError, match="audit_rate"):
        AutonomyConfig(audit_rate=1.1)


def test_complexity_profile_maps_intent_and_decision_dimensions() -> None:
    low = assess_complexity(
        ComplexityProfile(
            intent_ambiguity="low",
            context_variability="stable",
            outcome_observability="direct",
            decision_impact="low",
            reversibility="reversible",
        ),
        2,
    )
    standard = assess_complexity(ComplexityProfile(), 3, profile_source="default")
    high = assess_complexity(
        ComplexityProfile(
            intent_ambiguity="high",
            context_variability="dynamic",
            outcome_observability="subjective",
            decision_impact="medium",
            reversibility="costly",
        ),
        3,
    )
    critical = assess_complexity(
        ComplexityProfile(
            decision_impact="high",
            reversibility="irreversible",
        ),
        2,
    )

    assert low.tier == "low" and low.score == 0
    assert standard.tier == "standard" and standard.score == 6
    assert standard.profile_source == "default"
    assert high.tier == "high"
    assert "ambiguous_subjective_outcome" in high.risk_floors
    assert critical.tier == "critical"
    assert "high_impact_irreversible" in critical.risk_floors


def test_low_complexity_accepts_three_consistent_outcomes() -> None:
    store = InMemoryStore()
    snapshot = _policy(1)
    snapshot.metadata["complexity_profile"] = ComplexityProfile(
        intent_ambiguity="low",
        context_variability="stable",
        outcome_observability="direct",
        decision_impact="low",
        reversibility="reversible",
    ).to_dict()
    store.store_policy(snapshot)
    _store_outcomes(store, 3)

    assessment = assess_autonomy(store, snapshot)

    assert assessment.complexity.tier == "low"
    assert assessment.eligible
    assert assessment.criteria["minimum_outcomes"]["required"] == 3
    assert assessment.criteria["correctness_lower_bound"]["required"] == 0.40


def test_one_user_acceptance_grants_durable_autonomy() -> None:
    store = InMemoryStore()
    snapshot = _policy(1)
    store.store_policy(snapshot)
    store.store_episode(
        Episode(
            id="accepted-policy",
            agent_id="scout",
            task_id="choose-delegate",
            intent_summary="Choose a delegate",
            action_id="alternative",
            expected_outcome="Use the user's accepted delegate",
            execution_status="completed",
            result_summary="The user accepted the alternative",
            metadata={
                "feedback_status": "accepted",
                "correct_action_id": "alternative",
                "task_completed": True,
            },
        )
    )

    assessment = assess_autonomy(store, snapshot)

    assert assessment.eligible
    assert assessment.authorization_basis == "user_acceptance"
    assert assessment.user_feedback_status == "accepted"
    assert assessment.user_feedback_episode_id == "accepted-policy"
    assert assessment.recommended_action_id == "alternative"
    assert assessment.scored_outcomes == 0
    assert assessment.audit_rate == 0.0


def test_policy_metadata_keeps_acceptance_outside_episode_window() -> None:
    store = InMemoryStore()
    snapshot = _policy(1)
    snapshot.metadata["explicit_user_feedback"] = {
        "status": "accepted",
        "action_id": "alternative",
        "episode_id": "older-than-query-window",
        "recorded_at": "2026-08-09T10:00:00+00:00",
    }
    store.store_policy(snapshot)

    assessment = assess_autonomy(store, snapshot)

    assert assessment.eligible
    assert assessment.authorization_basis == "user_acceptance"
    assert assessment.recommended_action_id == "alternative"
    assert assessment.user_feedback_episode_id == "older-than-query-window"
    assert assessment.audit_rate == 0.0


def test_later_user_rejection_revokes_acceptance() -> None:
    store = InMemoryStore()
    snapshot = _policy(1)
    store.store_policy(snapshot)
    for episode_id, status, created_at in (
        ("accepted-policy", "accepted", "2026-08-09T10:00:00+00:00"),
        ("rejected-policy", "rejected", "2026-08-09T11:00:00+00:00"),
    ):
        store.store_episode(
            Episode(
                id=episode_id,
                agent_id="scout",
                task_id="choose-delegate",
                intent_summary="Choose a delegate",
                action_id="winner",
                expected_outcome="Use the user's chosen delegate",
                execution_status="completed",
                result_summary=f"The user {status} the recommendation",
                metadata={"feedback_status": status},
                created_at=created_at,
            )
        )

    assessment = assess_autonomy(store, snapshot)

    assert not assessment.eligible
    assert assessment.authorization_basis == "none"
    assert assessment.user_feedback_status == "rejected"
    assert assessment.user_feedback_episode_id == "rejected-policy"
    assert not assessment.criteria["latest_user_feedback_not_rejected"]["met"]


def test_newer_episode_rejection_overrides_persisted_acceptance() -> None:
    store = InMemoryStore()
    snapshot = _policy(1)
    snapshot.metadata["explicit_user_feedback"] = {
        "status": "accepted",
        "action_id": "winner",
        "episode_id": "accepted-policy",
        "recorded_at": "2026-08-09T10:00:00+00:00",
    }
    store.store_policy(snapshot)
    store.store_episode(
        Episode(
            id="rejected-policy",
            agent_id="scout",
            task_id="choose-delegate",
            intent_summary="Choose a delegate",
            action_id="winner",
            expected_outcome="Use the user's chosen delegate",
            execution_status="completed",
            result_summary="The user rejected the recommendation",
            metadata={"feedback_status": "rejected"},
            created_at="2026-08-09T11:00:00+00:00",
        )
    )

    assessment = assess_autonomy(store, snapshot)

    assert not assessment.eligible
    assert assessment.user_feedback_status == "rejected"
    assert assessment.user_feedback_episode_id == "rejected-policy"


def test_low_complexity_rejects_mixed_correctness_evidence() -> None:
    store = InMemoryStore()
    snapshot = _policy(1)
    snapshot.metadata["complexity_profile"] = ComplexityProfile(
        intent_ambiguity="low",
        context_variability="stable",
        outcome_observability="direct",
        decision_impact="low",
        reversibility="reversible",
    ).to_dict()
    store.store_policy(snapshot)
    _store_outcomes(store, 3)
    store.get_episode("episode-2", "scout").metadata["correct_action_id"] = (
        "alternative"
    )

    assessment = assess_autonomy(store, snapshot)

    assert not assessment.eligible
    assert not assessment.criteria["correctness_lower_bound"]["met"]


def test_pubsub_workload_recommendation_earns_autonomy_after_three_acceptances() -> None:
    store = InMemoryStore()
    policy = SoftmaxPolicy.from_actions(
        [
            Action(id="functions"),
            Action(id="container_apps"),
            Action(id="aks"),
        ],
        agent_id="scout",
        task_id="choose-azure-pubsub-message-processor",
    )
    snapshot = policy.snapshot()
    snapshot.metadata.update(
        {
            "policy_scope": "delegated_decision",
            "complexity_profile": ComplexityProfile(
                intent_ambiguity="low",
                context_variability="stable",
                outcome_observability="direct",
                decision_impact="low",
                reversibility="reversible",
            ).to_dict(),
        }
    )
    store.store_policy(snapshot)

    for index in range(3):
        episode = Episode(
            id=f"acceptance-{index}",
            agent_id="scout",
            task_id="choose-azure-pubsub-message-processor",
            intent_summary="Choose an Azure workload to process PubSub messages",
            action_id="functions",
            expected_outcome="Recommend the accepted workload",
            execution_status="completed",
            result_summary="The user accepted the Functions recommendation",
            metadata={"correct_action_id": "functions", "task_completed": True},
        )
        store.store_episode(episode)
        store.store_reward(
            Reward(
                episode_id=episode.id,
                agent_id=episode.agent_id,
                source=RewardSource.AGGREGATE,
                value=0.7,
            )
        )

    runner = LearningRunner(
        store=store,
        policy=SoftmaxPolicy.from_snapshot(snapshot),
        metrics=[],
    )
    runner.run_offline_batch(
        "scout",
        task_id="choose-azure-pubsub-message-processor",
        episode_limit=3,
        score_missing=False,
    )
    trained = store.get_active_policy(
        "scout", "choose-azure-pubsub-message-processor"
    )
    assert trained is not None

    assessment = assess_autonomy(store, trained)

    assert assessment.complexity.score == 1
    assert assessment.complexity.tier == "low"
    assert assessment.scored_outcomes == 3
    assert assessment.stable_snapshots == 1
    assert assessment.eligible


def test_same_evidence_requires_more_for_high_complexity() -> None:
    store = InMemoryStore()
    for version in range(1, 4):
        store.store_policy(_policy(version))
    snapshot = store.get_active_policy("scout", "choose-delegate")
    assert snapshot is not None
    _store_outcomes(store, 40)

    snapshot.metadata["complexity_profile"] = ComplexityProfile(
        intent_ambiguity="low",
        context_variability="stable",
        outcome_observability="direct",
        decision_impact="low",
        reversibility="reversible",
    ).to_dict()
    store.store_policy(snapshot)
    assert assess_autonomy(store, snapshot).eligible

    snapshot.metadata["complexity_profile"] = ComplexityProfile(
        intent_ambiguity="high",
        context_variability="dynamic",
        outcome_observability="subjective",
        decision_impact="high",
        reversibility="costly",
    ).to_dict()
    store.store_policy(snapshot)
    assessment = assess_autonomy(store, snapshot)

    assert assessment.complexity.tier == "high"
    assert not assessment.eligible
    assert assessment.criteria["minimum_outcomes"] == {
        "actual": 40,
        "required": 50,
        "met": False,
    }
    assert assessment.audit_rate == 0.25


def test_human_approval_profile_blocks_otherwise_eligible_policy() -> None:
    store = InMemoryStore()
    for version in range(1, 4):
        store.store_policy(_policy(version))
    snapshot = store.get_active_policy("scout", "choose-delegate")
    assert snapshot is not None
    _store_outcomes(store, 40)
    snapshot.metadata["complexity_profile"] = ComplexityProfile(
        intent_ambiguity="low",
        context_variability="stable",
        outcome_observability="direct",
        decision_impact="low",
        reversibility="reversible",
        requires_human_approval=True,
    ).to_dict()
    snapshot.metadata["explicit_user_feedback"] = {
        "status": "accepted",
        "action_id": "winner",
        "episode_id": "human-accepted-policy",
        "recorded_at": "2026-08-09T10:00:00+00:00",
    }
    store.store_policy(snapshot)

    assessment = assess_autonomy(store, snapshot)

    assert not assessment.eligible
    assert assessment.authorization_basis == "none"
    assert assessment.user_feedback_status == "accepted"
    assert not assessment.criteria["human_approval_not_required"]["met"]


def test_complexity_profile_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown complexity profile fields"):
        ComplexityProfile.from_dict({"intent_complexity": "low"})