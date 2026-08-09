"""Tests for evidence-gated autonomous decisions."""

from __future__ import annotations

import pytest

from agent_learning.autonomy import assess_autonomy, wilson_lower_bound
from agent_learning.config import AutonomyConfig
from agent_learning.storage import InMemoryStore
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