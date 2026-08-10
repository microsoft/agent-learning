"""Tests for reasoned and learned resolution through one task policy."""

import math
import random

import pytest

from agent_learning.decision import (
    DecisionAuthority,
    DecisionCriterion,
    DecisionFrame,
    DecisionOption,
    DecisionResult,
    DecisionStatus,
    EvidencePoint,
    TaskPolicy,
)
from agent_learning.types import Action, PolicySnapshot


def _snapshot(authority: DecisionAuthority, *, logits: dict[str, float] | None = None) -> PolicySnapshot:
    return PolicySnapshot(
        id="policy-1",
        agent_id="agent-1",
        task_id="choose-region",
        version=3,
        actions=[Action(id="east"), Action(id="west")],
        logits=logits or {"east": 0.0, "west": 0.0},
        metadata={"decision_authority": authority.value},
    )


def test_user_must_accept_or_reject_to_resolve_tied_options() -> None:
    snapshot = _snapshot(DecisionAuthority.FULL)
    frame = DecisionFrame(
        task="Choose a deployment region",
        criteria=[DecisionCriterion(id="fit")],
        options=[
            DecisionOption(
                action=Action(id="east"),
                evidence=[EvidencePoint(criterion_id="fit", source="capacity", support=0.9)],
            ),
            DecisionOption(
                action=Action(id="west"),
                evidence=[EvidencePoint(criterion_id="fit", source="capacity", support=0.9)],
            ),
        ],
    )

    policy = TaskPolicy(snapshot)
    result = policy.decide(frame)

    assert result.status is DecisionStatus.NEEDS_USER_TIE_BREAK
    assert result.proposed_action == Action(id="east")
    assert result.policy_id == snapshot.id
    assert result.policy_version == snapshot.version
    assert result.selection_basis == "bayesian_decision"

    result = policy.adjudicate(result, "reject")
    assert result.status is DecisionStatus.NEEDS_USER_TIE_BREAK
    assert result.proposed_action == Action(id="west")

    result = policy.adjudicate(result, "accept")
    assert result.status is DecisionStatus.RESOLVED
    assert result.selected_action == Action(id="west")
    assert result.authorization_basis == "user_acceptance"


def test_low_authority_uses_same_policy_and_rejection_becomes_learning_signal() -> None:
    snapshot = _snapshot(
        DecisionAuthority.LOW,
        logits={"east": 10.0, "west": -10.0},
    )
    policy = TaskPolicy(snapshot, rng=random.Random(7))

    recommendation = policy.decide()

    assert recommendation.status is DecisionStatus.NEEDS_USER_FEEDBACK
    assert recommendation.proposed_action == Action(id="east")
    assert recommendation.policy_id == snapshot.id
    assert recommendation.policy_version == snapshot.version
    assert recommendation.selection_basis == "learned_policy"

    rejected = policy.adjudicate(recommendation, "reject")
    assert rejected.status is DecisionStatus.REJECTED
    assert rejected.selected_action is None
    assert rejected.proposed_action is None
    assert rejected.authorization_basis == "user_rejection"


def test_low_authority_can_select_an_authorized_policy_action_deterministically() -> None:
    snapshot = _snapshot(
        DecisionAuthority.LOW,
        logits={"east": 2.0, "west": 0.0},
    )
    policy = TaskPolicy(snapshot)

    result = policy.decide(selected_action_id="west")

    assert result.proposed_action == Action(id="west")
    assert result.action_logprob == pytest.approx(
        math.log(result.action_probabilities["west"])
    )


def test_full_authority_frame_must_use_the_task_policy_action_space() -> None:
    policy = TaskPolicy(_snapshot(DecisionAuthority.FULL))
    frame = DecisionFrame(
        task="Choose a deployment region",
        criteria=[DecisionCriterion(id="fit")],
        options=[
            DecisionOption(
                action=Action(id="east"),
                evidence=[EvidencePoint(criterion_id="fit", source="capacity", support=0.9)],
            ),
            DecisionOption(
                action=Action(id="north"),
                evidence=[EvidencePoint(criterion_id="fit", source="capacity", support=0.8)],
            ),
        ],
    )

    with pytest.raises(ValueError, match="action space"):
        policy.decide(frame)


def test_missing_constraint_and_criterion_evidence_are_ranked_by_information_gain() -> None:
    policy = TaskPolicy(_snapshot(DecisionAuthority.FULL))
    frame = DecisionFrame(
        task="Choose a deployment region",
        criteria=[
            DecisionCriterion(id="capacity", weight=0.8, minimum_sources=2),
            DecisionCriterion(id="latency", weight=0.2),
        ],
        constraints=["residency"],
        options=[
            DecisionOption(action=Action(id="east")),
            DecisionOption(
                action=Action(id="west"),
                constraint_results={"residency": True},
            ),
        ],
    )

    result = policy.decide(frame)

    assert result.status is DecisionStatus.NEEDS_EVIDENCE
    assert [
        (need.kind, need.option_id, need.field_id)
        for need in result.information_needs
    ] == [
        ("constraint", "east", "residency"),
        ("criterion", "west", "capacity"),
        ("criterion", "west", "latency"),
    ]


def test_hard_constraints_can_rule_out_every_policy_action() -> None:
    policy = TaskPolicy(_snapshot(DecisionAuthority.FULL))
    frame = DecisionFrame(
        task="Choose a deployment region",
        criteria=[DecisionCriterion(id="fit")],
        constraints=["residency"],
        options=[
            DecisionOption(
                action=Action(id="east"),
                constraint_results={"residency": False},
            ),
            DecisionOption(
                action=Action(id="west"),
                constraint_results={"residency": False},
            ),
        ],
    )

    result = policy.decide(frame)

    assert result.status is DecisionStatus.NO_VIABLE_OPTION
    assert all(assessment.feasible is False for assessment in result.assessments)
    assert all(
        assessment.ruled_out_by == ("constraint:residency",)
        for assessment in result.assessments
    )


def test_independent_bayesian_evidence_selects_and_rules_out_dominated_action() -> None:
    policy = TaskPolicy(_snapshot(DecisionAuthority.FULL))
    frame = DecisionFrame(
        task="Choose a deployment region",
        criteria=[DecisionCriterion(id="fit", minimum_sources=2)],
        options=[
            DecisionOption(
                action=Action(id="east"),
                evidence=[
                    EvidencePoint(criterion_id="fit", source="capacity", support=0.8),
                    EvidencePoint(criterion_id="fit", source="quota", support=0.8),
                ],
            ),
            DecisionOption(
                action=Action(id="west"),
                evidence=[
                    EvidencePoint(criterion_id="fit", source="capacity", support=0.6),
                    EvidencePoint(criterion_id="fit", source="quota", support=0.6),
                ],
            ),
        ],
    )

    result = policy.decide(frame)

    assert result.status is DecisionStatus.RESOLVED
    assert result.selected_action == Action(id="east")
    east, west = result.assessments
    assert east.criteria[0].support > 0.8
    assert west.ruled_out_by == ("pareto_dominated_by:east",)
    assert result.authorization_basis == "reasoned_evidence"


def test_human_approval_blocks_unique_reasoned_winner_until_accept_or_reject() -> None:
    snapshot = _snapshot(DecisionAuthority.FULL)
    snapshot.metadata["complexity_profile"] = {"requires_human_approval": True}
    policy = TaskPolicy(snapshot)
    frame = DecisionFrame(
        task="Choose a deployment region",
        criteria=[DecisionCriterion(id="fit")],
        options=[
            DecisionOption(
                action=Action(id="east"),
                evidence=[EvidencePoint(criterion_id="fit", source="capacity", support=0.9)],
            ),
            DecisionOption(
                action=Action(id="west"),
                evidence=[EvidencePoint(criterion_id="fit", source="capacity", support=0.4)],
            ),
        ],
    )

    pending = policy.decide(frame)
    rejected = policy.adjudicate(pending, "reject")

    assert pending.status is DecisionStatus.NEEDS_USER_FEEDBACK
    assert pending.proposed_action == Action(id="east")
    assert rejected.status is DecisionStatus.REJECTED
    assert rejected.authorization_basis == "user_rejection"


def test_decision_certificate_round_trips_for_later_adjudication() -> None:
    policy = TaskPolicy(_snapshot(DecisionAuthority.FULL))
    frame = DecisionFrame(
        task="Choose a deployment region",
        criteria=[DecisionCriterion(id="fit")],
        options=[
            DecisionOption(
                action=Action(id=action_id),
                evidence=[EvidencePoint(criterion_id="fit", source="capacity", support=0.7)],
            )
            for action_id in ("east", "west")
        ],
    )

    pending = policy.decide(frame)
    restored = DecisionResult.from_dict(pending.to_dict())

    assert restored == pending
    assert policy.adjudicate(restored, "accept").selected_action == Action(id="east")


def test_frame_rejects_non_string_identifiers() -> None:
    policy = TaskPolicy(_snapshot(DecisionAuthority.FULL))
    frame = DecisionFrame(
        task="Choose a deployment region",
        criteria=[DecisionCriterion(id=1)],  # type: ignore[arg-type]
        options=[
            DecisionOption(action=Action(id="east")),
            DecisionOption(action=Action(id="west")),
        ],
    )

    with pytest.raises(TypeError, match="IDs must be strings"):
        policy.decide(frame)


def test_legacy_task_policy_defaults_to_low_decision_authority() -> None:
    snapshot = _snapshot(DecisionAuthority.LOW)
    snapshot.metadata = {"policy_scope": "delegated_decision"}

    policy = TaskPolicy(snapshot)

    assert policy.authority is DecisionAuthority.LOW
    assert policy.decide().selection_basis == "learned_policy"


def test_adjudication_rejects_resolved_result() -> None:
    policy = TaskPolicy(_snapshot(DecisionAuthority.FULL))
    frame = DecisionFrame(
        task="Choose a deployment region",
        criteria=[DecisionCriterion(id="fit")],
        options=[
            DecisionOption(
                action=Action(id="east"),
                evidence=[EvidencePoint(criterion_id="fit", source="capacity", support=0.9)],
            ),
            DecisionOption(
                action=Action(id="west"),
                evidence=[EvidencePoint(criterion_id="fit", source="capacity", support=0.4)],
            ),
        ],
    )
    resolved = policy.decide(frame)

    with pytest.raises(ValueError, match="only a pending user decision"):
        policy.adjudicate(resolved, "accept")


def test_adjudication_rejects_stale_or_tampered_certificate() -> None:
    snapshot = _snapshot(DecisionAuthority.FULL)
    policy = TaskPolicy(snapshot)
    frame = DecisionFrame(
        task="Choose a deployment region",
        criteria=[DecisionCriterion(id="fit")],
        options=[
            DecisionOption(
                action=Action(id=action_id),
                evidence=[EvidencePoint(criterion_id="fit", source="capacity", support=0.7)],
            )
            for action_id in ("east", "west")
        ],
    )
    pending = policy.decide(frame)

    stale_snapshot = PolicySnapshot.from_dict(snapshot.to_dict())
    stale_snapshot.version += 1
    with pytest.raises(ValueError, match="does not belong"):
        TaskPolicy(stale_snapshot).adjudicate(pending, "accept")

    tampered_payload = pending.to_dict()
    tampered_payload["policy_id"] = "different-policy"
    tampered = DecisionResult.from_dict(tampered_payload)
    with pytest.raises(ValueError, match="does not belong"):
        policy.adjudicate(tampered, "accept")


def test_decision_result_rejects_invalid_status() -> None:
    policy = TaskPolicy(_snapshot(DecisionAuthority.FULL))
    frame = DecisionFrame(
        task="Choose a deployment region",
        criteria=[DecisionCriterion(id="fit")],
        options=[
            DecisionOption(
                action=Action(id=action_id),
                evidence=[EvidencePoint(criterion_id="fit", source="capacity", support=0.7)],
            )
            for action_id in ("east", "west")
        ],
    )
    payload = policy.decide(frame).to_dict()
    payload["status"] = "unknown"

    with pytest.raises(ValueError, match="invalid decision status"):
        DecisionResult.from_dict(payload)