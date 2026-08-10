"""Reasoned and learned action selection through one task policy."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from .policy.softmax_bandit import SoftmaxPolicy
from .types import Action, PolicySnapshot


class DecisionAuthority(str, Enum):
    """How a task policy is allowed to select its next action."""

    LOW = "low"
    FULL = "full"


class DecisionStatus(str, Enum):
    """Possible outcomes from task-policy decision resolution."""

    RESOLVED = "resolved"
    NEEDS_EVIDENCE = "needs_evidence"
    NEEDS_USER_TIE_BREAK = "needs_user_tie_break"
    NEEDS_USER_FEEDBACK = "needs_user_feedback"
    REJECTED = "rejected"
    NO_VIABLE_OPTION = "no_viable_option"


class TieBreakDisposition(str, Enum):
    """The only user responses accepted by the tie-break protocol."""

    ACCEPT = "accept"
    REJECT = "reject"


@dataclass(frozen=True)
class DecisionCriterion:
    """A weighted objective supported by independent evidence sources."""

    id: str
    weight: float = 1.0
    minimum_sources: int = 1
    description: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "weight": self.weight,
            "minimum_sources": self.minimum_sources,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DecisionCriterion:
        return cls(
            id=data["id"],
            weight=float(data.get("weight", 1.0)),
            minimum_sources=int(data.get("minimum_sources", 1)),
            description=data.get("description"),
        )


@dataclass(frozen=True)
class EvidencePoint:
    """One source's normalized support for an option on one criterion."""

    criterion_id: str
    source: str
    support: float
    confidence: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "criterion_id": self.criterion_id,
            "source": self.source,
            "support": self.support,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvidencePoint:
        return cls(
            criterion_id=data["criterion_id"],
            source=data["source"],
            support=float(data["support"]),
            confidence=float(data.get("confidence", 1.0)),
        )


@dataclass(frozen=True)
class DecisionOption:
    """An executable action together with evidence and constraint results."""

    action: Action
    evidence: Sequence[EvidencePoint] = field(default_factory=tuple)
    constraint_results: Mapping[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "action_id": self.action.id,
            "evidence": [point.to_dict() for point in self.evidence],
            "constraint_results": dict(self.constraint_results),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        actions: Mapping[str, Action],
    ) -> DecisionOption:
        evidence = data.get("evidence", [])
        constraint_results = data.get("constraint_results", {})
        if not isinstance(evidence, list):
            raise TypeError("decision option evidence must be a JSON array")
        if not isinstance(constraint_results, dict):
            raise TypeError("decision option constraint_results must be a JSON object")
        action_id = data["action_id"]
        if action_id not in actions:
            raise ValueError(f"unknown task-policy action {action_id!r}")
        return cls(
            action=actions[action_id],
            evidence=tuple(
                EvidencePoint.from_dict(point) for point in evidence
            ),
            constraint_results=dict(constraint_results),
        )


@dataclass(frozen=True)
class DecisionFrame:
    """The information, alternatives, utility, and constraints for a decision."""

    task: str
    criteria: Sequence[DecisionCriterion]
    options: Sequence[DecisionOption]
    constraints: Sequence[str] = field(default_factory=tuple)
    minimum_margin: float = 0.05
    uncertainty_penalty: float = 0.10
    max_uncertainty: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "task": self.task,
            "criteria": [criterion.to_dict() for criterion in self.criteria],
            "options": [option.to_dict() for option in self.options],
            "constraints": list(self.constraints),
            "minimum_margin": self.minimum_margin,
            "uncertainty_penalty": self.uncertainty_penalty,
            "max_uncertainty": self.max_uncertainty,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        actions: Sequence[Action],
    ) -> DecisionFrame:
        known_fields = {
            "task",
            "criteria",
            "options",
            "constraints",
            "minimum_margin",
            "uncertainty_penalty",
            "max_uncertainty",
        }
        unknown_fields = set(data) - known_fields
        if unknown_fields:
            raise ValueError(f"unknown decision frame fields: {sorted(unknown_fields)}")
        criteria = data.get("criteria", [])
        options = data.get("options", [])
        constraints = data.get("constraints", [])
        if not isinstance(criteria, list):
            raise TypeError("decision frame criteria must be a JSON array")
        if not isinstance(options, list):
            raise TypeError("decision frame options must be a JSON array")
        if not isinstance(constraints, list):
            raise TypeError("decision frame constraints must be a JSON array")
        actions_by_id = {action.id: action for action in actions}
        return cls(
            task=data["task"],
            criteria=tuple(
                DecisionCriterion.from_dict(criterion)
                for criterion in criteria
            ),
            options=tuple(
                DecisionOption.from_dict(option, actions_by_id)
                for option in options
            ),
            constraints=tuple(constraints),
            minimum_margin=float(data.get("minimum_margin", 0.05)),
            uncertainty_penalty=float(data.get("uncertainty_penalty", 0.10)),
            max_uncertainty=float(data.get("max_uncertainty", 1.0)),
        )


@dataclass(frozen=True)
class CriterionAssessment:
    """Information-theoretic evidence summary for one criterion."""

    criterion_id: str
    support: float
    entropy_bits: float
    disagreement_bits: float
    source_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "criterion_id": self.criterion_id,
            "support": self.support,
            "entropy_bits": self.entropy_bits,
            "disagreement_bits": self.disagreement_bits,
            "source_count": self.source_count,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CriterionAssessment:
        return cls(
            criterion_id=data["criterion_id"],
            support=float(data["support"]),
            entropy_bits=float(data["entropy_bits"]),
            disagreement_bits=float(data["disagreement_bits"]),
            source_count=int(data["source_count"]),
        )


@dataclass(frozen=True)
class OptionAssessment:
    """Constraint, utility, and uncertainty assessment for one option."""

    action: Action
    feasible: bool | None
    criteria: tuple[CriterionAssessment, ...] = ()
    expected_utility: float | None = None
    uncertainty: float | None = None
    robust_utility: float | None = None
    ruled_out_by: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action.to_dict(),
            "feasible": self.feasible,
            "criteria": [criterion.to_dict() for criterion in self.criteria],
            "expected_utility": self.expected_utility,
            "uncertainty": self.uncertainty,
            "robust_utility": self.robust_utility,
            "ruled_out_by": list(self.ruled_out_by),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OptionAssessment:
        return cls(
            action=Action.from_dict(data["action"]),
            feasible=data.get("feasible"),
            criteria=tuple(
                CriterionAssessment.from_dict(criterion)
                for criterion in data.get("criteria", [])
            ),
            expected_utility=(
                float(data["expected_utility"])
                if data.get("expected_utility") is not None
                else None
            ),
            uncertainty=(
                float(data["uncertainty"])
                if data.get("uncertainty") is not None
                else None
            ),
            robust_utility=(
                float(data["robust_utility"])
                if data.get("robust_utility") is not None
                else None
            ),
            ruled_out_by=tuple(data.get("ruled_out_by", [])),
        )


@dataclass(frozen=True)
class InformationNeed:
    """The next observation with the greatest available decision value."""

    kind: str
    option_id: str
    field_id: str
    estimated_information_gain_bits: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "option_id": self.option_id,
            "field_id": self.field_id,
            "estimated_information_gain_bits": self.estimated_information_gain_bits,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> InformationNeed:
        return cls(
            kind=data["kind"],
            option_id=data["option_id"],
            field_id=data["field_id"],
            estimated_information_gain_bits=float(
                data["estimated_information_gain_bits"]
            ),
            reason=data["reason"],
        )


@dataclass(frozen=True)
class DecisionResult:
    """An auditable result tied to exactly one task-policy snapshot."""

    agent_id: str
    task_id: str
    policy_id: str
    policy_version: int
    status: DecisionStatus
    reason: str
    selection_basis: str
    selected_action: Action | None = None
    proposed_action: Action | None = None
    candidate_actions: tuple[Action, ...] = ()
    assessments: tuple[OptionAssessment, ...] = ()
    information_needs: tuple[InformationNeed, ...] = ()
    rejected_action_ids: tuple[str, ...] = ()
    authorization_basis: str | None = None
    action_probabilities: Mapping[str, float] = field(default_factory=dict)
    action_logprob: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "status": self.status.value,
            "reason": self.reason,
            "selection_basis": self.selection_basis,
            "selected_action": self.selected_action.to_dict() if self.selected_action else None,
            "proposed_action": self.proposed_action.to_dict() if self.proposed_action else None,
            "candidate_actions": [action.to_dict() for action in self.candidate_actions],
            "assessments": [assessment.to_dict() for assessment in self.assessments],
            "information_needs": [need.to_dict() for need in self.information_needs],
            "rejected_action_ids": list(self.rejected_action_ids),
            "authorization_basis": self.authorization_basis,
            "action_probabilities": dict(self.action_probabilities),
            "action_logprob": self.action_logprob,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DecisionResult:
        selected_action = data.get("selected_action")
        proposed_action = data.get("proposed_action")
        action_logprob = data.get("action_logprob")
        try:
            status = DecisionStatus(data["status"])
        except KeyError as exc:
            raise ValueError("decision status is required") from exc
        except ValueError as exc:
            raise ValueError(f"invalid decision status: {data.get('status')!r}") from exc
        return cls(
            agent_id=data["agent_id"],
            task_id=data["task_id"],
            policy_id=data["policy_id"],
            policy_version=int(data["policy_version"]),
            status=status,
            reason=data["reason"],
            selection_basis=data["selection_basis"],
            selected_action=(
                Action.from_dict(selected_action) if selected_action is not None else None
            ),
            proposed_action=(
                Action.from_dict(proposed_action) if proposed_action is not None else None
            ),
            candidate_actions=tuple(
                Action.from_dict(action) for action in data.get("candidate_actions", [])
            ),
            assessments=tuple(
                OptionAssessment.from_dict(assessment)
                for assessment in data.get("assessments", [])
            ),
            information_needs=tuple(
                InformationNeed.from_dict(need)
                for need in data.get("information_needs", [])
            ),
            rejected_action_ids=tuple(data.get("rejected_action_ids", [])),
            authorization_basis=data.get("authorization_basis"),
            action_probabilities={
                action_id: float(probability)
                for action_id, probability in data.get("action_probabilities", {}).items()
            },
            action_logprob=(
                float(action_logprob) if action_logprob is not None else None
            ),
        )


class DecisionResolver:
    """Resolve current evidence against an existing task policy."""

    _EPSILON = 1e-12

    def resolve(self, snapshot: PolicySnapshot, frame: DecisionFrame) -> DecisionResult:
        """Return a deterministic certificate, evidence request, or user tie-break."""
        self._validate(snapshot, frame)
        result_context = self._result_context(snapshot, "bayesian_decision")
        criteria_by_id = {criterion.id: criterion for criterion in frame.criteria}
        total_weight = sum(criterion.weight for criterion in frame.criteria)
        assessments = []
        information_needs = []

        for option in frame.options:
            failed_constraints = tuple(
                f"constraint:{constraint_id}"
                for constraint_id in frame.constraints
                if option.constraint_results.get(constraint_id) is False
            )
            missing_constraints = tuple(
                constraint_id
                for constraint_id in frame.constraints
                if constraint_id not in option.constraint_results
            )
            if failed_constraints:
                assessments.append(
                    OptionAssessment(
                        action=option.action,
                        feasible=False,
                        ruled_out_by=failed_constraints,
                    )
                )
                continue
            if missing_constraints:
                assessments.append(OptionAssessment(action=option.action, feasible=None))
                information_needs.extend(
                    InformationNeed(
                        kind="constraint",
                        option_id=option.action.id,
                        field_id=constraint_id,
                        estimated_information_gain_bits=1.0,
                        reason=f"Constraint {constraint_id!r} has not been evaluated.",
                    )
                    for constraint_id in missing_constraints
                )
                continue

            criterion_assessments = []
            missing_evidence = False
            for criterion in frame.criteria:
                points = tuple(
                    point for point in option.evidence if point.criterion_id == criterion.id
                )
                if len(points) < criterion.minimum_sources:
                    missing_evidence = True
                    information_needs.append(
                        InformationNeed(
                            kind="criterion",
                            option_id=option.action.id,
                            field_id=criterion.id,
                            estimated_information_gain_bits=criterion.weight / total_weight,
                            reason=(
                                f"Criterion {criterion.id!r} requires {criterion.minimum_sources} "
                                f"independent source(s); found {len(points)}."
                            ),
                        )
                    )
                    continue
                criterion_assessments.append(self._assess_criterion(criterion.id, points))

            if missing_evidence:
                assessments.append(
                    OptionAssessment(
                        action=option.action,
                        feasible=True,
                        criteria=tuple(criterion_assessments),
                    )
                )
                continue

            expected_utility = sum(
                criteria_by_id[item.criterion_id].weight * item.support
                for item in criterion_assessments
            ) / total_weight
            uncertainty = sum(
                criteria_by_id[item.criterion_id].weight * item.entropy_bits
                for item in criterion_assessments
            ) / total_weight
            assessments.append(
                OptionAssessment(
                    action=option.action,
                    feasible=True,
                    criteria=tuple(criterion_assessments),
                    expected_utility=expected_utility,
                    uncertainty=uncertainty,
                    robust_utility=expected_utility - frame.uncertainty_penalty * uncertainty,
                )
            )

        if information_needs:
            ordered_needs = tuple(
                sorted(
                    information_needs,
                    key=lambda need: (-need.estimated_information_gain_bits, need.option_id, need.field_id),
                )
            )
            return DecisionResult(
                **result_context,
                status=DecisionStatus.NEEDS_EVIDENCE,
                reason="The frame is missing evidence required to compare every viable option.",
                assessments=tuple(assessments),
                information_needs=ordered_needs,
            )

        viable = [assessment for assessment in assessments if assessment.feasible]
        if not viable:
            return DecisionResult(
                **result_context,
                status=DecisionStatus.NO_VIABLE_OPTION,
                reason="Every option violates at least one hard constraint.",
                assessments=tuple(assessments),
            )

        dominated_by = self._find_dominance(viable)
        if dominated_by:
            assessments = [
                replace(
                    assessment,
                    ruled_out_by=assessment.ruled_out_by
                    + ((f"pareto_dominated_by:{dominated_by[assessment.action.id]}",) if assessment.action.id in dominated_by else ()),
                )
                for assessment in assessments
            ]
            viable = [assessment for assessment in assessments if assessment.feasible and not assessment.ruled_out_by]

        viable.sort(key=lambda item: (-float(item.robust_utility), item.action.id))
        best = viable[0]
        close_candidates = tuple(
            assessment
            for assessment in viable
            if float(best.robust_utility) - float(assessment.robust_utility)
            <= frame.minimum_margin + self._EPSILON
        )

        uncertain = tuple(
            assessment
            for assessment in close_candidates
            if float(assessment.uncertainty) > frame.max_uncertainty
        )
        if uncertain:
            needs = tuple(self._uncertainty_need(assessment, criteria_by_id, total_weight) for assessment in uncertain)
            return DecisionResult(
                **result_context,
                status=DecisionStatus.NEEDS_EVIDENCE,
                reason="A leading option exceeds the allowed evidence uncertainty.",
                assessments=tuple(assessments),
                information_needs=tuple(
                    sorted(needs, key=lambda need: -need.estimated_information_gain_bits)
                ),
            )

        if len(close_candidates) == 1:
            if self._requires_human_approval(snapshot):
                return DecisionResult(
                    **result_context,
                    status=DecisionStatus.NEEDS_USER_FEEDBACK,
                    reason="The policy requires human approval; accept or reject the reasoned recommendation.",
                    proposed_action=best.action,
                    candidate_actions=(best.action,),
                    assessments=tuple(assessments),
                )
            return DecisionResult(
                **result_context,
                status=DecisionStatus.RESOLVED,
                reason="One option passed all constraints and exceeded the required utility margin.",
                selected_action=best.action,
                candidate_actions=(best.action,),
                assessments=tuple(assessments),
                authorization_basis="reasoned_evidence",
            )

        candidates = tuple(assessment.action for assessment in close_candidates)
        return DecisionResult(
            **result_context,
            status=DecisionStatus.NEEDS_USER_TIE_BREAK,
            reason="Multiple options remain within the required utility margin; accept or reject the proposed option.",
            proposed_action=candidates[0],
            candidate_actions=candidates,
            assessments=tuple(assessments),
        )

    def adjudicate(
        self,
        result: DecisionResult,
        disposition: TieBreakDisposition | str,
    ) -> DecisionResult:
        """Apply an explicit ``accept`` or ``reject`` to a pending tie."""
        if (
            result.status
            not in {
                DecisionStatus.NEEDS_USER_TIE_BREAK,
                DecisionStatus.NEEDS_USER_FEEDBACK,
            }
            or result.proposed_action is None
        ):
            raise ValueError("only a pending user decision can be adjudicated")
        try:
            resolved_disposition = TieBreakDisposition(disposition)
        except ValueError as exc:
            raise ValueError("tie-break disposition must be 'accept' or 'reject'") from exc

        if resolved_disposition is TieBreakDisposition.ACCEPT:
            return replace(
                result,
                status=DecisionStatus.RESOLVED,
                reason="The user accepted the proposed option to resolve the tie.",
                selected_action=result.proposed_action,
                proposed_action=None,
                authorization_basis="user_acceptance",
            )

        rejected_ids = result.rejected_action_ids + (result.proposed_action.id,)
        if result.status is DecisionStatus.NEEDS_USER_FEEDBACK:
            return replace(
                result,
                status=DecisionStatus.REJECTED,
                reason="The user rejected the recommendation; record the outcome before selecting again.",
                selected_action=None,
                proposed_action=None,
                rejected_action_ids=rejected_ids,
                authorization_basis="user_rejection",
            )
        remaining = tuple(
            action for action in result.candidate_actions if action.id not in rejected_ids
        )
        if not remaining:
            return replace(
                result,
                status=DecisionStatus.NO_VIABLE_OPTION,
                reason="The user rejected every tied option; the decision must be reframed.",
                proposed_action=None,
                candidate_actions=(),
                rejected_action_ids=rejected_ids,
            )
        return replace(
            result,
            reason="The user rejected the proposed option; accept or reject the next tied option.",
            proposed_action=remaining[0],
            candidate_actions=remaining,
            rejected_action_ids=rejected_ids,
        )

    def _validate(self, snapshot: PolicySnapshot, frame: DecisionFrame) -> None:
        if len(snapshot.actions) < 2:
            raise ValueError("task policy requires at least two actions")
        if not isinstance(frame.task, str):
            raise TypeError("decision task must be a string")
        if any(not isinstance(criterion, DecisionCriterion) for criterion in frame.criteria):
            raise TypeError("decision criteria must contain DecisionCriterion values")
        if any(not isinstance(option, DecisionOption) for option in frame.options):
            raise TypeError("decision options must contain DecisionOption values")
        policy_actions = {action.id: action for action in snapshot.actions}
        frame_actions = {option.action.id: option.action for option in frame.options}
        if set(policy_actions) != set(frame_actions) or any(
            policy_actions[action_id].to_dict() != frame_actions[action_id].to_dict()
            for action_id in set(policy_actions) & set(frame_actions)
        ):
            raise ValueError("decision frame action space must exactly match the task policy action space")
        if not frame.task.strip():
            raise ValueError("decision task must not be empty")
        if len(frame.options) < 2:
            raise ValueError("decision resolution requires at least two options")
        if not frame.criteria:
            raise ValueError("decision resolution requires at least one criterion")
        if not 0.0 <= frame.minimum_margin <= 1.0:
            raise ValueError("minimum_margin must be between 0 and 1")
        if frame.uncertainty_penalty < 0.0 or not math.isfinite(frame.uncertainty_penalty):
            raise ValueError("uncertainty_penalty must be finite and non-negative")
        if not 0.0 <= frame.max_uncertainty <= 1.0:
            raise ValueError("max_uncertainty must be between 0 and 1")

        criterion_ids = [criterion.id for criterion in frame.criteria]
        action_ids = [option.action.id for option in frame.options]
        constraint_ids = list(frame.constraints)
        if any(
            not isinstance(identifier, str)
            for identifier in criterion_ids + action_ids + constraint_ids
        ):
            raise TypeError("criterion, action, and constraint IDs must be strings")
        if any(not identifier.strip() for identifier in criterion_ids + action_ids + constraint_ids):
            raise ValueError("criterion, action, and constraint IDs must not be empty")
        if len(set(criterion_ids)) != len(criterion_ids):
            raise ValueError("criterion IDs must be unique")
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("action IDs must be unique")
        if len(set(constraint_ids)) != len(constraint_ids):
            raise ValueError("constraint IDs must be unique")

        for criterion in frame.criteria:
            if criterion.weight <= 0.0 or not math.isfinite(criterion.weight):
                raise ValueError("criterion weights must be finite and positive")
            if criterion.minimum_sources < 1:
                raise ValueError("criterion minimum_sources must be at least one")

        known_criteria = set(criterion_ids)
        known_constraints = set(constraint_ids)
        for option in frame.options:
            unknown_constraints = set(option.constraint_results) - known_constraints
            if unknown_constraints:
                raise ValueError(f"unknown constraints for action {option.action.id!r}: {sorted(unknown_constraints)}")
            if any(not isinstance(value, bool) for value in option.constraint_results.values()):
                raise ValueError("constraint results must be booleans")
            seen_sources = set()
            for point in option.evidence:
                if not isinstance(point, EvidencePoint):
                    raise TypeError("option evidence must contain EvidencePoint values")
                if not isinstance(point.criterion_id, str) or not isinstance(
                    point.source, str
                ):
                    raise TypeError("evidence criterion_id and source must be strings")
                if point.criterion_id not in known_criteria:
                    raise ValueError(f"unknown criterion {point.criterion_id!r}")
                if not point.source.strip():
                    raise ValueError("evidence source must not be empty")
                if not 0.0 <= point.support <= 1.0 or not math.isfinite(point.support):
                    raise ValueError("evidence support must be finite and between 0 and 1")
                if not 0.0 < point.confidence <= 1.0 or not math.isfinite(point.confidence):
                    raise ValueError("evidence confidence must be finite and in (0, 1]")
                source_key = (point.criterion_id, point.source)
                if source_key in seen_sources:
                    raise ValueError(
                        f"duplicate source {point.source!r} for criterion {point.criterion_id!r}"
                    )
                seen_sources.add(source_key)

    @staticmethod
    def _binary_entropy(probability: float) -> float:
        if probability <= 0.0 or probability >= 1.0:
            return 0.0
        return -probability * math.log2(probability) - (1.0 - probability) * math.log2(1.0 - probability)

    def _assess_criterion(
        self,
        criterion_id: str,
        points: Sequence[EvidencePoint],
    ) -> CriterionAssessment:
        confidence_total = sum(point.confidence for point in points)
        log_odds = 0.0
        for point in points:
            probability = min(1.0 - self._EPSILON, max(self._EPSILON, point.support))
            log_odds += point.confidence * math.log(probability / (1.0 - probability))
        support = (
            1.0 / (1.0 + math.exp(-log_odds))
            if log_odds >= 0.0
            else math.exp(log_odds) / (1.0 + math.exp(log_odds))
        )
        entropy = self._binary_entropy(support)
        mean_support = (
            sum(point.support * point.confidence for point in points) / confidence_total
        )
        source_entropy = sum(
            point.confidence * self._binary_entropy(point.support) for point in points
        ) / confidence_total
        return CriterionAssessment(
            criterion_id=criterion_id,
            support=support,
            entropy_bits=entropy,
            disagreement_bits=max(0.0, self._binary_entropy(mean_support) - source_entropy),
            source_count=len(points),
        )

    @staticmethod
    def _requires_human_approval(snapshot: PolicySnapshot) -> bool:
        profile = snapshot.metadata.get("complexity_profile")
        return bool(isinstance(profile, dict) and profile.get("requires_human_approval"))

    @staticmethod
    def _result_context(snapshot: PolicySnapshot, selection_basis: str) -> dict[str, object]:
        probabilities = SoftmaxPolicy.from_snapshot(
            PolicySnapshot.from_dict(snapshot.to_dict())
        ).probabilities()
        return {
            "agent_id": snapshot.agent_id,
            "task_id": snapshot.task_id,
            "policy_id": snapshot.id,
            "policy_version": snapshot.version,
            "selection_basis": selection_basis,
            "action_probabilities": {
                action.id: probability
                for action, probability in zip(snapshot.actions, probabilities)
            },
        }

    @staticmethod
    def _find_dominance(assessments: Sequence[OptionAssessment]) -> dict[str, str]:
        supports = {
            assessment.action.id: {
                criterion.criterion_id: criterion.support for criterion in assessment.criteria
            }
            for assessment in assessments
        }
        dominated_by = {}
        for candidate in assessments:
            candidate_support = supports[candidate.action.id]
            for challenger in assessments:
                if candidate.action.id == challenger.action.id:
                    continue
                challenger_support = supports[challenger.action.id]
                no_worse = all(
                    challenger_support[criterion_id] >= value
                    for criterion_id, value in candidate_support.items()
                )
                strictly_better = any(
                    challenger_support[criterion_id] > value
                    for criterion_id, value in candidate_support.items()
                )
                if no_worse and strictly_better:
                    dominated_by[candidate.action.id] = challenger.action.id
                    break
        return dominated_by

    @staticmethod
    def _uncertainty_need(
        assessment: OptionAssessment,
        criteria_by_id: Mapping[str, DecisionCriterion],
        total_weight: float,
    ) -> InformationNeed:
        criterion = max(
            assessment.criteria,
            key=lambda item: criteria_by_id[item.criterion_id].weight * item.entropy_bits,
        )
        information_gain = (
            criteria_by_id[criterion.criterion_id].weight * criterion.entropy_bits / total_weight
        )
        return InformationNeed(
            kind="uncertainty",
            option_id=assessment.action.id,
            field_id=criterion.criterion_id,
            estimated_information_gain_bits=information_gain,
            reason="Additional independent evidence here has the greatest entropy-reduction potential.",
        )


class TaskPolicy:
    """One agent-task decision policy with reasoned and learned selection routes."""

    def __init__(
        self,
        snapshot: PolicySnapshot,
        *,
        resolver: DecisionResolver | None = None,
        rng: random.Random | None = None,
    ) -> None:
        if len(snapshot.actions) < 2:
            raise ValueError("task policy requires at least two actions")
        try:
            self.authority = DecisionAuthority(
                snapshot.metadata.get("decision_authority", DecisionAuthority.LOW.value)
            )
        except ValueError as exc:
            raise ValueError("decision_authority must be 'low' or 'full'") from exc
        self._snapshot = PolicySnapshot.from_dict(snapshot.to_dict())
        self._resolver = resolver or DecisionResolver()
        self._rng = rng

    def snapshot(self) -> PolicySnapshot:
        """Return the unchanged durable policy snapshot used for decisions."""
        return PolicySnapshot.from_dict(self._snapshot.to_dict())

    def decide(
        self,
        frame: DecisionFrame | None = None,
        *,
        selected_action_id: str | None = None,
    ) -> DecisionResult:
        """Select through Bayesian resolution or the policy's learned logits."""
        if self.authority is DecisionAuthority.FULL:
            if selected_action_id is not None:
                raise ValueError("full decision authority cannot force a learned action")
            if frame is None:
                raise ValueError("full decision authority requires a decision frame")
            return self._resolver.resolve(self._snapshot, frame)
        if frame is not None:
            raise ValueError("low decision authority uses learned policy evidence, not a decision frame")

        policy = SoftmaxPolicy.from_snapshot(self.snapshot(), rng=self._rng)
        probabilities_list = policy.probabilities()
        if selected_action_id is None:
            decision = policy.choose()
            selected_action = decision.action
            action_logprob = decision.logprob
        else:
            selected_index = next(
                (
                    index
                    for index, action in enumerate(self._snapshot.actions)
                    if action.id == selected_action_id
                ),
                None,
            )
            if selected_index is None:
                raise ValueError("selected action is outside the task policy action space")
            selected_action = self._snapshot.actions[selected_index]
            action_logprob = math.log(max(probabilities_list[selected_index], 1e-12))
        probabilities = {
            action.id: probability
            for action, probability in zip(self._snapshot.actions, probabilities_list)
        }
        return DecisionResult(
            agent_id=self._snapshot.agent_id,
            task_id=self._snapshot.task_id,
            policy_id=self._snapshot.id,
            policy_version=self._snapshot.version,
            status=DecisionStatus.NEEDS_USER_FEEDBACK,
            reason=(
                "Low decision authority selected from learned policy evidence; "
                "accept, reject, or record an independently observable outcome."
            ),
            selection_basis="learned_policy",
            proposed_action=selected_action,
            candidate_actions=tuple(self._snapshot.actions),
            action_probabilities=probabilities,
            action_logprob=action_logprob,
        )

    def adjudicate(
        self,
        result: DecisionResult,
        disposition: TieBreakDisposition | str,
    ) -> DecisionResult:
        """Apply user feedback without changing the task-policy identity."""
        if (
            result.agent_id != self._snapshot.agent_id
            or result.task_id != self._snapshot.task_id
            or result.policy_id != self._snapshot.id
            or result.policy_version != self._snapshot.version
        ):
            raise ValueError("decision result does not belong to this task policy snapshot")
        policy_actions = {action.id: action.to_dict() for action in self._snapshot.actions}
        result_actions = [
            action
            for action in (
                result.selected_action,
                result.proposed_action,
                *result.candidate_actions,
                *(assessment.action for assessment in result.assessments),
            )
            if action is not None
        ]
        if any(
            action.id not in policy_actions
            or action.to_dict() != policy_actions[action.id]
            for action in result_actions
        ):
            raise ValueError("decision result contains an action outside this task policy")
        return self._resolver.adjudicate(result, disposition)


__all__ = [
    "CriterionAssessment",
    "DecisionAuthority",
    "DecisionCriterion",
    "DecisionFrame",
    "DecisionOption",
    "DecisionResolver",
    "DecisionResult",
    "DecisionStatus",
    "EvidencePoint",
    "InformationNeed",
    "OptionAssessment",
    "TaskPolicy",
    "TieBreakDisposition",
]