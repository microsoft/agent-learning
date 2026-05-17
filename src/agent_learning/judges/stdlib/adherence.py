"""Tier 1 stdlib task-adherence judge.

Deterministic rule engine over a per-task ``contract`` dict. No
training is required; the judge fires synchronously off the response
text and the contract's structural constraints.

Supported contract keys (all optional):

- ``required_substrings`` (``list[str]``): every entry must appear in
  the response. Matched case-insensitively.
- ``forbidden_substrings`` (``list[str]``): no entry may appear in
  the response. Matched case-insensitively.
- ``length_min`` (``int``): minimum response length in characters
  after trimming.
- ``length_max`` (``int``): maximum response length in characters
  after trimming.
- ``json_required`` (``bool``): when true, the response (after
  trimming) must be parseable as a single JSON value.

The score is the fraction of contract clauses satisfied, in
``[0, 1]``. With no constraints in the contract the judge returns
``1.0`` ("pass with full confidence"); this is the same permissive
default the other Tier 1 judges use when unfitted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from ..base import JudgeScore


@dataclass
class StdlibAdherenceJudge:
    """Stdlib rule-engine adherence judge."""

    name: str = "adherence"
    pass_threshold: float = 0.5

    @classmethod
    def load_or_default(
        cls,
        snapshot_dir: Optional[str] = None,
        *,
        pass_threshold: float = 0.5,
    ) -> "StdlibAdherenceJudge":
        """Match the load_or_default contract of the other judges.

        The rule engine carries no fitted state, so ``snapshot_dir``
        is accepted for symmetry but not consulted.
        """
        _ = snapshot_dir
        return cls(pass_threshold=pass_threshold)

    def score(
        self,
        *,
        response: Optional[str] = None,
        contract: Optional[dict] = None,
        **_: object,
    ) -> JudgeScore:
        """Score one response against the supplied contract."""
        if response is None:
            raise ValueError(
                "StdlibAdherenceJudge requires response"
            )
        contract = contract or {}
        clauses, violations = _evaluate_contract(
            str(response), contract
        )
        total = len(clauses)
        if total == 0:
            probability = 1.0
        else:
            satisfied = total - len(violations)
            probability = satisfied / total
        if probability >= self.pass_threshold:
            label = "pass"
            confidence = probability
        else:
            label = "fail"
            confidence = 1.0 - probability
        features = {
            "probability": probability,
            "clauses_total": total,
            "violations": list(violations),
        }
        return JudgeScore(
            label=label,
            confidence=confidence,
            normalized=probability,
            features=features,
        )


def _evaluate_contract(
    response: str, contract: dict
) -> Tuple[List[str], List[str]]:
    """Return ``(clauses_evaluated, violations)``."""
    clauses: List[str] = []
    violations: List[str] = []

    required = contract.get("required_substrings")
    if isinstance(required, Sequence) and not isinstance(
        required, (str, bytes)
    ):
        haystack = response.lower()
        for needle in required:
            label = f"required:{needle}"
            clauses.append(label)
            if str(needle).lower() not in haystack:
                violations.append(label)

    forbidden = contract.get("forbidden_substrings")
    if isinstance(forbidden, Sequence) and not isinstance(
        forbidden, (str, bytes)
    ):
        haystack = response.lower()
        for needle in forbidden:
            label = f"forbidden:{needle}"
            clauses.append(label)
            if str(needle).lower() in haystack:
                violations.append(label)

    trimmed = response.strip()

    length_min = contract.get("length_min")
    if isinstance(length_min, int) and length_min > 0:
        label = f"length_min:{length_min}"
        clauses.append(label)
        if len(trimmed) < length_min:
            violations.append(label)

    length_max = contract.get("length_max")
    if isinstance(length_max, int) and length_max > 0:
        label = f"length_max:{length_max}"
        clauses.append(label)
        if len(trimmed) > length_max:
            violations.append(label)

    if bool(contract.get("json_required")):
        label = "json_required"
        clauses.append(label)
        try:
            json.loads(trimmed)
        except (ValueError, TypeError):
            violations.append(label)

    return clauses, violations


# Silence "imported but unused" warnings on Iterable.
_ = Iterable

__all__ = ["StdlibAdherenceJudge"]
