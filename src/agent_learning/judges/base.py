"""Shared types for the judge backends.

Each backend produces the same :class:`JudgeScore` shape so the reward
shaper and learner stay backend-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Protocol, runtime_checkable


@dataclass(frozen=True)
class JudgeScore:
    """The contract every judge backend honors.

    Attributes:
        label: ``"pass"`` or ``"fail"``.
        confidence: probability of the predicted label in ``[0, 1]``.
        normalized: a continuous score in ``[0, 1]`` suitable for the
            reward shaper. Equals ``confidence`` when ``label == "pass"``
            and ``1.0 - confidence`` when ``label == "fail"``.
        features: optional debugging payload. Never consumed by the
            reward shaper or the learner.
    """

    label: str
    confidence: float
    normalized: float
    features: Dict[str, object] = field(default_factory=dict)


@runtime_checkable
class Judge(Protocol):
    """Structural type for both NLP and LLM judges."""

    name: str

    def score(self, **kwargs: object) -> JudgeScore:
        """Score one episode.

        NLP backends typically require ``phi`` and ``action_id``.
        LLM backends typically require ``request`` and ``response``.
        Unused keyword arguments are ignored by each backend so the
        same call site works regardless of mode.
        """


__all__ = ["Judge", "JudgeScore"]
