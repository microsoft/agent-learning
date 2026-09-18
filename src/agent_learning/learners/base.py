"""Learner interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable

from ..policy.base import Policy
from ..types import ConsumedInput, Episode, Reward


@dataclass
class LearnerResult:
    """Summary returned by :meth:`Learner.update`.

    A consumption receipt is optional for custom learners. None means no
    receipt was supplied; an empty list means the learner consumed nothing.
    """

    episodes_used: int
    mean_reward: float
    baseline_before: float
    baseline_after: float
    logit_deltas: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
    consumed_inputs: list[ConsumedInput] | None = None


class Learner(ABC):
    """Abstract base class for an agent-learning algorithm."""

    @abstractmethod
    def update(
        self,
        policy: Policy,
        episodes: Iterable[Episode],
        rewards: Iterable[Reward],
    ) -> LearnerResult:
        """Apply one update to ``policy`` using the supplied data."""


__all__ = ["Learner", "LearnerResult"]
