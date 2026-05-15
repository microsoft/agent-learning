"""Learner interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from ..policy.base import Policy
from ..types import Episode, Reward


@dataclass
class LearnerResult:
    """Summary returned by :meth:`Learner.update`."""

    episodes_used: int
    mean_reward: float
    baseline_before: float
    baseline_after: float
    logit_deltas: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


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
