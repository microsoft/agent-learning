"""Abstract policy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence

from ..types import Action, PolicySnapshot


@dataclass
class Decision:
    """Output of :meth:`Policy.choose`: which action and with what probability."""

    action: Action
    logprob: float
    probabilities: List[float]  # Probability assigned to every action (same order as snapshot.actions)


class Policy(ABC):
    """Abstract base class for native RL policies.

    The Policy is stateless from the caller's perspective: the durable
    state lives in the :class:`PolicySnapshot` returned by
    :meth:`snapshot`. Callers persist the snapshot themselves; the
    Policy never talks to the store directly.
    """

    @abstractmethod
    def choose(self) -> Decision:
        """Sample an action according to the current parameters."""

    @abstractmethod
    def actions(self) -> Sequence[Action]:
        """Return the action space."""

    @abstractmethod
    def snapshot(self) -> PolicySnapshot:
        """Return a deep-copyable snapshot of the policy state."""

    @abstractmethod
    def apply_update(self, deltas: dict, *, baseline: float, episodes_seen: int) -> None:
        """Apply per-action logit deltas and update bookkeeping."""


__all__ = ["Decision", "Policy"]
