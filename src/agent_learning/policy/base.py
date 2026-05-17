"""Abstract policy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

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

    Concrete policies may use an optional ``state`` argument on
    :meth:`choose` to condition action selection on contextual features
    (a request embedding, a session summary, etc.). Marginal policies
    such as :class:`SoftmaxPolicy` ignore the argument.
    """

    @abstractmethod
    def choose(self, state: Optional[Any] = None) -> Decision:
        """Sample an action according to the current parameters.

        Args:
            state: Optional contextual feature vector. Contextual
                policies (e.g. :class:`ContextualSoftmaxPolicy`) require
                a 1-D array of length ``feature_dim``. Marginal policies
                ignore the argument.
        """

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
