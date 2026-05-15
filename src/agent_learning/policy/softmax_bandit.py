"""Softmax bandit policy over a fixed set of discrete actions.

The policy stores one logit per action. Action probabilities are
computed via a numerically stable softmax. The action space is
fixed at construction time; the learner mutates only the logits and
the EMA baseline.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..types import Action, PolicySnapshot
from .base import Decision, Policy


class SoftmaxPolicy(Policy):
    """Discrete softmax policy with one logit per action.

    Construction:
        policy = SoftmaxPolicy.from_actions(actions, agent_id="dq")

    Continuation from a snapshot (e.g. after process restart):
        policy = SoftmaxPolicy.from_snapshot(snapshot)
    """

    def __init__(
        self,
        snapshot: PolicySnapshot,
        *,
        rng: Optional[random.Random] = None,
        max_logit_abs: float = 10.0,
    ) -> None:
        if not snapshot.actions:
            raise ValueError("SoftmaxPolicy requires at least one action.")
        self._snapshot = snapshot
        self._rng = rng or random.Random()
        self._max_logit_abs = max_logit_abs
        # Make sure every action has a logit
        for action in snapshot.actions:
            snapshot.logits.setdefault(action.id, 0.0)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_actions(
        cls,
        actions: Sequence[Action],
        *,
        agent_id: str = "default",
        rng: Optional[random.Random] = None,
        max_logit_abs: float = 10.0,
        initial_logits: Optional[Dict[str, float]] = None,
    ) -> "SoftmaxPolicy":
        logits = {a.id: 0.0 for a in actions}
        if initial_logits:
            for action_id, value in initial_logits.items():
                if action_id in logits:
                    logits[action_id] = float(value)
        snapshot = PolicySnapshot(
            agent_id=agent_id,
            version=0,
            actions=list(actions),
            logits=logits,
        )
        return cls(snapshot, rng=rng, max_logit_abs=max_logit_abs)

    @classmethod
    def from_snapshot(
        cls,
        snapshot: PolicySnapshot,
        *,
        rng: Optional[random.Random] = None,
        max_logit_abs: float = 10.0,
    ) -> "SoftmaxPolicy":
        return cls(snapshot, rng=rng, max_logit_abs=max_logit_abs)

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def actions(self) -> Sequence[Action]:
        return tuple(self._snapshot.actions)

    def probabilities(self) -> List[float]:
        """Softmax over the action logits in deterministic order."""
        logits = np.array(
            [self._snapshot.logits[a.id] for a in self._snapshot.actions],
            dtype=np.float64,
        )
        logits -= logits.max()  # numerical stability
        exp = np.exp(logits)
        denom = exp.sum()
        if denom <= 0.0 or not np.isfinite(denom):  # pragma: no cover
            uniform = 1.0 / len(self._snapshot.actions)
            return [uniform] * len(self._snapshot.actions)
        probs = exp / denom
        return probs.tolist()

    def choose(self) -> Decision:
        probs = self.probabilities()
        idx = self._weighted_sample(probs)
        chosen = self._snapshot.actions[idx]
        # log probability is clamped to avoid -inf for tiny probabilities
        prob = max(probs[idx], 1e-12)
        return Decision(action=chosen, logprob=math.log(prob), probabilities=probs)

    def snapshot(self) -> PolicySnapshot:
        return PolicySnapshot.from_dict(self._snapshot.to_dict())

    def apply_update(
        self,
        deltas: Dict[str, float],
        *,
        baseline: float,
        episodes_seen: int,
    ) -> None:
        """Apply per-action logit deltas, clip, and bump version."""
        for action_id, delta in deltas.items():
            if action_id not in self._snapshot.logits:
                continue
            new_value = self._snapshot.logits[action_id] + float(delta)
            new_value = max(-self._max_logit_abs, min(self._max_logit_abs, new_value))
            self._snapshot.logits[action_id] = new_value
        self._snapshot.baseline = float(baseline)
        self._snapshot.episodes_seen = int(episodes_seen)
        self._snapshot.updates_applied += 1
        self._snapshot.version += 1

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _weighted_sample(self, probs: Sequence[float]) -> int:
        """Sample an index from ``probs`` using the policy's RNG."""
        roll = self._rng.random()
        cumulative = 0.0
        for idx, p in enumerate(probs):
            cumulative += p
            if roll <= cumulative:
                return idx
        return len(probs) - 1  # pragma: no cover - rounding tail


__all__ = ["SoftmaxPolicy"]
