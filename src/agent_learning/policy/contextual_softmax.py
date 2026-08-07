"""Contextual softmax bandit policy.

The policy stores a weight matrix ``W`` of shape ``(K, d)`` where ``K``
is the number of actions and ``d`` is the feature dimension. Action
logits at inference time are computed as ``logits = W @ phi`` for a
caller-supplied feature vector ``phi`` of length ``d``. A numerically
stable softmax then turns the logits into a probability distribution
the policy samples from.

The weight matrix lives in :attr:`PolicySnapshot.metadata`:

- ``metadata["feature_dim"]`` (int) - the dimension ``d``.
- ``metadata["context_weights"]`` (dict[action_id, list[float]]) -
  one length-``d`` row per action.

Using ``metadata`` keeps the persisted snapshot schema unchanged
(the existing ``logits`` map stays untouched and unused). Storage
and Cosmos round-trip continue to work with no migration.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..types import Action, PolicySnapshot
from .base import Decision, Policy


_METADATA_WEIGHTS_KEY = "context_weights"
_METADATA_DIM_KEY = "feature_dim"


class ContextualSoftmaxPolicy(Policy):
    """Linear contextual softmax over a fixed action set.

    Construction:
        policy = ContextualSoftmaxPolicy.from_actions(
            actions, feature_dim=25, agent_id="dq"
        )

    Continuation from a snapshot:
        policy = ContextualSoftmaxPolicy.from_snapshot(snapshot)
    """

    def __init__(
        self,
        snapshot: PolicySnapshot,
        *,
        rng: Optional[random.Random] = None,
        max_weight_abs: float = 10.0,
    ) -> None:
        if not snapshot.actions:
            raise ValueError("ContextualSoftmaxPolicy requires at least one action.")
        feature_dim = int(snapshot.metadata.get(_METADATA_DIM_KEY, 0))
        if feature_dim <= 0:
            raise ValueError(
                "ContextualSoftmaxPolicy requires snapshot.metadata['feature_dim'] > 0."
            )
        weights = snapshot.metadata.get(_METADATA_WEIGHTS_KEY)
        if not isinstance(weights, dict):
            weights = {}
        # Materialise a row of zeros for any action missing a weight row.
        for action in snapshot.actions:
            row = weights.get(action.id)
            if row is None or len(row) != feature_dim:
                weights[action.id] = [0.0] * feature_dim
            else:
                weights[action.id] = [float(x) for x in row]
        snapshot.metadata[_METADATA_WEIGHTS_KEY] = weights
        snapshot.metadata[_METADATA_DIM_KEY] = feature_dim

        self._snapshot = snapshot
        self._rng = rng or random.Random()
        self._max_weight_abs = float(max_weight_abs)
        self._feature_dim = feature_dim

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_actions(
        cls,
        actions: Sequence[Action],
        *,
        feature_dim: int,
        agent_id: str = "default",
        task_id: str = "default",
        rng: Optional[random.Random] = None,
        max_weight_abs: float = 10.0,
        initial_weights: Optional[Dict[str, Sequence[float]]] = None,
    ) -> "ContextualSoftmaxPolicy":
        if feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer.")
        weights: Dict[str, List[float]] = {
            a.id: [0.0] * feature_dim for a in actions
        }
        if initial_weights:
            for action_id, row in initial_weights.items():
                if action_id in weights:
                    row_list = [float(x) for x in row]
                    if len(row_list) != feature_dim:
                        raise ValueError(
                            f"initial_weights['{action_id}'] has length "
                            f"{len(row_list)}, expected {feature_dim}."
                        )
                    weights[action_id] = row_list
        snapshot = PolicySnapshot(
            agent_id=agent_id,
            task_id=task_id,
            version=0,
            actions=list(actions),
            logits={a.id: 0.0 for a in actions},  # kept for back-compat, unused
            metadata={
                _METADATA_DIM_KEY: int(feature_dim),
                _METADATA_WEIGHTS_KEY: weights,
                "policy_kind": "contextual_softmax",
            },
        )
        return cls(snapshot, rng=rng, max_weight_abs=max_weight_abs)

    @classmethod
    def from_snapshot(
        cls,
        snapshot: PolicySnapshot,
        *,
        rng: Optional[random.Random] = None,
        max_weight_abs: float = 10.0,
    ) -> "ContextualSoftmaxPolicy":
        return cls(snapshot, rng=rng, max_weight_abs=max_weight_abs)

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def actions(self) -> Sequence[Action]:
        return tuple(self._snapshot.actions)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def _weights_matrix(self) -> np.ndarray:
        rows = [
            self._snapshot.metadata[_METADATA_WEIGHTS_KEY][a.id]
            for a in self._snapshot.actions
        ]
        return np.asarray(rows, dtype=np.float64)

    def probabilities(self, state: Optional[Any] = None) -> List[float]:
        """Softmax over ``W @ phi`` in deterministic action order.

        If ``state`` is None the policy returns the uniform distribution
        over actions (equivalent to ``phi == 0``).
        """
        n_actions = len(self._snapshot.actions)
        if state is None:
            uniform = 1.0 / n_actions
            return [uniform] * n_actions
        phi = np.asarray(state, dtype=np.float64).reshape(-1)
        if phi.shape[0] != self._feature_dim:
            raise ValueError(
                f"State vector has length {phi.shape[0]}, expected "
                f"{self._feature_dim}."
            )
        logits = self._weights_matrix() @ phi
        logits -= logits.max()  # numerical stability
        exp = np.exp(logits)
        denom = float(exp.sum())
        if denom <= 0.0 or not np.isfinite(denom):  # pragma: no cover
            uniform = 1.0 / n_actions
            return [uniform] * n_actions
        return (exp / denom).tolist()

    def choose(self, state: Optional[Any] = None) -> Decision:
        probs = self.probabilities(state)
        idx = self._weighted_sample(probs)
        chosen = self._snapshot.actions[idx]
        prob = max(probs[idx], 1e-12)
        return Decision(action=chosen, logprob=math.log(prob), probabilities=probs)

    def snapshot(self) -> PolicySnapshot:
        return PolicySnapshot.from_dict(self._snapshot.to_dict())

    def apply_update(
        self,
        deltas: Dict[str, Any],
        *,
        baseline: float,
        episodes_seen: int,
    ) -> None:
        """Apply additive updates to the context weights, clip, and bump version.

        ``deltas`` accepts two shapes for backward compatibility with
        the marginal :class:`SoftmaxPolicy` API:

        - ``{action_id: float}`` - apply ``delta`` to every dimension of
          that action's weight row (rare; useful only as a smoke test).
        - ``{action_id: Sequence[float]}`` of length ``feature_dim`` -
          element-wise add to the weight row.
        """
        weights: Dict[str, List[float]] = self._snapshot.metadata[_METADATA_WEIGHTS_KEY]
        for action_id, delta in deltas.items():
            if action_id not in weights:
                continue
            current = np.asarray(weights[action_id], dtype=np.float64)
            if isinstance(delta, (int, float)):
                current = current + float(delta)
            else:
                delta_arr = np.asarray(delta, dtype=np.float64).reshape(-1)
                if delta_arr.shape[0] != self._feature_dim:
                    raise ValueError(
                        f"Delta for '{action_id}' has length "
                        f"{delta_arr.shape[0]}, expected {self._feature_dim}."
                    )
                current = current + delta_arr
            current = np.clip(current, -self._max_weight_abs, self._max_weight_abs)
            weights[action_id] = current.tolist()
        self._snapshot.metadata[_METADATA_WEIGHTS_KEY] = weights
        self._snapshot.baseline = float(baseline)
        self._snapshot.episodes_seen = int(episodes_seen)
        self._snapshot.updates_applied += 1
        self._snapshot.advance_version()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _weighted_sample(self, probs: Sequence[float]) -> int:
        roll = self._rng.random()
        cumulative = 0.0
        for idx, p in enumerate(probs):
            cumulative += p
            if roll <= cumulative:
                return idx
        return len(probs) - 1  # pragma: no cover - rounding tail


__all__ = ["ContextualSoftmaxPolicy"]
