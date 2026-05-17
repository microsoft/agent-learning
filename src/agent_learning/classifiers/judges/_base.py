"""Shared base for binary judges.

All judges in this package are binary classifiers over the same
feature shape:

    [phi (feature_dim)] ++ [action one-hot (len(actions))] ++ [bias (1)]

Per-judge subclasses override :attr:`label_name` for the snapshot
payload. They are otherwise identical — the difference between an
intent, adherence, or completion judge lives entirely in the binary
labels used to train it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ..base import (
    ClassifierResult,
    _dot,
    _sigmoid,
    fit_binary_logreg,
)


@dataclass
class BinaryJudge:
    """Common base for the binary judges in this package.

    Both :attr:`feature_dim` and :attr:`actions` are inferred from
    the first training row when :meth:`fit` is called, so callers
    only need to set them explicitly if they want to build a judge
    without ever calling ``fit`` (e.g. when restoring from a
    snapshot).
    """

    label_name: str = "judge"
    feature_dim: int = 0
    actions: List[str] = field(default_factory=list)
    epochs: int = 20
    learning_rate: float = 0.10
    weight_decay: float = 1e-4
    batch_size: int = 256
    seed: int = 42

    # Filled by fit()
    weights: List[float] = field(default_factory=list)
    version: int = 0

    @property
    def vector_dim(self) -> int:
        """Length of the feature vector *excluding* the bias entry."""
        return self.feature_dim + len(self.actions)

    def _build_features(self, phi: List[float], action_id: str) -> List[float]:
        if len(phi) != self.feature_dim:
            raise ValueError(
                f"phi has length {len(phi)}, expected {self.feature_dim}"
            )
        action_onehot = [0.0] * len(self.actions)
        if action_id in self.actions:
            action_onehot[self.actions.index(action_id)] = 1.0
        return list(phi) + action_onehot + [1.0]  # bias

    def fit(self, training_rows: List[dict]) -> "BinaryJudge":
        if not training_rows:
            return self
        self._infer_dims(training_rows)
        prepared: List[dict] = []
        for row in training_rows:
            feats = self._build_features(row["phi"], row["action_id"])
            prepared.append({"features": feats, "label": int(row["label"])})
        weights = fit_binary_logreg(
            prepared,
            feature_dim=self.vector_dim,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            seed=self.seed,
        )
        self.weights = weights
        self.version += 1
        return self

    def _infer_dims(self, rows: List[dict]) -> None:
        if self.feature_dim == 0:
            for row in rows:
                phi = row.get("phi")
                if phi is not None:
                    self.feature_dim = len(phi)
                    break
        if not self.actions:
            seen: List[str] = []
            for row in rows:
                aid = row.get("action_id")
                if aid is not None and aid not in seen:
                    seen.append(aid)
            self.actions = sorted(seen)

    def predict(self, features: Dict[str, object]) -> ClassifierResult:
        phi = features.get("phi")
        action_id = features.get("action_id")
        if phi is None or action_id is None:
            raise ValueError(
                "judge predict requires features['phi'] and features['action_id']"
            )
        return self._predict(list(phi), str(action_id))  # type: ignore[arg-type]

    def score(self, *, phi: List[float], action_id: str) -> ClassifierResult:
        """Convenience wrapper matching a typical LLM judge call shape."""
        return self._predict(list(phi), action_id)

    def _predict(self, phi: List[float], action_id: str) -> ClassifierResult:
        if not self.weights:
            return ClassifierResult(label="fail", confidence=0.0)
        x = self._build_features(phi, action_id)
        p = _sigmoid(_dot(self.weights, x))
        if p >= 0.5:
            return ClassifierResult(
                label="pass",
                confidence=p,
                features={"probability": p, "action_id": _hash_str(action_id)},
            )
        return ClassifierResult(
            label="fail",
            confidence=1.0 - p,
            features={"probability": p, "action_id": _hash_str(action_id)},
        )

    def to_snapshot(self) -> Dict[str, object]:
        return {
            "type": "judge_snapshot",
            "label_name": self.label_name,
            "version": self.version,
            "feature_dim": self.feature_dim,
            "actions": list(self.actions),
            "weights": list(self.weights),
        }

    @classmethod
    def from_snapshot(cls, doc: Dict[str, object]) -> "BinaryJudge":
        inst = cls(
            label_name=str(doc.get("label_name", "judge")),
            feature_dim=int(doc.get("feature_dim", 0)),
            actions=list(doc.get("actions", [])),  # type: ignore[arg-type]
        )
        inst.weights = list(doc.get("weights", []))  # type: ignore[arg-type]
        inst.version = int(doc.get("version", 0))
        return inst


def _hash_str(s: str) -> float:
    """Stable deterministic hash for trace/explainability output."""
    h = 0
    for ch in s:
        h = (h * 31 + ord(ch)) & 0xFFFF
    return float(h)


__all__ = ["BinaryJudge"]
