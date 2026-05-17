"""Multi-class router/classifier over a context vector.

A :class:`RouterClassifier` maps a fixed-dimensional context vector
(usually called ``phi``) to one of a known set of class ids, plus a
calibrated confidence in ``[0, 1]``. It is the deterministic, in-SDK
replacement for any LLM-based or heuristic routing layer in front of
a policy.

Two inference modes are supported:

- ``mode="logreg"`` (set by :meth:`fit`) trains a multinomial
  logistic regression on whatever class ids appear in the training
  rows. This mode cannot route to a class id absent from the
  training set, so it is brittle on classes that have never been
  seen during training.
- ``mode="prototype"`` (set by :meth:`fit_from_catalog`) stores a
  ``phi`` prototype per class id. At inference time it picks the
  class whose prototype is closest (cosine similarity) to the query
  ``phi``. This mode generalises to every class in the catalog,
  including ones whose training examples are zero, because the
  catalog itself provides the per-class representation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

from .base import (
    ClassifierResult,
    _dot,
    _softmax,
    fit_multinomial_logreg,
)


@dataclass
class RouterClassifier:
    """Multi-class router over a context vector.

    Training rows for :meth:`fit` are dicts with:

    - ``"phi"``: ``list[float]`` of length :attr:`feature_dim`.
    - ``"class_id"``: ground-truth class id (``str``).

    Catalog rows for :meth:`fit_from_catalog` are dicts with the same
    two keys; the catalog enumerates every class id the router is
    allowed to predict.

    Inference takes a ``phi`` vector and returns the predicted
    ``class_id`` with its softmax (or cosine-derived) probability. If
    the top probability is below :attr:`refusal_threshold` the
    classifier emits ``label="refused"`` with the complement
    probability as confidence.
    """

    feature_dim: int = 0
    refusal_threshold: float = 0.40
    epochs: int = 20
    learning_rate: float = 0.10
    weight_decay: float = 1e-4
    batch_size: int = 256
    seed: int = 42

    # Filled in by fit() or fit_from_catalog().
    classes: List[str] = field(default_factory=list)
    weights: List[List[float]] = field(default_factory=list)
    prototypes: List[List[float]] = field(default_factory=list)
    mode: str = "logreg"  # "logreg" or "prototype"
    version: int = 0

    def fit(self, training_rows: List[dict]) -> "RouterClassifier":
        if not training_rows:
            return self
        self._infer_feature_dim(training_rows)
        unique = sorted({row["class_id"] for row in training_rows})
        cls_to_idx = {cid: i for i, cid in enumerate(unique)}
        prepared: List[dict] = []
        for row in training_rows:
            phi = row["phi"]
            self._check_phi(phi)
            features = list(phi) + [1.0]  # append bias
            prepared.append({
                "features": features,
                "label": cls_to_idx[row["class_id"]],
            })
        weights = fit_multinomial_logreg(
            prepared,
            feature_dim=self.feature_dim,
            num_classes=len(unique),
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            seed=self.seed,
        )
        self.classes = unique
        self.weights = weights
        self.mode = "logreg"
        self.version += 1
        return self

    def fit_from_catalog(self, catalog_rows: List[dict]) -> "RouterClassifier":
        """Build a prototype-based router from catalog rows.

        Each catalog row must carry ``"class_id"`` and ``"phi"``.
        At inference time the router picks the class whose prototype
        is closest (cosine similarity) to the query ``phi``.
        """
        if not catalog_rows:
            return self
        self._infer_feature_dim(catalog_rows)
        classes: List[str] = []
        prototypes: List[List[float]] = []
        for row in catalog_rows:
            cid = str(row["class_id"])
            phi = list(row["phi"])
            self._check_phi(phi)
            classes.append(cid)
            prototypes.append(phi)
        self.classes = classes
        self.prototypes = prototypes
        self.weights = []
        self.mode = "prototype"
        self.version += 1
        return self

    def predict(self, features: Dict[str, object]) -> ClassifierResult:
        phi = features.get("phi")
        if phi is None:
            raise ValueError("RouterClassifier.predict requires features['phi']")
        return self._predict_from_phi(list(phi))  # type: ignore[arg-type]

    def predict_from_phi(self, phi: List[float]) -> ClassifierResult:
        """Convenience: predict directly from a context vector."""
        return self._predict_from_phi(phi)

    def _predict_from_phi(self, phi: List[float]) -> ClassifierResult:
        if not self.classes:
            return ClassifierResult(label="refused", confidence=0.0)
        if self.mode == "prototype":
            return self._predict_prototype(phi)
        if not self.weights:
            return ClassifierResult(label="refused", confidence=0.0)
        x = list(phi) + [1.0]
        logits = [_dot(self.weights[k], x) for k in range(len(self.classes))]
        probs = _softmax(logits)
        # argmax
        best_idx = 0
        best_p = probs[0]
        for i in range(1, len(probs)):
            if probs[i] > best_p:
                best_p = probs[i]
                best_idx = i
        if best_p < self.refusal_threshold:
            return ClassifierResult(
                label="refused",
                confidence=1.0 - best_p,
                features={"top_candidate": float(best_idx), "top_probability": best_p},
            )
        return ClassifierResult(
            label=self.classes[best_idx],
            confidence=best_p,
            features={"top_probability": best_p},
        )

    def _predict_prototype(self, phi: List[float]) -> ClassifierResult:
        """Cosine-similarity nearest-prototype routing."""
        x_norm = math.sqrt(sum(v * v for v in phi)) or 1.0
        sims: List[float] = []
        for proto in self.prototypes:
            p_norm = math.sqrt(sum(v * v for v in proto)) or 1.0
            sims.append(_dot(phi, proto) / (x_norm * p_norm))
        probs = _softmax(sims)
        best_idx = 0
        best_p = probs[0]
        for i in range(1, len(probs)):
            if probs[i] > best_p:
                best_p = probs[i]
                best_idx = i
        if best_p < self.refusal_threshold:
            return ClassifierResult(
                label="refused",
                confidence=1.0 - best_p,
                features={"top_candidate": float(best_idx), "top_probability": best_p},
            )
        return ClassifierResult(
            label=self.classes[best_idx],
            confidence=best_p,
            features={"top_probability": best_p, "top_similarity": sims[best_idx]},
        )

    def _infer_feature_dim(self, rows: List[dict]) -> None:
        if self.feature_dim > 0:
            return
        for row in rows:
            phi = row.get("phi")
            if phi is not None:
                self.feature_dim = len(phi)
                return

    def _check_phi(self, phi) -> None:
        if self.feature_dim and len(phi) != self.feature_dim:
            raise ValueError(
                f"phi length {len(phi)} != feature_dim {self.feature_dim}"
            )

    def to_snapshot(self) -> Dict[str, object]:
        """Serialise to a JSON-friendly snapshot for persistence."""
        return {
            "type": "router_snapshot",
            "version": self.version,
            "mode": self.mode,
            "feature_dim": self.feature_dim,
            "refusal_threshold": self.refusal_threshold,
            "classes": list(self.classes),
            "weights": [list(w) for w in self.weights],
            "prototypes": [list(p) for p in self.prototypes],
        }

    @classmethod
    def from_snapshot(cls, doc: Dict[str, object]) -> "RouterClassifier":
        inst = cls(
            feature_dim=int(doc.get("feature_dim", 0)),
            refusal_threshold=float(doc.get("refusal_threshold", 0.40)),
        )
        inst.classes = list(doc.get("classes", []))  # type: ignore[arg-type]
        inst.weights = [list(w) for w in doc.get("weights", [])]  # type: ignore[arg-type]
        inst.prototypes = [list(p) for p in doc.get("prototypes", [])]  # type: ignore[arg-type]
        inst.mode = str(doc.get("mode", "logreg"))
        inst.version = int(doc.get("version", 0))
        return inst


__all__ = ["RouterClassifier"]
