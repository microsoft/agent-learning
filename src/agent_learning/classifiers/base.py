"""Base types shared by the router and the three scorers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Protocol


@dataclass(frozen=True)
class ClassifierResult:
    """Output of any classifier in this package.

    ``label`` is the predicted class id (a ``str`` for the router,
    ``"pass"`` / ``"fail"`` for the scorers). ``confidence`` is the
    model's calibrated probability for that label, in ``[0, 1]``.
    ``features`` exposes the per-feature contribution so callers can
    persist a per-decision explainability trace.
    """

    label: str
    confidence: float
    features: Dict[str, float] = field(default_factory=dict)


class Classifier(Protocol):
    """Common surface across the router and the three scorers.

    Implementations are deterministic — given the same fit input
    they produce the same weights, and given the same predict
    input they produce the same :class:`ClassifierResult`.
    """

    def fit(self, training_rows: List[dict]) -> "Classifier":
        ...

    def predict(self, features: Dict[str, float]) -> ClassifierResult:
        ...


# ---------------------------------------------------------------------------
# Pure-Python logistic regression primitives shared by the four classifiers
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    """Numerically stable logistic function."""
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _softmax(logits: List[float]) -> List[float]:
    """Numerically stable softmax over ``logits``."""
    if not logits:
        return []
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    if s == 0.0:
        n = len(exps)
        return [1.0 / n] * n
    return [v / s for v in exps]


def _dot(weights: List[float], features: List[float]) -> float:
    """Inner product of two equal-length vectors."""
    total = 0.0
    for w, x in zip(weights, features):
        total += w * x
    return total


def fit_binary_logreg(
    rows: List[dict],
    feature_dim: int,
    *,
    epochs: int = 20,
    learning_rate: float = 0.10,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    seed: int = 42,
) -> List[float]:
    """Fit a binary logistic regression with closed-form gradient.

    Each row in ``rows`` must carry:

    - ``"features"``: list[float] of length ``feature_dim + 1`` (the
      last entry is the bias term — caller appends 1.0).
    - ``"label"``: ``0`` or ``1``.

    Returns a weight vector of length ``feature_dim + 1``. Uses
    standard mini-batch gradient descent with an L2 penalty; pure
    Python, no numpy.
    """
    import random as _random

    rng = _random.Random(seed)
    n = feature_dim + 1  # bias column appended by caller
    weights = [0.0] * n
    indices = list(range(len(rows)))
    if not indices:
        return weights

    for _epoch in range(epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            grad = [0.0] * n
            for i in batch:
                row = rows[i]
                x = row["features"]
                y = float(row["label"])
                z = _dot(weights, x)
                p = _sigmoid(z)
                err = p - y
                for j in range(n):
                    grad[j] += err * x[j]
            inv_b = 1.0 / max(len(batch), 1)
            for j in range(n):
                grad[j] = grad[j] * inv_b + weight_decay * weights[j]
                weights[j] -= learning_rate * grad[j]
    return weights


def fit_multinomial_logreg(
    rows: List[dict],
    feature_dim: int,
    num_classes: int,
    *,
    epochs: int = 20,
    learning_rate: float = 0.10,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    seed: int = 42,
) -> List[List[float]]:
    """Fit a multinomial logistic regression (softmax classifier).

    Each row in ``rows`` must carry:

    - ``"features"``: list[float] of length ``feature_dim + 1``.
    - ``"label"``: integer in ``[0, num_classes)``.

    Returns a ``num_classes x (feature_dim + 1)`` weight matrix.
    """
    import random as _random

    rng = _random.Random(seed)
    n = feature_dim + 1
    weights: List[List[float]] = [[0.0] * n for _ in range(num_classes)]
    indices = list(range(len(rows)))
    if not indices:
        return weights

    for _epoch in range(epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            grads: List[List[float]] = [[0.0] * n for _ in range(num_classes)]
            for i in batch:
                row = rows[i]
                x = row["features"]
                y = int(row["label"])
                logits = [_dot(weights[k], x) for k in range(num_classes)]
                probs = _softmax(logits)
                for k in range(num_classes):
                    err = probs[k] - (1.0 if k == y else 0.0)
                    for j in range(n):
                        grads[k][j] += err * x[j]
            inv_b = 1.0 / max(len(batch), 1)
            for k in range(num_classes):
                for j in range(n):
                    grads[k][j] = grads[k][j] * inv_b + weight_decay * weights[k][j]
                    weights[k][j] -= learning_rate * grads[k][j]
    return weights


__all__ = [
    "Classifier",
    "ClassifierResult",
    "fit_binary_logreg",
    "fit_multinomial_logreg",
]
