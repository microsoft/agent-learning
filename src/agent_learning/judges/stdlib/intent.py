"""Tier 1 stdlib intent-resolution judge.

Hashing-trick bag-of-words features over the concatenated
``query`` + ``response`` token stream, fed into a binary logistic
regression head fit with the SDK's pure-Python
:func:`fit_binary_logreg`. No external dependencies.

The judge has two states:

- **Unfitted** (initial). ``score()`` returns ``label="pass"`` with
  confidence 0.5. The unfitted judge is intentionally permissive so
  the SDK ships with a working reward shaper even before any
  training data is collected.
- **Fitted**. After :meth:`fit`, ``score()`` returns a calibrated
  probability that the response addresses the requester's intent.

Snapshot format is the same JSON shape used by the existing
:class:`agent_learning.classifiers.judges._base.BinaryJudge`, so
operators can persist and reload a fitted judge via :meth:`save`
and :meth:`load`.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional

from ...classifiers.base import _sigmoid, fit_binary_logreg
from ..base import JudgeScore
from ._text import featurize_query_response


@dataclass
class StdlibIntentJudge:
    """Stdlib bag-of-words intent judge."""

    name: str = "intent"
    feature_dim: int = 1024
    pass_threshold: float = 0.5
    weights: List[float] = field(default_factory=list)

    @classmethod
    def load_or_default(
        cls,
        snapshot_dir: Optional[str] = None,
        *,
        feature_dim: int = 1024,
        pass_threshold: float = 0.5,
    ) -> "StdlibIntentJudge":
        """Load weights from a snapshot dir, or return an unfitted instance.

        The snapshot file is ``{snapshot_dir}/intent.stdlib.json``.
        """
        judge = cls(
            feature_dim=feature_dim,
            pass_threshold=pass_threshold,
        )
        if snapshot_dir:
            path = os.path.join(
                snapshot_dir, f"{judge.name}.stdlib.json"
            )
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as fh:
                    judge.load(json.load(fh))
        return judge

    def fit(self, training_rows: List[dict]) -> "StdlibIntentJudge":
        """Fit the binary logreg head on training rows.

        Each row must carry::

            {"query": str, "response": str, "label": 0 | 1}

        Rows missing either ``query`` or ``response`` are skipped.
        """
        prepared: List[dict] = []
        for row in training_rows:
            query = row.get("query")
            response = row.get("response")
            label = row.get("label")
            if query is None or response is None or label is None:
                continue
            features = featurize_query_response(
                str(query), str(response), dim=self.feature_dim
            )
            features.append(1.0)  # bias column
            prepared.append(
                {"features": features, "label": int(label)}
            )
        if not prepared:
            self.weights = [0.0] * (self.feature_dim + 1)
            return self
        self.weights = fit_binary_logreg(
            prepared, feature_dim=self.feature_dim
        )
        return self

    def _probability(self, query: str, response: str) -> float:
        if not self.weights:
            # Unfitted: permissive default.
            return self.pass_threshold
        features = featurize_query_response(
            query, response, dim=self.feature_dim
        )
        features.append(1.0)
        logit = 0.0
        for w, x in zip(self.weights, features):
            logit += w * x
        return _sigmoid(logit)

    def score(
        self,
        *,
        query: Optional[str] = None,
        response: Optional[str] = None,
        **_: object,
    ) -> JudgeScore:
        """Score one (query, response) pair."""
        if query is None or response is None:
            raise ValueError(
                "StdlibIntentJudge requires query and response"
            )
        probability = self._probability(str(query), str(response))
        if probability >= self.pass_threshold:
            label = "pass"
            confidence = probability
        else:
            label = "fail"
            confidence = 1.0 - probability
        features = {
            "probability": probability,
            "fitted": bool(self.weights),
            "feature_dim": self.feature_dim,
        }
        return JudgeScore(
            label=label,
            confidence=confidence,
            normalized=probability,
            features=features,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_snapshot(self) -> dict:
        return {
            "name": self.name,
            "feature_dim": self.feature_dim,
            "pass_threshold": self.pass_threshold,
            "weights": list(self.weights),
        }

    def load(self, snapshot: dict) -> "StdlibIntentJudge":
        self.name = snapshot.get("name", self.name)
        self.feature_dim = int(
            snapshot.get("feature_dim", self.feature_dim)
        )
        self.pass_threshold = float(
            snapshot.get("pass_threshold", self.pass_threshold)
        )
        self.weights = [float(w) for w in snapshot.get("weights", [])]
        return self

    def save(self, snapshot_dir: str) -> str:
        os.makedirs(snapshot_dir, exist_ok=True)
        path = os.path.join(
            snapshot_dir, f"{self.name}.stdlib.json"
        )
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_snapshot(), fh)
        return path


# Silence a "imported but unused" lint by re-exporting math for any
# downstream debugging hooks that import StdlibIntentJudge.math.
_ = math

__all__ = ["StdlibIntentJudge"]
