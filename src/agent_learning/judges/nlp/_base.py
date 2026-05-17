"""Shared wrapper that adapts a :class:`BinaryJudge` to the SDK
:class:`JudgeScore` contract.

The wrapper reads a snapshot from ``cfg.snapshot_dir/{name}.json`` at
load time. When no snapshot is present the judge stays in its unfitted
"always-fail with zero confidence" state, which keeps the reward
shaper safe to call but encourages operators to train the judges.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional

from ...classifiers.judges._base import BinaryJudge
from ...config import NlpJudgeConfig
from ..base import JudgeScore


@dataclass
class _NlpJudgeWrapper:
    """Common base for the three concrete NLP judges."""

    name: str
    label_name: str
    _impl: BinaryJudge
    _pass_threshold: float = 0.5

    @classmethod
    def _build(cls, name: str, cfg: NlpJudgeConfig) -> "_NlpJudgeWrapper":
        impl = BinaryJudge(label_name=name)
        snapshot_path = os.path.join(cfg.snapshot_dir, f"{name}.json")
        if os.path.isfile(snapshot_path):
            with open(snapshot_path, "r", encoding="utf-8") as fh:
                impl = BinaryJudge.from_snapshot(json.load(fh))
                impl.label_name = name
        return cls(
            name=name,
            label_name=name,
            _impl=impl,
            _pass_threshold=float(cfg.pass_threshold),
        )

    def fit(self, training_rows: List[dict]) -> "_NlpJudgeWrapper":
        """Fit the underlying logistic-regression head.

        Training rows must look like ``{"phi": [...], "action_id": str, "label": 0|1}``.
        """
        self._impl.fit(training_rows)
        return self

    def save(self, snapshot_dir: str) -> str:
        """Persist the fitted weights to ``{snapshot_dir}/{name}.json``."""
        os.makedirs(snapshot_dir, exist_ok=True)
        path = os.path.join(snapshot_dir, f"{self.name}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self._impl.to_snapshot(), fh)
        return path

    def score(
        self,
        *,
        phi: Optional[List[float]] = None,
        action_id: Optional[str] = None,
        **_: object,
    ) -> JudgeScore:
        """Score one episode given a context vector and chosen action.

        Extra keyword arguments are ignored so the same call site works
        when the LLM backend is swapped in.
        """
        if phi is None or action_id is None:
            raise ValueError(
                f"{self.name} NLP judge requires phi and action_id"
            )
        result = self._impl.score(phi=list(phi), action_id=str(action_id))
        probability = float(result.features.get("probability", result.confidence))
        label = "pass" if probability >= self._pass_threshold else "fail"
        if label == "pass":
            confidence = probability
            normalized = probability
        else:
            confidence = 1.0 - probability
            normalized = probability  # keep the raw "pass probability" for shaping
        features = {
            "probability": probability,
            "fitted": bool(self._impl.weights),
        }
        return JudgeScore(
            label=label,
            confidence=confidence,
            normalized=normalized,
            features=features,
        )


__all__ = ["_NlpJudgeWrapper"]
