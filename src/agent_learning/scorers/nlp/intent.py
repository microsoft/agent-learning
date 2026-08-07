"""NLP intent-resolution scorer."""

from __future__ import annotations

from ...config import NlpScoreConfig
from ._base import _NlpScorerWrapper


class NlpIntentScorer(_NlpScorerWrapper):
    """Predict whether the chosen action addresses the requester's intent."""

    @classmethod
    def load_or_default(cls, cfg: NlpScoreConfig) -> "NlpIntentScorer":
        return cls._build("intent", cfg)  # type: ignore[return-value]


__all__ = ["NlpIntentScorer"]
