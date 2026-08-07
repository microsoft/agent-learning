"""NLP task-adherence scorer."""

from __future__ import annotations

from ...config import NlpScoreConfig
from ._base import _NlpScorerWrapper


class NlpAdherenceScorer(_NlpScorerWrapper):
    """Predict whether the response adheres to the requested task contract."""

    @classmethod
    def load_or_default(cls, cfg: NlpScoreConfig) -> "NlpAdherenceScorer":
        return cls._build("adherence", cfg)  # type: ignore[return-value]


__all__ = ["NlpAdherenceScorer"]
