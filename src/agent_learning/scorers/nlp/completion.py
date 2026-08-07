"""NLP task-completion scorer."""

from __future__ import annotations

from ...config import NlpScoreConfig
from ._base import _NlpScorerWrapper


class NlpCompletionScorer(_NlpScorerWrapper):
    """Predict whether the response completes the requested task."""

    @classmethod
    def load_or_default(cls, cfg: NlpScoreConfig) -> "NlpCompletionScorer":
        return cls._build("completion", cfg)  # type: ignore[return-value]


__all__ = ["NlpCompletionScorer"]
