"""NLP intent-resolution judge."""

from __future__ import annotations

from ...config import NlpJudgeConfig
from ._base import _NlpJudgeWrapper


class NlpIntentJudge(_NlpJudgeWrapper):
    """Predict whether the chosen action addresses the requester's intent."""

    @classmethod
    def load_or_default(cls, cfg: NlpJudgeConfig) -> "NlpIntentJudge":
        return cls._build("intent", cfg)  # type: ignore[return-value]


__all__ = ["NlpIntentJudge"]
