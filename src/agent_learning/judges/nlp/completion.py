"""NLP task-completion judge."""

from __future__ import annotations

from ...config import NlpJudgeConfig
from ._base import _NlpJudgeWrapper


class NlpCompletionJudge(_NlpJudgeWrapper):
    """Predict whether the response completes the requested task."""

    @classmethod
    def load_or_default(cls, cfg: NlpJudgeConfig) -> "NlpCompletionJudge":
        return cls._build("completion", cfg)  # type: ignore[return-value]


__all__ = ["NlpCompletionJudge"]
