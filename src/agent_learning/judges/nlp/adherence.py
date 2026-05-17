"""NLP task-adherence judge."""

from __future__ import annotations

from ...config import NlpJudgeConfig
from ._base import _NlpJudgeWrapper


class NlpAdherenceJudge(_NlpJudgeWrapper):
    """Predict whether the response adheres to the requested task contract."""

    @classmethod
    def load_or_default(cls, cfg: NlpJudgeConfig) -> "NlpAdherenceJudge":
        return cls._build("adherence", cfg)  # type: ignore[return-value]


__all__ = ["NlpAdherenceJudge"]
