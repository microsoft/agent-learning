"""LLM-backed intent-resolution judge."""

from __future__ import annotations

from ...config import JudgeConfig
from ._base import _LlmJudgeWrapper


class LlmIntentJudge(_LlmJudgeWrapper):
    """Defers to ``azure.ai.evaluation.IntentResolutionEvaluator``."""

    def __init__(self, cfg: JudgeConfig):
        super().__init__(
            cfg=cfg,
            name="intent",
            label_name="intent",
            evaluator_attr="IntentResolutionEvaluator",
        )


__all__ = ["LlmIntentJudge"]
