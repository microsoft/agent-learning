"""LLM-backed task-adherence judge."""

from __future__ import annotations

from ...config import JudgeConfig
from ._base import _LlmJudgeWrapper


class LlmAdherenceJudge(_LlmJudgeWrapper):
    """Defers to ``azure.ai.evaluation.TaskAdherenceEvaluator``."""

    def __init__(self, cfg: JudgeConfig):
        super().__init__(
            cfg=cfg,
            name="adherence",
            label_name="adherence",
            evaluator_attr="TaskAdherenceEvaluator",
        )


__all__ = ["LlmAdherenceJudge"]
