"""LLM-backed task-completion judge."""

from __future__ import annotations

from ...config import JudgeConfig
from ._base import _LlmJudgeWrapper


class LlmCompletionJudge(_LlmJudgeWrapper):
    """Defers to ``azure.ai.evaluation.TaskCompletionEvaluator``."""

    def __init__(self, cfg: JudgeConfig):
        super().__init__(
            cfg=cfg,
            name="completion",
            label_name="completion",
            evaluator_attr="TaskCompletionEvaluator",
        )


__all__ = ["LlmCompletionJudge"]
