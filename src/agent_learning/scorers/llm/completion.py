"""LLM-backed task-completion scorer."""

from __future__ import annotations

from ...config import ScoreConfig
from ._base import _LlmScorerWrapper


class LlmCompletionScorer(_LlmScorerWrapper):
    """Defers to ``azure.ai.evaluation.TaskCompletionEvaluator``."""

    def __init__(self, cfg: ScoreConfig):
        super().__init__(
            cfg=cfg,
            name="completion",
            label_name="completion",
            evaluator_attr="TaskCompletionEvaluator",
        )


__all__ = ["LlmCompletionScorer"]
