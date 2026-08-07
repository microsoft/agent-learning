"""LLM-backed task-adherence scorer."""

from __future__ import annotations

from ...config import ScoreConfig
from ._base import _LlmScorerWrapper


class LlmAdherenceScorer(_LlmScorerWrapper):
    """Defers to ``azure.ai.evaluation.TaskAdherenceEvaluator``."""

    def __init__(self, cfg: ScoreConfig):
        super().__init__(
            cfg=cfg,
            name="adherence",
            label_name="adherence",
            evaluator_attr="TaskAdherenceEvaluator",
        )


__all__ = ["LlmAdherenceScorer"]
