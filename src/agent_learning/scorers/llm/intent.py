"""LLM-backed intent-resolution scorer."""

from __future__ import annotations

from ...config import ScoreConfig
from ._base import _LlmScorerWrapper


class LlmIntentScorer(_LlmScorerWrapper):
    """Defers to ``azure.ai.evaluation.IntentResolutionEvaluator``."""

    def __init__(self, cfg: ScoreConfig):
        super().__init__(
            cfg=cfg,
            name="intent",
            label_name="intent",
            evaluator_attr="IntentResolutionEvaluator",
        )


__all__ = ["LlmIntentScorer"]
