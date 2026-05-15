"""Intent resolution metric wrapper.

Score range from the underlying judge is ``1`` (very poor) to ``5``
(excellent). We normalise to ``[0, 1]`` via ``(score - 1) / 4`` so
that a perfect intent resolution becomes a maximum positive reward
signal after shaping.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..types import MetricName
from .base import MetricEvaluator, MetricRequest


class IntentResolutionMetric(MetricEvaluator):
    """Wraps :class:`azure.ai.evaluation.IntentResolutionEvaluator`."""

    NAME = MetricName.INTENT_RESOLUTION

    def _build_evaluator(self) -> Any:
        from azure.ai.evaluation import IntentResolutionEvaluator  # type: ignore

        return IntentResolutionEvaluator(
            model_config=self._judge_config.to_model_config(),
            threshold=self._judge_config.threshold,
        )

    def _build_kwargs(self, request: MetricRequest) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "query": request.query,
            "response": request.response,
        }
        if request.tool_definitions is not None:
            kwargs["tool_definitions"] = request.tool_definitions
        return kwargs

    def _normalize(self, raw: Dict[str, Any]) -> Optional[float]:
        # Judge produces an integer 1..5
        from .base import _extract_score  # type: ignore

        score = _extract_score(raw, self.NAME)
        if score is None:
            return None
        clamped = max(1.0, min(5.0, score))
        return (clamped - 1.0) / 4.0


__all__ = ["IntentResolutionMetric"]
