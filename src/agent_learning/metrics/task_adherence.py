"""Task adherence metric wrapper.

The underlying judge emits a binary score ``0`` (material failure) or
``1`` (pass). It is already in ``[0, 1]`` so normalisation is a no-op.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..types import MetricName
from .base import MetricEvaluator, MetricRequest


class TaskAdherenceMetric(MetricEvaluator):
    """Wraps :class:`azure.ai.evaluation.TaskAdherenceEvaluator`."""

    NAME = MetricName.TASK_ADHERENCE

    def _build_evaluator(self) -> Any:
        from azure.ai.evaluation import TaskAdherenceEvaluator  # type: ignore

        return TaskAdherenceEvaluator(model_config=self._judge_config.to_model_config())

    def _build_kwargs(self, request: MetricRequest) -> Dict[str, Any]:
        return {
            "query": request.query,
            "response": request.response,
            "system_message": request.system_message or "",
            "tool_calls": request.tool_calls or "[]",
        }

    def _normalize(self, raw: Dict[str, Any]) -> Optional[float]:
        from .base import _extract_score  # type: ignore

        score = _extract_score(raw, self.NAME)
        if score is None:
            return None
        return max(0.0, min(1.0, score))


__all__ = ["TaskAdherenceMetric"]
