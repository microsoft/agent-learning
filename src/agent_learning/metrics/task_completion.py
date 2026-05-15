"""Task completion metric wrapper.

The underlying judge emits ``1`` (task completed) or ``0`` (task
failed or incomplete). It is already in ``[0, 1]``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..types import MetricName
from .base import MetricEvaluator, MetricRequest


class TaskCompletionMetric(MetricEvaluator):
    """Wraps the (experimental) ``TaskCompletionEvaluator`` if available.

    Some releases of ``azure-ai-evaluation`` ship ``TaskCompletionEvaluator``
    and others ship ``CompletenessEvaluator``. We try the former first
    and fall back to the latter so the SDK stays compatible across
    minor versions.
    """

    NAME = MetricName.TASK_COMPLETION

    def _build_evaluator(self) -> Any:
        try:  # pragma: no cover - both paths are exercised in CI matrix
            from azure.ai.evaluation import TaskCompletionEvaluator  # type: ignore

            return TaskCompletionEvaluator(model_config=self._judge_config.to_model_config())
        except ImportError:
            from azure.ai.evaluation import CompletenessEvaluator  # type: ignore

            return CompletenessEvaluator(model_config=self._judge_config.to_model_config())

    def _build_kwargs(self, request: MetricRequest) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "query": request.query,
            "response": request.response,
        }
        if request.tool_definitions is not None:
            kwargs["tool_definitions"] = request.tool_definitions
        return kwargs

    def _normalize(self, raw: Dict[str, Any]) -> Optional[float]:
        from .base import _extract_score  # type: ignore

        score = _extract_score(raw, self.NAME)
        if score is None:
            return None
        return max(0.0, min(1.0, score))


__all__ = ["TaskCompletionMetric"]
