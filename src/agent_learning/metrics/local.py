"""Metric adapters for on-device scoring backends."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from ..scorers.base import Scorer, ScoreResult
from ..scorers.stdlib._text import tokenize
from ..types import MetricName, MetricResult
from .base import MetricEvaluator, MetricRequest

logger = logging.getLogger(__name__)


class LocalScorerMetric(MetricEvaluator):
    """Project a local :class:`Scorer` onto the metric interface."""

    NAME = MetricName.INTENT_RESOLUTION

    def __init__(self, metric: MetricName, scorer: Scorer) -> None:
        super().__init__(evaluator=scorer)
        self.NAME = metric
        self._scorer = scorer

    def _build_evaluator(self) -> Any:  # pragma: no cover - injected in __init__
        return self._scorer

    def _build_kwargs(self, request: MetricRequest) -> dict[str, Any]:
        metadata = request.extra.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        context_features = request.extra.get("context_features")
        if not isinstance(context_features, dict):
            context_features = {}
        contract = metadata.get("score_contract") or metadata.get("adherence_contract")
        if not isinstance(contract, dict):
            contract = {}
        expected_tokens = metadata.get("expected_tokens")
        if not isinstance(expected_tokens, Sequence) or isinstance(
            expected_tokens, (str, bytes)
        ):
            expected_tokens = tokenize(str(request.extra.get("expected_outcome") or ""))
        return {
            "query": request.query,
            "response": request.response,
            "system_message": request.system_message,
            "tool_calls": request.tool_calls,
            "action_id": request.extra.get("action_id"),
            "phi": context_features.get("phi"),
            "contract": contract,
            "expected_tokens": expected_tokens,
        }

    def _normalize(self, raw: dict[str, Any]) -> float | None:
        value = raw.get("normalized")
        return float(value) if value is not None else None

    def evaluate(self, request: MetricRequest) -> MetricResult:
        if not (request.query and request.response):
            return MetricResult(
                metric=self.NAME,
                score=None,
                normalized=None,
                status="skipped",
                reason="query or response is empty",
                evaluator=f"local:{self._scorer.name}",
            )

        authoritative = self._authoritative_completion(request)
        if authoritative is not None:
            value, reason = authoritative
            return MetricResult(
                metric=self.NAME,
                score=value,
                normalized=value,
                status="completed",
                reason=reason,
                evaluator="local:episode-outcome",
            )

        try:
            result = self._scorer.score(**self._build_kwargs(request))
        except Exception as exc:  # noqa: BLE001  # pragma: no cover
            logger.warning("Local metric %s failed: %s", self.NAME.value, exc)
            return MetricResult(
                metric=self.NAME,
                score=None,
                normalized=None,
                status="skipped",
                reason=f"local scorer error: {exc}",
                evaluator=f"local:{self._scorer.name}",
            )
        return _to_metric_result(self.NAME, self._scorer.name, result)

    def _authoritative_completion(
        self, request: MetricRequest
    ) -> tuple[float, str] | None:
        if self.NAME != MetricName.TASK_COMPLETION:
            return None
        metadata = request.extra.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        action_id = request.extra.get("action_id")
        correct_action_id = metadata.get("correct_action_id")
        if action_id and correct_action_id:
            return (
                float(action_id == correct_action_id),
                "derived from action_id and metadata.correct_action_id",
            )
        completed = metadata.get("task_completed")
        if isinstance(completed, bool):
            return float(completed), "derived from metadata.task_completed"
        status = str(request.extra.get("execution_status") or "").lower()
        if status == "completed":
            return 1.0, "derived from execution_status=completed"
        if status == "failed":
            return 0.0, "derived from execution_status=failed"
        if status == "partial":
            return 0.5, "derived from execution_status=partial"
        return None


def _to_metric_result(
    metric: MetricName, scorer_name: str, result: ScoreResult
) -> MetricResult:
    normalized = max(0.0, min(1.0, float(result.normalized)))
    return MetricResult(
        metric=metric,
        score=normalized,
        normalized=normalized,
        status="completed",
        reason=f"local {result.label}",
        properties=dict(result.features),
        evaluator=f"local:{scorer_name}",
        metadata={"label": result.label, "confidence": result.confidence},
    )


def local_metrics(scorers: tuple[Scorer, Scorer, Scorer]) -> list[MetricEvaluator]:
    """Return local metric adapters in reward-shaping order."""
    intent, adherence, completion = scorers
    return [
        LocalScorerMetric(MetricName.INTENT_RESOLUTION, intent),
        LocalScorerMetric(MetricName.TASK_ADHERENCE, adherence),
        LocalScorerMetric(MetricName.TASK_COMPLETION, completion),
    ]


__all__ = ["LocalScorerMetric", "local_metrics"]