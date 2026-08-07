"""Shared wrapper that adapts an ``azure-ai-evaluation`` evaluator to
the SDK :class:`ScoreResult` contract.

The evaluator import is lazy: the ``azure-ai-evaluation`` package is
only imported the first time :meth:`_evaluator` is invoked, so callers
who never use the LLM backend pay no import cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ...config import ScoreConfig
from ..base import ScoreResult


@dataclass
class _LlmScorerWrapper:
    """Common base for the three concrete LLM scorers."""

    cfg: ScoreConfig
    name: str = "llm-scorer"
    label_name: str = "llm-scorer"
    evaluator_attr: str = ""
    _cached_evaluator: Optional[Any] = field(default=None, repr=False)

    def _evaluator(self) -> Any:
        if self._cached_evaluator is not None:
            return self._cached_evaluator
        try:
            module = __import__("azure.ai.evaluation", fromlist=[self.evaluator_attr])
        except ImportError as exc:
            raise ImportError(
                "LLM scorers require the optional 'azure-ai-evaluation' package. "
                "Install it with: pip install azure-ai-evaluation"
            ) from exc
        evaluator_cls = getattr(module, self.evaluator_attr)
        self._cached_evaluator = evaluator_cls(model_config=self.cfg.to_model_config())
        return self._cached_evaluator

    def score(
        self,
        *,
        query: Optional[str] = None,
        response: Optional[str] = None,
        request: Optional[str] = None,
        **kwargs: object,
    ) -> ScoreResult:
        """Run the underlying evaluator and project to a ScoreResult.

        Either ``query`` or ``request`` may be used to supply the
        prompt. Extra keyword arguments are forwarded to the evaluator
        so callers can pass evaluator-specific fields like ``context``
        or ``tool_calls`` without the SDK needing to learn them.
        """
        if query is None:
            query = request
        if query is None or response is None:
            raise ValueError(
                f"{self.name} LLM scorer requires query (or request) and response"
            )
        payload = {"query": query, "response": response, **kwargs}
        # Drop NLP-specific kwargs that would confuse the evaluator.
        payload.pop("phi", None)
        payload.pop("action_id", None)
        evaluator = self._evaluator()
        result = evaluator(**payload)
        return _project_to_score(result, threshold=self.cfg.threshold, name=self.name)


def _project_to_score(
    result: Any, *, threshold: float, name: str
) -> ScoreResult:
    """Map an evaluator return value onto a :class:`ScoreResult`.

    ``azure-ai-evaluation`` evaluators return a mapping with a numeric
    score keyed by either ``"score"`` or ``"<evaluator>_score"``. The
    helper accepts either shape so the SDK works across evaluator
    versions.
    """
    if not isinstance(result, dict):
        raise TypeError(
            f"{name} evaluator returned {type(result).__name__}, expected mapping"
        )
    raw: Optional[float] = None
    for key in ("score", f"{name}_score", "result"):
        if key in result and isinstance(result[key], (int, float)):
            raw = float(result[key])
            break
    if raw is None:
        # Fall back to the first numeric value in the result mapping.
        for value in result.values():
            if isinstance(value, (int, float)):
                raw = float(value)
                break
    if raw is None:
        raise ValueError(
            f"{name} evaluator returned no numeric score; got keys {list(result)!r}"
        )
    normalized = _normalize(raw)
    label = "pass" if normalized >= threshold else "fail"
    confidence = normalized if label == "pass" else 1.0 - normalized
    return ScoreResult(
        label=label,
        confidence=confidence,
        normalized=normalized,
        features={"raw": raw, "evaluator_result": result},
    )


def _normalize(raw: float) -> float:
    """Project a raw evaluator score into ``[0, 1]``.

    Common ``azure-ai-evaluation`` evaluators emit scores on a 1-5
    Likert scale. Scores already in ``[0, 1]`` are passed through
    unchanged.
    """
    if 0.0 <= raw <= 1.0:
        return raw
    if 1.0 <= raw <= 5.0:
        return (raw - 1.0) / 4.0
    return max(0.0, min(1.0, raw))


__all__ = ["_LlmScorerWrapper"]
