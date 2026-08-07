"""Base classes for native score-based metrics."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import ScoreConfig
from ..types import Episode, MetricName, MetricResult

logger = logging.getLogger(__name__)


@dataclass
class MetricRequest:
    """Input passed to a metric evaluator.

    The fields mirror the prompty schemas used by azure-ai-evaluation:
    ``query`` is the user message (or the full conversation history
    formatted as a string), ``response`` is the agent reply, and
    ``tool_calls`` / ``system_message`` are populated when required by
    the underlying evaluator.
    """

    query: str
    response: str
    system_message: Optional[str] = None
    tool_calls: Optional[str] = None
    tool_definitions: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_episode(cls, episode: Episode) -> "MetricRequest":
        """Build a metric request from an :class:`Episode`."""
        query = _format_query(episode)
        return cls(
            query=query,
            response=episode.assistant_output or "",
            system_message=episode.system_message,
            tool_calls=_format_tool_calls(episode),
            tool_definitions=episode.metadata.get("tool_definitions"),
        )


def _format_query(episode: Episode) -> str:
    """Render the conversation history as the ``query`` field.

    The Azure prompts accept a free-form ``query`` string. We prefer
    the structured conversation history when present, falling back to
    the single ``user_input`` field.
    """
    if episode.conversation_history:
        lines: List[str] = []
        for msg in episode.conversation_history:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        if episode.user_input and (
            not lines or not lines[-1].lower().startswith("user:")
        ):
            lines.append(f"USER: {episode.user_input}")
        return "\n".join(lines)
    return episode.user_input or ""


def _format_tool_calls(episode: Episode) -> str:
    """Render tool calls as a JSON string consumable by the scorer."""
    if not episode.tool_calls:
        return "[]"
    payload = [
        {
            "name": tc.name,
            "arguments": tc.arguments,
            "result": tc.result,
            "error": tc.error,
        }
        for tc in episode.tool_calls
    ]
    return json.dumps(payload, ensure_ascii=False)


class MetricEvaluator(ABC):
    """Abstract base for all native scoring metrics."""

    #: Stable identifier - subclasses must override.
    NAME: MetricName

    def __init__(
        self,
        score_config: Optional[ScoreConfig] = None,
        *,
        evaluator: Optional[Any] = None,
    ) -> None:
        self._score_config = score_config or ScoreConfig()
        self._evaluator = evaluator  # Allow injection for tests

    # ------------------------------------------------------------------
    # Construction hooks (subclasses override _build_evaluator)
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_evaluator(self) -> Any:
        """Instantiate the underlying azure-ai-evaluation evaluator."""

    @abstractmethod
    def _normalize(self, raw: Dict[str, Any]) -> Optional[float]:
        """Map the evaluator's raw output to a ``[0, 1]`` score."""

    @abstractmethod
    def _build_kwargs(self, request: MetricRequest) -> Dict[str, Any]:
        """Build the call kwargs for the underlying evaluator."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def evaluator(self) -> Any:
        """The lazily-constructed underlying evaluator."""
        if self._evaluator is None:
            if not self._score_config.enabled:
                raise RuntimeError(
                    "Score model is not configured. Set AGENT_LEARNING_SCORE_ENDPOINT "
                    "and AGENT_LEARNING_SCORE_DEPLOYMENT (or pass an explicit "
                    "ScoreConfig) before evaluating metrics."
                )
            self._evaluator = self._build_evaluator()
        return self._evaluator

    def evaluate(self, request: MetricRequest) -> MetricResult:
        """Run the underlying scorer and return a normalised result.

        Errors are caught and returned as ``status=skipped`` so that a
        flaky scorer does not poison the learner. Callers that want
        strict semantics can inspect ``result.status``.
        """
        if not (request.query and request.response):
            return MetricResult(
                metric=self.NAME,
                score=None,
                normalized=None,
                status="skipped",
                reason="query or response is empty",
                evaluator=self._score_config.azure_deployment or None,
            )

        try:
            kwargs = self._build_kwargs(request)
            raw = self.evaluator(**kwargs)
            if not isinstance(raw, dict):
                raw = dict(raw)
        except Exception as exc:  # pragma: no cover - scoring runtime errors
            logger.warning("Metric %s failed: %s", self.NAME.value, exc)
            return MetricResult(
                metric=self.NAME,
                score=None,
                normalized=None,
                status="skipped",
                reason=f"evaluator error: {exc}",
                evaluator=self._score_config.azure_deployment or None,
            )

        return self._build_result(raw)

    # ------------------------------------------------------------------
    # Result extraction
    # ------------------------------------------------------------------

    def _build_result(self, raw: Dict[str, Any]) -> MetricResult:
        score = _extract_score(raw, self.NAME)
        normalized = self._normalize(raw)
        status = "skipped" if score is None else "completed"

        properties = raw.get("properties") if isinstance(raw.get("properties"), dict) else None
        reason_keys = (
            f"{self.NAME.value}_reason",
            "reason",
        )
        reason: Optional[str] = None
        for key in reason_keys:
            if isinstance(raw.get(key), str):
                reason = raw[key]
                break

        return MetricResult(
            metric=self.NAME,
            score=score,
            normalized=normalized,
            status=status,
            reason=reason,
            properties=properties,
            evaluator=self._score_config.azure_deployment or None,
            metadata={"raw": _safe_subset(raw)},
        )


def _extract_score(raw: Dict[str, Any], name: MetricName) -> Optional[float]:
    """Return the numeric score from a scorer response, regardless of key."""
    keys = (
        "score",
        f"{name.value}_score",
        f"{name.value}",
    )
    for key in keys:
        if key in raw and raw[key] is not None:
            try:
                return float(raw[key])
            except (TypeError, ValueError):
                continue
    return None


def _safe_subset(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Trim the raw score dict to a Cosmos-safe size for persistence."""
    keep_keys = {
        "score",
        "status",
        "reason",
        "properties",
        "intent_resolution_score",
        "intent_resolution_reason",
        "task_adherence_score",
        "task_adherence_reason",
        "task_completion_score",
        "task_completion_reason",
    }
    return {k: v for k, v in raw.items() if k in keep_keys}


__all__ = ["MetricEvaluator", "MetricRequest"]
