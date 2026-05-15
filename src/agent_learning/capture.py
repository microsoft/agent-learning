"""Episode capture helpers.

The capture hook is the orchestrator-facing entry point. It wraps a
single agent turn, lets the caller record tool calls during the
turn, and persists a complete :class:`Episode` at the end.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import CaptureConfig
from .storage.base import LearningStore
from .storage.cosmos import get_default_store
from .types import Episode, ToolCall

logger = logging.getLogger(__name__)


# Conservative redaction patterns (mirrors lightning's set)
_REDACT_PATTERNS = [
    (re.compile(r"(bearer\s+)[a-zA-Z0-9\-_\.]+", re.IGNORECASE), r"\1[REDACTED]"),
    (re.compile(r"(api[_-]?key[\"\s:=]+)[a-zA-Z0-9\-_]+", re.IGNORECASE), r"\1[REDACTED]"),
    (re.compile(r"(password[\"\s:=]+)[^\s\"]+", re.IGNORECASE), r"\1[REDACTED]"),
    (re.compile(r"(secret[\"\s:=]+)[^\s\"]+", re.IGNORECASE), r"\1[REDACTED]"),
    (re.compile(r"(token[\"\s:=]+)[a-zA-Z0-9\-_\.]+", re.IGNORECASE), r"\1[REDACTED]"),
    (re.compile(r"(connection[_-]?string[\"\s:=]+)[^\s\"]+", re.IGNORECASE), r"\1[REDACTED]"),
    (re.compile(r"(AccountKey=)[^;]+", re.IGNORECASE), r"\1[REDACTED]"),
    (re.compile(r"(SharedAccessSignature=)[^;]+", re.IGNORECASE), r"\1[REDACTED]"),
]


def redact(text: Optional[str]) -> Optional[str]:
    """Strip well-known secret patterns from a string."""
    if not text:
        return text
    out = text
    for pattern, replacement in _REDACT_PATTERNS:
        out = pattern.sub(replacement, out)
    return out


@dataclass
class CaptureContext:
    """In-flight state for a single capture."""

    episode_id: str
    agent_id: str
    start_time: float
    user_input: str
    system_message: Optional[str] = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    policy_id: Optional[str] = None
    policy_version: Optional[int] = None
    action_id: Optional[str] = None
    action_logprob: Optional[float] = None
    context_features: Dict[str, Any] = field(default_factory=dict)
    model_deployment: Optional[str] = None
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EpisodeCapture:
    """Capture one episode at a time and persist it via the store."""

    def __init__(
        self,
        config: Optional[CaptureConfig] = None,
        store: Optional[LearningStore] = None,
    ) -> None:
        self._config = config or CaptureConfig()
        self._store = store

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------

    @property
    def store(self) -> LearningStore:
        if self._store is None:
            self._store = get_default_store()
        return self._store

    @property
    def config(self) -> CaptureConfig:
        return self._config

    def is_enabled(self) -> bool:
        return self._config.enabled

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(
        self,
        user_input: str,
        *,
        system_message: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        model_deployment: Optional[str] = None,
        correlation_id: Optional[str] = None,
        session_id: Optional[str] = None,
        policy_id: Optional[str] = None,
        policy_version: Optional[int] = None,
        action_id: Optional[str] = None,
        action_logprob: Optional[float] = None,
        context_features: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CaptureContext:
        return CaptureContext(
            episode_id=str(uuid.uuid4()),
            agent_id=self._config.agent_id,
            start_time=time.time(),
            user_input=user_input,
            system_message=system_message,
            conversation_history=conversation_history or [],
            policy_id=policy_id,
            policy_version=policy_version,
            action_id=action_id,
            action_logprob=action_logprob,
            context_features=context_features or {},
            model_deployment=model_deployment,
            correlation_id=correlation_id,
            session_id=session_id,
            metadata=metadata or {},
        )

    def record_tool_call(
        self,
        ctx: CaptureContext,
        name: str,
        arguments: Dict[str, Any],
        result: Optional[str] = None,
        *,
        duration_ms: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        if not self.is_enabled():
            return

        safe_args = arguments
        if self._config.redact_secrets:
            safe_args = {
                k: redact(v) if isinstance(v, str) else v for k, v in arguments.items()
            }

        safe_result = result
        if safe_result and self._config.redact_secrets:
            safe_result = redact(safe_result)
        if safe_result and len(safe_result) > self._config.max_output_length:
            safe_result = safe_result[: self._config.max_output_length] + "...[TRUNCATED]"

        ctx.tool_calls.append(
            ToolCall(
                name=name,
                arguments=safe_args or {},
                result=safe_result,
                duration_ms=duration_ms,
                error=error,
            )
        )

    def end(
        self,
        ctx: CaptureContext,
        assistant_output: str,
        *,
        token_usage: Optional[Dict[str, int]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Episode]:
        if not self.is_enabled():
            return None

        latency_ms = int((time.time() - ctx.start_time) * 1000)
        output = assistant_output
        if output and self._config.redact_secrets:
            output = redact(output) or output

        metadata = dict(ctx.metadata)
        if extra_metadata:
            metadata.update(extra_metadata)

        episode = Episode(
            id=ctx.episode_id,
            agent_id=ctx.agent_id,
            user_input=ctx.user_input,
            assistant_output=output or "",
            system_message=ctx.system_message,
            conversation_history=ctx.conversation_history,
            tool_calls=ctx.tool_calls,
            policy_id=ctx.policy_id,
            policy_version=ctx.policy_version,
            action_id=ctx.action_id,
            action_logprob=ctx.action_logprob,
            context_features=ctx.context_features,
            model_deployment=ctx.model_deployment,
            correlation_id=ctx.correlation_id,
            session_id=ctx.session_id,
            request_latency_ms=latency_ms,
            token_usage=token_usage,
            metadata=metadata,
        )

        try:
            self.store.store_episode(episode)
            logger.info("Captured episode %s for agent %s", episode.id, episode.agent_id)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to persist episode %s: %s", episode.id, exc)

        return episode


_default_capture: Optional[EpisodeCapture] = None


def get_capture() -> EpisodeCapture:
    """Return a process-wide singleton capture hook."""
    global _default_capture
    if _default_capture is None:
        _default_capture = EpisodeCapture()
    return _default_capture


__all__ = ["CaptureContext", "EpisodeCapture", "get_capture", "redact"]
