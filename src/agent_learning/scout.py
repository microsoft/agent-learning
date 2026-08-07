"""Local audit adapter for Scout action executions."""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar, cast

from .capture import redact
from .judges.base import Judge, JudgeScore
from .judges.stdlib import StdlibAdherenceJudge, StdlibCompletionJudge, StdlibIntentJudge

T = TypeVar("T")


class ScoutAuditAdapter:
    """Execute Scout actions and append inspectable audit records to JSONL."""

    def __init__(
        self,
        audit_path: str | Path = "scout-audit.jsonl",
        *,
        judges: tuple[Judge, Judge, Judge] | None = None,
    ) -> None:
        self.audit_path = Path(audit_path).expanduser()
        defaults = cast(
            tuple[Judge, Judge, Judge],
            (StdlibIntentJudge(), StdlibAdherenceJudge(), StdlibCompletionJudge()),
        )
        self._judges = judges or defaults
        self._write_lock = threading.Lock()

    def execute(
        self,
        *,
        intent: str,
        action_path: Sequence[str],
        action: Callable[..., T],
        args: Sequence[Any] = (),
        action_kwargs: Mapping[str, Any] | None = None,
        contract: dict[str, Any] | None = None,
        expected_tokens: Sequence[str] | None = None,
    ) -> T:
        """Run one synchronous Scout action, audit it, and return its result."""
        path = _action_path(action_path)
        started = time.monotonic()
        try:
            result = action(*args, **dict(action_kwargs or {}))
        except Exception as exc:
            self._record(
                intent=intent,
                action_path=path,
                started=started,
                status="failed",
                result=None,
                error=exc,
                contract=contract,
                expected_tokens=expected_tokens,
            )
            raise

        self._record(
            intent=intent,
            action_path=path,
            started=started,
            status="succeeded",
            result=result,
            error=None,
            contract=contract,
            expected_tokens=expected_tokens,
        )
        return result

    async def execute_async(
        self,
        *,
        intent: str,
        action_path: Sequence[str],
        action: Callable[..., Awaitable[T]],
        args: Sequence[Any] = (),
        action_kwargs: Mapping[str, Any] | None = None,
        contract: dict[str, Any] | None = None,
        expected_tokens: Sequence[str] | None = None,
    ) -> T:
        """Run one asynchronous Scout action, audit it, and return its result."""
        path = _action_path(action_path)
        started = time.monotonic()
        try:
            result = await action(*args, **dict(action_kwargs or {}))
        except Exception as exc:
            self._record(
                intent=intent,
                action_path=path,
                started=started,
                status="failed",
                result=None,
                error=exc,
                contract=contract,
                expected_tokens=expected_tokens,
            )
            raise

        self._record(
            intent=intent,
            action_path=path,
            started=started,
            status="succeeded",
            result=result,
            error=None,
            contract=contract,
            expected_tokens=expected_tokens,
        )
        return result

    def _record(
        self,
        *,
        intent: str,
        action_path: Sequence[str],
        started: float,
        status: str,
        result: Any,
        error: Exception | None,
        contract: dict[str, Any] | None,
        expected_tokens: Sequence[str] | None,
    ) -> None:
        safe_intent = redact(intent) or ""
        safe_result = _json_safe(result)
        error_message = redact(str(error)) if error is not None else None
        response = _response_text(safe_result) if error is None else (error_message or "")
        signals = self._evaluate(
            query=safe_intent,
            response=response,
            contract=contract or {},
            expected_tokens=expected_tokens or (),
        )
        record = {
            "schema_version": 1,
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "intent": safe_intent,
            "action_path": [redact(str(segment)) or "" for segment in action_path],
            "outcome": {
                "status": status,
                "result": safe_result,
                "error": error_message,
                "error_type": type(error).__name__ if error is not None else None,
                "duration_ms": int((time.monotonic() - started) * 1000),
            },
            "judge_signals": signals,
        }
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._write_lock, self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _evaluate(
        self,
        *,
        query: str,
        response: str,
        contract: dict[str, Any],
        expected_tokens: Sequence[str],
    ) -> dict[str, dict[str, Any]]:
        signals: dict[str, dict[str, Any]] = {}
        for name, judge in zip(("intent", "adherence", "completion"), self._judges):
            try:
                score = judge.score(
                    query=query,
                    response=response,
                    contract=contract,
                    expected_tokens=expected_tokens,
                )
                signals[name] = _score_dict(score)
            except Exception as exc:  # noqa: BLE001 - judge failures belong in the audit record
                signals[name] = {
                    "status": "error",
                    "error": redact(str(exc)),
                }
        return signals


def _score_dict(score: JudgeScore) -> dict[str, Any]:
    return {
        "status": "completed",
        "label": score.label,
        "confidence": score.confidence,
        "normalized": score.normalized,
        "features": _json_safe(score.features),
    }


def _response_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _action_path(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not value:
        raise ValueError("action_path must be a non-empty sequence of path segments")
    return tuple(str(segment) for segment in value)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return redact(str(value))


__all__ = ["ScoutAuditAdapter"]
