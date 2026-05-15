"""Core domain types for the agent-learning SDK.

These dataclasses are the durable shapes that flow between capture →
metric evaluation → reward shaping → policy learning → persistence.

All types are JSON-serialisable via ``to_dict``/``from_dict`` so they
can round-trip through Cosmos DB or local JSONL fallbacks without any
special encoder.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (timezone-aware)."""
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    """Return a new opaque UUID4 identifier as a string."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class MetricName(str, Enum):
    """Identifiers for the supported native judge metrics."""

    INTENT_RESOLUTION = "intent_resolution"
    TASK_ADHERENCE = "task_adherence"
    TASK_COMPLETION = "task_completion"


class RewardSource(str, Enum):
    """Source of a reward record."""

    METRIC = "metric"  # Derived from a single judge metric
    AGGREGATE = "aggregate"  # Combined scalar reward across multiple metrics
    HUMAN_APPROVAL = "human_approval"
    TEST_RESULT = "test_result"
    LATENCY_PENALTY = "latency_penalty"
    COST_PENALTY = "cost_penalty"


class TrainingStatus(str, Enum):
    """Lifecycle states for a learner training run."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """A single tool invocation captured inside an episode."""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None
    duration_ms: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "result": self.result,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        return cls(
            name=data["name"],
            arguments=data.get("arguments", {}),
            result=data.get("result"),
            duration_ms=data.get("duration_ms"),
            error=data.get("error"),
        )


@dataclass
class Episode:
    """A complete agent interaction: prompt → tool calls → response.

    The episode is the unit of evaluation and the unit on which rewards
    are attached. ``context_features`` carries the input the policy used
    to choose its action (so the learner can replay the decision).
    """

    id: str = field(default_factory=_new_id)
    agent_id: str = "default"
    user_input: str = ""
    assistant_output: str = ""
    system_message: Optional[str] = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    # Policy decision metadata - what the policy chose for this episode
    policy_id: Optional[str] = None
    policy_version: Optional[int] = None
    action_id: Optional[str] = None
    action_logprob: Optional[float] = None
    context_features: Dict[str, Any] = field(default_factory=dict)
    # Operational metadata
    model_deployment: Optional[str] = None
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    request_latency_ms: Optional[int] = None
    token_usage: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utcnow_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "user_input": self.user_input,
            "assistant_output": self.assistant_output,
            "system_message": self.system_message,
            "conversation_history": self.conversation_history,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "action_id": self.action_id,
            "action_logprob": self.action_logprob,
            "context_features": self.context_features,
            "model_deployment": self.model_deployment,
            "correlation_id": self.correlation_id,
            "session_id": self.session_id,
            "request_latency_ms": self.request_latency_ms,
            "token_usage": self.token_usage,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        return cls(
            id=data["id"],
            agent_id=data.get("agent_id", "default"),
            user_input=data.get("user_input", ""),
            assistant_output=data.get("assistant_output", ""),
            system_message=data.get("system_message"),
            conversation_history=data.get("conversation_history", []),
            tool_calls=[ToolCall.from_dict(tc) for tc in data.get("tool_calls", [])],
            policy_id=data.get("policy_id"),
            policy_version=data.get("policy_version"),
            action_id=data.get("action_id"),
            action_logprob=data.get("action_logprob"),
            context_features=data.get("context_features", {}),
            model_deployment=data.get("model_deployment"),
            correlation_id=data.get("correlation_id"),
            session_id=data.get("session_id"),
            request_latency_ms=data.get("request_latency_ms"),
            token_usage=data.get("token_usage"),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", _utcnow_iso()),
        )


# ---------------------------------------------------------------------------
# Metric & Reward
# ---------------------------------------------------------------------------


@dataclass
class MetricResult:
    """The output of a single judge metric evaluation."""

    metric: MetricName
    score: Optional[float]  # Raw judge score (e.g. 1-5 or 0/1); None if skipped
    normalized: Optional[float]  # Mapped to [0, 1]; None if skipped
    status: str  # "completed" | "skipped"
    reason: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None
    evaluator: Optional[str] = None  # Judge model deployment used
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric.value,
            "score": self.score,
            "normalized": self.normalized,
            "status": self.status,
            "reason": self.reason,
            "properties": self.properties,
            "evaluator": self.evaluator,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricResult":
        return cls(
            metric=MetricName(data["metric"]),
            score=data.get("score"),
            normalized=data.get("normalized"),
            status=data.get("status", "completed"),
            reason=data.get("reason"),
            properties=data.get("properties"),
            evaluator=data.get("evaluator"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Reward:
    """A scalar reward attached to an episode."""

    id: str = field(default_factory=_new_id)
    episode_id: str = ""
    agent_id: str = "default"
    source: RewardSource = RewardSource.METRIC
    value: float = 0.0  # Always in [-1, 1]
    raw_value: Optional[Any] = None
    metric: Optional[MetricName] = None  # Populated when source == METRIC
    rubric: Optional[str] = None
    evaluator: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utcnow_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "episode_id": self.episode_id,
            "agent_id": self.agent_id,
            "source": self.source.value,
            "value": self.value,
            "raw_value": self.raw_value,
            "metric": self.metric.value if self.metric else None,
            "rubric": self.rubric,
            "evaluator": self.evaluator,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Reward":
        metric_val = data.get("metric")
        return cls(
            id=data["id"],
            episode_id=data["episode_id"],
            agent_id=data.get("agent_id", "default"),
            source=RewardSource(data.get("source", "metric")),
            value=float(data.get("value", 0.0)),
            raw_value=data.get("raw_value"),
            metric=MetricName(metric_val) if metric_val else None,
            rubric=data.get("rubric"),
            evaluator=data.get("evaluator"),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", _utcnow_iso()),
        )


# ---------------------------------------------------------------------------
# Policy & Training
# ---------------------------------------------------------------------------


@dataclass
class Action:
    """A discrete action in the policy's action space.

    An action represents a concrete agent configuration choice (for
    example, "use prompt variant A" or "use retrieval_k=8"). The
    learner does not care about the semantics — it only sees the
    ``id``; consumers map the id back to a real configuration.
    """

    id: str
    description: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        return cls(
            id=data["id"],
            description=data.get("description"),
            parameters=data.get("parameters", {}),
        )


@dataclass
class PolicySnapshot:
    """A versioned snapshot of policy parameters.

    The snapshot is the authoritative record of the policy that was
    used to generate a set of episodes and is what the learner mutates
    when it applies an update.
    """

    id: str = field(default_factory=_new_id)
    agent_id: str = "default"
    version: int = 0
    actions: List[Action] = field(default_factory=list)
    logits: Dict[str, float] = field(default_factory=dict)
    baseline: float = 0.0  # EMA value baseline
    episodes_seen: int = 0
    updates_applied: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utcnow_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "version": self.version,
            "actions": [a.to_dict() for a in self.actions],
            "logits": self.logits,
            "baseline": self.baseline,
            "episodes_seen": self.episodes_seen,
            "updates_applied": self.updates_applied,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicySnapshot":
        return cls(
            id=data["id"],
            agent_id=data.get("agent_id", "default"),
            version=int(data.get("version", 0)),
            actions=[Action.from_dict(a) for a in data.get("actions", [])],
            logits={k: float(v) for k, v in data.get("logits", {}).items()},
            baseline=float(data.get("baseline", 0.0)),
            episodes_seen=int(data.get("episodes_seen", 0)),
            updates_applied=int(data.get("updates_applied", 0)),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", _utcnow_iso()),
        )


@dataclass
class TrainingRun:
    """Record of a native learner training run."""

    id: str = field(default_factory=_new_id)
    agent_id: str = "default"
    policy_id: str = ""
    algorithm: str = "reinforce"
    status: TrainingStatus = TrainingStatus.PENDING
    episode_ids: List[str] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utcnow_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "policy_id": self.policy_id,
            "algorithm": self.algorithm,
            "status": self.status.value,
            "episode_ids": self.episode_ids,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "error_message": self.error_message,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingRun":
        return cls(
            id=data["id"],
            agent_id=data.get("agent_id", "default"),
            policy_id=data.get("policy_id", ""),
            algorithm=data.get("algorithm", "reinforce"),
            status=TrainingStatus(data.get("status", "pending")),
            episode_ids=data.get("episode_ids", []),
            hyperparameters=data.get("hyperparameters", {}),
            metrics=data.get("metrics", {}),
            error_message=data.get("error_message"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", _utcnow_iso()),
        )


__all__ = [
    "Action",
    "Episode",
    "MetricName",
    "MetricResult",
    "PolicySnapshot",
    "Reward",
    "RewardSource",
    "ToolCall",
    "TrainingRun",
    "TrainingStatus",
]
