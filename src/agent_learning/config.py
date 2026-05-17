"""Environment-driven configuration for the agent-learning SDK.

All settings can be overridden via environment variables; programmatic
overrides are also accepted via dataclass field assignment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable using common conventions."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Cosmos DB
# ---------------------------------------------------------------------------


@dataclass
class CosmosConfig:
    """Cosmos DB connection + container configuration."""

    endpoint: str = field(default_factory=lambda: os.getenv("AGENT_LEARNING_COSMOS_ENDPOINT", ""))
    database_name: str = field(default_factory=lambda: os.getenv("AGENT_LEARNING_COSMOS_DATABASE", "dq_rl"))
    auth_mode: str = field(default_factory=lambda: os.getenv("AGENT_LEARNING_COSMOS_AUTH_MODE", "aad"))
    account_key: str = field(default_factory=lambda: os.getenv("AGENT_LEARNING_COSMOS_KEY", ""))
    partition_key_field: str = field(
        default_factory=lambda: os.getenv("AGENT_LEARNING_PARTITION_KEY_FIELD", "agent_id")
    )

    container_episodes: str = field(
        default_factory=lambda: os.getenv("AGENT_LEARNING_CONTAINER_EPISODES", "learning_episodes")
    )
    container_rewards: str = field(
        default_factory=lambda: os.getenv("AGENT_LEARNING_CONTAINER_REWARDS", "learning_rewards")
    )
    container_metrics: str = field(
        default_factory=lambda: os.getenv("AGENT_LEARNING_CONTAINER_METRICS", "learning_metrics")
    )
    container_policies: str = field(
        default_factory=lambda: os.getenv("AGENT_LEARNING_CONTAINER_POLICIES", "learning_policies")
    )
    container_runs: str = field(
        default_factory=lambda: os.getenv("AGENT_LEARNING_CONTAINER_RUNS", "learning_runs")
    )

    @property
    def enabled(self) -> bool:
        return bool(self.endpoint)


# ---------------------------------------------------------------------------
# Judge / evaluator model
# ---------------------------------------------------------------------------


@dataclass
class JudgeConfig:
    """Configuration for the LLM judge used by metric evaluators."""

    azure_endpoint: str = field(default_factory=lambda: os.getenv("AGENT_LEARNING_JUDGE_ENDPOINT", ""))
    azure_deployment: str = field(default_factory=lambda: os.getenv("AGENT_LEARNING_JUDGE_DEPLOYMENT", ""))
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("AGENT_LEARNING_JUDGE_API_KEY") or None)
    api_version: str = field(default_factory=lambda: os.getenv("AGENT_LEARNING_JUDGE_API_VERSION", "2024-10-21"))
    # Pass-through threshold for IntentResolutionEvaluator (defaults to its own default)
    threshold: int = field(default_factory=lambda: _env_int("AGENT_LEARNING_INTENT_THRESHOLD", 3))

    @property
    def enabled(self) -> bool:
        return bool(self.azure_endpoint and self.azure_deployment)

    def to_model_config(self) -> dict:
        """Build the dict expected by azure-ai-evaluation evaluators."""
        cfg: dict = {
            "azure_endpoint": self.azure_endpoint,
            "azure_deployment": self.azure_deployment,
            "api_version": self.api_version,
        }
        if self.api_key:
            cfg["api_key"] = self.api_key
        return cfg


# ---------------------------------------------------------------------------
# Capture / shaping
# ---------------------------------------------------------------------------


@dataclass
class CaptureConfig:
    """Configuration for episode capture."""

    enabled: bool = field(default_factory=lambda: _env_bool("AGENT_LEARNING_ENABLE_CAPTURE", False))
    agent_id: str = field(default_factory=lambda: os.getenv("AGENT_LEARNING_AGENT_ID", "default"))
    local_fallback_dir: str = field(
        default_factory=lambda: os.getenv("AGENT_LEARNING_DATA_DIR", "./data/agent-learning")
    )
    max_output_length: int = field(default_factory=lambda: _env_int("AGENT_LEARNING_MAX_OUTPUT_LEN", 10000))
    redact_secrets: bool = field(default_factory=lambda: _env_bool("AGENT_LEARNING_REDACT_SECRETS", True))


@dataclass
class ShapingConfig:
    """Weights and penalties used to combine metric scores into a scalar reward.

    Each weight is multiplied against the metric's normalized score (in
    [0, 1]) before summing. Weights need not sum to 1 — the shaper will
    return the raw weighted sum, clamped to [-1, 1].

    The defaults reflect the credit-assignment design in
    ``AGENTS_LEARNING_DESIGN.md`` §5: task completion gets the largest
    weight because the action template directly controls it; intent
    resolution gets the smallest weight because it is computed by an
    upstream classifier the policy cannot influence.
    """

    intent_resolution_weight: float = field(
        default_factory=lambda: _env_float("AGENT_LEARNING_W_INTENT", 0.10)
    )
    task_adherence_weight: float = field(
        default_factory=lambda: _env_float("AGENT_LEARNING_W_ADHERENCE", 0.20)
    )
    task_completion_weight: float = field(
        default_factory=lambda: _env_float("AGENT_LEARNING_W_COMPLETION", 0.50)
    )
    latency_penalty_threshold_ms: int = field(
        default_factory=lambda: _env_int("AGENT_LEARNING_LATENCY_THRESHOLD_MS", 15000)
    )
    latency_penalty_value: float = field(
        default_factory=lambda: _env_float("AGENT_LEARNING_LATENCY_PENALTY", -0.1)
    )
    # Routing terms (§5.2). Computed in the orchestrator and passed to
    # ``RewardShaper.shape(..., routing_reward=..., routing_penalty=...,
    # hallucination_penalty=...)``. The values below are the magnitudes
    # used by the orchestrator when the relevant condition fires.
    route_correct_reward: float = field(
        default_factory=lambda: _env_float("AGENT_LEARNING_R_ROUTE_CORRECT", 0.20)
    )
    route_wrong_penalty: float = field(
        default_factory=lambda: _env_float("AGENT_LEARNING_P_ROUTE_WRONG", -0.30)
    )
    hallucinated_member_penalty: float = field(
        default_factory=lambda: _env_float("AGENT_LEARNING_P_HALLU", -0.25)
    )


# ---------------------------------------------------------------------------
# Learner
# ---------------------------------------------------------------------------


@dataclass
class LearnerConfig:
    """Hyperparameters for the REINFORCE-with-baseline learner."""

    learning_rate: float = field(default_factory=lambda: _env_float("AGENT_LEARNING_LR", 0.05))
    baseline_decay: float = field(default_factory=lambda: _env_float("AGENT_LEARNING_BASELINE_DECAY", 0.9))
    entropy_bonus: float = field(default_factory=lambda: _env_float("AGENT_LEARNING_ENTROPY_BONUS", 0.01))
    max_logit_abs: float = field(default_factory=lambda: _env_float("AGENT_LEARNING_MAX_LOGIT", 10.0))
    importance_clip: float = field(
        default_factory=lambda: _env_float("AGENT_LEARNING_IMPORTANCE_CLIP", 5.0)
    )


__all__ = [
    "CaptureConfig",
    "CosmosConfig",
    "JudgeConfig",
    "LearnerConfig",
    "ShapingConfig",
]
