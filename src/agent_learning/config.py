"""Environment-driven configuration for the agent-learning SDK.

All settings can be overridden via environment variables; programmatic
overrides are also accepted via dataclass field assignment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

ScoreMode = Literal["nlp", "llm"]
ScoreTier = Literal["stdlib", "nlp", "slm", "llm"]
AutonomyTier = Literal["low", "standard", "high", "critical"]
CredentialMode = Literal[
    "default",
    "managed-identity",
    "workload-identity",
    "environment",
    "azure-cli",
    "none",
]
_VALID_CREDENTIAL_MODES: frozenset[str] = frozenset(
    {"default", "managed-identity", "workload-identity", "environment", "azure-cli", "none"}
)
_VALID_SCORE_TIERS: frozenset[str] = frozenset(
    {"stdlib", "nlp", "slm", "llm"}
)
_AUTONOMY_TIER_DEFAULTS: dict[str, dict[str, float | int]] = {
    "low": {
        "min_outcomes": 3,
        "min_correctness_lower_bound": 0.40,
        "min_mean_reward": 0.0,
        "min_action_probability": 0.0,
        "min_probability_margin": 0.0,
        "stable_snapshots": 1,
        "audit_rate": 0.05,
    },
    "standard": {
        "min_outcomes": 20,
        "min_correctness_lower_bound": 0.90,
        "min_mean_reward": 0.0,
        "min_action_probability": 0.60,
        "min_probability_margin": 0.15,
        "stable_snapshots": 3,
        "audit_rate": 0.10,
    },
    "high": {
        "min_outcomes": 50,
        "min_correctness_lower_bound": 0.95,
        "min_mean_reward": 0.10,
        "min_action_probability": 0.75,
        "min_probability_margin": 0.30,
        "stable_snapshots": 4,
        "audit_rate": 0.25,
    },
    "critical": {
        "min_outcomes": 100,
        "min_correctness_lower_bound": 0.975,
        "min_mean_reward": 0.20,
        "min_action_probability": 0.85,
        "min_probability_margin": 0.45,
        "stable_snapshots": 5,
        "audit_rate": 0.50,
    },
}


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable using common conventions."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_credential_mode() -> Optional[CredentialMode]:
    """Read AGENT_LEARNING_SCORE_CREDENTIAL_MODE and validate."""
    raw = os.getenv("AGENT_LEARNING_SCORE_CREDENTIAL_MODE")
    if raw is None or raw == "":
        return None
    value = raw.strip().lower()
    if value in _VALID_CREDENTIAL_MODES:
        return value  # type: ignore[return-value]
    return None


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
# Score / evaluator model
# ---------------------------------------------------------------------------


@dataclass
class ScoreConfig:
    """Configuration for the LLM scorer used by metric evaluators.

    Authentication paths, in priority order:

    1. ``credential`` set explicitly to a ``TokenCredential`` instance.
       Wins over every other field. The SDK passes a bearer-token
       provider built from it to the underlying evaluator.
    2. ``credential_mode`` set to one of ``"default"``,
       ``"managed-identity"``, ``"workload-identity"``,
       ``"environment"``, ``"azure-cli"`` (or ``"none"`` to opt out
       even when an api_key is also present). The SDK lazy-imports
       ``azure-identity`` and builds the matching credential.
       ``user_assigned_client_id`` is forwarded when provided.
    3. ``api_key`` (legacy path). Used only when no credential is
       resolved.

    Environment variables ``AGENT_LEARNING_SCORE_CREDENTIAL_MODE`` and
    ``AGENT_LEARNING_SCORE_USER_ASSIGNED_CLIENT_ID`` mirror the two
    string fields so the same SDK build switches between developer
    laptop, CI, and production AKS without code changes.
    """

    azure_endpoint: str = field(default_factory=lambda: os.getenv("AGENT_LEARNING_SCORE_ENDPOINT", ""))
    azure_deployment: str = field(default_factory=lambda: os.getenv("AGENT_LEARNING_SCORE_DEPLOYMENT", ""))
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("AGENT_LEARNING_SCORE_API_KEY") or None)
    api_version: str = field(default_factory=lambda: os.getenv("AGENT_LEARNING_SCORE_API_VERSION", "2024-10-21"))
    # Pass-through threshold for IntentResolutionEvaluator (defaults to its own default)
    threshold: int = field(default_factory=lambda: _env_int("AGENT_LEARNING_INTENT_THRESHOLD", 3))
    # Managed-identity / TokenCredential authentication (optional).
    credential: Optional[Any] = field(default=None, repr=False)
    credential_mode: Optional[CredentialMode] = field(default_factory=_env_credential_mode)
    user_assigned_client_id: Optional[str] = field(
        default_factory=lambda: os.getenv("AGENT_LEARNING_SCORE_USER_ASSIGNED_CLIENT_ID") or None
    )
    # Azure AD token scope for the resolved credential. Defaults to the
    # Cognitive Services scope used by Azure OpenAI and Azure AI Foundry.
    credential_scope: str = field(
        default_factory=lambda: os.getenv(
            "AGENT_LEARNING_SCORE_CREDENTIAL_SCOPE",
            "https://cognitiveservices.azure.com/.default",
        )
    )

    @property
    def enabled(self) -> bool:
        return bool(self.azure_endpoint and self.azure_deployment)

    def resolve_credential(self) -> Optional[Any]:
        """Resolve the active TokenCredential, or ``None``.

        Returns the explicit ``credential`` field when set. Otherwise
        builds a credential from ``credential_mode`` via a lazy import
        of ``azure-identity``. Returns ``None`` when no credential is
        configured (caller falls back to ``api_key``).
        """
        if self.credential is not None:
            return self.credential
        mode = self.credential_mode
        if mode is None or mode == "none":
            return None
        try:
            import azure.identity as az_id  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Credential-based auth for the LLM scorer requires the optional "
                "'azure-identity' package. Install it with: pip install azure-identity"
            ) from exc
        if mode == "default":
            kwargs: dict = {}
            if self.user_assigned_client_id:
                kwargs["managed_identity_client_id"] = self.user_assigned_client_id
            return az_id.DefaultAzureCredential(**kwargs)
        if mode == "managed-identity":
            if self.user_assigned_client_id:
                return az_id.ManagedIdentityCredential(client_id=self.user_assigned_client_id)
            return az_id.ManagedIdentityCredential()
        if mode == "workload-identity":
            if self.user_assigned_client_id:
                return az_id.WorkloadIdentityCredential(client_id=self.user_assigned_client_id)
            return az_id.WorkloadIdentityCredential()
        if mode == "environment":
            return az_id.EnvironmentCredential()
        if mode == "azure-cli":
            return az_id.AzureCliCredential()
        raise ValueError(f"Unknown credential_mode: {mode!r}")

    def to_model_config(self) -> dict:
        """Build the dict expected by azure-ai-evaluation evaluators.

        When a TokenCredential is resolved, the dict carries an
        ``azure_ad_token_provider`` callable (built via
        ``azure.identity.get_bearer_token_provider``) and the
        ``api_key`` field is omitted. Otherwise the legacy ``api_key``
        path is used.
        """
        cfg: dict = {
            "azure_endpoint": self.azure_endpoint,
            "azure_deployment": self.azure_deployment,
            "api_version": self.api_version,
        }
        credential = self.resolve_credential()
        if credential is not None:
            try:
                from azure.identity import (  # type: ignore[import-not-found]
                    get_bearer_token_provider,
                )
            except ImportError as exc:
                raise ImportError(
                    "Credential-based auth for the LLM scorer requires the optional "
                    "'azure-identity' package. Install it with: pip install azure-identity"
                ) from exc
            cfg["azure_ad_token_provider"] = get_bearer_token_provider(
                credential, self.credential_scope
            )
        elif self.api_key:
            cfg["api_key"] = self.api_key
        return cfg


@dataclass
class NlpScoreConfig:
    """Configuration for the in-SDK NLP scoring stack (Tier 0, pure stdlib).

    The NLP scorers read their fitted weights from ``snapshot_dir`` at
    load time and fall back to an unfitted (always-pass) policy when no
    snapshot is present. Optional dependencies (scikit-learn,
    sentence-transformers, etc.) are detected at import time. Tier 0
    requires nothing beyond the Python standard library.
    """

    snapshot_dir: str = field(
        default_factory=lambda: os.getenv(
            "AGENT_LEARNING_NLP_SCORE_DIR", "./data/agent-learning/nlp-scores"
        )
    )
    pass_threshold: float = field(
        default_factory=lambda: _env_float("AGENT_LEARNING_NLP_PASS_THRESHOLD", 0.5)
    )
    enable_semantic: bool = field(
        default_factory=lambda: _env_bool("AGENT_LEARNING_NLP_SEMANTIC", False)
    )


def _env_score_mode(default: ScoreMode = "llm") -> ScoreMode:
    raw = os.getenv("AGENT_LEARNING_SCORE_MODE")
    if raw is None or raw == "":
        return default
    value = raw.strip().lower()
    if value in ("nlp", "llm"):
        return value  # type: ignore[return-value]
    return default


def _env_score_tier() -> Optional[ScoreTier]:
    """Read AGENT_LEARNING_SCORE_TIER and validate.

    Returns one of ``"stdlib"``, ``"nlp"``, ``"slm"``, ``"llm"`` or
    ``None`` when the env var is unset or invalid. The factory in
    scoring factory falls back to ``mode`` when ``tier``
    is None.
    """
    raw = os.getenv("AGENT_LEARNING_SCORE_TIER")
    if raw is None or raw == "":
        return None
    value = raw.strip().lower()
    if value in _VALID_SCORE_TIERS:
        return value  # type: ignore[return-value]
    return None


@dataclass
class StdlibScoreConfig:
    """Configuration for the Tier 1 stdlib scorers.

    All Tier 1 scorers run on the Python standard library alone. The
    intent scorer optionally loads fitted bag-of-words weights from
    ``snapshot_dir``; the adherence and completion scorers are pure
    rule engines with no persisted state.
    """

    snapshot_dir: str = field(
        default_factory=lambda: os.getenv(
            "AGENT_LEARNING_STDLIB_SCORE_DIR",
            "./data/agent-learning/stdlib-scores",
        )
    )
    pass_threshold: float = field(
        default_factory=lambda: _env_float(
            "AGENT_LEARNING_STDLIB_PASS_THRESHOLD", 0.5
        )
    )
    feature_dim: int = field(
        default_factory=lambda: _env_int(
            "AGENT_LEARNING_STDLIB_FEATURE_DIM", 1024
        )
    )


@dataclass
class NlpTextScoreConfig:
    """Configuration for the Tier 2 NLP text scorers.

    Tier 2 wraps a TF-IDF vectorizer + scikit-learn logistic
    regression around the response text. The fitted vectorizer and
    classifier are persisted to ``{snapshot_dir}/{name}.nlp_text.joblib``
    with a sibling JSON header at ``{name}.nlp_text.json``. When no
    snapshot is present the scorers fall back to a rule-engine signal
    only (adherence, completion) or the pass threshold (intent).

    Requires the ``[nlp]`` extra (``pip install
    agent-learning[nlp]``).
    """

    snapshot_dir: str = field(
        default_factory=lambda: os.getenv(
            "AGENT_LEARNING_NLP_TEXT_SCORE_DIR",
            "./data/agent-learning/nlp-text-scores",
        )
    )
    pass_threshold: float = field(
        default_factory=lambda: _env_float(
            "AGENT_LEARNING_NLP_TEXT_PASS_THRESHOLD", 0.5
        )
    )
    max_features: int = field(
        default_factory=lambda: _env_int(
            "AGENT_LEARNING_NLP_TEXT_MAX_FEATURES", 20000
        )
    )
    ngram_min: int = field(
        default_factory=lambda: _env_int("AGENT_LEARNING_NLP_TEXT_NGRAM_MIN", 1)
    )
    ngram_max: int = field(
        default_factory=lambda: _env_int("AGENT_LEARNING_NLP_TEXT_NGRAM_MAX", 2)
    )


@dataclass
class SlmScoreConfig:
    """Configuration for the Tier 3 small-language-model scorers.

    Tier 3 wraps a locally-hosted instance of Microsoft
    Phi-4-mini-instruct (3.8 B parameters, 4-bit ONNX) via
    ``onnxruntime-genai``. The model is loaded from ``model_dir`` on
    first use and reused across the three scorers in the same process.

    Requires the ``[slm]`` extra (``pip install
    agent-learning[slm]``). The default ``model_dir``
    expects a Phi-4-mini-instruct INT4 ONNX bundle laid out under
    ``./models/phi-4-mini-instruct-int4-onnx``; set
    ``AGENT_LEARNING_SLM_MODEL_DIR`` to point at any local path.
    """

    model_dir: str = field(
        default_factory=lambda: os.getenv(
            "AGENT_LEARNING_SLM_MODEL_DIR",
            "./models/phi-4-mini-instruct-int4-onnx",
        )
    )
    pass_threshold: float = field(
        default_factory=lambda: _env_float(
            "AGENT_LEARNING_SLM_PASS_THRESHOLD", 0.5
        )
    )
    max_new_tokens: int = field(
        default_factory=lambda: _env_int(
            "AGENT_LEARNING_SLM_MAX_NEW_TOKENS", 64
        )
    )
    temperature: float = field(
        default_factory=lambda: _env_float(
            "AGENT_LEARNING_SLM_TEMPERATURE", 0.0
        )
    )


@dataclass
class ScoreRuntimeConfig:
    """Top-level switch across the four scoring tiers.

    Two selectors exist, in priority order:

    1. ``tier`` (preferred). One of ``"stdlib"`` (Tier 1, zero deps),
       ``"nlp"`` (Tier 2, scikit-learn + rapidfuzz, ``[nlp]`` extra),
       ``"slm"`` (Tier 3, Phi-4-mini-instruct ONNX, ``[slm]`` extra),
       or ``"llm"`` (Tier 4, azure-ai-evaluation, ``[llm]`` extra).
    2. ``mode`` (legacy). One of ``"nlp"`` (the existing phi+action_id
    binary scoring stack) or ``"llm"`` (Azure AI Evaluation).
       Used when ``tier`` is ``None``.

    The default ``mode`` stays at ``"llm"`` for backwards compatibility
    with callers already wired through azure-ai-evaluation. Set
    ``AGENT_LEARNING_SCORE_TIER=stdlib`` to opt in to the new Tier 1
    text-based scorers.
    """

    mode: ScoreMode = field(default_factory=_env_score_mode)
    tier: Optional[ScoreTier] = field(default_factory=_env_score_tier)
    stdlib: StdlibScoreConfig = field(default_factory=StdlibScoreConfig)
    nlp: NlpScoreConfig = field(default_factory=NlpScoreConfig)
    nlp_text: NlpTextScoreConfig = field(default_factory=NlpTextScoreConfig)
    slm: SlmScoreConfig = field(default_factory=SlmScoreConfig)
    llm: ScoreConfig = field(default_factory=ScoreConfig)


# ---------------------------------------------------------------------------
# Decision autonomy
# ---------------------------------------------------------------------------


@dataclass
class AutonomyConfig:
    """Evidence thresholds for executing a learned decision autonomously."""

    min_outcomes: int = field(
        default_factory=lambda: _env_int("AGENT_LEARNING_AUTONOMY_MIN_OUTCOMES", 20)
    )
    min_correctness_lower_bound: float = field(
        default_factory=lambda: _env_float(
            "AGENT_LEARNING_AUTONOMY_MIN_CORRECTNESS_LOWER_BOUND", 0.90
        )
    )
    min_mean_reward: float = field(
        default_factory=lambda: _env_float(
            "AGENT_LEARNING_AUTONOMY_MIN_MEAN_REWARD", 0.0
        )
    )
    min_action_probability: float = field(
        default_factory=lambda: _env_float(
            "AGENT_LEARNING_AUTONOMY_MIN_ACTION_PROBABILITY", 0.60
        )
    )
    min_probability_margin: float = field(
        default_factory=lambda: _env_float(
            "AGENT_LEARNING_AUTONOMY_MIN_PROBABILITY_MARGIN", 0.15
        )
    )
    stable_snapshots: int = field(
        default_factory=lambda: _env_int(
            "AGENT_LEARNING_AUTONOMY_STABLE_SNAPSHOTS", 3
        )
    )
    audit_rate: float = field(
        default_factory=lambda: _env_float(
            "AGENT_LEARNING_AUTONOMY_AUDIT_RATE", 0.10
        )
    )
    wilson_z: float = field(
        default_factory=lambda: _env_float("AGENT_LEARNING_AUTONOMY_WILSON_Z", 1.96)
    )

    @classmethod
    def for_tier(cls, tier: AutonomyTier) -> "AutonomyConfig":
        """Resolve tier defaults, then apply global environment overrides."""
        defaults = _AUTONOMY_TIER_DEFAULTS[tier]
        return cls(
            min_outcomes=_env_int(
                "AGENT_LEARNING_AUTONOMY_MIN_OUTCOMES",
                int(defaults["min_outcomes"]),
            ),
            min_correctness_lower_bound=_env_float(
                "AGENT_LEARNING_AUTONOMY_MIN_CORRECTNESS_LOWER_BOUND",
                float(defaults["min_correctness_lower_bound"]),
            ),
            min_mean_reward=_env_float(
                "AGENT_LEARNING_AUTONOMY_MIN_MEAN_REWARD",
                float(defaults["min_mean_reward"]),
            ),
            min_action_probability=_env_float(
                "AGENT_LEARNING_AUTONOMY_MIN_ACTION_PROBABILITY",
                float(defaults["min_action_probability"]),
            ),
            min_probability_margin=_env_float(
                "AGENT_LEARNING_AUTONOMY_MIN_PROBABILITY_MARGIN",
                float(defaults["min_probability_margin"]),
            ),
            stable_snapshots=_env_int(
                "AGENT_LEARNING_AUTONOMY_STABLE_SNAPSHOTS",
                int(defaults["stable_snapshots"]),
            ),
            audit_rate=_env_float(
                "AGENT_LEARNING_AUTONOMY_AUDIT_RATE",
                float(defaults["audit_rate"]),
            ),
            wilson_z=_env_float("AGENT_LEARNING_AUTONOMY_WILSON_Z", 1.96),
        )

    def __post_init__(self) -> None:
        if self.min_outcomes < 1:
            raise ValueError("min_outcomes must be at least 1")
        if self.stable_snapshots < 1:
            raise ValueError("stable_snapshots must be at least 1")
        for name in (
            "min_correctness_lower_bound",
            "min_action_probability",
            "min_probability_margin",
            "audit_rate",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if not -1.0 <= self.min_mean_reward <= 1.0:
            raise ValueError("min_mean_reward must be between -1 and 1")
        if self.wilson_z <= 0.0:
            raise ValueError("wilson_z must be positive")


# ---------------------------------------------------------------------------
# Capture / shaping
# ---------------------------------------------------------------------------


@dataclass
class CaptureConfig:
    """Configuration for episode capture."""

    enabled: bool = field(default_factory=lambda: _env_bool("AGENT_LEARNING_ENABLE_CAPTURE", False))
    agent_id: str = field(default_factory=lambda: os.getenv("AGENT_LEARNING_AGENT_ID", "default"))
    agent_name: Optional[str] = field(
        default_factory=lambda: os.getenv("AGENT_LEARNING_AGENT_NAME") or None
    )
    task_id: str = field(default_factory=lambda: os.getenv("AGENT_LEARNING_TASK_ID", "default"))
    task_name: Optional[str] = field(
        default_factory=lambda: os.getenv("AGENT_LEARNING_TASK_NAME") or None
    )
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

    The defaults bias the credit-assignment toward task completion (the
    metric the action template directly controls) and away from intent
    resolution (typically computed by an upstream classifier the policy
    cannot influence).
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
    # Routing terms. Computed by the calling system and passed to
    # ``RewardShaper.shape(..., routing_correct=..., hallucinated_class=...)``.
    # The values below are the magnitudes used when the relevant condition
    # fires.
    route_correct_reward: float = field(
        default_factory=lambda: _env_float("AGENT_LEARNING_R_ROUTE_CORRECT", 0.20)
    )
    route_wrong_penalty: float = field(
        default_factory=lambda: _env_float("AGENT_LEARNING_P_ROUTE_WRONG", -0.30)
    )
    hallucinated_class_penalty: float = field(
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
    min_train_episodes: int = field(
        default_factory=lambda: _env_int("AGENT_LEARNING_MIN_TRAIN_EPISODES", 5)
    )


__all__ = [
    "AutonomyConfig",
    "AutonomyTier",
    "CaptureConfig",
    "CosmosConfig",
    "CredentialMode",
    "ScoreConfig",
    "ScoreMode",
    "ScoreRuntimeConfig",
    "ScoreTier",
    "LearnerConfig",
    "NlpScoreConfig",
    "NlpTextScoreConfig",
    "ShapingConfig",
    "SlmScoreConfig",
    "StdlibScoreConfig",
]
