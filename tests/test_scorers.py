"""Tests for the swappable scoring stack (`score_mode = "nlp" | "llm"`).

These cover:

* the env-driven default for :class:`ScoreRuntimeConfig`,
* the factory branches for each backend,
* the NLP backend round-trip (fit → score → save → reload),
* the LLM backend's contract when ``azure-ai-evaluation`` is absent,
* the projection helper that maps evaluator output to a ScoreResult.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Iterator

import pytest

from agent_learning.config import (
    NlpScoreConfig,
    ScoreConfig,
    ScoreRuntimeConfig,
)
from agent_learning.scorers import Scorer, ScoreResult, build_scorers
from agent_learning.scorers.llm._base import _project_to_score
from agent_learning.scorers.nlp import (
    NlpAdherenceScorer,
    NlpCompletionScorer,
    NlpIntentScorer,
)


@pytest.fixture
def nlp_cfg(tmp_path) -> NlpScoreConfig:
    return NlpScoreConfig(snapshot_dir=str(tmp_path), pass_threshold=0.5)


def test_runtime_config_defaults_to_llm(monkeypatch) -> None:
    monkeypatch.delenv("AGENT_LEARNING_SCORE_MODE", raising=False)
    cfg = ScoreRuntimeConfig()
    assert cfg.mode == "llm"


def test_runtime_config_respects_env(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_LEARNING_SCORE_MODE", "nlp")
    cfg = ScoreRuntimeConfig()
    assert cfg.mode == "nlp"


def test_runtime_config_rejects_unknown_mode(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_LEARNING_SCORE_MODE", "magic")
    cfg = ScoreRuntimeConfig()
    # Unknown values fall back to the documented default.
    assert cfg.mode == "llm"


def test_build_scorers_nlp_returns_three_scorers(nlp_cfg) -> None:
    cfg = ScoreRuntimeConfig(mode="nlp", nlp=nlp_cfg)
    intent, adherence, completion = build_scorers(cfg)
    assert isinstance(intent, NlpIntentScorer)
    assert isinstance(adherence, NlpAdherenceScorer)
    assert isinstance(completion, NlpCompletionScorer)
    for scorer in (intent, adherence, completion):
        assert isinstance(scorer, Scorer)


def test_build_scorers_unknown_mode_raises() -> None:
    cfg = ScoreRuntimeConfig()
    cfg.mode = "wat"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="unknown score_mode"):
        build_scorers(cfg)


def test_nlp_scorer_unfitted_returns_zero_probability(nlp_cfg) -> None:
    scorer = NlpIntentScorer.load_or_default(nlp_cfg)
    score = scorer.score(phi=[0.1, 0.2, 0.3], action_id="ack")
    assert isinstance(score, ScoreResult)
    assert score.label == "fail"
    assert score.normalized == 0.0
    assert score.features["fitted"] is False


def test_nlp_scorer_requires_phi_and_action_id(nlp_cfg) -> None:
    scorer = NlpIntentScorer.load_or_default(nlp_cfg)
    with pytest.raises(ValueError, match="phi and action_id"):
        scorer.score(phi=None, action_id="ack")


def test_nlp_scorer_fit_predict_separates_classes(nlp_cfg) -> None:
    rows = []
    # Action "good" is always positive, action "bad" is always negative.
    for _ in range(64):
        rows.append({"phi": [1.0, 0.0, 0.0], "action_id": "good", "label": 1})
        rows.append({"phi": [0.0, 1.0, 0.0], "action_id": "bad", "label": 0})
    scorer = NlpIntentScorer.load_or_default(nlp_cfg).fit(rows)
    good = scorer.score(phi=[1.0, 0.0, 0.0], action_id="good")
    bad = scorer.score(phi=[0.0, 1.0, 0.0], action_id="bad")
    assert good.label == "pass"
    assert bad.label == "fail"
    assert good.normalized > 0.5
    assert bad.normalized < 0.5


def test_nlp_scorer_save_and_reload_round_trips(nlp_cfg) -> None:
    rows = [
        {"phi": [1.0, 0.0], "action_id": "a", "label": 1},
        {"phi": [0.0, 1.0], "action_id": "b", "label": 0},
    ] * 32
    scorer = NlpIntentScorer.load_or_default(nlp_cfg).fit(rows)
    path = scorer.save(nlp_cfg.snapshot_dir)
    assert os.path.isfile(path)
    payload = json.loads(open(path, "r", encoding="utf-8").read())
    assert payload["label_name"] == "intent"
    assert len(payload["weights"]) > 0

    # A new instance loaded from the same dir should be already fitted.
    reloaded = NlpIntentScorer.load_or_default(nlp_cfg)
    reloaded_score = reloaded.score(phi=[1.0, 0.0], action_id="a")
    original_score = scorer.score(phi=[1.0, 0.0], action_id="a")
    assert reloaded_score.normalized == pytest.approx(original_score.normalized)
    assert reloaded_score.features["fitted"] is True


def test_llm_scorer_raises_clear_error_without_dependency(monkeypatch) -> None:
    """The LLM backend must surface a helpful ImportError when
    ``azure-ai-evaluation`` is not installed."""
    cfg = ScoreRuntimeConfig(mode="llm")
    intent, _, _ = build_scorers(cfg)

    # Force the lazy import to fail regardless of whether the optional
    # dependency happens to be present in this environment.
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("azure.ai.evaluation"):
            raise ImportError("simulated missing optional dep")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(ImportError, match="azure-ai-evaluation"):
        intent.score(query="hello", response="world")


def test_project_to_score_handles_likert_scale() -> None:
    result = {"score": 4.0, "reason": "looks good"}
    score = _project_to_score(result, threshold=0.5, name="intent")
    assert score.label == "pass"
    # 4 on a 1-5 scale maps to 0.75.
    assert score.normalized == pytest.approx(0.75)
    assert score.features["raw"] == 4.0


def test_project_to_score_handles_zero_to_one_score() -> None:
    result = {"intent_score": 0.2}
    score = _project_to_score(result, threshold=0.5, name="intent")
    assert score.label == "fail"
    assert score.normalized == pytest.approx(0.2)
    assert score.confidence == pytest.approx(0.8)


def test_project_to_score_rejects_non_mapping() -> None:
    with pytest.raises(TypeError):
        _project_to_score(0.5, threshold=0.5, name="intent")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ScoreConfig managed-identity / TokenCredential resolution
# ---------------------------------------------------------------------------


class _FakeCredential:
    """Stand-in for any ``azure.identity`` credential class."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def get_token(self, *scopes, **_kwargs):  # pragma: no cover - not exercised
        class _Token:
            token = "fake-token"

        return _Token()


class _FakeAzureIdentity:
    """Stub module that mimics the subset of ``azure.identity`` the SDK uses."""

    class DefaultAzureCredential(_FakeCredential):
        pass

    class ManagedIdentityCredential(_FakeCredential):
        pass

    class WorkloadIdentityCredential(_FakeCredential):
        pass

    class EnvironmentCredential(_FakeCredential):
        pass

    class AzureCliCredential(_FakeCredential):
        pass

    @staticmethod
    def get_bearer_token_provider(credential, scope):
        def _provider() -> str:
            return f"fake-token:{scope}"

        _provider.credential = credential  # type: ignore[attr-defined]
        _provider.scope = scope  # type: ignore[attr-defined]
        return _provider


@pytest.fixture
def fake_identity(monkeypatch) -> Iterator[_FakeAzureIdentity]:
    """Inject a fake ``azure.identity`` module into ``sys.modules``.

    Because ``azure-identity`` is already importable in this venv, we
    must replace both ``sys.modules["azure.identity"]`` AND the
    ``.identity`` attribute on the parent ``azure`` package. Otherwise
    ``import azure.identity as az_id`` would resolve via the attribute
    lookup and bypass the stub.
    """
    import azure  # type: ignore[import-not-found]

    fake = _FakeAzureIdentity()
    monkeypatch.setitem(sys.modules, "azure.identity", fake)
    monkeypatch.setattr(azure, "identity", fake, raising=False)
    yield fake


def test_score_config_no_credential_uses_api_key() -> None:
    """Default path stays unchanged: api_key is forwarded."""
    cfg = ScoreConfig(
        api_key="secret",
        azure_endpoint="https://x",
        azure_deployment="d",
        credential_mode=None,
    )
    out = cfg.to_model_config()
    assert out["api_key"] == "secret"
    assert "azure_ad_token_provider" not in out


def test_score_config_explicit_credential_wins_over_api_key(fake_identity) -> None:
    cred = _FakeCredential(tag="explicit")
    cfg = ScoreConfig(
        api_key="secret",
        credential=cred,
        azure_endpoint="https://x",
        azure_deployment="d",
    )
    out = cfg.to_model_config()
    assert "api_key" not in out
    provider = out["azure_ad_token_provider"]
    assert provider() == f"fake-token:{cfg.credential_scope}"
    assert provider.credential is cred


def test_score_config_mode_default_passes_managed_identity_client_id(fake_identity) -> None:
    cfg = ScoreConfig(
        credential_mode="default",
        user_assigned_client_id="abc-123",
        azure_endpoint="https://x",
        azure_deployment="d",
    )
    cred = cfg.resolve_credential()
    assert isinstance(cred, fake_identity.DefaultAzureCredential)
    assert cred.kwargs == {"managed_identity_client_id": "abc-123"}


def test_score_config_mode_managed_identity_with_client_id(fake_identity) -> None:
    cfg = ScoreConfig(
        credential_mode="managed-identity",
        user_assigned_client_id="abc-123",
    )
    cred = cfg.resolve_credential()
    assert isinstance(cred, fake_identity.ManagedIdentityCredential)
    assert cred.kwargs == {"client_id": "abc-123"}


def test_score_config_mode_workload_identity_no_client_id(fake_identity) -> None:
    cfg = ScoreConfig(credential_mode="workload-identity")
    cred = cfg.resolve_credential()
    assert isinstance(cred, fake_identity.WorkloadIdentityCredential)
    assert cred.kwargs == {}


def test_score_config_mode_environment_and_azure_cli(fake_identity) -> None:
    env_cfg = ScoreConfig(credential_mode="environment")
    cli_cfg = ScoreConfig(credential_mode="azure-cli")
    assert isinstance(env_cfg.resolve_credential(), fake_identity.EnvironmentCredential)
    assert isinstance(cli_cfg.resolve_credential(), fake_identity.AzureCliCredential)


def test_score_config_mode_none_falls_back_to_api_key(fake_identity) -> None:
    cfg = ScoreConfig(
        credential_mode="none",
        api_key="secret",
        azure_endpoint="https://x",
        azure_deployment="d",
    )
    assert cfg.resolve_credential() is None
    out = cfg.to_model_config()
    assert out["api_key"] == "secret"
    assert "azure_ad_token_provider" not in out


def test_score_config_unknown_credential_mode_raises(fake_identity) -> None:
    cfg = ScoreConfig(credential_mode="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="credential_mode"):
        cfg.resolve_credential()


def test_score_config_credential_mode_missing_azure_identity(monkeypatch) -> None:
    """When azure-identity is not importable, resolve_credential raises a
    helpful ImportError."""
    real_import = (
        __builtins__["__import__"]
        if isinstance(__builtins__, dict)
        else __import__
    )

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("azure.identity"):
            raise ImportError("simulated missing azure-identity")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)
    cfg = ScoreConfig(credential_mode="default")
    with pytest.raises(ImportError, match="azure-identity"):
        cfg.resolve_credential()


def test_score_config_env_var_drives_credential_mode(monkeypatch, fake_identity) -> None:
    monkeypatch.setenv("AGENT_LEARNING_SCORE_CREDENTIAL_MODE", "azure-cli")
    cfg = ScoreConfig()
    assert cfg.credential_mode == "azure-cli"
    assert isinstance(cfg.resolve_credential(), fake_identity.AzureCliCredential)


def test_score_config_env_var_invalid_mode_falls_back_to_none(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_LEARNING_SCORE_CREDENTIAL_MODE", "garbage")
    cfg = ScoreConfig()
    assert cfg.credential_mode is None
    assert cfg.resolve_credential() is None
