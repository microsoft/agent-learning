"""Public scorer interface for the agent-learning SDK.

Two selectors on :class:`agent_learning.config.ScoreRuntimeConfig`
decide which backend ``build_scorers`` returns:

- ``cfg.tier`` (preferred, new). One of:
        - ``"stdlib"``: Tier 1, pure-stdlib text scorers. Zero external
      dependencies.
        - ``"nlp"``: Tier 2, in-SDK feature-based scorers over
      ``(phi, action_id)``. Currently routed to the existing
      :mod:`.nlp` package.
    - ``"slm"``: Tier 3, Microsoft Phi-4-mini-instruct via the
      ``[slm]`` extra. Stub raises until the extra is wired.
    - ``"llm"``: Tier 4, ``azure-ai-evaluation`` evaluators behind
      Azure OpenAI. Requires the ``[llm]`` extra.
- ``cfg.mode`` (legacy). Used only when ``cfg.tier`` is None. Values
  ``"nlp"`` and ``"llm"`` map to the same backends Tier 2 and Tier 4
  resolve to, preserving the v0.1 API.

Callers never branch on the tier themselves; the factory hides the
choice so the reward shaper and learner stay backend-agnostic.
"""

from __future__ import annotations

from typing import Tuple

from ..config import ScoreRuntimeConfig
from .base import Scorer, ScoreResult


def _build_stdlib(cfg: ScoreRuntimeConfig) -> Tuple[Scorer, Scorer, Scorer]:
    from .stdlib import (
        StdlibAdherenceScorer,
        StdlibCompletionScorer,
        StdlibIntentScorer,
    )
    return (
        StdlibIntentScorer.load_or_default(
            cfg.stdlib.snapshot_dir,
            feature_dim=cfg.stdlib.feature_dim,
            pass_threshold=cfg.stdlib.pass_threshold,
        ),
        StdlibAdherenceScorer.load_or_default(
            cfg.stdlib.snapshot_dir,
            pass_threshold=cfg.stdlib.pass_threshold,
        ),
        StdlibCompletionScorer.load_or_default(
            cfg.stdlib.snapshot_dir,
            pass_threshold=cfg.stdlib.pass_threshold,
        ),
    )


def _build_nlp(cfg: ScoreRuntimeConfig) -> Tuple[Scorer, Scorer, Scorer]:
    """Tier 2: TF-IDF + scikit-learn scorers over query/response text.

    Requires the ``[nlp]`` extra. The scorers are imported lazily so
    callers that never select ``tier="nlp"`` don't need scikit-learn
    installed.
    """
    from .nlp_text import (
        NlpTextAdherenceScorer,
        NlpTextCompletionScorer,
        NlpTextIntentScorer,
    )
    nlp_text_cfg = cfg.nlp_text
    return (
        NlpTextIntentScorer.load_or_default(
            nlp_text_cfg.snapshot_dir,
            pass_threshold=nlp_text_cfg.pass_threshold,
            max_features=nlp_text_cfg.max_features,
            ngram_min=nlp_text_cfg.ngram_min,
            ngram_max=nlp_text_cfg.ngram_max,
        ),
        NlpTextAdherenceScorer.load_or_default(
            nlp_text_cfg.snapshot_dir,
            pass_threshold=nlp_text_cfg.pass_threshold,
            max_features=nlp_text_cfg.max_features,
            ngram_min=nlp_text_cfg.ngram_min,
            ngram_max=nlp_text_cfg.ngram_max,
        ),
        NlpTextCompletionScorer.load_or_default(
            nlp_text_cfg.snapshot_dir,
            pass_threshold=nlp_text_cfg.pass_threshold,
            max_features=nlp_text_cfg.max_features,
            ngram_min=nlp_text_cfg.ngram_min,
            ngram_max=nlp_text_cfg.ngram_max,
        ),
    )


def _build_nlp_legacy(cfg: ScoreRuntimeConfig) -> Tuple[Scorer, Scorer, Scorer]:
    """Back-compat path for callers using ``mode="nlp"``.

    Routes to the original :class:`agent_learning.classifiers.scorers.BinaryScorer`
    stack over ``(phi, action_id)``. Preserves the v0.1 API for callers
    that haven't migrated to the tier-based selector yet.
    """
    from .nlp import (
        NlpAdherenceScorer,
        NlpCompletionScorer,
        NlpIntentScorer,
    )
    return (
        NlpIntentScorer.load_or_default(cfg.nlp),
        NlpAdherenceScorer.load_or_default(cfg.nlp),
        NlpCompletionScorer.load_or_default(cfg.nlp),
    )


def _build_slm(cfg: ScoreRuntimeConfig) -> Tuple[Scorer, Scorer, Scorer]:
    """Tier 3: Phi-4-mini-instruct INT4 ONNX scorers.

    Requires the ``[slm]`` extra and a local copy of the
    Phi-4-mini-instruct INT4 ONNX bundle. Scorers are imported lazily so
    callers that never select ``tier="slm"`` don't need
    ``onnxruntime-genai`` installed.
    """
    from .slm import (
        SlmAdherenceScorer,
        SlmCompletionScorer,
        SlmIntentScorer,
    )
    slm_cfg = cfg.slm
    return (
        SlmIntentScorer.load_or_default(
            slm_cfg.model_dir,
            pass_threshold=slm_cfg.pass_threshold,
            max_new_tokens=slm_cfg.max_new_tokens,
            temperature=slm_cfg.temperature,
        ),
        SlmAdherenceScorer.load_or_default(
            slm_cfg.model_dir,
            pass_threshold=slm_cfg.pass_threshold,
            max_new_tokens=slm_cfg.max_new_tokens,
            temperature=slm_cfg.temperature,
        ),
        SlmCompletionScorer.load_or_default(
            slm_cfg.model_dir,
            pass_threshold=slm_cfg.pass_threshold,
            max_new_tokens=slm_cfg.max_new_tokens,
            temperature=slm_cfg.temperature,
        ),
    )


def _build_llm(cfg: ScoreRuntimeConfig) -> Tuple[Scorer, Scorer, Scorer]:
    from .llm import (
        LlmAdherenceScorer,
        LlmCompletionScorer,
        LlmIntentScorer,
    )
    return (
        LlmIntentScorer(cfg.llm),
        LlmAdherenceScorer(cfg.llm),
        LlmCompletionScorer(cfg.llm),
    )


def build_scorers(cfg: ScoreRuntimeConfig) -> Tuple[Scorer, Scorer, Scorer]:
    """Return the ``(intent, adherence, completion)`` scorer trio.

    Routing order: ``cfg.tier`` if set, else ``cfg.mode``.
    """
    tier = cfg.tier
    if tier is None:
        # Legacy mode fallback. mode="nlp" routes to the BinaryScorer
        # stack, NOT the new TF-IDF scorers (those are reachable via
        # tier="nlp").
        if cfg.mode == "nlp":
            return _build_nlp_legacy(cfg)
        if cfg.mode == "llm":
            return _build_llm(cfg)
        raise ValueError(f"unknown score_mode: {cfg.mode!r}")
    if tier == "stdlib":
        return _build_stdlib(cfg)
    if tier == "nlp":
        return _build_nlp(cfg)
    if tier == "slm":
        return _build_slm(cfg)
    if tier == "llm":
        return _build_llm(cfg)
    raise ValueError(f"unknown score tier: {tier!r}")


__all__ = ["Scorer", "ScoreResult", "build_scorers"]
