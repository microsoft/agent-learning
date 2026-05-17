"""Public judge interface for the agent-learning SDK.

Two selectors on :class:`agent_learning.config.JudgeRuntimeConfig`
decide which backend ``build_judges`` returns:

- ``cfg.tier`` (preferred, new). One of:
    - ``"stdlib"``: Tier 1, pure-stdlib text judges. Zero external
      dependencies.
    - ``"nlp"``: Tier 2, in-SDK feature-based judges over
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

from ..config import JudgeRuntimeConfig
from .base import Judge, JudgeScore


def _build_stdlib(cfg: JudgeRuntimeConfig) -> Tuple[Judge, Judge, Judge]:
    from .stdlib import (
        StdlibAdherenceJudge,
        StdlibCompletionJudge,
        StdlibIntentJudge,
    )
    return (
        StdlibIntentJudge.load_or_default(
            cfg.stdlib.snapshot_dir,
            feature_dim=cfg.stdlib.feature_dim,
            pass_threshold=cfg.stdlib.pass_threshold,
        ),
        StdlibAdherenceJudge.load_or_default(
            cfg.stdlib.snapshot_dir,
            pass_threshold=cfg.stdlib.pass_threshold,
        ),
        StdlibCompletionJudge.load_or_default(
            cfg.stdlib.snapshot_dir,
            pass_threshold=cfg.stdlib.pass_threshold,
        ),
    )


def _build_nlp(cfg: JudgeRuntimeConfig) -> Tuple[Judge, Judge, Judge]:
    """Tier 2: TF-IDF + scikit-learn judges over query/response text.

    Requires the ``[nlp]`` extra. The judges are imported lazily so
    callers that never select ``tier="nlp"`` don't need scikit-learn
    installed.
    """
    from .nlp_text import (
        NlpTextAdherenceJudge,
        NlpTextCompletionJudge,
        NlpTextIntentJudge,
    )
    nlp_text_cfg = cfg.nlp_text
    return (
        NlpTextIntentJudge.load_or_default(
            nlp_text_cfg.snapshot_dir,
            pass_threshold=nlp_text_cfg.pass_threshold,
            max_features=nlp_text_cfg.max_features,
            ngram_min=nlp_text_cfg.ngram_min,
            ngram_max=nlp_text_cfg.ngram_max,
        ),
        NlpTextAdherenceJudge.load_or_default(
            nlp_text_cfg.snapshot_dir,
            pass_threshold=nlp_text_cfg.pass_threshold,
            max_features=nlp_text_cfg.max_features,
            ngram_min=nlp_text_cfg.ngram_min,
            ngram_max=nlp_text_cfg.ngram_max,
        ),
        NlpTextCompletionJudge.load_or_default(
            nlp_text_cfg.snapshot_dir,
            pass_threshold=nlp_text_cfg.pass_threshold,
            max_features=nlp_text_cfg.max_features,
            ngram_min=nlp_text_cfg.ngram_min,
            ngram_max=nlp_text_cfg.ngram_max,
        ),
    )


def _build_nlp_legacy(cfg: JudgeRuntimeConfig) -> Tuple[Judge, Judge, Judge]:
    """Back-compat path for callers using ``mode="nlp"``.

    Routes to the original :class:`agent_learning.classifiers.judges.BinaryJudge`
    stack over ``(phi, action_id)``. Preserves the v0.1 API for callers
    that haven't migrated to the tier-based selector yet.
    """
    from .nlp import (
        NlpAdherenceJudge,
        NlpCompletionJudge,
        NlpIntentJudge,
    )
    return (
        NlpIntentJudge.load_or_default(cfg.nlp),
        NlpAdherenceJudge.load_or_default(cfg.nlp),
        NlpCompletionJudge.load_or_default(cfg.nlp),
    )


def _build_slm(cfg: JudgeRuntimeConfig) -> Tuple[Judge, Judge, Judge]:
    """Tier 3: Phi-4-mini-instruct INT4 ONNX judges.

    Requires the ``[slm]`` extra and a local copy of the
    Phi-4-mini-instruct INT4 ONNX bundle. Judges are imported lazily so
    callers that never select ``tier="slm"`` don't need
    ``onnxruntime-genai`` installed.
    """
    from .slm import (
        SlmAdherenceJudge,
        SlmCompletionJudge,
        SlmIntentJudge,
    )
    slm_cfg = cfg.slm
    return (
        SlmIntentJudge.load_or_default(
            slm_cfg.model_dir,
            pass_threshold=slm_cfg.pass_threshold,
            max_new_tokens=slm_cfg.max_new_tokens,
            temperature=slm_cfg.temperature,
        ),
        SlmAdherenceJudge.load_or_default(
            slm_cfg.model_dir,
            pass_threshold=slm_cfg.pass_threshold,
            max_new_tokens=slm_cfg.max_new_tokens,
            temperature=slm_cfg.temperature,
        ),
        SlmCompletionJudge.load_or_default(
            slm_cfg.model_dir,
            pass_threshold=slm_cfg.pass_threshold,
            max_new_tokens=slm_cfg.max_new_tokens,
            temperature=slm_cfg.temperature,
        ),
    )


def _build_llm(cfg: JudgeRuntimeConfig) -> Tuple[Judge, Judge, Judge]:
    from .llm import (
        LlmAdherenceJudge,
        LlmCompletionJudge,
        LlmIntentJudge,
    )
    return (
        LlmIntentJudge(cfg.llm),
        LlmAdherenceJudge(cfg.llm),
        LlmCompletionJudge(cfg.llm),
    )


def build_judges(cfg: JudgeRuntimeConfig) -> Tuple[Judge, Judge, Judge]:
    """Return the ``(intent, adherence, completion)`` judge trio.

    Routing order: ``cfg.tier`` if set, else ``cfg.mode``.
    """
    tier = cfg.tier
    if tier is None:
        # Legacy mode fallback. mode="nlp" routes to the BinaryJudge
        # stack, NOT the new TF-IDF judges (those are reachable via
        # tier="nlp").
        if cfg.mode == "nlp":
            return _build_nlp_legacy(cfg)
        if cfg.mode == "llm":
            return _build_llm(cfg)
        raise ValueError(f"unknown judge_mode: {cfg.mode!r}")
    if tier == "stdlib":
        return _build_stdlib(cfg)
    if tier == "nlp":
        return _build_nlp(cfg)
    if tier == "slm":
        return _build_slm(cfg)
    if tier == "llm":
        return _build_llm(cfg)
    raise ValueError(f"unknown judge tier: {tier!r}")


__all__ = ["Judge", "JudgeScore", "build_judges"]
