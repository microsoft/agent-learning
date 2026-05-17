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
    raise NotImplementedError(
        "Tier 3 (slm) judges are not yet packaged. Install the [slm] "
        "extra and select tier='slm' when the package ships."
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
        # Legacy mode fallback.
        if cfg.mode == "nlp":
            return _build_nlp(cfg)
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
