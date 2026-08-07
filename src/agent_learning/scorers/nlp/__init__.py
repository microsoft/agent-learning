"""Native, in-process scorers (Tier 0, pure standard library).

These wrap the existing :class:`agent_learning.classifiers.scorers.BinaryScorer`
implementations so the SDK can ship a working scorer stack with no
external service dependencies. The wrappers translate the underlying
:class:`agent_learning.classifiers.base.ClassifierResult` into the
backend-neutral :class:`agent_learning.scorers.base.ScoreResult`.
"""

from __future__ import annotations

from .adherence import NlpAdherenceScorer
from .completion import NlpCompletionScorer
from .intent import NlpIntentScorer

__all__ = ["NlpAdherenceScorer", "NlpCompletionScorer", "NlpIntentScorer"]
