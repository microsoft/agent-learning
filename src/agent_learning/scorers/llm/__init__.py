"""LLM-backed scorers (Tier 4, requires ``azure-ai-evaluation``).

These wrappers stay shallow on purpose: each one defers to the
matching evaluator from the ``azure-ai-evaluation`` package and
projects the evaluator's verdict onto the SDK-neutral
:class:`agent_learning.scorers.base.ScoreResult` shape.

The evaluator import is deferred until first call so installing the
SDK without ``azure-ai-evaluation`` stays inexpensive when callers
only use the NLP backend.
"""

from __future__ import annotations

from .adherence import LlmAdherenceScorer
from .completion import LlmCompletionScorer
from .intent import LlmIntentScorer

__all__ = ["LlmAdherenceScorer", "LlmCompletionScorer", "LlmIntentScorer"]
