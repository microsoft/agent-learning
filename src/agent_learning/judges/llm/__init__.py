"""LLM-backed judges (Tier 1, requires ``azure-ai-evaluation``).

These wrappers stay shallow on purpose: each one defers to the
matching evaluator from the ``azure-ai-evaluation`` package and
projects the evaluator's verdict onto the SDK-neutral
:class:`agent_learning.judges.base.JudgeScore` shape.

The evaluator import is deferred until first call so installing the
SDK without ``azure-ai-evaluation`` stays inexpensive when callers
only use the NLP backend.
"""

from __future__ import annotations

from .adherence import LlmAdherenceJudge
from .completion import LlmCompletionJudge
from .intent import LlmIntentJudge

__all__ = ["LlmAdherenceJudge", "LlmCompletionJudge", "LlmIntentJudge"]
