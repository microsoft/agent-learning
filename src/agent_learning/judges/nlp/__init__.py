"""Native, in-process judges (Tier 0, pure standard library).

These wrap the existing :class:`agent_learning.classifiers.judges.BinaryJudge`
implementations so the SDK can ship a working judge stack with no
external service dependencies. The wrappers translate the underlying
:class:`agent_learning.classifiers.base.ClassifierResult` into the
backend-neutral :class:`agent_learning.judges.base.JudgeScore`.
"""

from __future__ import annotations

from .adherence import NlpAdherenceJudge
from .completion import NlpCompletionJudge
from .intent import NlpIntentJudge

__all__ = ["NlpAdherenceJudge", "NlpCompletionJudge", "NlpIntentJudge"]
