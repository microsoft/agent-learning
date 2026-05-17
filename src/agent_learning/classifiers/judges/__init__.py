"""Binary ``{pass, fail}`` judges for RL reward shaping.

Each judge is a binary classifier over a ``(context, action)`` pair:

- :class:`agent_learning.classifiers.judges.intent.IntentJudge` —
  did the action address the requester's intent?
- :class:`agent_learning.classifiers.judges.adherence.AdherenceJudge`
  — did the action respect the contract / constraints of the task?
- :class:`agent_learning.classifiers.judges.completion.CompletionJudge`
  — did the action surface every required output?

All three judges share the same surface:

- ``fit(training_rows)`` learns a binary logistic regression. Each
  row carries ``"phi"`` (the context vector), ``"action_id"`` (the
  chosen action), and ``"label"`` (``0`` or ``1``).
- ``predict(features)`` returns a :class:`ClassifierResult`.
- ``score(phi=..., action_id=...)`` is the convenience wrapper
  matching the call shape any LLM-based judge would expose.

The three judges are intentionally identical in structure — they
differ only in their training labels and in the meaning callers
assign to their outputs.
"""

from .adherence import AdherenceJudge
from .completion import CompletionJudge
from .intent import IntentJudge

__all__ = [
    "AdherenceJudge",
    "CompletionJudge",
    "IntentJudge",
]
