"""Binary ``{pass, fail}`` scorers for RL reward shaping.

Each scorer is a binary classifier over a ``(context, action)`` pair:

- :class:`agent_learning.classifiers.scorers.intent.IntentScorer` —
  did the action address the requester's intent?
- :class:`agent_learning.classifiers.scorers.adherence.AdherenceScorer`
  — did the action respect the contract / constraints of the task?
- :class:`agent_learning.classifiers.scorers.completion.CompletionScorer`
  — did the action surface every required output?

All three scorers share the same surface:

- ``fit(training_rows)`` learns a binary logistic regression. Each
  row carries ``"phi"`` (the context vector), ``"action_id"`` (the
  chosen action), and ``"label"`` (``0`` or ``1``).
- ``predict(features)`` returns a :class:`ClassifierResult`.
- ``score(phi=..., action_id=...)`` is the convenience wrapper
  matching the call shape any LLM-based scorer would expose.

The three scorers are intentionally identical in structure — they
differ only in their training labels and in the meaning callers
assign to their outputs.
"""

from .adherence import AdherenceScorer
from .completion import CompletionScorer
from .intent import IntentScorer

__all__ = [
  "AdherenceScorer",
  "CompletionScorer",
  "IntentScorer",
]
