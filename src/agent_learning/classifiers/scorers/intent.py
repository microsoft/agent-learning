"""Intent-resolution scorer.

Binary ``{pass, fail}`` classifier that predicts whether the chosen
action addresses the requester's intent given a context vector. A
deterministic, non-LLM drop-in for any LLM-based intent evaluator.
"""

from __future__ import annotations

from ._base import BinaryScorer


class IntentScorer(BinaryScorer):
    """Predict whether the chosen action addresses the requester's intent.

    Training rows expect ``label = 1`` when the action is scored as
    address the intent and ``label = 0`` otherwise.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("label_name", "intent")
        super().__init__(**kwargs)


__all__ = ["IntentScorer"]
