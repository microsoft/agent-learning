"""Task-completion scorer.

Binary ``{pass, fail}`` classifier that predicts whether the chosen
action produced a complete result. "Complete" is whatever the
training labels encode; the classifier itself sees only the context
vector, the action id, and the binary label.
"""

from __future__ import annotations

from ._base import BinaryScorer


class CompletionScorer(BinaryScorer):
    """Predict whether an action produced a complete result.

    Training rows expect ``label = 1`` when the action's output is
    scored as complete and ``label = 0`` otherwise.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("label_name", "completion")
        super().__init__(**kwargs)


__all__ = ["CompletionScorer"]
