"""Task-adherence judge.

Binary ``{pass, fail}`` classifier that predicts whether the chosen
action respected the task's contract or constraints. Domain-specific
contracts are entirely the caller's concern — this class only sees
the context vector, the action id, and the training labels.
"""

from __future__ import annotations

from ._base import BinaryJudge


class AdherenceJudge(BinaryJudge):
    """Predict whether an action respects the task's contract.

    Training rows expect ``label = 1`` when the action is judged to
    adhere to the contract and ``label = 0`` otherwise.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("label_name", "adherence")
        super().__init__(**kwargs)


__all__ = ["AdherenceJudge"]
