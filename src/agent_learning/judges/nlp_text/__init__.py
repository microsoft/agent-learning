"""Tier 2 NLP judges (TF-IDF + scikit-learn) over raw query/response text.

These judges operate on the same ``(query, response)`` pair the Tier 1
stdlib judges accept and the same Tier 4 LLM judges accept, but use a
TF-IDF vectorizer + logistic-regression head from scikit-learn for the
intent and adherence/completion classifiers. The adherence judge also
applies the same deterministic rule engine the stdlib backend uses;
the rule-engine score is combined with the learned probability.

Requires the ``[nlp]`` extra (``pip install
agents-learning-sdk[nlp]``). All scikit-learn imports are lazy so
the package can still be imported without the extra; calling
:meth:`fit` or :meth:`score` raises a helpful ``ImportError`` then.
"""

from __future__ import annotations

from .adherence import NlpTextAdherenceJudge
from .completion import NlpTextCompletionJudge
from .intent import NlpTextIntentJudge

__all__ = [
    "NlpTextAdherenceJudge",
    "NlpTextCompletionJudge",
    "NlpTextIntentJudge",
]
