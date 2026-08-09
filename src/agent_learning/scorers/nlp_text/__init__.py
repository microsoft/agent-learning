"""Tier 2 NLP scorers (TF-IDF + scikit-learn) over raw query/response text.

These scorers operate on the same ``(query, response)`` pair the Tier 1
stdlib scorers accept and the same Tier 4 LLM scorers accept, but use a
TF-IDF vectorizer + logistic-regression head from scikit-learn for the
intent and adherence/completion classifiers. The adherence scorer also
applies the same deterministic rule engine the stdlib backend uses;
the rule-engine score is combined with the learned probability.

Requires the ``[nlp]`` extra (``pip install
agent-learning[nlp]``). All scikit-learn imports are lazy so
the package can still be imported without the extra; calling
:meth:`fit` or :meth:`score` raises a helpful ``ImportError`` then.
"""

from __future__ import annotations

from .adherence import NlpTextAdherenceScorer
from .completion import NlpTextCompletionScorer
from .intent import NlpTextIntentScorer

__all__ = [
    "NlpTextAdherenceScorer",
    "NlpTextCompletionScorer",
    "NlpTextIntentScorer",
]
