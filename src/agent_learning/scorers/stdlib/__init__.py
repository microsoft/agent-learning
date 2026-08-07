"""Tier 1 stdlib scorers (zero external dependencies).

These scorers score the rendered response text directly, using only
the Python standard library:

- :class:`StdlibIntentScorer` is a bag-of-words binary classifier with
  hashing-trick features and the SDK's pure-Python logistic regression
  primitive. It returns an unfitted ``pass``-prior policy until fitted.
- :class:`StdlibAdherenceScorer` is a deterministic rule engine over
  required substrings, forbidden substrings, length bounds, and an
  optional JSON-shape requirement. No training is required.
- :class:`StdlibCompletionScorer` is a token-coverage scorer that
  measures how many of the expected tokens the response contains.
  No training is required.

The wire-up contract is the same as every other scorer backend in the
SDK: each scorer exposes ``score(**kwargs) -> ScoreResult`` and ignores
extra keyword arguments. The three scorers expect:

- intent: ``query: str``, ``response: str``
- adherence: ``response: str``, ``contract: dict``
- completion: ``response: str``, ``expected_tokens: list[str]``

The factory in :mod:`agent_learning.scorers` routes ``tier="stdlib"``
to this package.
"""

from __future__ import annotations

from .adherence import StdlibAdherenceScorer
from .completion import StdlibCompletionScorer
from .intent import StdlibIntentScorer

__all__ = [
  "StdlibAdherenceScorer",
  "StdlibCompletionScorer",
  "StdlibIntentScorer",
]
