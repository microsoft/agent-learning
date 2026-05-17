"""Tier 1 stdlib judges (zero external dependencies).

These judges score the rendered response text directly, using only
the Python standard library:

- :class:`StdlibIntentJudge` is a bag-of-words binary classifier with
  hashing-trick features and the SDK's pure-Python logistic regression
  primitive. It returns an unfitted ``pass``-prior policy until fitted.
- :class:`StdlibAdherenceJudge` is a deterministic rule engine over
  required substrings, forbidden substrings, length bounds, and an
  optional JSON-shape requirement. No training is required.
- :class:`StdlibCompletionJudge` is a token-coverage scorer that
  measures how many of the expected tokens the response contains.
  No training is required.

The wire-up contract is the same as every other judge backend in the
SDK: each judge exposes ``score(**kwargs) -> JudgeScore`` and ignores
extra keyword arguments. The three judges expect:

- intent: ``query: str``, ``response: str``
- adherence: ``response: str``, ``contract: dict``
- completion: ``response: str``, ``expected_tokens: list[str]``

The factory in :mod:`agent_learning.judges` routes ``tier="stdlib"``
to this package.
"""

from __future__ import annotations

from .adherence import StdlibAdherenceJudge
from .completion import StdlibCompletionJudge
from .intent import StdlibIntentJudge

__all__ = [
    "StdlibAdherenceJudge",
    "StdlibCompletionJudge",
    "StdlibIntentJudge",
]
