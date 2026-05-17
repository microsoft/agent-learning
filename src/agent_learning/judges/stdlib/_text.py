"""Stdlib-only text helpers used by the Tier 1 judges.

Implements:

- :func:`tokenize`: a deterministic, lower-cased, alphanumeric tokenizer
  that emits unigrams. Adequate for short healthcare-grader prompts.
- :func:`hash_bow`: hashing-trick bag-of-words feature vector. Returns
  ``list[float]`` of length ``dim`` so the SDK's pure-Python
  :func:`fit_binary_logreg` consumes it directly. The hashing trick
  removes the need to persist a vocabulary.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, List

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lower-case, alphanumeric unigram tokenizer."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def _token_hash(token: str, dim: int) -> int:
    """Hash ``token`` to a non-negative bucket in ``[0, dim)``."""
    digest = hashlib.md5(token.encode("utf-8")).digest()
    # First 4 bytes give us 32 bits of entropy; modulo into the bucket.
    raw = int.from_bytes(digest[:4], byteorder="big", signed=False)
    return raw % dim


def hash_bow(
    tokens: Iterable[str],
    dim: int,
    *,
    binary: bool = False,
) -> List[float]:
    """Hashing-trick bag-of-words feature vector.

    Args:
        tokens: iterable of tokens.
        dim: feature dimension (number of hash buckets).
        binary: if True, set buckets to ``1.0`` instead of accumulating
            counts. Useful for short responses where one occurrence
            already saturates the signal.

    Returns:
        ``list[float]`` of length ``dim``.
    """
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    vec = [0.0] * dim
    for tok in tokens:
        bucket = _token_hash(tok, dim)
        if binary:
            vec[bucket] = 1.0
        else:
            vec[bucket] += 1.0
    return vec


def featurize_query_response(
    query: str,
    response: str,
    *,
    dim: int = 1024,
) -> List[float]:
    """Concatenate query and response token streams, then hash-BOW.

    The returned vector is suitable as the ``features`` field for
    :func:`agent_learning.classifiers.base.fit_binary_logreg` once the
    caller appends the ``1.0`` bias column.
    """
    tokens = tokenize(query) + tokenize(response)
    return hash_bow(tokens, dim, binary=False)


__all__ = [
    "featurize_query_response",
    "hash_bow",
    "tokenize",
]
