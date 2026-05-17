"""Tier 1 stdlib task-completion judge.

Token-coverage scorer with no training. Given a list of expected
tokens (or a single string that the judge splits on whitespace),
returns the fraction present in the response text. The match is
case-insensitive and lemma-free (unigram only); callers wanting
stemming should supply already-normalized expected tokens.

The score collapses to ``1.0`` when no expected tokens are provided,
matching the permissive default of the other Tier 1 judges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from ..base import JudgeScore
from ._text import tokenize


@dataclass
class StdlibCompletionJudge:
    """Stdlib token-coverage completion judge."""

    name: str = "completion"
    pass_threshold: float = 0.5

    @classmethod
    def load_or_default(
        cls,
        snapshot_dir: Optional[str] = None,
        *,
        pass_threshold: float = 0.5,
    ) -> "StdlibCompletionJudge":
        """Match the load_or_default contract; no state to load."""
        _ = snapshot_dir
        return cls(pass_threshold=pass_threshold)

    def score(
        self,
        *,
        response: Optional[str] = None,
        expected_tokens: Optional[Sequence[str]] = None,
        contract: Optional[dict] = None,
        **_: object,
    ) -> JudgeScore:
        """Score one response against expected tokens.

        Either ``expected_tokens`` (preferred) or
        ``contract["completion_tokens"]`` supplies the target set.
        """
        if response is None:
            raise ValueError(
                "StdlibCompletionJudge requires response"
            )
        targets: List[str] = []
        if expected_tokens is not None:
            targets = [str(t) for t in expected_tokens]
        elif contract is not None:
            raw = contract.get("completion_tokens") or []
            if isinstance(raw, Sequence) and not isinstance(
                raw, (str, bytes)
            ):
                targets = [str(t) for t in raw]
        targets = [t for t in targets if t and t.strip()]

        present_tokens = set(tokenize(str(response)))
        if not targets:
            probability = 1.0
            hits = 0
            misses: List[str] = []
        else:
            hits = 0
            misses = []
            for tok in targets:
                normalized = tok.strip().lower()
                if not normalized:
                    continue
                # Single-token target: bucket lookup.
                if " " not in normalized:
                    if normalized in present_tokens:
                        hits += 1
                    else:
                        misses.append(tok)
                else:
                    # Multi-word target: substring match on the lowered
                    # response. Keeps the judge useful for short keyed
                    # phrases like "blood pressure" without requiring
                    # n-gram tokenization.
                    if normalized in str(response).lower():
                        hits += 1
                    else:
                        misses.append(tok)
            probability = hits / max(len(targets), 1)

        if probability >= self.pass_threshold:
            label = "pass"
            confidence = probability
        else:
            label = "fail"
            confidence = 1.0 - probability
        features = {
            "probability": probability,
            "hits": hits,
            "misses": misses,
            "total_targets": len(targets),
        }
        return JudgeScore(
            label=label,
            confidence=confidence,
            normalized=probability,
            features=features,
        )


__all__ = ["StdlibCompletionJudge"]
