"""Tier 2 NLP text completion scorer.

Combines the deterministic token-coverage scorer from
:class:`agent_learning.scorers.stdlib.StdlibCompletionScorer` with a
TF-IDF + logistic-regression probability over the response text. The
final probability is the mean of the two signals (rule + classifier)
when both are available; otherwise the available signal is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from ..base import ScoreResult
from ..stdlib.completion import StdlibCompletionScorer
from ._base import _NlpTextScorerBase


@dataclass
class NlpTextCompletionScorer(_NlpTextScorerBase):
    """Completion scorer backed by TF-IDF + token-coverage."""

    name: str = "completion"

    @classmethod
    def load_or_default(
        cls,
        snapshot_dir: Optional[str] = None,
        *,
        pass_threshold: float = 0.5,
        max_features: int = 20000,
        ngram_min: int = 1,
        ngram_max: int = 2,
    ) -> "NlpTextCompletionScorer":
        instance = cls(
            pass_threshold=pass_threshold,
            max_features=max_features,
            ngram_min=ngram_min,
            ngram_max=ngram_max,
        )
        cls._load_into(instance, snapshot_dir)
        return instance  # type: ignore[return-value]

    def _pair_text(
        self, *, query: Optional[str], response: str
    ) -> str:
        return response.strip()

    def score(
        self,
        *,
        response: Optional[str] = None,
        expected_tokens: Optional[Sequence[str]] = None,
        contract: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None,
        **_: object,
    ) -> ScoreResult:
        if response is None:
            raise ValueError(
                "completion NLP text scorer requires a response"
            )
        rule_scorer = StdlibCompletionScorer(pass_threshold=self.pass_threshold)
        rule_score = rule_scorer.score(
            response=response,
            expected_tokens=expected_tokens,
            contract=contract,
        )
        rule_probability = float(rule_score.normalized)

        learned_probability: Optional[float]
        if self.fitted:
            learned_probability = self._predict_probability(
                query=query, response=response
            )
            combined = 0.5 * (rule_probability + learned_probability)
        else:
            learned_probability = None
            combined = rule_probability

        features: Dict[str, Any] = {
            "rule_probability": rule_probability,
            "learned_probability": learned_probability,
            "hits": rule_score.features.get("hits", 0),
            "misses": rule_score.features.get("misses", []),
            "total_targets": rule_score.features.get("total_targets", 0),
        }
        return self._to_score(combined, features=features)


__all__ = ["NlpTextCompletionScorer"]
