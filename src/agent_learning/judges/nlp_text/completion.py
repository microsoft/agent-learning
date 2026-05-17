"""Tier 2 NLP text completion judge.

Combines the deterministic token-coverage scorer from
:class:`agent_learning.judges.stdlib.StdlibCompletionJudge` with a
TF-IDF + logistic-regression probability over the response text. The
final probability is the mean of the two signals (rule + classifier)
when both are available; otherwise the available signal is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from ..base import JudgeScore
from ..stdlib.completion import StdlibCompletionJudge
from ._base import _NlpTextJudgeBase


@dataclass
class NlpTextCompletionJudge(_NlpTextJudgeBase):
    """Completion judge backed by TF-IDF + token-coverage."""

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
    ) -> "NlpTextCompletionJudge":
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
    ) -> JudgeScore:
        if response is None:
            raise ValueError(
                "completion NLP text judge requires a response"
            )
        rule_judge = StdlibCompletionJudge(pass_threshold=self.pass_threshold)
        rule_score = rule_judge.score(
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


__all__ = ["NlpTextCompletionJudge"]
