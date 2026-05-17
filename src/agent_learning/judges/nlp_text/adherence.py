"""Tier 2 NLP text adherence judge.

Combines the deterministic rule engine that ships in
:class:`agent_learning.judges.stdlib.StdlibAdherenceJudge` with a
TF-IDF + logistic-regression probability over the response text. The
final probability is the mean of:

- the rule-engine ratio (clauses satisfied / clauses total), and
- the classifier's predicted probability of the positive class.

When the classifier is unfitted the rule-engine ratio is used
unchanged. When no contract is supplied the rule-engine ratio is 1.0
and the score degenerates to the learned probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..base import JudgeScore
from ..stdlib.adherence import StdlibAdherenceJudge
from ._base import _NlpTextJudgeBase


@dataclass
class NlpTextAdherenceJudge(_NlpTextJudgeBase):
    """Adherence judge backed by TF-IDF + a deterministic rule engine."""

    name: str = "adherence"

    @classmethod
    def load_or_default(
        cls,
        snapshot_dir: Optional[str] = None,
        *,
        pass_threshold: float = 0.5,
        max_features: int = 20000,
        ngram_min: int = 1,
        ngram_max: int = 2,
    ) -> "NlpTextAdherenceJudge":
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
        # Adherence is about the response itself; ignore the query.
        return response.strip()

    def score(
        self,
        *,
        response: Optional[str] = None,
        contract: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None,
        **_: object,
    ) -> JudgeScore:
        if response is None:
            raise ValueError(
                "adherence NLP text judge requires a response"
            )
        # Rule engine via the Tier 1 backend.
        rule_judge = StdlibAdherenceJudge(pass_threshold=self.pass_threshold)
        rule_score = rule_judge.score(response=response, contract=contract)
        rule_probability = float(rule_score.normalized)

        # Classifier signal.
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
            "violations": rule_score.features.get("violations", []),
            "clauses_total": rule_score.features.get("clauses_total", 0),
        }
        return self._to_score(combined, features=features)


__all__ = ["NlpTextAdherenceJudge"]
