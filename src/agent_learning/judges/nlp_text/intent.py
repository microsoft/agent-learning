"""Tier 2 NLP text intent judge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..base import JudgeScore
from ._base import _NlpTextJudgeBase


@dataclass
class NlpTextIntentJudge(_NlpTextJudgeBase):
    """Predicts whether the response addresses the requester's intent.

    Feeds TF-IDF a concatenation of ``query`` and ``response`` so the
    classifier can learn topical alignment between the question and
    the agent's answer.
    """

    name: str = "intent"

    @classmethod
    def load_or_default(
        cls,
        snapshot_dir: Optional[str] = None,
        *,
        pass_threshold: float = 0.5,
        max_features: int = 20000,
        ngram_min: int = 1,
        ngram_max: int = 2,
    ) -> "NlpTextIntentJudge":
        instance = cls(
            pass_threshold=pass_threshold,
            max_features=max_features,
            ngram_min=ngram_min,
            ngram_max=ngram_max,
        )
        cls._load_into(instance, snapshot_dir)
        return instance  # type: ignore[return-value]

    def score(
        self,
        *,
        query: Optional[str] = None,
        response: Optional[str] = None,
        **_: object,
    ) -> JudgeScore:
        if not query or not response:
            raise ValueError(
                "intent NLP text judge requires both query and response"
            )
        probability = self._predict_probability(
            query=query, response=response
        )
        return self._to_score(probability)


__all__ = ["NlpTextIntentJudge"]
