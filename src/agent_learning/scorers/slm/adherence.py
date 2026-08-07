"""Tier 3 SLM adherence scorer.

Asks Phi-4-mini-instruct whether the response adheres to the supplied
contract (required substrings, forbidden substrings, length bounds, JSON
shape). The scorer is read-only: it never modifies the contract or the
response, only scores them against each other.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..base import ScoreResult
from ._base import SlmRunner, SlmScorerBase


_SYSTEM_PROMPT = (
    "You are a strict evaluator that checks whether an agent's response "
    "complies with a structural contract. Reply only with a single-line "
    "JSON object: {\"verdict\": \"pass\" | \"fail\", \"confidence\": "
    "<number in [0, 1]>}. Output no other text."
)


def _format_contract(contract: Optional[Dict[str, Any]]) -> str:
    if not contract:
        return (
            "(no explicit contract; the response is acceptable as long "
            "as it is on-topic and free of obvious violations)"
        )
    try:
        return json.dumps(contract, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(contract)


_USER_PROMPT_TEMPLATE = (
    "CONTRACT:\n{contract}\n\nRESPONSE:\n{response}\n\n"
    "Does the response satisfy the contract? Reply with the JSON verdict "
    "only."
)


@dataclass
class SlmAdherenceScorer(SlmScorerBase):
    """Phi-4-mini-instruct adherence scorer."""

    name: str = "adherence"
    system_prompt: str = field(default=_SYSTEM_PROMPT, repr=False)

    @classmethod
    def load_or_default(
        cls,
        model_dir: Optional[str] = None,
        *,
        pass_threshold: float = 0.5,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        runner: Optional[SlmRunner] = None,
    ) -> "SlmAdherenceScorer":
        return cls(
            model_dir=model_dir or "",
            pass_threshold=pass_threshold,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            runner=runner,
        )

    def _validate(self, **inputs: object) -> None:
        if inputs.get("response") is None:
            raise ValueError(
                "adherence SLM scorer requires a response"
            )

    def _build_user_prompt(self, **inputs: object) -> str:
        return _USER_PROMPT_TEMPLATE.format(
            contract=_format_contract(inputs.get("contract")),
            response=str(inputs["response"]).strip(),
        )

    def score(
        self,
        *,
        response: Optional[str] = None,
        contract: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None,
        **_: object,
    ) -> ScoreResult:
        return self._score_from_inputs(
            response=response,
            contract=contract,
            query=query,
        )


__all__ = ["SlmAdherenceScorer"]
