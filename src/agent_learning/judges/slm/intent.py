"""Tier 3 SLM intent judge.

Asks Phi-4-mini-instruct whether the model's response actually addresses
the user's question. The judge requires both ``query`` and ``response``;
this matches the intent-evaluator surface area used by Tiers 1, 2, and 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..base import JudgeScore
from ._base import SlmJudgeBase, SlmRunner


_SYSTEM_PROMPT = (
    "You are a strict evaluator that checks whether an agent's response "
    "addresses the user's question. Reply only with a single-line JSON "
    "object: {\"verdict\": \"pass\" | \"fail\", \"confidence\": <number "
    "in [0, 1]>}. Output no other text."
)

_USER_PROMPT_TEMPLATE = (
    "QUERY:\n{query}\n\nRESPONSE:\n{response}\n\n"
    "Does the response actually address the query? Reply with the JSON "
    "verdict only."
)


@dataclass
class SlmIntentJudge(SlmJudgeBase):
    """Phi-4-mini-instruct intent judge."""

    name: str = "intent"
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
    ) -> "SlmIntentJudge":
        return cls(
            model_dir=model_dir or "",
            pass_threshold=pass_threshold,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            runner=runner,
        )

    def _validate(self, **inputs: object) -> None:
        query = inputs.get("query")
        response = inputs.get("response")
        if not query or not response:
            raise ValueError(
                "intent SLM judge requires both query and response"
            )

    def _build_user_prompt(self, **inputs: object) -> str:
        return _USER_PROMPT_TEMPLATE.format(
            query=str(inputs["query"]).strip(),
            response=str(inputs["response"]).strip(),
        )

    def score(
        self,
        *,
        query: Optional[str] = None,
        response: Optional[str] = None,
        **_: object,
    ) -> JudgeScore:
        return self._score_from_inputs(query=query, response=response)


__all__ = ["SlmIntentJudge"]
