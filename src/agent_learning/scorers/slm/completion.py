"""Tier 3 SLM completion scorer.

Asks Phi-4-mini-instruct whether every required token / sub-task is
present in the response. The expected tokens can be supplied either
through ``expected_tokens`` directly or through ``contract.completion_tokens``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..base import ScoreResult
from ._base import SlmRunner, SlmScorerBase


_SYSTEM_PROMPT = (
    "You are a strict evaluator that checks whether an agent's response "
    "covers every required item. Reply only with a single-line JSON "
    "object: {\"verdict\": \"pass\" | \"fail\", \"confidence\": <number "
    "in [0, 1]>}. Output no other text."
)


_USER_PROMPT_TEMPLATE = (
    "REQUIRED ITEMS (every one must be covered):\n{items}\n\n"
    "RESPONSE:\n{response}\n\n"
    "Does the response cover every required item? Reply with the JSON "
    "verdict only."
)


def _resolve_targets(
    expected_tokens: Optional[Sequence[str]],
    contract: Optional[Dict[str, Any]],
) -> List[str]:
    if expected_tokens:
        return [str(t).strip() for t in expected_tokens if str(t).strip()]
    if contract:
        tokens = contract.get("completion_tokens")
        if isinstance(tokens, Sequence) and not isinstance(tokens, (str, bytes)):
            return [str(t).strip() for t in tokens if str(t).strip()]
    return []


@dataclass
class SlmCompletionScorer(SlmScorerBase):
    """Phi-4-mini-instruct completion scorer."""

    name: str = "completion"
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
    ) -> "SlmCompletionScorer":
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
                "completion SLM scorer requires a response"
            )

    def _build_user_prompt(self, **inputs: object) -> str:
        targets = _resolve_targets(
            inputs.get("expected_tokens"),  # type: ignore[arg-type]
            inputs.get("contract"),  # type: ignore[arg-type]
        )
        if targets:
            items_str = "\n".join(f"- {t}" for t in targets)
        else:
            items_str = (
                "(no explicit checklist; treat the response as complete "
                "if it answers the apparent task)"
            )
        return _USER_PROMPT_TEMPLATE.format(
            items=items_str,
            response=str(inputs["response"]).strip(),
        )

    def score(
        self,
        *,
        response: Optional[str] = None,
        expected_tokens: Optional[Sequence[str]] = None,
        contract: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None,
        **_: object,
    ) -> ScoreResult:
        return self._score_from_inputs(
            response=response,
            expected_tokens=expected_tokens,
            contract=contract,
            query=query,
        )


__all__ = ["SlmCompletionScorer"]
