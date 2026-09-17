"""Values shared through MAF invocation context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...decision import DecisionResult

USER_INPUT_KEY = "_agent_learning_user_input"
PENDING_DECISIONS_KEY = "agent_learning.pending_decisions"


@dataclass(frozen=True)
class PendingDecision:
    """A policy decision suspended while MAF obtains user approval."""

    decision: DecisionResult
    action_id: str
    decision_context: dict[str, Any]
    action_inputs: dict[str, dict[str, Any]]
    user_input: str
    autonomy: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        expected_action = self.decision.selected_action or self.decision.proposed_action
        if expected_action is None or expected_action.id != self.action_id:
            raise ValueError("pending decision action does not match its decision")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "decision": self.decision.to_dict(),
            "action_id": self.action_id,
            "decision_context": self.decision_context,
            "action_inputs": self.action_inputs,
            "user_input": self.user_input,
            "autonomy": self.autonomy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PendingDecision:
        if data.get("version") != 1:
            raise ValueError("unsupported pending decision state version")
        return cls(
            decision=DecisionResult.from_dict(data["decision"]),
            action_id=data["action_id"],
            decision_context=dict(data["decision_context"]),
            action_inputs={
                name: dict(arguments)
                for name, arguments in data["action_inputs"].items()
            },
            user_input=data["user_input"],
            autonomy=(dict(data["autonomy"]) if data.get("autonomy") else None),
        )


__all__ = [
    "PENDING_DECISIONS_KEY",
    "USER_INPUT_KEY",
    "PendingDecision",
]
