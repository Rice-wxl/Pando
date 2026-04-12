"""Simple baseline agents that don't use budget."""

from typing import Any

from . import register_agent
from .base import BaseAgent, AgentResult


@register_agent("always_true")
class AlwaysTrueAgent(BaseAgent):
    """Agent that always predicts True (yes).

    Uses no budget. Simple baseline.
    """

    name = "always_true"

    def predict(self, test_inputs: list[dict[str, Any]]) -> AgentResult:
        predictions = [True] * len(test_inputs)
        return AgentResult(
            predictions=predictions,
            metadata={"strategy": "always_true", "budget_used": 0}
        )


@register_agent("always_false")
class AlwaysFalseAgent(BaseAgent):
    """Agent that always predicts False (no).

    Uses no budget. Simple baseline.
    """

    name = "always_false"

    def predict(self, test_inputs: list[dict[str, Any]]) -> AgentResult:
        predictions = [False] * len(test_inputs)
        return AgentResult(
            predictions=predictions,
            metadata={"strategy": "always_false", "budget_used": 0}
        )
