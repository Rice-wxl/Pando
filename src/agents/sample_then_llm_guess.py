"""Sample then LLM guess agent.

Queries random samples within budget, uses GPT-5.1 to find a pattern,
then uses GPT-5-mini to predict each remaining sample.

This is the simplest InterpLLMAgent: no interp tools, just I/O pairs.
"""

from . import register_agent
from .interp_llm_base import InterpLLMAgent, InterpContext


@register_agent("blackbox")
class SampleThenLLMGuessAgent(InterpLLMAgent):
    """Agent that samples randomly, finds pattern with LLM, then predicts.

    Strategy:
    1. Randomly select samples to query (within budget)
    2. Query the model for those samples
    3. Send queried samples to GPT-5.1 to identify a pattern/rule
    4. Use GPT-4.1 with the pattern to predict each remaining sample

    No interpretability tools are used -- this is the blackbox LLM baseline.
    """

    name = "blackbox"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_interp(self, ctx: InterpContext) -> dict:
        """No interp data for this agent."""
        return {}

    def format_interp_results(self) -> str:
        """No interp data for this agent."""
        return ""

    def _format_esk_interp(self) -> str:
        """No interp data for this agent."""
        return ""
