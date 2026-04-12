"""Blackbox prefill agent - queries prediction + reasoning, then LLM pattern match.

Queries samples to get yes/no prediction, then prefills "yes/no, because" and
generates reasoning tokens. Uses this to find patterns via GPT.
"""

from typing import Any

from . import register_agent
from .interp_llm_base import InterpLLMAgent, InterpContext
from ..budget import BudgetExceededError


def format_sample_with_reasoning(inputs: dict[str, Any], label: bool, reasoning: str) -> str:
    """Format a single sample with its model-generated reasoning."""
    fields = ", ".join(f"{k}={v}" for k, v in inputs.items())
    return f"Input: {{{fields}}} -> Output: {'Yes' if label else 'No'}\n  Model's reasoning: \"{reasoning}\""


@register_agent("prefill")
class BlackboxPrefillAgent(InterpLLMAgent):
    """Agent that queries prediction + reasoning, finds pattern with LLM.

    Strategy:
    1. Query samples: get yes/no + prefill "yes/no, because" + generate 10 tokens
    2. Send samples + reasoning to GPT-5.1 to identify the decision rule
    3. Use GPT-4.1 to predict remaining samples
    """

    name = "prefill"

    def run_interp(self, ctx: InterpContext) -> dict:
        """Prefill and generate reasoning tokens.

        Normal path: prefill "yes/no, because" after prediction.
        ESK path: prefill task-specific text after model response.
        """
        if ctx.response is not None:
            # ESK: prefill task-specific text after model response
            td = self.task_descriptor
            if not td or not td.prefill_text:
                return {"reasoning": ""}
            formatted = self.model.apply_chat_template(ctx.prompt)
            prefilled = formatted + ctx.response + "\n" + td.prefill_text
            # charge=False: the response generation already paid for this
            # inference slot; the prefill continuation is a second forward
            # pass on the same prompt and is treated as free interp data.
            continuation = self.model.generate(
                prefilled, max_new_tokens=20, temperature=0.0, charge=False,
            )
            return {"reasoning": td.prefill_text + continuation}

        # Normal: prefill "yes/no, because" after prediction
        prefill = "yes, because" if ctx.prediction else "no, because"
        formatted = self.model.apply_chat_template(ctx.prompt)
        prefilled = formatted + prefill
        try:
            continuation = self.model.generate(
                prefilled,
                max_new_tokens=10,
                temperature=0.0,
                # predict_yes_no already charged prompt tokens;
                # in fixed_prompt_budget mode the inference is already counted
                charge=not self.model.budget.fixed_prompt_budget,
                charge_prompt=False,
            )
            return {"reasoning": prefill + continuation}
        except BudgetExceededError:
            return {"reasoning": prefill + " [truncated]"}

    def format_interp_results(self) -> str:
        """Format reasoning data from samples for the GPT prompt."""
        if not self.interp_results:
            return ""

        samples = []
        for inp, label, interp_data in zip(
            self.queried_inputs, self.queried_results, self.interp_results
        ):
            reasoning = interp_data.get("reasoning", "")
            if reasoning:
                samples.append(format_sample_with_reasoning(inp, label, reasoning))

        if not samples:
            return ""

        samples_text = "\n\n".join(samples)

        lines = [
            "## Model Self-Reasoning",
            "",
            "Each sample includes what the model generated when asked to explain its decision:",
            "",
            samples_text,
            "",
            'The "Model\'s reasoning" shows what the model generated when asked to explain its decision. Use this to understand which fields matter.',
        ]
        return "\n".join(lines)

    # --- ESK overrides ---

    def _format_esk_interp(self) -> str:
        """Format prefilled responses for ESK discovery prompt."""
        reasonings = [r.get("reasoning", "") for r in self.interp_results]
        if not any(reasonings):
            return ""

        lines = ["## Model Self-Reasoning (via prefill)", ""]
        lines.append("We prefilled the model with task-specific text and let it continue. This may reveal hidden knowledge.")
        lines.append("")
        for i, reasoning in enumerate(reasonings):
            if reasoning:
                lines.append(f"Query {i+1} prefill continuation: \"{reasoning}\"")
        return "\n".join(lines)

    def get_metadata(self) -> dict[str, Any]:
        """Get metadata including pattern and reasoning."""
        metadata = super().get_metadata()
        metadata["strategy"] = "prefill"
        metadata["queried_reasoning"] = [r.get("reasoning", "") for r in self.interp_results]
        return metadata
