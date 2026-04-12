"""Autoresearch interp agent -- editable by meta-agent.

Only edit run_interp() and format_interp_results().
Do NOT add extra litellm/GPT calls, modify the base class, or change the budget.
"""

import sys
from pathlib import Path
from typing import Any

# Add package root to path for src/ imports
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.interp_llm_base import InterpLLMAgent, InterpContext
from src.budget import BudgetExceededError


class AutoresearchAgent(InterpLLMAgent):
    """Interp agent being optimized by autoresearch loop.

    Constraints:
    - run_interp() receives ctx with .prompt, .prediction, .inputs
    - All model calls inside run_interp() must use charge=False
    - Available tools (via self.model):
      * self.model.generate(prompt, max_new_tokens=20, temperature=0.0, charge=False)
      * self.model.get_embedding_gradients(prompt, charge=False)
      * self.model.get_logit_lens(prompt, top_k=50, charge=False)
      * self.model.get_next_token_probs(prompt, tokens=None, charge=False)
      * self.model.get_hidden_states(prompt, layers=None, charge=False)  # raw activations
      * self.model.apply_chat_template(prompt)
      * self.model.model.tokenizer (for token manipulation)
    - Cannot make litellm/GPT calls
    - Cannot do inference on new inputs (only test prompts from ctx)
    - Budget: 10 inferences in fixed_prompt_budget mode
    """

    name = "autoresearch"

    def run_interp(self, ctx: InterpContext) -> dict:
        """Collect interpretability data from one sample.

        Args:
            ctx.prompt: The scenario prompt (e.g., car purchase description)
            ctx.prediction: Model's yes/no prediction (bool)
            ctx.inputs: Dict of field_name -> value

        Returns:
            Dict of interp data (will be stored in self.interp_results)
        """
        result = {}

        # 1. Prefill continuation: "yes/no, because ..."
        prefill = "yes, because" if ctx.prediction else "no, because"
        formatted = self.model.apply_chat_template(ctx.prompt)
        prefilled = formatted + prefill
        try:
            continuation = self.model.generate(
                prefilled, max_new_tokens=20, temperature=0.0,
                charge=False,
            )
            result["reasoning"] = prefill + continuation
        except BudgetExceededError:
            result["reasoning"] = prefill

        # 2. Embedding gradients -> per-field importance
        grad_data = self.model.get_embedding_gradients(ctx.prompt, charge=False)
        result["tokens"] = grad_data["tokens"]
        result["token_grad_norms"] = grad_data["token_grad_norms"]
        if ctx.inputs:
            field_grads = self._compute_field_grads(
                ctx.inputs, grad_data["tokens"], grad_data["token_grad_norms"]
            )
            result["field_grads"] = field_grads

        return result

    def _compute_field_grads(
        self,
        inputs: dict[str, Any],
        tokens: list[str],
        grad_norms: list[float],
    ) -> dict[str, float]:
        """Compute per-field gradient importance.

        Replicates logic from src/agents/sample_then_gradient_llm.py:99-140.
        Maps tokens to fields using scenario.format(annotated=True).
        """
        annotated = self.scenario.format(inputs, style=self.format_style, annotated=True)
        field_grads = {}
        token_idx = 0

        for field_name, line_text in annotated.items():
            cur_grads = []
            cur_text = ""

            while token_idx < len(tokens):
                tok_stripped = tokens[token_idx].strip()
                if tok_stripped and tok_stripped in line_text:
                    cur_grads.append(grad_norms[token_idx])
                cur_text += tokens[token_idx]
                token_idx += 1

                if line_text in cur_text:
                    field_grads[field_name] = sum(cur_grads)
                    break
            else:
                field_grads[field_name] = sum(cur_grads) if cur_grads else 0.0

        return field_grads

    def format_interp_results(self) -> str:
        """Format collected interp data for GPT-5.1 pattern discovery.

        This text is appended to the I/O pairs in the find_pattern prompt.
        Make it informative but concise.
        """
        if not self.interp_results:
            return ""

        sections = []

        # --- Reasoning section ---
        reasoning_lines = []
        for inp, label, interp in zip(
            self.queried_inputs, self.queried_results, self.interp_results
        ):
            reasoning = interp.get("reasoning", "")
            if reasoning:
                fields = ", ".join(f"{k}={v}" for k, v in inp.items())
                reasoning_lines.append(
                    f"Input: {{{fields}}} -> {'Yes' if label else 'No'}\n"
                    f'  Model says: "{reasoning}"'
                )

        if reasoning_lines:
            sections.append(
                "## Model Self-Reasoning\n\n"
                "After each prediction, the model was asked to explain. "
                "Use this to identify which fields matter:\n\n"
                + "\n\n".join(reasoning_lines)
            )

        # --- Field gradient importance ---
        all_field_grads = [r.get("field_grads", {}) for r in self.interp_results]
        if any(all_field_grads):
            field_totals: dict[str, float] = {}
            field_counts: dict[str, int] = {}
            for fg in all_field_grads:
                for f, g in fg.items():
                    if f == "header":
                        continue
                    field_totals[f] = field_totals.get(f, 0) + g
                    field_counts[f] = field_counts.get(f, 0) + 1

            if field_totals:
                sorted_fields = sorted(
                    [(f, field_totals[f] / field_counts[f]) for f in field_totals],
                    key=lambda x: x[1],
                    reverse=True,
                )
                field_lines = [f"  {f}: {imp:.2f}" for f, imp in sorted_fields]
                sections.append(
                    "## Field Importance (Gradient Attribution)\n\n"
                    "Mean gradient magnitude per field (higher = more influential):\n"
                    + "\n".join(field_lines)
                    + "\n\nFields with higher gradient importance are more likely part of the rule."
                )

        return "\n\n".join(sections)
