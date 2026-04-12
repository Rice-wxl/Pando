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
from src.agents.relp import relp_mode
from src.budget import BudgetExceededError


class AutoresearchAgent(InterpLLMAgent):
    """Interp agent: prefill + RelP gradients + contrastive summary."""

    name = "autoresearch"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._relp_logged = False

    def _get_hf_model(self):
        return self.model.model.model

    def run_interp(self, ctx: InterpContext) -> dict:
        result = {}

        # 1. Prefill continuation
        prefill = "yes, because" if ctx.prediction else "no, because"
        formatted = self.model.apply_chat_template(ctx.prompt)
        prefilled = formatted + prefill
        try:
            continuation = self.model.generate(
                prefilled, max_new_tokens=20, temperature=0.0, charge=False,
            )
            result["reasoning"] = prefill + continuation
        except BudgetExceededError:
            result["reasoning"] = prefill

        # 2. RelP gradients
        verbose = not self._relp_logged
        try:
            with relp_mode(self._get_hf_model(), rules=["LN", "Identity", "Half", "AH"], verbose=verbose):
                self._relp_logged = True
                grad_data = self.model.get_embedding_gradients(ctx.prompt, charge=False)
        except (ValueError, RuntimeError):
            grad_data = self.model.get_embedding_gradients(ctx.prompt, charge=False)

        result["tokens"] = grad_data["tokens"]
        result["token_grad_norms"] = grad_data["token_grad_norms"]
        if ctx.inputs:
            result["field_grads"] = self._compute_field_grads(
                ctx.inputs, grad_data["tokens"], grad_data["token_grad_norms"]
            )

        return result

    def _compute_field_grads(self, inputs, tokens, grad_norms):
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
        if not self.interp_results:
            return ""

        sections = []

        # --- Per-sample field attribution + reasoning (proven format) ---
        sample_lines = []
        for inp, label, interp in zip(
            self.queried_inputs, self.queried_results, self.interp_results
        ):
            fields = ", ".join(f"{k}={v}" for k, v in inp.items())
            fg = interp.get("field_grads", {})
            reasoning = interp.get("reasoning", "")

            sorted_fg = sorted(
                [(f, g) for f, g in fg.items() if f != "header"],
                key=lambda x: x[1], reverse=True,
            )
            attr_str = ", ".join(f"{f}({g:.1f})" for f, g in sorted_fg) if sorted_fg else ""

            line = f"Input: {{{fields}}} -> {'Yes' if label else 'No'}"
            if attr_str:
                line += f"\n  Field attribution (higher=more important): {attr_str}"
            if reasoning:
                line += f'\n  Model reasoning: "{reasoning}"'
            sample_lines.append(line)

        if sample_lines:
            sections.append(
                "## Per-Sample Analysis\n\n"
                "Each sample shows the model's prediction, which fields had the highest "
                "attribution scores (indicating importance to the decision), and the model's "
                "self-reported reasoning:\n\n"
                + "\n\n".join(sample_lines)
            )

        # --- Aggregated field importance ---
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
                    key=lambda x: x[1], reverse=True,
                )
                field_lines = [f"  {f}: {imp:.2f}" for f, imp in sorted_fields]
                sections.append(
                    "## Field Importance Summary\n\n"
                    "Mean attribution magnitude per field across all samples (higher = more likely part of the rule):\n"
                    + "\n".join(field_lines)
                    + "\n\nFocus on the top-ranked fields when inferring the decision rule. "
                    "Fields with low attribution are unlikely to be part of the rule."
                )

        # --- Contrastive summary: Yes vs No field values ---
        yes_inputs = [inp for inp, label in zip(self.queried_inputs, self.queried_results) if label]
        no_inputs = [inp for inp, label in zip(self.queried_inputs, self.queried_results) if not label]

        if yes_inputs and no_inputs:
            # Get top fields by attribution
            top_fields = []
            if field_totals:
                top_fields = [f for f, _ in sorted_fields[:6]]  # top 6 fields
            else:
                top_fields = list(self.queried_inputs[0].keys())

            contrast_lines = []
            for field in top_fields:
                yes_vals = [str(inp.get(field, "")) for inp in yes_inputs]
                no_vals = [str(inp.get(field, "")) for inp in no_inputs]
                contrast_lines.append(f"  {field}: Yes samples={yes_vals}, No samples={no_vals}")

            sections.append(
                "## Yes vs No Comparison (top fields)\n\n"
                "Comparing field values between Yes and No predictions for the most important fields:\n"
                + "\n".join(contrast_lines)
            )

        return "\n\n".join(sections)
