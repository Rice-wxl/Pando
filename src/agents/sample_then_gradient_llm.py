"""Sample then gradient LLM agent.

Queries random samples with gradient computation (2x cost), uses gradient norms
to identify important tokens/fields, then uses GPT to find a pattern and predict.
"""

from typing import Any

from dotenv import load_dotenv

from . import register_agent
from .interp_llm_base import InterpLLMAgent, InterpContext

# Load environment variables
load_dotenv()


def format_sample_with_field_gradients(
    inputs: dict[str, Any],
    label: bool,
    tokens: list[str],
    grad_norms: list[float],
    field_grads: dict[str, float],
) -> str:
    """Format a single sample with token gradients and per-field importance."""
    # Input fields
    fields = ", ".join(f"{k}={v}" for k, v in inputs.items())

    # Per-token gradients
    token_info = " ".join(f"{repr(t)}({g:.2f})" for t, g in zip(tokens, grad_norms))

    # Per-field importance (sorted)
    sorted_fields = sorted(
        [(f, g) for f, g in field_grads.items() if f != "header"],
        key=lambda x: x[1],
        reverse=True,
    )
    field_info = ", ".join(f"{f}({g:.2f})" for f, g in sorted_fields)

    return (
        f"Input: {{{fields}}} -> Output: {'Yes' if label else 'No'}\n"
        f"  Token gradients: {token_info}\n"
        f"  Field importance: {field_info}"
    )


def aggregate_field_gradients(
    all_field_grads: list[dict[str, float]],
    all_labels: list[bool],
) -> str:
    """Aggregate per-field gradients across all samples."""
    if not all_field_grads:
        return "No gradient data available."

    # Sum gradients per field across all samples
    field_totals: dict[str, float] = {}
    field_counts: dict[str, int] = {}

    for field_grads in all_field_grads:
        for field, grad in field_grads.items():
            if field == "header":
                continue
            field_totals[field] = field_totals.get(field, 0) + grad
            field_counts[field] = field_counts.get(field, 0) + 1

    # Compute mean gradient per field
    field_means = {f: field_totals[f] / field_counts[f] for f in field_totals}

    # Sort by importance
    sorted_fields = sorted(field_means.items(), key=lambda x: x[1], reverse=True)

    if not sorted_fields:
        return "Could not determine field importance from gradients."

    lines = ["Field importance (mean gradient magnitude across samples):"]
    for field, importance in sorted_fields:
        lines.append(f"  {field}: {importance:.2f}")

    return "\n".join(lines)


@register_agent("gradient_v1")
class SampleThenGradientLLMAgent(InterpLLMAgent):
    """Agent that samples with gradients, finds pattern with LLM, then predicts.

    Strategy:
    1. Randomly select samples to query with gradient computation (2x cost)
    2. Collect gradient norms per token to identify important fields
    3. Send samples + gradient info to GPT-5.1 to identify a pattern/rule
    4. Use GPT-4.1 with the pattern to predict each remaining sample
    """

    name = "gradient_v1"
    _query_with_backward = True

    def __init__(self, *args, format_style: str = "structured", **kwargs):
        super().__init__(*args, format_style=format_style, **kwargs)

    def compute_field_gradients(
        self,
        inputs: dict[str, Any],
        tokens: list[str],
        grad_norms: list[float],
    ) -> dict[str, float]:
        """Compute per-field gradient importance using structured annotation.

        Uses scenario.format() to map tokens to fields by
        accumulating tokens until each field's text is found.
        """
        annotated = self.scenario.format(inputs, style=self.format_style, annotated=True)

        field_grads = {}
        token_idx = 0

        for field_name, line_text in annotated.items():
            # Accumulate tokens until we find this line's text
            cur_tokens = []
            cur_grads = []
            cur_text = ""

            while token_idx < len(tokens):
                cur_tokens.append(tokens[token_idx])
                tok_stripped = tokens[token_idx].strip()
                if tok_stripped and tok_stripped in line_text:
                    # print(f'{repr(tokens[token_idx])}',end=' ')  # DONT REMOVE
                    cur_grads.append(grad_norms[token_idx])
                cur_text += tokens[token_idx]
                token_idx += 1

                # Check if we've accumulated enough to contain this line
                if line_text in cur_text:
                    # Sum gradients for tokens that contributed to this field
                    field_grads[field_name] = sum(cur_grads)
                    break
            else:
                # Ran out of tokens - use what we have
                field_grads[field_name] = sum(cur_grads) if cur_grads else 0.0
            # print('<- end of field',field_name)  # DONT REMOVE

        return field_grads

    def run_interp(self, ctx: InterpContext) -> dict:
        """Extract gradient data from one prompt.

        Charges 1x for the backward pass (honor code) when not fixed_prompt_budget.
        The forward pass (predict_yes_no) was already charged 1x by the generic loop.
        """
        result = self.model.get_embedding_gradients(ctx.prompt, charge=False)
        # Charge 1x for backward pass (honor code) when not fixed_prompt_budget
        if not self.model.budget.fixed_prompt_budget:
            tokens = self.model.count_tokens(ctx.prompt)
            self.model.budget.charge(tokens)
        out = {"tokens": result["tokens"], "token_grad_norms": result["token_grad_norms"]}
        if ctx.inputs is not None:
            out["field_grads"] = self.compute_field_gradients(
                ctx.inputs, result["tokens"], result["token_grad_norms"]
            )
        return out

    def format_interp_results(self) -> str:
        """Format gradient data from self.interp_results into human-readable text."""
        if not self.interp_results:
            return ""

        # Collect samples that have gradient data
        samples = []
        all_field_grads = []
        for i, interp_data in enumerate(self.interp_results):
            tokens = interp_data.get("tokens", [])
            grad_norms = interp_data.get("token_grad_norms", [])
            field_grads = interp_data.get("field_grads", {})
            if not tokens:
                continue
            samples.append(format_sample_with_field_gradients(
                self.queried_inputs[i],
                self.queried_results[i],
                tokens,
                grad_norms,
                field_grads,
            ))
            if field_grads:
                all_field_grads.append(field_grads)

        if not samples:
            return ""

        samples_text = "\n\n".join(samples)

        # Aggregate field gradients across samples
        grad_stats = aggregate_field_gradients(
            all_field_grads,
            self.queried_results,
        )

        return (
            f"## Gradient Analysis ({len(samples)} samples)\n"
            f"\n"
            f"Each example shows:\n"
            f"- Token gradients: gradient norm per input token (higher = more influential on the yes/no decision)\n"
            f"- Field importance: total gradient per field (higher = field matters more)\n"
            f"\n"
            f"{samples_text}\n"
            f"\n"
            f"{grad_stats}\n"
            f"\n"
            f"Analyze these examples carefully. Fields with higher gradient importance are more likely to be part of the decision rule."
        )
