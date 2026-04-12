"""Sample then logit lens LLM agent.

Queries random samples with logit lens analysis (1x cost, logit lens is free),
uses layer-wise predictions to identify when the model "decides", then uses
GPT to find a pattern and predict.
"""

from typing import Any

from . import register_agent
from .interp_llm_base import InterpLLMAgent, InterpContext


# ---------------------------------------------------------------------------
# Module-level helper functions (also used by sample_then_logit_lens_field_llm)
# ---------------------------------------------------------------------------

def format_logit_lens_summary(
    layer_predictions: list[dict],
) -> str:
    """Format logit lens data into a readable summary.

    Shows top tokens at every layer with their logit values.
    Format: 'token' (logit), e.g., 'yes' (2.5), 'approved' (2.1)
    """
    if not layer_predictions:
        return "No logit lens data available."

    lines = []
    for layer_data in layer_predictions:
        layer_num = layer_data["layer"]

        # Format all top tokens as 'token' (logit)
        top_tokens = layer_data["top_tokens"]
        top_str = ", ".join(f"'{t[0].strip()}' ({t[1]:.1f})" for t in top_tokens)

        lines.append(f"  Layer {layer_num}: {top_str}")

    return "\n".join(lines)


def analyze_decision_layer(layer_predictions: list[dict]) -> dict:
    """Analyze at which layer the model makes its decision.

    Returns info about when yes/no preference emerges and stabilizes.
    """
    if not layer_predictions:
        return {"decision_layer": None, "stable_from": None}

    # Track when the sign of (yes - no) first becomes consistent
    final_diff = layer_predictions[-1]["yes_logit"] - layer_predictions[-1]["no_logit"]
    final_sign = 1 if final_diff > 0 else -1

    first_match = None
    stable_from = None

    for i, layer_data in enumerate(layer_predictions):
        diff = layer_data["yes_logit"] - layer_data["no_logit"]
        sign = 1 if diff > 0 else -1

        if sign == final_sign:
            if first_match is None:
                first_match = i
            # Check if it stays consistent from here
            is_stable = all(
                (lp["yes_logit"] - lp["no_logit"]) * final_sign > 0
                for lp in layer_predictions[i:]
            )
            if is_stable and stable_from is None:
                stable_from = i
        else:
            first_match = None  # Reset if sign changes

    return {
        "decision_layer": first_match,
        "stable_from": stable_from,
        "total_layers": len(layer_predictions),
    }


def format_sample_with_logit_lens(
    inputs: dict[str, Any],
    label: bool,
    layer_predictions: list[dict],
) -> str:
    """Format a single sample with logit lens information for LLM context."""
    fields = ", ".join(f"{k}={v}" for k, v in inputs.items())

    # Logit lens summary - every layer, top tokens with logits
    lens_summary = format_logit_lens_summary(layer_predictions)

    return f"Input: {{{fields}}} -> Output: {'Yes' if label else 'No'}\nLogit lens (top tokens per layer):\n{lens_summary}"


def aggregate_logit_lens_patterns(
    all_layer_predictions: list[list[dict]],
    all_labels: list[bool],
) -> str:
    """Aggregate logit lens patterns to find common decision characteristics."""
    if not all_layer_predictions:
        return "No patterns available."

    yes_decision_layers = []
    no_decision_layers = []

    for preds, label in zip(all_layer_predictions, all_labels):
        info = analyze_decision_layer(preds)
        if info["stable_from"] is not None:
            if label:
                yes_decision_layers.append(info["stable_from"])
            else:
                no_decision_layers.append(info["stable_from"])

    lines = ["Decision layer analysis:"]
    if yes_decision_layers:
        avg_yes = sum(yes_decision_layers) / len(yes_decision_layers)
        lines.append(f"  YES outputs: decision stabilizes around layer {avg_yes:.1f} (n={len(yes_decision_layers)})")
    if no_decision_layers:
        avg_no = sum(no_decision_layers) / len(no_decision_layers)
        lines.append(f"  NO outputs: decision stabilizes around layer {avg_no:.1f} (n={len(no_decision_layers)})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@register_agent("logit_lens")
class SampleThenLogitLensLLMAgent(InterpLLMAgent):
    """Agent that samples with logit lens, finds pattern with LLM, then predicts.

    Strategy:
    1. Randomly select samples to query with logit lens (1x cost)
    2. Collect layer-wise yes/no logits to see how decision evolves
    3. Send samples + logit lens info to GPT-5.1 to identify a pattern/rule
    4. Use GPT-4.1 with the pattern to predict each remaining sample
    """

    name = "logit_lens"

    def run_interp(self, ctx: InterpContext) -> dict:
        """Run logit lens on the prompt (FREE -- charge=False)."""
        result = self.model.get_logit_lens(ctx.prompt, top_k=50, charge=False)
        return {"layer_predictions": result["layer_predictions"]}

    def format_interp_results(self) -> str:
        """Format logit lens data from self.interp_results for the GPT prompt."""
        # Extract layer predictions from interp_results
        all_layer_preds = [
            r.get("layer_predictions", []) for r in self.interp_results
        ]

        if not all_layer_preds or not any(all_layer_preds):
            return ""

        # Detailed per-sample logit lens
        detailed_samples = []
        for inp, label, layer_preds in zip(
            self.queried_inputs,
            self.queried_results,
            all_layer_preds,
        ):
            detailed_samples.append(format_sample_with_logit_lens(inp, label, layer_preds))
        detailed_text = "\n\n".join(detailed_samples)

        # Aggregate logit lens patterns
        lens_patterns = aggregate_logit_lens_patterns(all_layer_preds, self.queried_results)

        lines = [
            f"## Logit Lens Analysis ({len(self.queried_inputs)} samples)",
            "",
            'Each example shows how the model\'s prediction evolves through its layers using "logit lens" - the top predicted output tokens at each layer depth. Early layers show intermediate processing, later layers show the final decision forming.',
            "",
            detailed_text,
            "",
            lens_patterns,
            "",
            "Analyze carefully. Use the logit lens for clues about decision timing, then look at all the input-output pairs to find patterns.",
        ]
        return "\n".join(lines)

    # --- ESK overrides ---

    def _format_esk_interp(self) -> str:
        """Format logit lens info for ESK discovery prompt."""
        if not self.interp_results or not any(r.get("layer_predictions") for r in self.interp_results):
            return ""

        lines = ["## Logit Lens Analysis", ""]
        lines.append("Shows top predicted tokens at each layer depth. Later layers show the final output forming.")
        lines.append("")
        for i, (prompt, interp_data) in enumerate(
            zip(self.queried_prompts, self.interp_results)
        ):
            layer_preds = interp_data.get("layer_predictions", [])
            if not layer_preds:
                continue
            lines.append(f"Query {i+1}:")
            lines.append(format_logit_lens_summary(layer_preds))
            lines.append("")
        return "\n".join(lines)
