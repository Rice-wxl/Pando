"""Sample then logit lens with field token highlighting LLM agent.

Extends the logit lens agent to also display logits for field name tokens
at each layer. For example, if field "brand" tokenizes to ["br", "and"],
shows the logits of these tokens alongside the top-k tokens for each layer.
"""

from typing import Any

from dotenv import load_dotenv

from . import register_agent
from .interp_llm_base import InterpContext
from .sample_then_logit_lens_llm import (
    SampleThenLogitLensLLMAgent,
    analyze_decision_layer,
    aggregate_logit_lens_patterns,
)

# Load environment variables
load_dotenv()


def format_logit_lens_summary_with_fields(
    layer_predictions: list[dict],
    field_token_info: dict[str, dict] | None = None,
) -> str:
    """Format logit lens data with field token logits at end of each layer.

    Shows top tokens at every layer with their logit values, plus field token logits.
    Format: 'token' (logit), e.g., 'yes' (2.5), 'approved' (2.1)
    Field tokens: field_name='tok1'(logit1),'tok2'(logit2) | ...
    """
    if not layer_predictions:
        return "No logit lens data available."

    lines = []
    for layer_data in layer_predictions:
        layer_num = layer_data["layer"]

        # Format all top tokens as 'token' (logit)
        top_tokens = layer_data["top_tokens"]
        top_str = ", ".join(f"'{t[0].strip()}' ({t[1]:.1f})" for t in top_tokens)

        line = f"  Layer {layer_num}: {top_str}"

        # Field token logits (new)
        if "field_logits" in layer_data and layer_data["field_logits"]:
            field_parts = []
            for field_name, token_logits in layer_data["field_logits"].items():
                tokens_str = ",".join(f"'{tok}'({logit:.1f})" for tok, logit in token_logits)
                field_parts.append(f"{field_name}={tokens_str}")
            line += f"\n    Field tokens: {' | '.join(field_parts)}"

        lines.append(line)
    return "\n".join(lines)


def format_sample_with_logit_lens_fields(
    inputs: dict[str, Any],
    label: bool,
    layer_predictions: list[dict],
    field_token_info: dict[str, dict] | None = None,
) -> str:
    """Format a single sample with logit lens and field token information for LLM context."""
    fields = ", ".join(f"{k}={v}" for k, v in inputs.items())

    # Logit lens summary - every layer, top tokens with logits, plus field tokens
    lens_summary = format_logit_lens_summary_with_fields(layer_predictions, field_token_info)

    return f"Input: {{{fields}}} -> Output: {'Yes' if label else 'No'}\nLogit lens (top tokens per layer):\n{lens_summary}"


@register_agent("logit_lens_field")
class SampleThenLogitLensFieldLLMAgent(SampleThenLogitLensLLMAgent):
    """Logit lens agent with field name token highlighting.

    Extends SampleThenLogitLensLLMAgent to also show logits for field name tokens
    at each layer. This helps identify which fields the model is paying attention to
    at different processing stages.

    Strategy:
    1. Randomly select samples to query with logit lens (1x cost)
    2. Collect layer-wise yes/no logits + field token logits
    3. Send samples + logit lens info to GPT-5.1 to identify a pattern/rule
    4. Use GPT-4.1 with the pattern to predict each remaining sample
    """

    name = "logit_lens_field"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.field_token_info: dict[str, dict] | None = None

    def run_interp(self, ctx: InterpContext) -> dict:
        """Run logit lens with field name tracking on the prompt (FREE -- charge=False)."""
        field_names = self.scenario.field_names()
        result = self.model.get_logit_lens(
            ctx.prompt,
            top_k=50,
            field_names=field_names,
            charge=False,
        )
        # Store field_token_info (same for all samples, capture on first)
        if self.field_token_info is None and "field_token_info" in result:
            self.field_token_info = result["field_token_info"]
        return {"layer_predictions": result["layer_predictions"]}

    def format_interp_results(self) -> str:
        """Format logit lens data with field token info for the GPT prompt."""
        # Extract layer predictions from interp_results
        all_layer_preds = [
            r.get("layer_predictions", []) for r in self.interp_results
        ]

        if not all_layer_preds or not any(all_layer_preds):
            return ""

        # Detailed per-sample logit lens with field tokens
        detailed_samples = []
        for inp, label, layer_preds in zip(
            self.queried_inputs,
            self.queried_results,
            all_layer_preds,
        ):
            detailed_samples.append(
                format_sample_with_logit_lens_fields(inp, label, layer_preds, self.field_token_info)
            )
        detailed_text = "\n\n".join(detailed_samples)

        # Aggregate logit lens patterns
        lens_patterns = aggregate_logit_lens_patterns(all_layer_preds, self.queried_results)

        # Add field tokenization info
        field_info_text = ""
        if self.field_token_info:
            field_info_lines = ["Field name tokenization:"]
            for field_name, info in self.field_token_info.items():
                tokens_str = ", ".join(f"'{t}'" for t in info["tokens"])
                field_info_lines.append(f"  {field_name} -> [{tokens_str}]")
            field_info_text = "\n".join(field_info_lines) + "\n\n"

        lines = [
            f"{field_info_text}## Logit Lens Analysis ({len(self.queried_inputs)} samples)",
            "",
            'Each example shows how the model\'s prediction evolves through its layers using "logit lens" - the top predicted output tokens at each layer depth. Early layers show intermediate processing, later layers show the final decision forming.',
            "",
            "Additionally, for each layer we show the logits for field name tokens. High logits for a field token suggest the model is \"thinking about\" that field at that layer.",
            "",
            detailed_text,
            "",
            lens_patterns,
            "",
            "Analyze carefully. Use the logit lens for clues about decision timing, and look at the field token logits to see which fields the model focuses on at different layers. Then look at all the input-output pairs to find patterns.",
        ]
        return "\n".join(lines)

    def get_metadata(self) -> dict[str, Any]:
        """Get metadata including field token info."""
        metadata = super().get_metadata()
        metadata["strategy"] = "logit_lens_field"
        if self.field_token_info:
            metadata["field_token_info"] = self.field_token_info
        return metadata
