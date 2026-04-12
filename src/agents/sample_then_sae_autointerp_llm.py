"""Sample then SAE autointerp LLM agent.

Queries random samples with SAE analysis across all layers (1x cost, SAE is free),
fetches feature descriptions from Neuronpedia, then uses GPT to find a pattern.

In .env: supply SAE_LAYERS, SAE_FEATURES_CACHE_PATH.
"""

from typing import Any

from . import register_agent
from .sae_base import SAEAgentBase, TOP_N
from .interp_llm_base import InterpContext


def format_sae_features_summary(
    layer_features: dict[int, list[dict]],
    feature_descriptions: dict[tuple[int, int], str],
    tokens: list[str],
    all_token_activations: dict[tuple[int, int], list[tuple[int, float]]],
    top_n: int = TOP_N,
) -> str:
    """Format SAE features with descriptions and token activations.

    Args:
        layer_features: Dict mapping layer -> list of {feature_idx, activation}
        feature_descriptions: Dict mapping (layer, feature_idx) -> description
        tokens: List of input token strings.
        all_token_activations: Dict mapping (layer, feature_idx) -> list of (token_idx, activation)
        top_n: Max features to show per layer.

    Returns:
        Formatted string summary with per-feature token activations.
    """
    lines = []
    for layer in sorted(layer_features.keys()):
        features = layer_features[layer][:top_n]
        if not features:
            continue

        layer_lines = [f"  Layer {layer}:"]
        for feat in features:
            idx = feat["feature_idx"]
            act = feat["activation"]
            desc = feature_descriptions.get((layer, idx), "No description")

            # Get token activations for this feature (ALL tokens)
            tok_acts = all_token_activations.get((layer, idx), [])
            layer_lines.append(f"    [{idx}] ({act:.1f}): {desc}")
            if tok_acts:
                # Show ALL tokens with their activations (excluding last which is prediction position)
                tok_str = " ".join(f"{repr(tokens[ti])}({ta:.1f})" for ti, ta in tok_acts)
                # layer_lines.append(f"        Activates on: {tok_str}")   # too long, comment out for now

        lines.extend(layer_lines)

    return "\n".join(lines) if lines else "No SAE features found."


def format_sample_with_sae(
    inputs: dict[str, Any],
    label: bool,
    tokens: list[str],
    all_token_activations: dict[tuple[int, int], list[tuple[int, float]]],
    layer_features: dict[int, list[dict]],
    feature_descriptions: dict[tuple[int, int], str],
) -> str:
    """Format a single sample with SAE feature information for LLM context.

    Args:
        inputs: Input field values.
        label: Model prediction.
        tokens: List of input token strings.
        all_token_activations: Dict mapping (layer, feature_idx) -> list of (token_idx, activation)
        layer_features: Dict mapping layer -> list of top features at last token.
        feature_descriptions: Dict mapping (layer, feature_idx) -> description.
    """
    fields = ", ".join(f"{k}={v}" for k, v in inputs.items())

    # Detailed last token SAE features (with descriptions and token activations)
    sae_summary = format_sae_features_summary(
        layer_features, feature_descriptions, tokens, all_token_activations
    )

    return (
        f"Input: {{{fields}}} -> Output: {'Yes' if label else 'No'}\n"
        f"Top SAE features at last token (by layer):\n{sae_summary}"
    )


@register_agent("sae_autointerp")
class SampleThenSAEAutoInterpLLMAgent(SAEAgentBase):
    """Agent that samples with SAE analysis, finds pattern with LLM, then predicts.

    Strategy:
    1. Load SAEs for all layers (using Gemma Scope)
    2. Query samples and get SAE features at last token (1x cost)
    3. Fetch feature descriptions from Neuronpedia
    4. Send samples + feature descriptions to GPT-5.1 to identify pattern
    5. Use GPT-4.1 with the pattern to predict remaining samples
    """

    name = "sae_autointerp"

    def run_interp(self, ctx: InterpContext) -> dict:
        """Run SAE analysis on one prompt. FREE (charge=False)."""
        result = self.model.get_sae_features_all_tokens(
            ctx.prompt, saes=self.saes, top_k_last_token=TOP_N, charge=False,
        )
        return {
            "tokens": result["tokens"],
            "all_token_activations": result["all_token_activations"],
            "layer_features": result["layer_features"],
        }

    def format_interp_results(self) -> str:
        """Format SAE feature data from self.interp_results for GPT prompt."""
        if not self.interp_results:
            return ""

        detailed_samples = []
        for inp, label, interp_data in zip(
            self.queried_inputs,
            self.queried_results,
            self.interp_results,
        ):
            tokens = interp_data.get("tokens", [])
            all_tok_acts = interp_data.get("all_token_activations", {})
            layer_feats = interp_data.get("layer_features", {})
            detailed_samples.append(format_sample_with_sae(
                inp, label, tokens, all_tok_acts, layer_feats, self.feature_descriptions
            ))
        detailed_text = "\n\n".join(detailed_samples)

        return f"""## SAE Feature Analysis ({len(self.interp_results)} samples)

Each example shows top SAE features at the last token position (which determines yes/no). For each feature:
- Feature description from Neuronpedia (auto-generated)
- "Activates on:" shows this feature's activation on ALL input tokens

Look for features that activate strongly on specific input values (e.g., a "car brand" feature activating on "BMW").

{detailed_text}

Analyze carefully. Match SAE feature activations with input token values to understand what the model focuses on."""

    def _format_esk_interp(self) -> str:
        """Format SAE feature info for ESK discovery prompt."""
        if not self.interp_results or not any(self.interp_results):
            return ""

        lines = ["## SAE Feature Analysis", ""]
        lines.append("Top SAE features at the last token position, with auto-generated descriptions from Neuronpedia.")
        lines.append("")
        for i, (prompt, interp_data) in enumerate(
            zip(self.queried_prompts, self.interp_results)
        ):
            tokens = interp_data.get("tokens", [])
            all_tok_acts = interp_data.get("all_token_activations", {})
            layer_feats = interp_data.get("layer_features", {})
            if not layer_feats:
                continue
            lines.append(f"Query {i+1}:")
            lines.append(format_sae_features_summary(
                layer_feats, self.feature_descriptions, tokens, all_tok_acts
            ))
            lines.append("")
        return "\n".join(lines)
