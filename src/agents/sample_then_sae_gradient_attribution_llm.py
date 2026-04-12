"""Sample then SAE gradient attribution LLM agent.

A variant of the SAE autointerp agent that ranks features by gradient attribution
instead of raw activation magnitude.

Attribution formula: delta_{i,t} = -(g_t . d_i)(a_t . d_i)
where:
    - g_t = gradient of (yes_logit - no_logit) w.r.t. hidden state at token t
    - d_i = SAE decoder direction for feature i
    - a_t = hidden state activation at token t

Features are ranked by |delta| to identify which features most influence the decision.
Costs 2x per sample due to backward pass.
"""

from typing import Any

from . import register_agent
from .sae_base import SAEAgentBase, TOP_N
from .interp_llm_base import InterpContext


def format_sae_attribution_summary(
    layer_features: dict[int, list[dict]],
    feature_descriptions: dict[tuple[int, int], str],
    top_n: int = TOP_N,
) -> str:
    """Format SAE features with attribution scores and descriptions.

    Args:
        layer_features: Dict mapping layer -> list of {feature_idx, attribution, activation}
        feature_descriptions: Dict mapping (layer, feature_idx) -> description
        top_n: Max features to show per layer.

    Returns:
        Formatted string summary with attribution scores.
    """
    lines = []
    for layer in sorted(layer_features.keys()):
        features = layer_features[layer][:top_n]
        if not features:
            continue

        layer_lines = [f"  Layer {layer}:"]
        for feat in features:
            idx = feat["feature_idx"]
            attr = feat["attribution"]
            desc = feature_descriptions.get((layer, idx), "No description")

            # Sign indicates direction: + pushes toward yes, - pushes toward no
            sign = "+" if attr >= 0 else ""
            layer_lines.append(f"    [{idx}] (attr={sign}{attr:.2f}): {desc}")

        lines.extend(layer_lines)

    return "\n".join(lines) if lines else "No SAE features found."


def format_sample_with_sae_attribution(
    inputs: dict[str, Any],
    label: bool,
    layer_features: dict[int, list[dict]],
    feature_descriptions: dict[tuple[int, int], str],
) -> str:
    """Format a single sample with SAE attribution information for LLM context.

    Args:
        inputs: Input field values.
        label: Model prediction.
        layer_features: Dict mapping layer -> list of top features.
        feature_descriptions: Dict mapping (layer, feature_idx) -> description.
    """
    fields = ", ".join(f"{k}={v}" for k, v in inputs.items())

    sae_summary = format_sae_attribution_summary(
        layer_features, feature_descriptions
    )

    return (
        f"Input: {{{fields}}} -> Output: {'Yes' if label else 'No'}\n"
        f"Top SAE features by gradient attribution:\n{sae_summary}"
    )


@register_agent("sae_gradient")
class SampleThenSAEGradientAttributionLLMAgent(SAEAgentBase):
    """Agent that samples with SAE gradient attribution analysis, finds pattern with LLM, then predicts.

    Strategy:
    1. Load SAEs for specified layers (using Gemma Scope)
    2. Query samples and get SAE features ranked by gradient attribution (2x cost due to backward pass)
    3. Fetch feature descriptions from Neuronpedia
    4. Send samples + feature descriptions to GPT to identify pattern
    5. Use GPT with the pattern to predict remaining samples

    Key difference from SAE autointerp agent: ranks features by gradient attribution
    (how much they influence yes/no decision) instead of raw activation magnitude.
    """

    name = "sae_gradient"
    _query_with_backward = True

    def run_interp(self, ctx: InterpContext) -> dict:
        """Run SAE gradient attribution on one prompt.

        Charges 1x for the backward pass when not fixed_prompt_budget.
        The forward pass (predict_yes_no) was already charged 1x by the generic loop.
        """
        result = self.model.get_sae_attribution(
            ctx.prompt, saes=self.saes, top_k=TOP_N, charge=False,
        )
        # Charge 1x for backward pass
        if not self.model.budget.fixed_prompt_budget:
            tokens = self.model.count_tokens(ctx.prompt)
            self.model.budget.charge(tokens)
        return {"layer_features": result["layer_features"]}

    def format_interp_results(self) -> str:
        """Format SAE attribution data from self.interp_results into human-readable text."""
        if not self.interp_results:
            return ""

        # Detailed SAE attribution info for all samples
        detailed_samples = []
        for i, interp_data in enumerate(self.interp_results):
            layer_features = interp_data.get("layer_features", {})
            if not layer_features:
                continue
            detailed_samples.append(format_sample_with_sae_attribution(
                self.queried_inputs[i],
                self.queried_results[i],
                layer_features,
                self.feature_descriptions,
            ))

        if not detailed_samples:
            return ""

        detailed_text = "\n\n".join(detailed_samples)

        return (
            f"## SAE Gradient Attribution Analysis ({len(detailed_samples)} samples)\n"
            f"\n"
            f"Each example shows top SAE features ranked by GRADIENT ATTRIBUTION - how much each feature influences the yes/no decision.\n"
            f"- Positive attribution (attr=+X) means the feature pushes toward \"Yes\"\n"
            f"- Negative attribution (attr=-X) means the feature pushes toward \"No\"\n"
            f"- Features with larger |attribution| have more influence on the decision\n"
            f"\n"
            f"The feature descriptions are from Neuronpedia (auto-generated interpretations).\n"
            f"\n"
            f"{detailed_text}\n"
            f"\n"
            f"Analyze carefully. Match SAE feature attributions with input field values to understand what the model focuses on. Features with high attribution are most important for understanding the decision rule."
        )

    def _format_esk_interp(self) -> str:
        """Format SAE gradient attribution info for ESK discovery prompt."""
        if not self.interp_results or not any(self.interp_results):
            return ""

        lines = ["## SAE Gradient Attribution Analysis", ""]
        lines.append("Top SAE features ranked by gradient attribution (how much each feature influences the output).")
        lines.append("Positive attribution pushes one direction, negative another.")
        lines.append("")
        for i, interp_data in enumerate(self.interp_results):
            layer_feats = interp_data.get("layer_features", {})
            if not layer_feats:
                continue
            lines.append(f"Query {i+1}:")
            lines.append(format_sae_attribution_summary(
                layer_feats, self.feature_descriptions
            ))
            lines.append("")
        return "\n".join(lines)
