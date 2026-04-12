"""Sample then SAE token similarity LLM agent.

Queries random samples with SAE analysis, computes cosine similarity between
SAE feature decoder vectors and token embeddings to find what tokens each
feature represents, then uses GPT to find a pattern.

This approach doesn't require external API calls (unlike Neuronpedia autointerp).
"""

from typing import Any

from . import register_agent
from .sae_base import SAEAgentBase, load_saes, TOP_N
from .interp_llm_base import InterpContext

TOP_K_TOKENS = 10  # Number of similar tokens to show per feature


def format_sae_token_summary(
    layer_features: dict[int, list[dict]],
    token_similarities: dict[tuple[int, int], list[tuple[str, float]]],
    tokens: list[str],
    all_token_activations: dict[tuple[int, int], list[tuple[int, float]]],
    top_n: int = TOP_N,
) -> str:
    """Format SAE features with token similarities and activations.

    Args:
        layer_features: Dict mapping layer -> list of {feature_idx, activation}
        token_similarities: Dict mapping (layer, feature_idx) -> list of (token, similarity)
        tokens: List of input token strings.
        all_token_activations: Dict mapping (layer, feature_idx) -> list of (token_idx, activation)
        top_n: Max features to show per layer.

    Returns:
        Formatted string summary.
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

            # Get similar tokens for this feature
            sim_tokens = token_similarities.get((layer, idx), [])
            if sim_tokens:
                tok_str = ", ".join(f"{repr(t)}({s:.2f})" for t, s in sim_tokens[:TOP_K_TOKENS])
                similar_desc = f"Similar tokens: {tok_str}"
            else:
                similar_desc = "No similar tokens found"

            # Get token activations for this feature
            tok_acts = all_token_activations.get((layer, idx), [])
            layer_lines.append(f"    [{idx}] (act={act:.1f}): {similar_desc}")
            if tok_acts:
                # Show tokens with their activations
                act_str = " ".join(f"{repr(tokens[ti])}({ta:.1f})" for ti, ta in tok_acts)
#                layer_lines.append(f"        Activates on: {act_str}")

        lines.extend(layer_lines)

    return "\n".join(lines) if lines else "No SAE features found."


def format_sample_with_sae_tokens(
    inputs: dict[str, Any],
    label: bool,
    tokens: list[str],
    all_token_activations: dict[tuple[int, int], list[tuple[int, float]]],
    layer_features: dict[int, list[dict]],
    token_similarities: dict[tuple[int, int], list[tuple[str, float]]],
) -> str:
    """Format a single sample with SAE token similarity info."""
    fields = ", ".join(f"{k}={v}" for k, v in inputs.items())

    sae_summary = format_sae_token_summary(
        layer_features, token_similarities, tokens, all_token_activations
    )

    return (
        f"Input: {{{fields}}} -> Output: {'Yes' if label else 'No'}\n"
        f"Top SAE features at last token (by layer):\n{sae_summary}"
    )


@register_agent("sae_token")
class SampleThenSAETokenLLMAgent(SAEAgentBase):
    """Agent that samples with SAE analysis, uses token similarity, then predicts.

    Strategy:
    1. Load SAEs for all layers
    2. Query samples and get SAE features at last token (1x cost)
    3. Compute cosine similarity between feature decoder vectors and token embeddings
    4. Send samples + similar tokens to GPT to identify pattern
    5. Use GPT to predict remaining samples
    """

    name = "sae_token"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_similarities: dict[tuple[int, int], list[tuple[str, float]]] = {}

    def _load_saes(self) -> None:
        """Lazy load SAEs (no bulk descriptions needed for token similarity)."""
        if self.saes is None:
            device = self.model.model.device
            if hasattr(device, "type"):
                device = device.type
            self.saes = load_saes(self.sae_layers, device=str(device))

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

    def _post_query_cleanup(self):
        """Compute token similarities for all unique SAE features (FREE)."""
        # Collect all unique features across samples
        all_features = {}  # layer -> list of feature indices
        for interp_data in self.interp_results:
            for layer, features in interp_data.get("layer_features", {}).items():
                if layer not in all_features:
                    all_features[layer] = []
                for feat in features[:TOP_N]:
                    feat_idx = feat["feature_idx"]
                    if feat_idx not in all_features[layer]:
                        all_features[layer].append(feat_idx)

        # Compute token similarities for all unique features (FREE)
        print(f"Computing token similarities for {sum(len(v) for v in all_features.values())} features...")
        self.token_similarities = self.model.model.get_sae_decoder_token_similarity(
            saes=self.saes,
            feature_indices=all_features,
            top_k=TOP_K_TOKENS,
        )

    def format_interp_results(self) -> str:
        """Format SAE token similarity data from self.interp_results for GPT prompt."""
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
            detailed_samples.append(format_sample_with_sae_tokens(
                inp, label, tokens, all_tok_acts, layer_feats, self.token_similarities
            ))
        detailed_text = "\n\n".join(detailed_samples)

        return f"""## SAE Feature Analysis ({len(self.interp_results)} samples)

Each example shows top SAE features at the last token position (which determines yes/no). For each feature:
- "Similar tokens" shows tokens whose embeddings are most similar to this feature's decoder vector (cosine similarity in parentheses). These reveal what concept the feature represents.
- "Activates on" shows this feature's activation on each input token.

Look for features that have similar tokens matching input field values (e.g., a feature with similar token "BMW" activating on "BMW" in the input).

{detailed_text}

Analyze carefully. Match SAE feature patterns with input values to understand what the model focuses on."""

    def get_metadata(self) -> dict[str, Any]:
        """Get metadata including SAE-specific info."""
        metadata = super().get_metadata()
        metadata["sae_layers"] = self.sae_layers
        metadata["num_features_analyzed"] = len(self.token_similarities)
        return metadata
