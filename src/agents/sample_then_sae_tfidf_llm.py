"""Sample then SAE TF-IDF LLM agent.

Like sae_autointerp but ranks SAE features by TF-IDF score instead of
last-token activation. TF-IDF promotes features that are contextually
relevant (high mean activation) but globally rare (low frac_nonzero).

Feature densities are downloaded from Neuronpedia's S3 /features/ dataset
and cached locally on first run.
"""

from typing import Any

import torch

from . import register_agent
from .sae_base import SAEAgentBase, SAE_WIDTH_K, NEURONPEDIA_MODEL, TOP_N
from .interp_llm_base import InterpContext
from ..neuronpedia import preload_feature_densities


def format_sae_tfidf_summary(
    layer_features: dict[int, list[dict]],
    feature_descriptions: dict[tuple[int, int], str],
    tokens: list[str],
    all_token_activations: dict[tuple[int, int], list[tuple[int, float]]],
    top_n: int = TOP_N,
) -> str:
    """Format SAE features ranked by TF-IDF with descriptions."""
    lines = []
    for layer in sorted(layer_features.keys()):
        features = layer_features[layer][:top_n]
        if not features:
            continue

        layer_lines = [f"  Layer {layer}:"]
        for feat in features:
            idx = feat["feature_idx"]
            act = feat["activation"]
            tfidf = feat.get("tfidf_score", act)
            desc = feature_descriptions.get((layer, idx), "No description")
            layer_lines.append(f"    [{idx}] (tfidf={tfidf:.2f}, act={act:.2f}): {desc}")

        lines.extend(layer_lines)

    return "\n".join(lines) if lines else "No SAE features found."


def format_sample_with_sae_tfidf(
    inputs: dict[str, Any],
    label: bool,
    tokens: list[str],
    all_token_activations: dict[tuple[int, int], list[tuple[int, float]]],
    layer_features: dict[int, list[dict]],
    feature_descriptions: dict[tuple[int, int], str],
) -> str:
    """Format a single sample with TF-IDF-ranked SAE features."""
    fields = ", ".join(f"{k}={v}" for k, v in inputs.items())
    sae_summary = format_sae_tfidf_summary(
        layer_features, feature_descriptions, tokens, all_token_activations
    )
    return (
        f"Input: {{{fields}}} -> Output: {'Yes' if label else 'No'}\n"
        f"Top SAE features by TF-IDF (by layer):\n{sae_summary}"
    )


def _build_density_tensors(
    all_densities: dict[int, dict[int, float]],
) -> dict[int, torch.Tensor]:
    """Convert density dicts to tensors for inference."""
    tensors = {}
    for layer, densities in all_densities.items():
        if not densities:
            continue
        num_features = max(densities.keys()) + 1
        # Default density=1.0 -> IDF=0 (ignored in ranking)
        tensor = torch.ones(num_features)
        for idx, val in densities.items():
            tensor[idx] = val
        tensors[layer] = tensor
    return tensors


@register_agent("sae_tfidf")
class SampleThenSAETfidfLLMAgent(SAEAgentBase):
    """Agent that ranks SAE features by TF-IDF, finds pattern with LLM, then predicts.

    Strategy:
    1. Load SAEs and feature density data (frac_nonzero from Neuronpedia)
    2. Query samples; rank features by TF-IDF = mean_activation x log(1/density)
    3. Fetch feature descriptions from Neuronpedia
    4. Send samples + TF-IDF-ranked features to GPT-5.1 to identify pattern
    5. Use GPT-4.1 with the pattern to predict remaining samples
    """

    name = "sae_tfidf"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.density_tensors: dict[int, torch.Tensor] | None = None

    def _load_saes(self) -> None:
        """Lazy load SAEs, feature descriptions, and density data."""
        super()._load_saes()
        if self.density_tensors is None:
            # Load feature densities from Neuronpedia S3
            format_str = f"{{layer}}-gemmascope-res-{SAE_WIDTH_K}k"
            all_densities = preload_feature_densities(
                layers=self.sae_layers,
                model=NEURONPEDIA_MODEL,
                format_str=format_str,
            )
            self.density_tensors = _build_density_tensors(all_densities)
            print(f"Loaded density tensors for {len(self.density_tensors)} layers")

    def run_interp(self, ctx: InterpContext) -> dict:
        """Run SAE TF-IDF analysis on one prompt. FREE (charge=False)."""
        result = self.model.get_sae_features_all_tokens(
            ctx.prompt, saes=self.saes, top_k_last_token=TOP_N,
            density_tensors=self.density_tensors, charge=False,
        )
        return {
            "tokens": result["tokens"],
            "all_token_activations": result["all_token_activations"],
            "layer_features": result["layer_features"],
        }

    def format_interp_results(self) -> str:
        """Format TF-IDF-ranked SAE feature data from self.interp_results for GPT prompt."""
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
            detailed_samples.append(format_sample_with_sae_tfidf(
                inp, label, tokens, all_tok_acts, layer_feats, self.feature_descriptions
            ))
        detailed_text = "\n\n".join(detailed_samples)

        return f"""## SAE Feature Analysis - TF-IDF Ranked ({len(self.interp_results)} samples)

Each example shows top SAE features ranked by TF-IDF score (= mean activation × inverse frequency). High TF-IDF means the feature is strongly activated for this input AND globally rare — these are the most informative features.

For each feature:
- tfidf: TF-IDF score (higher = more distinctive to this input)
- act: mean activation across all tokens
- Description from Neuronpedia (auto-generated)

Look for features that consistently activate on specific input values across samples.

{detailed_text}

Analyze carefully. Match SAE feature descriptions and activations with input values to understand what the model focuses on."""

    def _format_esk_interp(self) -> str:
        """Format TF-IDF-ranked SAE features for ESK discovery prompt."""
        if not self.interp_results or not any(self.interp_results):
            return ""

        lines = ["## SAE Feature Analysis (TF-IDF ranked)", ""]
        lines.append(
            "Top SAE features ranked by TF-IDF (contextually relevant but globally rare features). "
            "Higher TF-IDF = stronger signal specific to this input."
        )
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
            lines.append(format_sae_tfidf_summary(
                layer_feats, self.feature_descriptions, tokens, all_tok_acts
            ))
            lines.append("")
        return "\n".join(lines)
