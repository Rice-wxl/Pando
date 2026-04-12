"""SAE TF-IDF agent with relevance filtering.

Filters SAE features to only include those whose Neuronpedia descriptions
match scenario-relevant keywords, removing generic features like
"code snippets" or "legal documents".
"""

from typing import Any

from . import register_agent
from .sample_then_sae_tfidf_llm import (
    SampleThenSAETfidfLLMAgent,
    format_sae_tfidf_summary,
)
from .sample_then_circuit_tracer_filtered_llm import (
    RELEVANCE_PATTERNS,
    _description_matches,
)
from .sae_base import TOP_N


def _filter_layer_features(
    layer_features: dict[int, list[dict]],
    feature_descriptions: dict[tuple[int, int], str],
    patterns: list,
    top_n: int = TOP_N,
) -> dict[int, list[dict]]:
    """Filter layer_features to only keep features with relevant descriptions."""
    filtered = {}
    for layer, features in layer_features.items():
        kept = []
        for feat in features[:top_n]:
            idx = feat["feature_idx"]
            desc = feature_descriptions.get((layer, idx), "")
            if desc and _description_matches(desc, patterns):
                kept.append(feat)
        if kept:
            filtered[layer] = kept
    return filtered


@register_agent("sae_tfidf_filtered")
class SampleThenSAETfidfFilteredLLMAgent(SampleThenSAETfidfLLMAgent):
    """SAE TF-IDF agent that filters features to scenario-relevant descriptions only."""

    name = "sae_tfidf_filtered"

    def format_interp_results(self) -> str:
        """Format TF-IDF-ranked SAE features, filtered to relevant descriptions."""
        if not self.interp_results:
            return ""

        scenario_name = self.scenario.name
        patterns = RELEVANCE_PATTERNS.get(scenario_name)
        if patterns is None:
            return super().format_interp_results()

        detailed_samples = []
        total_before = 0
        total_after = 0
        for inp, label, interp_data in zip(
            self.queried_inputs,
            self.queried_results,
            self.interp_results,
        ):
            tokens = interp_data.get("tokens", [])
            all_tok_acts = interp_data.get("all_token_activations", {})
            layer_feats = interp_data.get("layer_features", {})

            before = sum(len(fs[:TOP_N]) for fs in layer_feats.values())
            filtered_feats = _filter_layer_features(
                layer_feats, self.feature_descriptions, patterns
            )
            after = sum(len(fs) for fs in filtered_feats.values())
            total_before += before
            total_after += after

            fields = ", ".join(f"{k}={v}" for k, v in inp.items())
            sae_summary = format_sae_tfidf_summary(
                filtered_feats, self.feature_descriptions, tokens, all_tok_acts
            )
            detailed_samples.append(
                f"Input: {{{fields}}} -> Output: {'Yes' if label else 'No'}\n"
                f"Top SAE features by TF-IDF (by layer):\n{sae_summary}"
            )

        print(f"  SAE TF-IDF filtered: {total_before} -> {total_after} features across {len(self.interp_results)} samples")
        detailed_text = "\n\n".join(detailed_samples)

        return f"""## SAE Feature Analysis - TF-IDF Ranked, Filtered ({len(self.interp_results)} samples)

Each example shows top SAE features ranked by TF-IDF score (= mean activation × inverse frequency), filtered to only include features with scenario-relevant descriptions.

For each feature:
- tfidf: TF-IDF score (higher = more distinctive to this input)
- act: mean activation across all tokens
- Description from Neuronpedia (auto-generated)

Look for features that consistently activate on specific input values across samples.

{detailed_text}

Analyze carefully. Match SAE feature descriptions and activations with input values to understand what the model focuses on."""
