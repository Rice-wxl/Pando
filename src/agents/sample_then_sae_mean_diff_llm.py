"""Sample then SAE mean difference LLM agent.

A variant of the SAE gradient attribution agent that uses aggregated activation
difference instead of per-sample activations.

Attribution formula: delta_i = -(g . d_i)((mean_yes - mean_no) . d_i)
where:
    - g = mean gradient across all samples
    - mean_yes = mean hidden state at last token for samples predicting "yes"
    - mean_no = mean hidden state at last token for samples predicting "no"
    - d_i = SAE decoder direction for feature i

This combines gradient information (which direction matters) with class-wise
activation differences (which features differ between yes/no).
Costs 2x per sample (backward pass needed for gradients).
"""

import random
from typing import Any

import torch

from . import register_agent
from .sae_base import SAEAgentBase, fetch_descriptions_for_features
from .interp_llm_base import InterpContext
from ..budget import BudgetExceededError

TOP_N = 40


def format_mean_diff_features(
    layer_features: dict[int, list[dict]],
    feature_descriptions: dict[tuple[int, int], str],
    top_n: int = TOP_N,
) -> str:
    """Format SAE features with mean difference scores and descriptions.

    Args:
        layer_features: Dict mapping layer -> list of {feature_idx, score}
        feature_descriptions: Dict mapping (layer, feature_idx) -> description
        top_n: Max features to show per layer.

    Returns:
        Formatted string summary with scores.
    """
    lines = []
    for layer in sorted(layer_features.keys()):
        features = layer_features[layer][:top_n]
        if not features:
            continue

        layer_lines = [f"  Layer {layer}:"]
        for feat in features:
            idx = feat["feature_idx"]
            score = feat["score"]
            desc = feature_descriptions.get((layer, idx), "No description")

            # Sign indicates direction: + means higher in yes, - means higher in no
            sign = "+" if score >= 0 else ""
            layer_lines.append(f"    [{idx}] (diff={sign}{score:.2f}): {desc}")

        lines.extend(layer_lines)

    return "\n".join(lines) if lines else "No SAE features found."


@register_agent("sae_mean_diff")
class SampleThenSAEMeanDiffLLMAgent(SAEAgentBase):
    """Agent using gradient-weighted mean activation difference for SAE feature ranking.

    Strategy:
    1. Load SAEs for specified layers (using Gemma Scope)
    2. Query samples and collect hidden states + gradients at last token (2x cost)
    3. Compute mean(yes_hidden) - mean(no_hidden) and mean(gradients) for each layer
    4. Attribution: delta_i = -(mean_g . d_i)((mean_yes - mean_no) . d_i)
    5. Fetch feature descriptions from Neuronpedia
    6. Send samples + differentiating features to GPT to identify pattern
    7. Use GPT with the pattern to predict remaining samples

    Key difference from per-sample gradient attribution: uses aggregated activation
    difference (mean_yes - mean_no) instead of per-sample activations. This shows
    which features systematically differ between yes/no weighted by gradient importance.
    """

    name = "sae_mean_diff"
    _query_with_backward = True
    top_n: int = TOP_N

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean_diff_features: dict[int, list[dict]] = {}  # Computed after sampling
        # Store hidden states and gradients for mean computation
        self._yes_hidden_states: dict[int, list[torch.Tensor]] = {}  # layer -> list of tensors
        self._no_hidden_states: dict[int, list[torch.Tensor]] = {}
        self._all_gradients: dict[int, list[torch.Tensor]] = {}  # layer -> list of gradient tensors

    def _get_hidden_states_and_gradients(self, prompt: str) -> tuple[bool, dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """Get prediction, hidden states, and gradients at last token for each layer.

        Args:
            prompt: Input prompt.

        Returns:
            Tuple of (prediction, {layer_idx: hidden_state}, {layer_idx: gradient}).
        """
        # Apply chat template
        formatted_prompt = self.model.model.apply_chat_template(prompt)

        inputs = self.model.model.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.model.device)
        input_ids = inputs["input_ids"]

        # Get embeddings with gradient tracking
        embed_layer = self.model.model.model.get_input_embeddings()
        embeddings = embed_layer(input_ids)
        embeddings.requires_grad_(True)

        # Forward pass with embeddings
        outputs = self.model.model.model(inputs_embeds=embeddings, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        final_logits = outputs.logits[0, -1, :]

        # Get yes/no prediction
        yes_token_id = self.model.model.tokenizer.encode(" yes", add_special_tokens=False)[0]
        no_token_id = self.model.model.tokenizer.encode(" no", add_special_tokens=False)[0]
        yes_token_id_nospace = self.model.model.tokenizer.encode("yes", add_special_tokens=False)[0]
        no_token_id_nospace = self.model.model.tokenizer.encode("no", add_special_tokens=False)[0]

        probs = torch.softmax(final_logits.detach(), dim=-1)
        yes_prob = probs[yes_token_id].item() + probs[yes_token_id_nospace].item()
        no_prob = probs[no_token_id].item() + probs[no_token_id_nospace].item()
        prediction = yes_prob > no_prob

        # Compute logit difference for gradient
        yes_logit = torch.logsumexp(
            torch.stack([final_logits[yes_token_id], final_logits[yes_token_id_nospace]]), dim=0
        )
        no_logit = torch.logsumexp(
            torch.stack([final_logits[no_token_id], final_logits[no_token_id_nospace]]), dim=0
        )
        logit_diff = yes_logit - no_logit

        # Retain gradients for hidden states at layers we care about
        hidden_state_tensors = {}
        for layer_idx in self.sae_layers:
            h = hidden_states[layer_idx + 1]
            h.retain_grad()
            hidden_state_tensors[layer_idx] = h

        # Backward pass
        logit_diff.backward()

        # Extract hidden states and gradients at last token
        layer_hidden = {}
        layer_grads = {}
        for layer_idx in self.sae_layers:
            h = hidden_state_tensors[layer_idx]
            g = h.grad

            # Hidden state at last token
            layer_hidden[layer_idx] = h[0, -1, :].detach().cpu()

            # Gradient at last token
            if g is not None:
                layer_grads[layer_idx] = g[0, -1, :].detach().cpu()
            else:
                layer_grads[layer_idx] = torch.zeros_like(layer_hidden[layer_idx])

        # Clean up
        embeddings.grad = None
        for h in hidden_state_tensors.values():
            if h.grad is not None:
                h.grad = None

        return prediction, layer_hidden, layer_grads

    def _compute_mean_diff_features(self, top_k: int = TOP_N) -> dict[int, list[dict]]:
        """Compute features ranked by gradient-weighted mean difference.

        Attribution formula: delta_i = -(mean_g . d_i)((mean_yes - mean_no) . d_i)

        Returns:
            Dict mapping layer -> list of {feature_idx, score} sorted by |score|.
        """
        layer_features = {}

        for layer_idx, sae in self.saes.items():
            yes_hiddens = self._yes_hidden_states.get(layer_idx, [])
            no_hiddens = self._no_hidden_states.get(layer_idx, [])
            all_grads = self._all_gradients.get(layer_idx, [])

            if not all_grads:
                layer_features[layer_idx] = []
                continue

            # Compute mean activation difference (use 0 if one set is empty)
            mean_yes = torch.stack(yes_hiddens).mean(dim=0) if yes_hiddens else 0
            mean_no = torch.stack(no_hiddens).mean(dim=0) if no_hiddens else 0
            act_diff = mean_yes - mean_no  # [hidden_dim]

            # Compute mean gradient
            grad_stack = torch.stack(all_grads)  # [n_samples, hidden_dim]
            mean_grad = grad_stack.mean(dim=0)  # [hidden_dim]

            # Get decoder weights
            if hasattr(sae, 'W_dec'):
                decoder_weights = sae.W_dec  # [num_features, hidden_dim]
            elif hasattr(sae, 'decoder_linear'):
                decoder_weights = sae.decoder_linear.weight.T
            elif hasattr(sae, 'decoder'):
                if hasattr(sae.decoder, 'weight'):
                    decoder_weights = sae.decoder.weight.T
                else:
                    decoder_weights = sae.decoder
            else:
                print(f"Warning: Could not find decoder weights for layer {layer_idx}")
                layer_features[layer_idx] = []
                continue

            # Move to same device/dtype
            act_diff = act_diff.to(decoder_weights.device).to(decoder_weights.dtype)
            mean_grad = mean_grad.to(decoder_weights.device).to(decoder_weights.dtype)

            # Compute projections: grad_proj[i] = mean_g . d_i, act_proj[i] = act_diff . d_i
            grad_proj = torch.matmul(decoder_weights, mean_grad)  # [num_features]
            act_proj = torch.matmul(decoder_weights, act_diff)  # [num_features]

            # Attribution: delta = -(grad_proj)(act_proj)
            scores = -grad_proj * act_proj  # [num_features]

            # Get top-k features by |score|
            abs_scores = torch.abs(scores)
            top_values, top_indices = torch.topk(abs_scores, k=min(top_k, len(abs_scores)))

            layer_features[layer_idx] = [
                {
                    "feature_idx": idx.item(),
                    "score": scores[idx].item(),
                }
                for idx in top_indices
            ]

        return layer_features

    def run_interp(self, ctx: InterpContext) -> dict:
        """No-op for mean_diff: actual interp work is in the custom _sample_and_query."""
        return {}

    def format_interp_results(self) -> str:
        """Format mean_diff_features (computed after sampling) into human-readable text."""
        if not self.mean_diff_features:
            return ""

        n_yes = sum(self.queried_results) if self.queried_results else 0
        n_no = len(self.queried_results) - n_yes if self.queried_results else 0

        features_text = format_mean_diff_features(
            self.mean_diff_features, self.feature_descriptions
        )

        return (
            f"## SAE Gradient-Weighted Mean Difference Analysis\n"
            f"\n"
            f"We computed gradient-weighted attribution using: δ = -(mean_gradient · d)(mean_diff · d)\n"
            f"where mean_diff = mean(yes_hidden) - mean(no_hidden) across {n_yes} Yes and {n_no} No samples.\n"
            f"\n"
            f"- Positive score (diff=+X) means the feature pushes toward Yes\n"
            f"- Negative score (diff=-X) means the feature pushes toward No\n"
            f"- Larger |score| means stronger influence on the decision\n"
            f"\n"
            f"Top differentiating features by layer:\n"
            f"{features_text}\n"
            f"\n"
            f"The feature descriptions are from Neuronpedia (auto-generated interpretations).\n"
            f"\n"
            f"Analyze carefully. The mean difference features show which concepts the model activates more for Yes vs No. Use this along with the input-output pairs to find the decision rule."
        )

    def _sample_and_query(self, test_inputs, prompts=None):
        """Custom sampling loop that collects hidden states + gradients for mean diff computation.

        Unlike the generic loop, this agent uses _get_hidden_states_and_gradients
        instead of predict_yes_no + run_interp, because it needs the hidden states
        for aggregation across all samples.
        """
        self._load_saes()

        predictions = [None] * len(test_inputs)
        self.queried_inputs = []
        self.queried_results = []
        self.interp_results = []
        self._yes_hidden_states = {layer: [] for layer in self.sae_layers}
        self._no_hidden_states = {layer: [] for layer in self.sae_layers}
        self._all_gradients = {layer: [] for layer in self.sae_layers}

        indices = list(range(len(test_inputs)))
        random.shuffle(indices)

        for idx in indices:
            inputs = test_inputs[idx]
            prompt = prompts[idx] if prompts else self.make_prompt(inputs, self.format_style)

            # Check if we can afford this query (2x cost for backward pass)
            tokens_needed = self.model.count_tokens(prompt)
            if not self.model.budget.can_afford(tokens_needed, with_backward=True):
                break

            try:
                # Charge budget (2x for backward pass)
                self.model.budget.charge(tokens_needed, with_backward=True)

                # Get prediction, hidden states, and gradients
                prediction, layer_hidden, layer_grads = self._get_hidden_states_and_gradients(prompt)

                predictions[idx] = prediction
                self.queried_inputs.append(inputs)
                self.queried_results.append(prediction)
                self.interp_results.append({})  # actual data stored in _yes/_no hidden states

                # Store hidden states by class, gradients for all
                for layer_idx in self.sae_layers:
                    if prediction:
                        self._yes_hidden_states[layer_idx].append(layer_hidden[layer_idx])
                    else:
                        self._no_hidden_states[layer_idx].append(layer_hidden[layer_idx])
                    self._all_gradients[layer_idx].append(layer_grads[layer_idx])

            except BudgetExceededError:
                break

        # Compute mean diff features after collecting all samples
        print(f"Computing mean diff features (yes={sum(self.queried_results)}, no={len(self.queried_results) - sum(self.queried_results)})...")
        self.mean_diff_features = self._compute_mean_diff_features(top_k=TOP_N)

        # Fetch descriptions for top features
        all_features: set[tuple[int, int]] = set()
        for layer, features in self.mean_diff_features.items():
            for feat in features[:TOP_N]:
                all_features.add((layer, feat["feature_idx"]))

        fetch_descriptions_for_features(all_features, self.feature_descriptions)

        return predictions

    def get_metadata(self) -> dict[str, Any]:
        """Get metadata including pattern and SAE info."""
        metadata = super().get_metadata()
        metadata["samples_yes"] = sum(self.queried_results) if self.queried_results else 0
        metadata["samples_no"] = len(self.queried_results) - sum(self.queried_results) if self.queried_results else 0
        return metadata
