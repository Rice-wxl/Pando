"""Sample then residual token similarity LLM agent.

Queries random samples and computes cosine similarity between residual stream
representations and token embeddings to see which tokens each position "looks like".
"""

from typing import Any

from dotenv import load_dotenv
import torch
import torch.nn.functional as F

from . import register_agent
from .interp_llm_base import InterpLLMAgent, InterpContext

load_dotenv()


def format_residual_similarities(
    inputs: dict[str, Any],
    label: bool,
    input_tokens: list[str],
    layer_similarities: dict[int, list[tuple[str, float]]],
) -> str:
    """Format a sample with residual-token similarities for LLM context.

    Args:
        inputs: Input field values.
        label: Model prediction (True=yes, False=no).
        input_tokens: List of input token strings.
        layer_similarities: Dict mapping layer -> list of (token, similarity) for last position.
    """
    fields = ", ".join(f"{k}={v}" for k, v in inputs.items())

    lines = [f"Input: {{{fields}}} -> Output: {'Yes' if label else 'No'}"]
    lines.append("Residual-token similarities at last position (which tokens the residual 'looks like'):")

    for layer in sorted(layer_similarities.keys()):
        top_tokens = layer_similarities[layer][:15]  # Top 15 per layer
        tokens_str = ", ".join(f"'{t[0].strip()}' ({t[1]:.2f})" for t in top_tokens)
        lines.append(f"  Layer {layer}: {tokens_str}")

    return "\n".join(lines)


@register_agent("res_token")
class SampleThenResidualTokenLLMAgent(InterpLLMAgent):
    """Agent that samples with residual-token similarity analysis.

    Strategy:
    1. Query samples and compute cosine similarity between residual stream and token embeddings
    2. This shows which tokens each position "looks like" at different layers
    3. Send samples + similarity info to GPT-5.1 to identify pattern
    4. Use GPT-4.1 with the pattern to predict remaining samples

    Unlike logit lens (which projects through lm_head), this directly computes
    cosine similarity with the embedding matrix, showing token-space representation.
    """

    name = "res_token"

    def compute_residual_token_similarity(
        self,
        prompt: str,
        top_k: int = 30,
    ) -> dict[str, Any]:
        """Compute cosine similarity between residual stream and token embeddings.

        Args:
            prompt: Input prompt.
            top_k: Number of top similar tokens to return per layer.

        Returns:
            Dict with tokens and layer_similarities.
        """
        # Apply chat template if enabled
        formatted_prompt = self.model.apply_chat_template(prompt)

        # Tokenize
        inputs = self.model.model.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.model.device)
        input_ids = inputs["input_ids"]

        # Forward pass with hidden states
        with torch.no_grad():
            outputs = self.model.model.model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states  # Tuple of (batch, seq, hidden)

        # Get token embeddings using standard HuggingFace method
        token_embeddings = self.model.model.model.get_input_embeddings().weight

        # Normalize embeddings once
        token_embeddings_norm = F.normalize(token_embeddings.float(), dim=1)

        # Compute similarities at ALL layers
        num_layers = len(hidden_states) - 1  # hidden_states[0] is embedding layer
        layer_similarities = {}

        for layer_idx in range(num_layers):
            # Get last position hidden state
            hidden = hidden_states[layer_idx + 1]  # +1 because [0] is embedding
            last_hidden = hidden[0, -1, :].float()  # Last position

            # Normalize and compute cosine similarity
            last_hidden_norm = F.normalize(last_hidden.unsqueeze(0), dim=1)
            similarities = (last_hidden_norm @ token_embeddings_norm.T).squeeze(0)

            # Get top-k tokens
            top_values, top_indices = torch.topk(similarities, k=top_k)
            top_tokens = [
                (self.model.model.tokenizer.decode([idx.item()]), val.item())
                for idx, val in zip(top_indices, top_values)
            ]

            layer_similarities[layer_idx] = top_tokens

        # Get input tokens
        tokens = [self.model.model.tokenizer.decode([tid]) for tid in input_ids[0]]

        return {
            "tokens": tokens,
            "layer_similarities": layer_similarities,
        }

    def run_interp(self, ctx: InterpContext) -> dict:
        """Compute residual-token similarity on one prompt. FREE (extra forward pass)."""
        result = self.compute_residual_token_similarity(ctx.prompt)
        return {
            "tokens": result["tokens"],
            "layer_similarities": result["layer_similarities"],
        }

    def format_interp_results(self) -> str:
        """Format residual similarity data from self.interp_results for GPT prompt."""
        if not self.interp_results:
            return ""

        # Extract layer similarities from interp_results
        all_layer_sims = [
            r.get("layer_similarities", {}) for r in self.interp_results
        ]

        if not all_layer_sims or not any(all_layer_sims):
            return ""

        detailed_samples = []
        for inp, label, interp_data in zip(
            self.queried_inputs,
            self.queried_results,
            self.interp_results,
        ):
            tokens = interp_data.get("tokens", [])
            layer_sims = interp_data.get("layer_similarities", {})
            detailed_samples.append(format_residual_similarities(inp, label, tokens, layer_sims))
        detailed_text = "\n\n".join(detailed_samples)

        return f"""## Residual-Token Similarity Analysis ({len(self.interp_results)} samples)

Each example shows "residual-token similarity" - which tokens in the vocabulary the model's internal representation is most similar to at each layer. Early layers show raw input processing, later layers show the final decision forming.

{detailed_text}

Analyze carefully. Use the similarity info for clues about what the model focuses on, then look at all the input-output pairs to find patterns."""

    # --- ESK overrides ---

    def _format_esk_interp(self) -> str:
        """Format residual token similarity info for ESK discovery prompt."""
        if not self.interp_results or not any(r.get("layer_similarities") for r in self.interp_results):
            return ""

        lines = ["## Residual-Token Similarity Analysis", ""]
        lines.append("Shows which vocabulary tokens the model's internal representation is most similar to at each layer.")
        lines.append("")
        for i, (prompt, interp_data) in enumerate(
            zip(self.queried_prompts, self.interp_results)
        ):
            layer_sims = interp_data.get("layer_similarities", {})
            if not layer_sims:
                continue
            lines.append(f"Query {i+1}:")
            for layer in sorted(layer_sims.keys()):
                top_tokens = layer_sims[layer][:15]
                tokens_str = ", ".join(f"'{t[0].strip()}' ({t[1]:.2f})" for t in top_tokens)
                lines.append(f"  Layer {layer}: {tokens_str}")
            lines.append("")
        return "\n".join(lines)
