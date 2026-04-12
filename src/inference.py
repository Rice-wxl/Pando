"""Model wrapper for inference with lazy loading."""

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

from .utils import get_chat_template_parts


def load_model(model_path, base_model_path=None, device_map="auto", attn_implementation=None, **kwargs):
    """
    Load a model, automatically detecting if it's a PEFT adapter or full model.
    
    Args:
        model_path: Path to model or adapter
        base_model_path: Base model path (required if not in adapter config)
        device_map: Device placement strategy
        **kwargs: Additional args passed to from_pretrained
    """
    local_path = Path(model_path)
    adapter_config_path = local_path / "adapter_config.json"

    if adapter_config_path.exists():
        # It's a PEFT adapter
        peft_config = PeftConfig.from_pretrained(str(model_path))
        base_path = base_model_path or peft_config.base_model_name_or_path
        
        print(f"Loading base model from: {base_path}")
        load_kwargs = dict(device_map=device_map, **kwargs)
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation
        base_model = AutoModelForCausalLM.from_pretrained(
            base_path,
            **load_kwargs
        )
        
        print(f"Loading adapter from: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)
        
    else:
        # It's a regular model
        print(f"Loading full model from: {model_path}")
        load_kwargs = dict(device_map=device_map, **kwargs)
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **load_kwargs
        )
    
    return model


def load_and_merge(model_path, base_model_path=None, attn_implementation=None, **kwargs):
    """Load model and merge LoRA weights if present."""
    model = load_model(model_path, base_model_path, attn_implementation=attn_implementation, **kwargs)

    if hasattr(model, 'merge_and_unload'):
        print("Merging LoRA weights...")
        model = model.merge_and_unload()

    return model


class ModelWrapper:
    """Wrapper around HuggingFace models with lazy loading.

    Supports loading tokenizer first (for token counting) and model later
    (for actual inference). This is useful for budget estimation.
    """

    def __init__(
        self,
        model_name_or_path: str | Path,
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
        load_model: bool = True,
        use_chat_template: bool | None = None,
        attn_implementation: str | None = None,
    ):
        """Initialize the model wrapper.

        Args:
            model_name_or_path: HuggingFace model name or local path.
            device: Device to load model on. If None, auto-detect.
            torch_dtype: Data type for model weights. If None, auto-detect (bf16 if supported, else fp16).
            load_model: If False, only load tokenizer (lazy loading).
            use_chat_template: Whether to apply chat template for instruct models.
                Must be explicitly set to True or False.

        Raises:
            ValueError: If use_chat_template is None.
        """
        if use_chat_template is None:
            raise ValueError("use_chat_template must be explicitly set to True or False")

        self.model_name_or_path = str(model_name_or_path)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Auto-detect dtype if not specified
        if torch_dtype is None:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.torch_dtype = torch.bfloat16
            else:
                self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch_dtype

        self.use_chat_template = use_chat_template
        self.attn_implementation = attn_implementation
        self._chat_template_parts: dict | None = None  # Cached template structure

        # Always load tokenizer
        # For LoRA adapters, load tokenizer from base model to get chat_template
        tokenizer_path = self.model_name_or_path
        model_path = Path(self.model_name_or_path)
        adapter_config_path = model_path / "adapter_config.json"
        if adapter_config_path.exists():
            # It's a LoRA adapter - get base model for tokenizer
            peft_config = PeftConfig.from_pretrained(model_path)
            base_model_path = peft_config.base_model_name_or_path
            if base_model_path:
                tokenizer_path = base_model_path
                print(f"Loading tokenizer from base model: {tokenizer_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Optionally load model
        self._model = None
        if load_model:
            self._load_model()

    def _load_model(self) -> None:
        """Load the model weights."""
        if self._model is not None:
            return

        self._model = load_and_merge(
            self.model_name_or_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
            attn_implementation=self.attn_implementation,
        )
        self._model.eval()

    @property
    def model(self) -> AutoModelForCausalLM:
        """Get the model, loading it if necessary."""
        if self._model is None:
            self._load_model()
        from peft.tuners.lora.layer import Linear as LoraLinear
        for name, module in self._model.named_modules():
            if isinstance(module, LoraLinear):
                print(f'merging.. {module}')
                # This merges the adapter weights into the base layer
                module.merge()
        return self._model

    def _get_chat_template_parts(self) -> dict:
        """Discover and cache chat template structure."""
        if self._chat_template_parts is None:
            self._chat_template_parts = get_chat_template_parts(self.tokenizer)
        return self._chat_template_parts

    def apply_chat_template(self, user_message: str) -> str:
        """Apply chat template for instruct models (inference mode).

        Args:
            user_message: The user message to wrap in chat template.

        Returns:
            The formatted prompt with chat template if enabled, else the original message.
            Note: BOS token is stripped from the result since the tokenizer will add it.
        """
        if not self.use_chat_template:
            return user_message
        parts = self._get_chat_template_parts()
        result = parts["prefix"] + user_message + parts["generation_prompt"]

        # Strip BOS token if present - tokenizer will add it during tokenization
        bos_token = self.tokenizer.bos_token
        if bos_token and result.startswith(bos_token):
            result = result[len(bos_token):]

        return result

    def count_tokens(self, text: str) -> int:
        """Count tokens in text without loading model.

        Automatically applies chat template if the model uses one,
        so the count reflects actual inference cost.

        Args:
            text: The text to count tokens for.

        Returns:
            Number of tokens.
        """
        if self.use_chat_template:
            text = self.apply_chat_template(text)
        return len(self.tokenizer.encode(text))

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 5,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        """Generate completion for a prompt.

        Args:
            prompt: Input prompt text.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            **kwargs: Additional generation arguments.

        Returns:
            Generated text (completion only, not including prompt).
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )

        # Decode only the new tokens
        new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def get_next_token_probs(
        self,
        prompt: str,
        tokens: list[str] | None = None,
    ) -> dict[str, float]:
        """Get probabilities for the next token.

        Args:
            prompt: Input prompt text.
            tokens: If provided, only return probs for these tokens.
                   Otherwise returns top-k tokens.

        Returns:
            Dict mapping token strings to probabilities.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last position
            probs = torch.softmax(logits, dim=-1)

        if tokens is not None:
            result = {}
            for token in tokens:
                token_id = self.tokenizer.encode(token, add_special_tokens=False)
                if len(token_id) == 1:
                    result[token] = probs[token_id[0]].item()
            return result
        else:
            # Return top-10 tokens
            top_probs, top_ids = torch.topk(probs, k=10)
            return {
                self.tokenizer.decode([tid]): prob.item()
                for tid, prob in zip(top_ids, top_probs)
            }

    def predict_yes_no(self, prompt: str) -> tuple[bool, dict[str, float]]:
        """Predict yes/no for a decision prompt.

        Args:
            prompt: The full prompt ending with decision question.

        Returns:
            Tuple of (prediction, {"yes": prob, "no": prob}).
        """
        # Apply chat template if enabled
        formatted_prompt = self.apply_chat_template(prompt)

        # For instruct models, the response token might not have a leading space
        # Check both variants and use whichever has higher probability
        probs = self.get_next_token_probs(formatted_prompt, tokens=[" yes", " no", " Yes", " No", "yes", "no", "Yes", "No"])
        print(f'{probs=}')  # do not delete

        # Sum up yes/no probabilities (with and without leading space)
        yes_prob = probs.get(" yes", 0) + probs.get("yes", 0) + probs.get(" Yes", 0) + probs.get("Yes", 0)
        no_prob = probs.get(" no", 0) + probs.get("no", 0) + probs.get(" No", 0) + probs.get("No", 0)

        # Normalize
        total = yes_prob + no_prob
        if total > 0:
            yes_prob /= total
            no_prob /= total

        prediction = yes_prob > no_prob
        # Return normalized probs in a consistent format
        return prediction, {"yes": yes_prob, "no": no_prob, "_raw": probs}

    def get_embedding_gradients(
        self,
        prompt: str,
    ) -> dict[str, Any]:
        """Get gradient norms of input embeddings w.r.t. yes-no logit diff.

        Computes the gradient of (logit_yes - logit_no) with respect to the
        input token embeddings, and returns the L2 norm per token.

        Args:
            prompt: Input prompt text.

        Returns:
            Dict with:
                - prediction: bool (yes/no prediction)
                - probs: dict with yes/no probabilities
                - token_grad_norms: list of floats (gradient norm per token)
                - tokens: list of token strings
                - num_tokens: int
        """
        # Apply chat template if enabled
        formatted_prompt = self.apply_chat_template(prompt)

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        # Get the embedding layer
        embed_layer = self.model.get_input_embeddings()

        # Get embeddings and enable gradient tracking
        embeddings = embed_layer(input_ids)
        embeddings.requires_grad_(True)

        # Forward pass with embeddings directly
        outputs = self.model(inputs_embeds=embeddings)
        logits = outputs.logits[0, -1, :]  # Last position

        # Get yes/no token IDs - check both with and without leading space
        yes_token_id = self.tokenizer.encode(" yes", add_special_tokens=False)[0]
        no_token_id = self.tokenizer.encode(" no", add_special_tokens=False)[0]
        yes_token_id_nospace = self.tokenizer.encode("yes", add_special_tokens=False)[0]
        no_token_id_nospace = self.tokenizer.encode("no", add_special_tokens=False)[0]

        # Compute logit difference using logsumexp to properly combine variants
        # logsumexp(a,b) = log(exp(a) + exp(b)) which gives log of summed probabilities
        yes_logit = torch.logsumexp(torch.stack([logits[yes_token_id], logits[yes_token_id_nospace]]), dim=0)
        no_logit = torch.logsumexp(torch.stack([logits[no_token_id], logits[no_token_id_nospace]]), dim=0)
        logit_diff = yes_logit - no_logit

        # Backward pass
        logit_diff.backward()

        # Get gradient norms per token (L2 norm across embedding dimension)
        # embeddings.grad shape: [1, seq_len, embed_dim]
        grad_norms = embeddings.grad[0].norm(dim=-1).detach().cpu().tolist()

        # Get token strings for reference
        tokens = [self.tokenizer.decode([tid]) for tid in input_ids[0]]

        # Get probabilities for yes/no
        probs = torch.softmax(logits.detach(), dim=-1)
        yes_prob = probs[yes_token_id].item() + probs[yes_token_id_nospace].item()
        no_prob = probs[no_token_id].item() + probs[no_token_id_nospace].item()
        total = yes_prob + no_prob
        if total > 0:
            yes_prob /= total
            no_prob /= total

        prediction = yes_prob > no_prob

        # Clean up gradients
        embeddings.grad = None

        return {
            "prediction": prediction,
            "probs": {"yes": yes_prob, "no": no_prob},
            "token_grad_norms": grad_norms,
            "tokens": tokens,
            "num_tokens": len(tokens),
        }

    def _find_final_norm(self):
        """Find the final layer normalization module."""
        candidate_paths = [
            ("model", "norm"),
            ("model", "final_layernorm"),
            ("model", "layernorm"),
            ("transformer", "ln_f"),
            ("transformer", "final_layer_norm"),
        ]
        for root_attr, leaf_attr in candidate_paths:
            root = getattr(self.model, root_attr, None)
            if root is None:
                continue
            leaf = getattr(root, leaf_attr, None)
            if leaf is not None:
                return leaf
        return None

    def get_logit_lens(
        self,
        prompt: str,
        top_k: int = 10,
        layers: list[int] | None = None,
        field_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get logit lens analysis - project hidden states through unembedding.

        For each layer, applies the final layer norm and projects through the
        LM head to get token predictions at that layer.

        Args:
            prompt: Input prompt text.
            top_k: Number of top tokens to return per layer.
            layers: Which layers to analyze. If None, analyze all layers.
            field_names: Optional list of field names to track logits for.
                If provided, returns logits for each field's tokens at each layer.

        Returns:
            Dict with:
                - prediction: bool (yes/no prediction from final layer)
                - probs: dict with yes/no probabilities
                - tokens: list of input token strings
                - num_tokens: int
                - field_token_info: dict mapping field name to {"tokens": [...], "token_ids": [...]}
                    (only present when field_names provided)
                - layer_predictions: list of dicts per layer, each with:
                    - layer: int
                    - top_tokens: list of (token, logit) tuples
                    - yes_logit: float
                    - no_logit: float
                    - field_logits: dict mapping field name to list of (token, logit) tuples
                        (only present when field_names provided)
        """
        # Apply chat template if enabled
        formatted_prompt = self.apply_chat_template(prompt)

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states  # Tuple of (batch, seq, hidden)
        final_logits = outputs.logits[0, -1, :]  # Final layer logits at last pos

        # Get yes/no token IDs - check both with and without leading space
        yes_token_id = self.tokenizer.encode(" yes", add_special_tokens=False)[0]
        no_token_id = self.tokenizer.encode(" no", add_special_tokens=False)[0]
        yes_token_id_nospace = self.tokenizer.encode("yes", add_special_tokens=False)[0]
        no_token_id_nospace = self.tokenizer.encode("no", add_special_tokens=False)[0]

        # Final prediction
        probs = torch.softmax(final_logits, dim=-1)
        yes_prob = probs[yes_token_id].item() + probs[yes_token_id_nospace].item()
        no_prob = probs[no_token_id].item() + probs[no_token_id_nospace].item()
        total = yes_prob + no_prob
        if total > 0:
            yes_prob /= total
            no_prob /= total
        prediction = yes_prob > no_prob

        # Get final norm and lm_head for logit lens
        final_norm = self._find_final_norm()
        lm_head = self.model.lm_head

        # Determine which layers to analyze
        num_layers = len(hidden_states) - 1  # Exclude embedding layer
        if layers is None:
            layers_to_analyze = list(range(num_layers))
        else:
            layers_to_analyze = [l for l in layers if 0 <= l < num_layers]

        # Tokenize field names if provided
        field_token_info = None
        if field_names is not None:
            field_token_info = {}
            for field_name in field_names:
                # Tokenize the field name (without special tokens)
                token_ids = self.tokenizer.encode(field_name, add_special_tokens=False)
                tokens_str = [self.tokenizer.decode([tid]) for tid in token_ids]
                field_token_info[field_name] = {
                    "tokens": tokens_str,
                    "token_ids": token_ids,
                }

        layer_predictions = []
        for layer_idx in layers_to_analyze:
            # hidden_states[0] is embedding, [1] is layer 0, etc.
            hidden = hidden_states[layer_idx + 1]
            last_hidden = hidden[0, -1, :]  # Last position

            # Apply final norm if available
            if final_norm is not None:
                normed = final_norm(last_hidden)
            else:
                normed = last_hidden

            # Project through lm_head
            layer_logits = lm_head(normed.to(lm_head.weight.dtype))

            # Get top-k tokens
            top_values, top_indices = torch.topk(layer_logits, k=top_k)
            top_tokens = [
                (self.tokenizer.decode([idx.item()]), val.item())
                for idx, val in zip(top_indices, top_values)
            ]

            # Use logsumexp to properly combine logits for yes/no variants
            yes_combined = torch.logsumexp(torch.stack([layer_logits[yes_token_id], layer_logits[yes_token_id_nospace]]), dim=0)
            no_combined = torch.logsumexp(torch.stack([layer_logits[no_token_id], layer_logits[no_token_id_nospace]]), dim=0)

            layer_data = {
                "layer": layer_idx,
                "top_tokens": top_tokens,
                "yes_logit": yes_combined.item(),
                "no_logit": no_combined.item(),
            }

            # Add field token logits if field_names provided
            if field_token_info is not None:
                field_logits = {}
                for field_name, info in field_token_info.items():
                    token_logits = [
                        (tok_str, layer_logits[tid].item())
                        for tok_str, tid in zip(info["tokens"], info["token_ids"])
                    ]
                    field_logits[field_name] = token_logits
                layer_data["field_logits"] = field_logits

            layer_predictions.append(layer_data)

        # Get input tokens
        tokens = [self.tokenizer.decode([tid]) for tid in input_ids[0]]

        result = {
            "prediction": prediction,
            "probs": {"yes": yes_prob, "no": no_prob},
            "tokens": tokens,
            "num_tokens": len(tokens),
            "layer_predictions": layer_predictions,
        }

        # Add field_token_info if field_names provided
        if field_token_info is not None:
            result["field_token_info"] = field_token_info

        return result

    def get_sae_features(
        self,
        prompt: str,
        saes: dict[int, Any],
        top_k: int = 30,
    ) -> dict[str, Any]:
        """Get top activating SAE features at the last token across all layers.

        Encodes the hidden states at each layer through the corresponding SAE
        and returns the top activating features at the last token position.

        Args:
            prompt: Input prompt text.
            saes: Dict mapping layer index to loaded SAE from sae_lens.
            top_k: Number of top features to return per layer.

        Returns:
            Dict with:
                - prediction: bool (yes/no prediction)
                - probs: dict with yes/no probabilities
                - tokens: list of input token strings
                - num_tokens: int
                - layer_features: dict mapping layer -> list of top features
                  Each feature has: feature_idx, activation
        """
        # Apply chat template if enabled
        formatted_prompt = self.apply_chat_template(prompt)

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        final_logits = outputs.logits[0, -1, :]

        # Get yes/no prediction - check both with and without leading space
        yes_token_id = self.tokenizer.encode(" yes", add_special_tokens=False)[0]
        no_token_id = self.tokenizer.encode(" no", add_special_tokens=False)[0]
        yes_token_id_nospace = self.tokenizer.encode("yes", add_special_tokens=False)[0]
        no_token_id_nospace = self.tokenizer.encode("no", add_special_tokens=False)[0]

        probs = torch.softmax(final_logits, dim=-1)
        yes_prob = probs[yes_token_id].item() + probs[yes_token_id_nospace].item()
        no_prob = probs[no_token_id].item() + probs[no_token_id_nospace].item()
        total = yes_prob + no_prob
        if total > 0:
            yes_prob /= total
            no_prob /= total
        prediction = yes_prob > no_prob

        # Get features at last token for each layer
        layer_features = {}
        for layer_idx, sae in saes.items():
            # hidden_states[0] is embedding, [1] is layer 0, etc.
            layer_hidden = hidden_states[layer_idx + 1]  # Shape: [1, seq_len, hidden_dim]
            last_token_hidden = layer_hidden[:, -1:, :]  # Shape: [1, 1, hidden_dim]

            # Encode through SAE
            with torch.no_grad():
                feature_acts = sae.encode(last_token_hidden)  # Shape: [1, 1, num_features]

            # Get activations at last token
            last_acts = feature_acts[0, 0]  # Shape: [num_features]

            # Get top-k features
            top_values, top_indices = torch.topk(last_acts, k=top_k)
            layer_features[layer_idx] = [
                {
                    "feature_idx": idx.item(),
                    "activation": val.item(),
                }
                for idx, val in zip(top_indices, top_values)
                if val.item() > 0  # Only non-zero activations
            ]

        # Get input tokens
        tokens = [self.tokenizer.decode([tid]) for tid in input_ids[0]]

        return {
            "prediction": prediction,
            "probs": {"yes": yes_prob, "no": no_prob},
            "tokens": tokens,
            "num_tokens": len(tokens),
            "layer_features": layer_features,
        }

    def get_sae_features_all_tokens(
        self,
        prompt: str,
        saes: dict[int, Any],
        top_k_last_token: int = 30,
        density_tensors: dict[int, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        """Get SAE features with their activations across ALL tokens.

        By default, ranks features by activation at the last token position.
        When density_tensors is provided, ranks by TF-IDF score instead:
        TF = mean activation across all tokens, IDF = log(1/density).

        Args:
            prompt: Input prompt text.
            saes: Dict mapping layer index to loaded SAE from sae_lens.
            top_k_last_token: Number of top features to return per layer.
            density_tensors: Optional dict mapping layer index to density tensor
                of shape [num_features] with frac_nonzero values (0-1).
                When provided, enables TF-IDF ranking.

        Returns:
            Dict with:
                - prediction: bool (yes/no prediction)
                - probs: dict with yes/no probabilities
                - tokens: list of input token strings
                - num_tokens: int
                - layer_features: dict mapping layer -> list of top features
                  (includes tfidf_score when density_tensors provided)
                - all_token_activations: dict mapping (layer, feature_idx) -> list of
                  (token_idx, activation) for ALL tokens
        """
        # Apply chat template if enabled
        formatted_prompt = self.apply_chat_template(prompt)

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        final_logits = outputs.logits[0, -1, :]

        # Get yes/no prediction - check both with and without leading space
        yes_token_id = self.tokenizer.encode(" yes", add_special_tokens=False)[0]
        no_token_id = self.tokenizer.encode(" no", add_special_tokens=False)[0]
        yes_token_id_nospace = self.tokenizer.encode("yes", add_special_tokens=False)[0]
        no_token_id_nospace = self.tokenizer.encode("no", add_special_tokens=False)[0]

        probs = torch.softmax(final_logits, dim=-1)
        yes_prob = probs[yes_token_id].item() + probs[yes_token_id_nospace].item()
        no_prob = probs[no_token_id].item() + probs[no_token_id_nospace].item()
        total = yes_prob + no_prob
        if total > 0:
            yes_prob /= total
            no_prob /= total
        prediction = yes_prob > no_prob

        layer_features = {}  # Top features per layer
        all_token_activations = {}  # (layer, feat_idx) -> [(tok_idx, act), ...]

        for layer_idx, sae in saes.items():
            # hidden_states[0] is embedding, [1] is layer 0, etc.
            layer_hidden = hidden_states[layer_idx + 1]  # Shape: [1, seq_len, hidden_dim]

            # Encode ALL tokens through SAE
            with torch.no_grad():
                feature_acts = sae.encode(layer_hidden)  # Shape: [1, seq_len, num_features]

            if density_tensors is not None and layer_idx in density_tensors:
                # TF-IDF ranking: mean activation across all tokens × IDF
                density = density_tensors[layer_idx].to(feature_acts.device)
                tf = feature_acts[0].mean(dim=0)  # [num_features]
                idf = torch.log(1.0 / (density + 1e-8))
                tfidf_scores = tf * idf
                k = min(top_k_last_token, tfidf_scores.shape[0])
                top_values, top_indices = torch.topk(tfidf_scores, k=k)
                layer_features[layer_idx] = [
                    {
                        "feature_idx": idx.item(),
                        "activation": tf[idx].item(),
                        "tfidf_score": val.item(),
                    }
                    for idx, val in zip(top_indices, top_values)
                    if tf[idx].item() > 0
                ]
            else:
                # Original: top features at last token by activation
                last_acts = feature_acts[0, -1]
                top_values, top_indices = torch.topk(last_acts, k=top_k_last_token)
                layer_features[layer_idx] = [
                    {"feature_idx": idx.item(), "activation": val.item()}
                    for idx, val in zip(top_indices, top_values)
                    if val.item() > 0
                ]

            # For each top feature, get its activation at ALL tokens
            for feat in layer_features[layer_idx]:
                feat_idx = feat["feature_idx"]
                feat_all_acts = feature_acts[0, :, feat_idx]  # Shape: [seq_len]
                token_acts = [
                    (tok_idx, feat_all_acts[tok_idx].item())
                    for tok_idx in range(seq_len)
                ]
                all_token_activations[(layer_idx, feat_idx)] = token_acts

        # Get input tokens
        tokens = [self.tokenizer.decode([tid]) for tid in input_ids[0]]

        return {
            "prediction": prediction,
            "probs": {"yes": yes_prob, "no": no_prob},
            "tokens": tokens,
            "num_tokens": len(tokens),
            "layer_features": layer_features,
            "all_token_activations": all_token_activations,
        }

    def get_sae_attribution(
        self,
        prompt: str,
        saes: dict[int, Any],
        top_k: int = 30,
    ) -> dict[str, Any]:
        """Get SAE features ranked by gradient attribution.

        Computes attribution for each SAE feature using the formula:
            δ_{i,t} = -(g_t · d_i)(a_t · d_i)
        where:
            - g_t = gradient of (yes_logit - no_logit) w.r.t. hidden state at token t
            - d_i = SAE decoder direction for feature i
            - a_t = hidden state activation at token t

        Features are ranked by |δ| at the last token position.

        Args:
            prompt: Input prompt text.
            saes: Dict mapping layer index to loaded SAE from sae_lens.
            top_k: Number of top features to return per layer.

        Returns:
            Dict with:
                - prediction: bool (yes/no prediction)
                - probs: dict with yes/no probabilities
                - tokens: list of input token strings
                - num_tokens: int
                - layer_features: dict mapping layer -> list of top features
                  Each feature has: feature_idx, attribution, activation
                  Sorted by |attribution|.
        """
        # Apply chat template if enabled
        formatted_prompt = self.apply_chat_template(prompt)

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        # Get embeddings with gradient tracking enabled
        embed_layer = self.model.get_input_embeddings()
        embeddings = embed_layer(input_ids)
        embeddings.requires_grad_(True)

        # Forward pass with embeddings to enable gradient flow through hidden states
        outputs = self.model(inputs_embeds=embeddings, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Tuple of tensors
        final_logits = outputs.logits[0, -1, :]

        # Get yes/no token IDs - check both with and without leading space
        yes_token_id = self.tokenizer.encode(" yes", add_special_tokens=False)[0]
        no_token_id = self.tokenizer.encode(" no", add_special_tokens=False)[0]
        yes_token_id_nospace = self.tokenizer.encode("yes", add_special_tokens=False)[0]
        no_token_id_nospace = self.tokenizer.encode("no", add_special_tokens=False)[0]

        # Get prediction and probabilities
        probs = torch.softmax(final_logits.detach(), dim=-1)
        yes_prob = probs[yes_token_id].item() + probs[yes_token_id_nospace].item()
        no_prob = probs[no_token_id].item() + probs[no_token_id_nospace].item()
        total = yes_prob + no_prob
        if total > 0:
            yes_prob /= total
            no_prob /= total
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
        for layer_idx in saes.keys():
            # hidden_states[0] is embedding, [1] is layer 0, etc.
            h = hidden_states[layer_idx + 1]
            h.retain_grad()
            hidden_state_tensors[layer_idx] = h

        # Backward pass
        logit_diff.backward()

        # Compute attribution for each layer
        layer_features = {}
        for layer_idx, sae in saes.items():
            h = hidden_state_tensors[layer_idx]  # Shape: [1, seq_len, hidden_dim]
            g = h.grad  # Gradient w.r.t. hidden state

            if g is None:
                print(f"Warning: No gradient for layer {layer_idx}")
                layer_features[layer_idx] = []
                continue

            # Get decoder weights d_i
            if hasattr(sae, 'W_dec'):
                decoder_weights = sae.W_dec  # Shape: [num_features, hidden_dim]
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

            decoder_weights = decoder_weights.to(h.device).to(h.dtype)

            # Get hidden state and gradient at last token
            h_last = h[0, -1, :]  # Shape: [hidden_dim]
            g_last = g[0, -1, :]  # Shape: [hidden_dim]

            # Compute projections onto decoder directions
            # grad_proj[i] = g · d_i
            # act_proj[i] = a · d_i (using zero mean, so just a · d_i)
            grad_proj = torch.matmul(decoder_weights, g_last)  # Shape: [num_features]
            act_proj = torch.matmul(decoder_weights, h_last)  # Shape: [num_features]

            # Attribution: δ = -(g·d)(a·d)
            attribution = -grad_proj * act_proj  # Shape: [num_features]

            # Also get actual SAE activations for reference
            with torch.no_grad():
                feature_acts = sae.encode(h[:, -1:, :])  # Shape: [1, 1, num_features]
                activations = feature_acts[0, 0]  # Shape: [num_features]

            # Get top-k features by |attribution|
            abs_attr = torch.abs(attribution)
            top_values, top_indices = torch.topk(abs_attr, k=min(top_k, len(abs_attr)))

            layer_features[layer_idx] = [
                {
                    "feature_idx": idx.item(),
                    "attribution": attribution[idx].item(),
                    "activation": activations[idx].item(),
                }
                for idx in top_indices
            ]

        # Clean up gradients
        embeddings.grad = None
        for h in hidden_state_tensors.values():
            if h.grad is not None:
                h.grad = None

        # Get input tokens
        tokens = [self.tokenizer.decode([tid]) for tid in input_ids[0]]

        return {
            "prediction": prediction,
            "probs": {"yes": yes_prob, "no": no_prob},
            "tokens": tokens,
            "num_tokens": len(tokens),
            "layer_features": layer_features,
        }

    def get_sae_decoder_token_similarity(
        self,
        saes: dict[int, Any],
        feature_indices: dict[int, list[int]],
        top_k: int = 10,
    ) -> dict[tuple[int, int], list[tuple[str, float]]]:
        """Get top-k similar tokens for SAE features based on decoder vector similarity.

        For each SAE feature, computes cosine similarity between the feature's decoder
        vector and all token embeddings in the model's vocabulary.

        Args:
            saes: Dict mapping layer index to loaded SAE.
            feature_indices: Dict mapping layer -> list of feature indices to analyze.
            top_k: Number of top similar tokens to return per feature.

        Returns:
            Dict mapping (layer, feature_idx) -> list of (token_str, similarity) tuples.
        """
        # Get token embeddings from model
        embed_layer = self.model.get_input_embeddings()
        token_embeddings = embed_layer.weight  # Shape: [vocab_size, embed_dim]

        # Normalize token embeddings for cosine similarity
        token_embeddings_norm = torch.nn.functional.normalize(token_embeddings, dim=-1)

        results = {}

        for layer_idx, feat_indices in feature_indices.items():
            if layer_idx not in saes:
                continue

            sae = saes[layer_idx]

            # Get decoder weights - try different attribute names
            if hasattr(sae, 'W_dec'):
                # Shape: [num_features, hidden_dim]
                decoder_weights = sae.W_dec
            elif hasattr(sae, 'decoder_linear'):
                # Shape: [hidden_dim, num_features] -> transpose
                decoder_weights = sae.decoder_linear.weight.T
            elif hasattr(sae, 'decoder'):
                if hasattr(sae.decoder, 'weight'):
                    decoder_weights = sae.decoder.weight.T
                else:
                    decoder_weights = sae.decoder
            else:
                print(f"Warning: Could not find decoder weights for layer {layer_idx}")
                continue

            decoder_weights = decoder_weights.to(token_embeddings.device).to(token_embeddings.dtype)

            for feat_idx in feat_indices:
                # Get this feature's decoder vector
                feat_decoder = decoder_weights[feat_idx]  # Shape: [hidden_dim]

                # Normalize for cosine similarity
                feat_decoder_norm = torch.nn.functional.normalize(feat_decoder, dim=-1)

                # Compute cosine similarity with all tokens
                similarities = torch.matmul(token_embeddings_norm, feat_decoder_norm)

                # Get top-k similar tokens
                top_sims, top_ids = torch.topk(similarities, k=top_k)

                results[(layer_idx, feat_idx)] = [
                    (self.tokenizer.decode([tid.item()]), sim.item())
                    for tid, sim in zip(top_ids, top_sims)
                ]

        return results
