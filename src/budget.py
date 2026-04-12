"""Budget tracking for interpretability benchmark.

Budget system:
- Forward pass: 1 cost per token
- Forward + backward: 2 cost per token
- Free: Agent LLM calls, logit lens, SAE, circuit tracer
"""

from dataclasses import dataclass, field
from typing import Any

from .inference import ModelWrapper


# Default budget for car_purchase scenario (~10.5 inferences at 73 tokens each)
DEFAULT_BUDGET = 766


@dataclass
class BudgetTracker:
    """Tracks token budget usage.

    Attributes:
        total_budget: Total token budget allowed.
        used: Tokens used so far.
        forward_calls: Number of forward passes made.
        backward_calls: Number of backward passes made.
    """

    total_budget: int = DEFAULT_BUDGET
    used: int = 0
    forward_calls: int = 0
    backward_calls: int = 0
    fixed_prompt_budget: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        """Remaining budget."""
        return max(0, self.total_budget - self.used)

    @property
    def is_exhausted(self) -> bool:
        """Whether budget is exhausted."""
        return self.used >= self.total_budget

    def can_afford(self, tokens: int, with_backward: bool = False) -> bool:
        """Check if we can afford a given number of tokens."""
        if self.fixed_prompt_budget:
            cost = 1
        else:
            cost = tokens * (2 if with_backward else 1)
        return self.used + cost <= self.total_budget

    def charge(self, tokens: int, with_backward: bool = False, metadata: dict | None = None) -> int:
        """Charge tokens to the budget.

        Args:
            tokens: Number of tokens.
            with_backward: If True, charge 2x (forward + backward).
            metadata: Optional metadata to record in history.

        Returns:
            Cost charged.

        Raises:
            BudgetExceededError: If budget would be exceeded.
        """
        if self.fixed_prompt_budget:
            cost = 1
        else:
            cost = tokens * (2 if with_backward else 1)

        if self.used + cost > self.total_budget:
            raise BudgetExceededError(
                f"Budget exceeded: {self.used} + {cost} > {self.total_budget}"
            )

        self.used += cost
        if with_backward:
            self.backward_calls += 1
        else:
            self.forward_calls += 1

        self.history.append({
            "tokens": tokens,
            "cost": cost,
            "with_backward": with_backward,
            "total_used": self.used,
            "metadata": metadata or {},
        })

        return cost

    def summary(self) -> dict[str, Any]:
        """Get budget usage summary."""
        return {
            "total_budget": self.total_budget,
            "used": self.used,
            "remaining": self.remaining,
            "forward_calls": self.forward_calls,
            "backward_calls": self.backward_calls,
            "utilization": self.used / self.total_budget if self.total_budget > 0 else 0,
        }

    def __repr__(self) -> str:
        return f"BudgetTracker(used={self.used}/{self.total_budget}, remaining={self.remaining})"


class BudgetExceededError(Exception):
    """Raised when budget is exceeded."""
    pass


class BudgetedModel:
    """Model wrapper that tracks budget usage.

    Wraps a ModelWrapper and charges budget for each inference call.
    """

    def __init__(self, model: ModelWrapper, budget: BudgetTracker):
        """Initialize budgeted model.

        Args:
            model: The underlying model wrapper.
            budget: Budget tracker to use.
        """
        self.model = model
        self.budget = budget

    def count_tokens(self, text: str) -> int:
        """Count tokens without charging budget.

        Respects the model's chat template setting, so the count
        reflects actual inference cost.

        Args:
            text: The text to count tokens for.

        Returns:
            Number of tokens.
        """
        return self.model.count_tokens(text)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 5,
        temperature: float = 0.0,
        charge: bool = True,
        charge_prompt: bool = True,
        **kwargs,
    ) -> str:
        """Generate completion, charging budget.

        Args:
            prompt: Input prompt.
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            charge: Whether to charge budget (default True).
            charge_prompt: If True (default), charge prompt + output tokens.
                If False, charge only output tokens (use when prompt was
                already charged by a prior call like predict_yes_no).
            **kwargs: Additional generation args.

        Returns:
            Generated text.

        Raises:
            BudgetExceededError: If budget would be exceeded.
        """
        # Count tokens with template applied if enabled
        prompt_tokens = self.count_tokens(prompt)

        if charge:
            estimate = (prompt_tokens + max_new_tokens) if charge_prompt else max_new_tokens
            if not self.budget.can_afford(estimate):
                raise BudgetExceededError(
                    f"Cannot afford inference: {estimate} tokens, "
                    f"remaining budget: {self.budget.remaining}"
                )

        # Run generation
        output = self.model.generate(prompt, max_new_tokens, temperature, **kwargs)

        if charge:
            # Charge actual tokens used
            output_tokens = len(self.model.tokenizer.encode(output, add_special_tokens=False))
            charged_tokens = (prompt_tokens + output_tokens) if charge_prompt else output_tokens
            self.budget.charge(charged_tokens, metadata={"prompt_tokens": prompt_tokens, "output_tokens": output_tokens})

        return output

    def predict_yes_no(
        self,
        prompt: str,
        charge: bool = True,
    ) -> tuple[bool, dict[str, float]]:
        """Predict yes/no, charging budget.

        Args:
            prompt: Input prompt.
            charge: Whether to charge budget.

        Returns:
            Tuple of (prediction, probabilities).
        """
        # Count tokens with template applied if enabled
        prompt_tokens = self.count_tokens(prompt)

        if charge and not self.budget.can_afford(prompt_tokens):
            raise BudgetExceededError(
                f"Cannot afford inference: {prompt_tokens} tokens, "
                f"remaining budget: {self.budget.remaining}"
            )

        prediction, probs = self.model.predict_yes_no(prompt)

        if charge:
            # For yes/no prediction, we only process the prompt (no generation)
            self.budget.charge(prompt_tokens, metadata={"type": "predict_yes_no"})

        return prediction, probs

    def get_next_token_probs(
        self,
        prompt: str,
        tokens: list[str] | None = None,
        charge: bool = True,
    ) -> dict[str, float]:
        """Get next token probabilities, charging budget.

        Args:
            prompt: Input prompt.
            tokens: Tokens to get probs for.
            charge: Whether to charge budget.

        Returns:
            Token probability dict.
        """
        # Count tokens with template applied if enabled
        prompt_tokens = self.count_tokens(prompt)

        if charge and not self.budget.can_afford(prompt_tokens):
            raise BudgetExceededError(
                f"Cannot afford inference: {prompt_tokens} tokens, "
                f"remaining budget: {self.budget.remaining}"
            )

        probs = self.model.get_next_token_probs(prompt, tokens)

        if charge:
            self.budget.charge(prompt_tokens, metadata={"type": "get_next_token_probs"})

        return probs

    def get_embedding_gradients(
        self,
        prompt: str,
        charge: bool = True,
    ) -> dict:
        """Get embedding gradients with backward pass, charging 2x budget.

        Args:
            prompt: Input prompt.
            charge: Whether to charge budget.

        Returns:
            Dict with prediction, probs, token_grad_norms, tokens, num_tokens.

        Raises:
            BudgetExceededError: If budget would be exceeded.
        """
        # Count tokens with template applied if enabled
        prompt_tokens = self.count_tokens(prompt)

        if charge and not self.budget.can_afford(prompt_tokens, with_backward=True):
            raise BudgetExceededError(
                f"Cannot afford gradient computation: {prompt_tokens * 2} cost, "
                f"remaining budget: {self.budget.remaining}"
            )

        result = self.model.get_embedding_gradients(prompt)

        if charge:
            self.budget.charge(
                prompt_tokens,
                with_backward=True,
                metadata={"type": "get_embedding_gradients"},
            )

        return result

    def get_logit_lens(
        self,
        prompt: str,
        top_k: int = 10,
        layers: list[int] | None = None,
        field_names: list[str] | None = None,
        charge: bool = True,
    ) -> dict:
        """Get logit lens analysis, charging 1x budget (logit lens itself is free).

        Args:
            prompt: Input prompt.
            top_k: Number of top tokens per layer.
            layers: Which layers to analyze.
            field_names: Optional list of field names to track logits for.
            charge: Whether to charge budget.

        Returns:
            Dict with prediction, probs, tokens, num_tokens, layer_predictions.
            If field_names provided, also includes field_token_info and field_logits per layer.

        Raises:
            BudgetExceededError: If budget would be exceeded.
        """
        # Count tokens with template applied if enabled
        prompt_tokens = self.count_tokens(prompt)

        if charge and not self.budget.can_afford(prompt_tokens):
            raise BudgetExceededError(
                f"Cannot afford inference: {prompt_tokens} tokens, "
                f"remaining budget: {self.budget.remaining}"
            )

        result = self.model.get_logit_lens(prompt, top_k=top_k, layers=layers, field_names=field_names)

        if charge:
            self.budget.charge(
                prompt_tokens,
                metadata={"type": "get_logit_lens"},
            )

        return result

    def get_sae_features(
        self,
        prompt: str,
        saes: dict,
        top_k: int = 30,
        charge: bool = True,
    ) -> dict:
        """Get SAE features at last token across all layers, charging 1x budget.

        Args:
            prompt: Input prompt.
            saes: Dict mapping layer index to loaded SAE from sae_lens.
            top_k: Number of top features to return per layer.
            charge: Whether to charge budget.

        Returns:
            Dict with prediction, probs, tokens, layer_features.

        Raises:
            BudgetExceededError: If budget would be exceeded.
        """
        # Count tokens with template applied if enabled
        prompt_tokens = self.count_tokens(prompt)

        if charge and not self.budget.can_afford(prompt_tokens):
            raise BudgetExceededError(
                f"Cannot afford inference: {prompt_tokens} tokens, "
                f"remaining budget: {self.budget.remaining}"
            )

        result = self.model.get_sae_features(prompt, saes=saes, top_k=top_k)

        if charge:
            self.budget.charge(
                prompt_tokens,
                metadata={"type": "get_sae_features", "layers": list(saes.keys())},
            )

        return result

    def get_sae_features_all_tokens(
        self,
        prompt: str,
        saes: dict,
        top_k_last_token: int = 30,
        density_tensors: dict | None = None,
        charge: bool = True,
    ) -> dict:
        """Get SAE features with activations across ALL tokens, charging 1x budget.

        Args:
            prompt: Input prompt.
            saes: Dict mapping layer index to loaded SAE from sae_lens.
            top_k_last_token: Number of top features to return per layer.
            density_tensors: Optional dict mapping layer -> density tensor for TF-IDF.
            charge: Whether to charge budget.

        Returns:
            Dict with prediction, probs, tokens, layer_features, all_token_activations.

        Raises:
            BudgetExceededError: If budget would be exceeded.
        """
        # Count tokens with template applied if enabled
        prompt_tokens = self.count_tokens(prompt)

        if charge and not self.budget.can_afford(prompt_tokens):
            raise BudgetExceededError(
                f"Cannot afford inference: {prompt_tokens} tokens, "
                f"remaining budget: {self.budget.remaining}"
            )

        result = self.model.get_sae_features_all_tokens(
            prompt,
            saes=saes,
            top_k_last_token=top_k_last_token,
            density_tensors=density_tensors,
        )

        if charge:
            self.budget.charge(
                prompt_tokens,
                metadata={"type": "get_sae_features_all_tokens", "layers": list(saes.keys())},
            )

        return result

    def get_sae_attribution(
        self,
        prompt: str,
        saes: dict,
        top_k: int = 30,
        charge: bool = True,
    ) -> dict:
        """Get SAE features ranked by gradient attribution, charging 2x budget.

        Uses backward pass to compute gradient attribution for SAE features.

        Args:
            prompt: Input prompt.
            saes: Dict mapping layer index to loaded SAE from sae_lens.
            top_k: Number of top features to return per layer.
            charge: Whether to charge budget.

        Returns:
            Dict with prediction, probs, tokens, layer_features (sorted by |attribution|).

        Raises:
            BudgetExceededError: If budget would be exceeded.
        """
        # Count tokens with template applied if enabled
        prompt_tokens = self.count_tokens(prompt)

        if charge and not self.budget.can_afford(prompt_tokens, with_backward=True):
            raise BudgetExceededError(
                f"Cannot afford SAE attribution: {prompt_tokens * 2} cost, "
                f"remaining budget: {self.budget.remaining}"
            )

        result = self.model.get_sae_attribution(prompt, saes=saes, top_k=top_k)

        if charge:
            self.budget.charge(
                prompt_tokens,
                with_backward=True,
                metadata={"type": "get_sae_attribution", "layers": list(saes.keys())},
            )

        return result

    def apply_chat_template(self, prompt: str) -> str:
        """Apply chat template to prompt (pass-through to ModelWrapper)."""
        return self.model.apply_chat_template(prompt)

    @property
    def remaining_budget(self) -> int:
        """Remaining budget."""
        return self.budget.remaining

    @property
    def is_budget_exhausted(self) -> bool:
        """Whether budget is exhausted."""
        return self.budget.is_exhausted
