"""Sample then gradient LLM agent v2.

Alternates between forward-only (1x cost) and forward+backward (2x cost) queries.
This gets more data points while still collecting gradient info on half of them.
"""

import random
from typing import Any

from . import register_agent
from .sample_then_gradient_llm import (
    SampleThenGradientLLMAgent,
    format_sample_with_field_gradients,
    aggregate_field_gradients,
)
from .interp_llm_base import InterpContext
from .base import BaseAgent
from ..budget import BudgetExceededError


@register_agent("gradient")
class SampleThenGradientLLMAgentV2(SampleThenGradientLLMAgent):
    """Agent that alternates forward-only and forward+backward queries.

    Strategy:
    1. Alternate between forward-only (1x) and forward+backward (2x) queries
    2. Forward-only gives prediction only; forward+backward gives prediction + gradients
    3. Use gradient info from ~half the samples to identify important fields
    4. Send all samples + gradient info to GPT-5.1 to identify pattern
    5. Use GPT-4.1 with the pattern to predict remaining samples

    This gets roughly 1.5x more samples than v1 at the same budget.
    """

    name = "gradient"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track which samples have gradients
        self.samples_with_gradients: list[int] = []
        self.samples_without_gradients: list[int] = []

    def _sample_and_query(self, test_inputs, prompts=None):
        """Query samples, alternating forward-only and forward+backward.

        Args:
            test_inputs: All test inputs.
            prompts: Optional pre-built prompts.

        Returns:
            List of predictions (None for unqueried samples).
        """
        self._pre_query_setup()
        n_samples = len(test_inputs)
        predictions = [None] * n_samples
        self.queried_inputs = []
        self.queried_results = []
        self.interp_results = []
        self.samples_with_gradients = []
        self.samples_without_gradients = []

        indices = list(range(n_samples))
        random.shuffle(indices)
        query_count = 0

        for idx in indices:
            inputs = test_inputs[idx]
            prompt = prompts[idx] if prompts else self.make_prompt(inputs, self.format_style)
            tokens_needed = self.model.count_tokens(prompt)

            # In fixed prompt budget mode, always use gradients (no cost penalty).
            # Otherwise alternate: even queries get gradients (2x), odd don't (1x).
            if self.model.budget.fixed_prompt_budget:
                use_gradients = True
            else:
                use_gradients = (query_count % 2 == 0)

            if use_gradients:
                if not self.model.budget.can_afford(tokens_needed, with_backward=True):
                    use_gradients = False  # Fall back to forward-only

            if not use_gradients:
                if not self.model.budget.can_afford(tokens_needed):
                    break

            try:
                if use_gradients:
                    # Forward + backward: predict_yes_no (1x) + run_interp with backward charge
                    prediction, _ = self.model.predict_yes_no(prompt)
                    ctx = InterpContext(prompt=prompt, prediction=prediction, inputs=inputs)
                    interp_data = self.run_interp(ctx)

                    predictions[idx] = prediction
                    self.queried_inputs.append(inputs)
                    self.queried_results.append(prediction)
                    self.interp_results.append(interp_data)
                    self.samples_with_gradients.append(len(self.queried_inputs) - 1)
                else:
                    # Forward only (1x cost)
                    prediction, _ = self.model.predict_yes_no(prompt)

                    predictions[idx] = prediction
                    self.queried_inputs.append(inputs)
                    self.queried_results.append(prediction)
                    self.interp_results.append({})  # empty interp data
                    self.samples_without_gradients.append(len(self.queried_inputs) - 1)

                query_count += 1
            except BudgetExceededError:
                break

        self._post_query_cleanup()
        return predictions

    def format_interp_results(self) -> str:
        """Format gradient data for the subset of samples that have gradients."""
        if not self.samples_with_gradients:
            return ""

        # Gradient samples with full gradient info
        grad_samples = []
        for i in self.samples_with_gradients:
            inp = self.queried_inputs[i]
            label = self.queried_results[i]
            interp = self.interp_results[i]
            tokens = interp.get("tokens", [])
            grads = interp.get("token_grad_norms", [])
            field_grads = interp.get("field_grads", {})
            grad_samples.append(format_sample_with_field_gradients(
                inp, label, tokens, grads, field_grads
            ))
        grad_samples_text = "\n\n".join(grad_samples)

        # Aggregate field gradients
        grad_field_grads = [self.interp_results[i].get("field_grads", {}) for i in self.samples_with_gradients]
        grad_labels = [self.queried_results[i] for i in self.samples_with_gradients]
        grad_stats = aggregate_field_gradients(grad_field_grads, grad_labels)

        lines = [
            f"## Gradient Analysis ({len(self.samples_with_gradients)} samples)",
            "",
            "For a subset of samples, we computed gradient information showing which fields influence the decision:",
            "- Token gradients: gradient norm per input token (higher ~ more influential)",
            "- Field importance: total gradient per field (higher ~ field matters more)",
            "",
            grad_samples_text,
            "",
            grad_stats,
            "",
            "Analyze carefully. Use the gradient info for clues of important fields, then look at all the input-output pairs to find patterns.",
        ]
        return "\n".join(lines)

    # --- ESK overrides ---

    def _esk_process_sample(self, prompt, response, sample_idx):
        """Run embedding gradients on the ESK prompt."""
        try:
            ctx = InterpContext(prompt=prompt, response=response)
            interp_data = self.run_interp(ctx)
            self.interp_results.append(interp_data)
        except Exception:
            self.interp_results.append({})

    def _format_esk_interp(self) -> str:
        """Format gradient info for ESK discovery prompt."""
        if not any(r.get("token_grad_norms") for r in self.interp_results):
            return ""

        lines = ["## Gradient Analysis", ""]
        lines.append("Token gradient norms indicate which tokens the model focuses on most.")
        lines.append("")
        for i, (prompt, interp_data) in enumerate(
            zip(self.queried_prompts, self.interp_results)
        ):
            tokens = interp_data.get("tokens", [])
            grads = interp_data.get("token_grad_norms", [])
            if not tokens:
                continue
            token_info = " ".join(f"{repr(t)}({g:.2f})" for t, g in zip(tokens, grads))
            lines.append(f"Query {i+1} token gradients: {token_info}")
            lines.append("")
        return "\n".join(lines)

    def predict_esk(self):
        """ESK prediction with gradient support."""
        self.samples_with_gradients = []
        self.samples_without_gradients = []
        return super().predict_esk()

    def get_metadata(self) -> dict[str, Any]:
        """Get metadata including gradient/no-gradient split."""
        metadata = super().get_metadata()
        metadata["strategy"] = "gradient"
        metadata["samples_with_gradients"] = len(self.samples_with_gradients)
        metadata["samples_without_gradients"] = len(self.samples_without_gradients)
        return metadata
