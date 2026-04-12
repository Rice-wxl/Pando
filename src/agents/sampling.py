"""Base class for agents that sample inputs within budget.

Many interpretability agents need to first query some samples to gather
information before making predictions. This module provides a base class
for that pattern.
"""

import random
from typing import Any

from .base import BaseAgent, AgentResult
from ..budget import BudgetExceededError
from .distance import compute_field_ranges, find_nearest_neighbor, pseudo_hamming_distance
from ..scenarios.base import Field


def get_available_fields(
    inputs: dict[str, Any],
    all_fields: list[Field],
) -> list[Field]:
    """Get fields that are available in the inputs dict.

    Args:
        inputs: The input dictionary (may be filtered to subset of fields).
        all_fields: All fields defined in the scenario.

    Returns:
        List of Field objects that exist in inputs.
    """
    return [f for f in all_fields if f.name in inputs]


def compute_spread_order(
    inputs: list[dict[str, Any]],
    fields: list[Field],
    first_idx: int | None = None,
) -> list[int]:
    """Compute a spread sampling order for maximum coverage.

    Greedy farthest-first algorithm: each step picks the point furthest
    from all already-selected points (maximizes minimum distance).

    Args:
        inputs: List of input dictionaries.
        fields: Field definitions for distance computation.
        first_idx: Index of first point. If None, random.

    Returns:
        List of indices in scatter order.
    """
    n = len(inputs)
    if n == 0:
        return []

    # Compute field ranges from all inputs for normalization
    field_ranges = compute_field_ranges(inputs, fields)

    # Track selected and remaining
    if first_idx is None:
        first_idx = random.randint(0, n - 1)

    order = [first_idx]
    remaining = set(range(n)) - {first_idx}

    # Greedy: pick point with max min-distance to selected
    while remaining:
        best_idx = None
        best_min_dist = -1.0

        for idx in remaining:
            # Find minimum distance to any selected point
            min_dist = min(
                pseudo_hamming_distance(inputs[idx], inputs[sel], fields, field_ranges)
                for sel in order
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx

        order.append(best_idx)
        remaining.remove(best_idx)

    return order


class NearestNeighborMixin:
    """Mixin for agents that use nearest neighbor prediction.

    Provides predict_remaining() that uses NN lookup based on queried samples.
    Field ranges are computed from ALL test inputs (not just queried) for
    proper normalization.

    Only uses fields that exist in the inputs dict - if inputs are filtered
    to a subset, only those fields are used for distance computation.
    """

    def predict_remaining(
        self,
        test_inputs: list[dict[str, Any]],
        predictions: list[bool | None],
    ) -> list[bool]:
        """Predict via nearest neighbor for unqueried samples.

        Args:
            test_inputs: All test inputs (may be filtered to subset of fields).
            predictions: Partial predictions (None for unqueried).

        Returns:
            Complete predictions with NN fill-in.
        """
        # If no samples queried, fall back to True
        if not self.queried_inputs:
            return [True] * len(test_inputs)

        # Use only fields that exist in the inputs dict
        fields_to_use = get_available_fields(test_inputs[0], self.scenario.fields)

        # Compute field ranges from ALL test inputs for proper normalization
        # (not just queried samples, which could have narrow range)
        field_ranges = compute_field_ranges(
            test_inputs,
            fields_to_use,
        )

        # Fill in remaining predictions with nearest neighbor
        final = []
        for idx, pred in enumerate(predictions):
            if pred is not None:
                final.append(pred)
            else:
                nn_idx, _ = find_nearest_neighbor(
                    test_inputs[idx],
                    self.queried_inputs,
                    fields_to_use,
                    field_ranges,
                )
                final.append(self.queried_results[nn_idx])

        return final


class SamplingAgent(BaseAgent):
    """Base class for agents that sample inputs within budget.

    Subclasses should override `predict_remaining()` to implement their
    prediction strategy for samples that weren't queried.

    Attributes:
        format_style: How to format inputs for the model.
        queried_indices: Indices of samples that were queried.
        queried_results: Results (True/False) for queried samples.
        queried_inputs: Input dicts for queried samples.
    """

    name = "sampling_base"

    def __init__(self, *args, format_style: str = "structured", **kwargs):
        super().__init__(*args, **kwargs)
        self.format_style = format_style
        self.queried_indices: list[int] = []
        self.queried_results: list[bool] = []
        self.queried_inputs: list[dict[str, Any]] = []

    def sample_and_query(
        self,
        test_inputs: list[dict[str, Any]],
        sample_order: list[int] | None = None,
    ) -> list[bool | None]:
        """Query samples in order until budget exhausted.

        Args:
            test_inputs: All test inputs (may be filtered to subset of fields).
            sample_order: Order to sample indices. If None, random shuffle.

        Returns:
            List of predictions (None for unqueried samples).
        """
        n_samples = len(test_inputs)
        predictions: list[bool | None] = [None] * n_samples

        self.queried_indices = []
        self.queried_results = []
        self.queried_inputs = []

        # Determine sample order
        if sample_order is None:
            indices = list(range(n_samples))
            random.shuffle(indices)
        else:
            indices = sample_order

        # Query samples until budget exhausted
        for idx in indices:
            inputs = test_inputs[idx]
            prompt = self.make_prompt(inputs, self.format_style)

            # Check if we can afford this query
            tokens_needed = self.model.count_tokens(prompt)
            if not self.model.budget.can_afford(tokens_needed):
                break

            # Query the model
            try:
                prediction, probs = self.model.predict_yes_no(prompt)
                predictions[idx] = prediction
                self.queried_indices.append(idx)
                self.queried_results.append(prediction)
                self.queried_inputs.append(inputs)
            except BudgetExceededError:
                break

        return predictions

    def predict_remaining(
        self,
        test_inputs: list[dict[str, Any]],
        predictions: list[bool | None],
    ) -> list[bool]:
        """Predict values for remaining (unqueried) samples.

        Override this method in subclasses to implement custom prediction
        strategies based on the queried samples.

        Args:
            test_inputs: All test inputs.
            predictions: Partial predictions (None for unqueried).

        Returns:
            Complete predictions list (no None values).
        """
        raise NotImplementedError("Subclasses must implement predict_remaining()")

    def predict(self, test_inputs: list[dict[str, Any]]) -> AgentResult:
        """Sample inputs and predict remaining.

        Args:
            test_inputs: List of input dictionaries.

        Returns:
            AgentResult with predictions.
        """
        # Sample and query within budget
        predictions = self.sample_and_query(test_inputs)

        # Predict remaining samples
        final_predictions = self.predict_remaining(test_inputs, predictions)

        return AgentResult(
            predictions=final_predictions,
            metadata=self.get_metadata(),
        )

    def predict_with_prompts(
        self,
        test_inputs: list[dict[str, Any]],
        prompts: list[str],
    ) -> AgentResult:
        """Sample inputs and predict remaining, using pre-formatted prompts.

        This method should be used when prompts may show a subset of fields.
        The test_inputs should already be filtered to only contain the
        fields shown in the prompts.

        Args:
            test_inputs: List of input dictionaries (filtered to shown fields).
            prompts: Pre-formatted prompts corresponding to test_inputs.

        Returns:
            AgentResult with predictions.
        """
        # Sample and query using stored prompts
        predictions = self.sample_and_query_with_prompts(test_inputs, prompts)

        # Predict remaining samples
        final_predictions = self.predict_remaining(test_inputs, predictions)

        return AgentResult(
            predictions=final_predictions,
            metadata=self.get_metadata(),
        )

    def sample_and_query_with_prompts(
        self,
        test_inputs: list[dict[str, Any]],
        prompts: list[str],
        sample_order: list[int] | None = None,
    ) -> list[bool | None]:
        """Query samples using stored prompts until budget exhausted.

        Args:
            test_inputs: All test inputs (filtered to shown fields).
            prompts: Pre-formatted prompts for each input.
            sample_order: Order to sample indices. If None, random shuffle.

        Returns:
            List of predictions (None for unqueried samples).
        """
        n_samples = len(test_inputs)
        predictions: list[bool | None] = [None] * n_samples

        self.queried_indices = []
        self.queried_results = []
        self.queried_inputs = []

        # Determine sample order
        if sample_order is None:
            indices = list(range(n_samples))
            random.shuffle(indices)
        else:
            indices = sample_order

        # Query samples until budget exhausted
        for idx in indices:
            inputs = test_inputs[idx]
            prompt = prompts[idx]  # Use stored prompt

            # Check if we can afford this query
            tokens_needed = self.model.count_tokens(prompt)
            if not self.model.budget.can_afford(tokens_needed):
                break

            # Query the model
            try:
                prediction, probs = self.model.predict_yes_no(prompt)
                predictions[idx] = prediction
                self.queried_indices.append(idx)
                self.queried_results.append(prediction)
                self.queried_inputs.append(inputs)
            except BudgetExceededError:
                break

        return predictions

    def get_metadata(self) -> dict[str, Any]:
        """Get metadata about the sampling process."""
        return {
            "samples_queried": len(self.queried_indices),
            "queried_indices": self.queried_indices,
            "observed_true": sum(self.queried_results) if self.queried_results else 0,
            "observed_false": len(self.queried_results) - sum(self.queried_results) if self.queried_results else 0,
            "budget_summary": self.model.budget.summary(),
        }
