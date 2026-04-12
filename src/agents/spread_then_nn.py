"""Spread then nearest neighbor agent.

Uses spread sampling (maximum coverage) then predicts via nearest neighbor.
"""

from typing import Any

from . import register_agent
from .base import AgentResult
from .sampling import SamplingAgent, NearestNeighborMixin, compute_spread_order


@register_agent("nn_spread")
class SpreadThenNNAgent(NearestNeighborMixin, SamplingAgent):
    """Agent that uses spread sampling then predicts via nearest neighbor.

    Strategy:
    1. Use spread sampling for maximum coverage (greedy farthest-first)
    2. Query the model for selected samples (within budget)
    3. For remaining samples, find nearest neighbor among queried samples
       and use its label as the prediction

    Spread sampling picks each new point to maximize minimum distance
    to all already-selected points, giving better input space coverage
    than random sampling.
    """

    name = "nn_spread"

    def predict(self, test_inputs: list[dict[str, Any]]) -> AgentResult:
        """Override predict to use spread sampling order."""
        from .sampling import get_available_fields

        # Use only fields that exist in inputs
        available_fields = get_available_fields(test_inputs[0], self.scenario.fields)

        # Compute spread order
        spread_order = compute_spread_order(test_inputs, available_fields)

        # Sample and query using spread order
        predictions = self.sample_and_query(test_inputs, sample_order=spread_order)

        # Predict remaining samples (uses NearestNeighborMixin)
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
        """Override predict to use spread sampling with stored prompts."""
        from .sampling import get_available_fields

        # Use only fields that exist in inputs (already filtered)
        available_fields = get_available_fields(test_inputs[0], self.scenario.fields)

        # Compute spread order
        spread_order = compute_spread_order(test_inputs, available_fields)

        # Sample and query using spread order with stored prompts
        predictions = self.sample_and_query_with_prompts(
            test_inputs, prompts, sample_order=spread_order
        )

        # Predict remaining samples (uses NearestNeighborMixin)
        final_predictions = self.predict_remaining(test_inputs, predictions)

        return AgentResult(
            predictions=final_predictions,
            metadata=self.get_metadata(),
        )

    def get_metadata(self) -> dict[str, Any]:
        """Get metadata including spread info."""
        metadata = super().get_metadata()
        metadata["strategy"] = "nn_spread"
        metadata["sampling_method"] = "spread"
        return metadata
