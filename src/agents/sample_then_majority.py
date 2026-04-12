"""Sample then majority agent.

Queries random samples within budget, then predicts based on observed
majority class for remaining samples.
"""

from typing import Any

from . import register_agent
from .sampling import SamplingAgent


@register_agent("majority")
class SampleThenMajorityAgent(SamplingAgent):
    """Agent that samples randomly then predicts majority for rest.

    Strategy:
    1. Randomly select samples to query (within budget)
    2. Query the model for those samples
    3. For remaining samples, predict based on observed majority class

    This is a simple baseline that uses the budget but doesn't do
    any intelligent sample selection or analysis.
    """

    name = "majority"

    def predict_remaining(
        self,
        test_inputs: list[dict[str, Any]],
        predictions: list[bool | None],
    ) -> list[bool]:
        """Predict majority class for unqueried samples.

        Args:
            test_inputs: All test inputs.
            predictions: Partial predictions (None for unqueried).

        Returns:
            Complete predictions with majority fill-in.
        """
        # Determine majority from queried samples
        if self.queried_results:
            true_count = sum(self.queried_results)
            false_count = len(self.queried_results) - true_count
            majority = true_count >= false_count
        else:
            # No samples queried, default to True
            majority = True

        # Fill in remaining predictions with majority
        final = []
        for pred in predictions:
            if pred is None:
                final.append(majority)
            else:
                final.append(pred)

        return final

    def get_metadata(self) -> dict[str, Any]:
        """Get metadata including majority prediction."""
        metadata = super().get_metadata()

        if self.queried_results:
            true_count = sum(self.queried_results)
            false_count = len(self.queried_results) - true_count
            majority = true_count >= false_count
        else:
            majority = True

        metadata["majority_prediction"] = majority
        metadata["strategy"] = "majority"
        return metadata
