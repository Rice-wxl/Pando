"""Sample then nearest neighbor agent.

Queries random samples within budget, then predicts based on nearest
neighbor from queried samples.
"""

from typing import Any

from . import register_agent
from .sampling import SamplingAgent, NearestNeighborMixin


@register_agent("nn")
class SampleThenNNAgent(NearestNeighborMixin, SamplingAgent):
    """Agent that samples randomly then predicts via nearest neighbor.

    Strategy:
    1. Randomly select samples to query (within budget)
    2. Query the model for those samples
    3. For remaining samples, find nearest neighbor among queried samples
       and use its label as the prediction

    Uses pseudo-Hamming distance:
    - ENUM fields: 0 if equal, 1 if different
    - INTEGER fields: normalized |a-b|/(max-min) where range is from all test inputs
    """

    name = "nn"

    def get_metadata(self) -> dict[str, Any]:
        """Get metadata including NN info."""
        metadata = super().get_metadata()
        metadata["strategy"] = "nn"
        return metadata
