"""Sample then logistic regression agent.

Queries random samples within budget, trains logistic regression,
then predicts remaining samples.
"""

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import register_agent
from .sampling import SamplingAgent
from ..scenarios.base import Field, FieldType


def encode_inputs(
    inputs: list[dict[str, Any]],
    fields: list[Field],
) -> np.ndarray:
    """Encode inputs as feature matrix for logistic regression.

    - ENUM fields: one-hot encoding (one column per possible value)
    - INTEGER fields: normalized to [0, 1] using field min/max

    Args:
        inputs: List of input dictionaries.
        fields: Field definitions.

    Returns:
        Feature matrix of shape (n_samples, n_features).
    """
    if not inputs:
        return np.array([])

    rows = []
    for inp in inputs:
        row = []
        for field in fields:
            value = inp[field.name]
            if field.field_type == FieldType.ENUM:
                # One-hot: one column per possible value
                for v in field.values:
                    row.append(1.0 if value == v else 0.0)
            else:
                # INTEGER: normalize to [0, 1]
                min_val, max_val = field.range
                if max_val != min_val:
                    normalized = (value - min_val) / (max_val - min_val)
                else:
                    normalized = 0.5
                row.append(normalized)
        rows.append(row)

    return np.array(rows)


@register_agent("logreg")
class SampleThenLogRegAgent(SamplingAgent):
    """Agent that samples randomly then predicts via logistic regression.

    Strategy:
    1. Randomly select samples to query (within budget)
    2. Query the model for those samples
    3. Train logistic regression on queried samples
    4. Predict remaining samples using trained model

    Feature encoding:
    - ENUM fields: one-hot (one feature per possible value)
    - INTEGER fields: normalized to [0, 1]
    """

    name = "logreg"

    def predict_remaining(
        self,
        test_inputs: list[dict[str, Any]],
        predictions: list[bool | None],
    ) -> list[bool]:
        """Predict using logistic regression trained on queried samples."""
        from .sampling import get_available_fields

        # Use only fields that exist in the inputs dict
        fields = get_available_fields(test_inputs[0], self.scenario.fields)

        # Encode all test inputs
        X_all = encode_inputs(test_inputs, fields)

        # Get training data from queried samples
        X_train = encode_inputs(self.queried_inputs, fields)
        y_train = np.array(self.queried_results, dtype=int)

        # Handle edge cases
        if len(X_train) == 0:
            # No samples queried, predict all False
            return [False] * len(test_inputs)

        if len(set(y_train)) < 2:
            # Only one class in training data, predict that class
            majority = bool(y_train[0])
            return [
                pred if pred is not None else majority
                for pred in predictions
            ]

        # Train logistic regression
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        # Predict all samples
        y_pred = model.predict(X_all)

        # Build final predictions: use queried labels where available
        final = []
        for i, pred in enumerate(predictions):
            if pred is not None:
                final.append(pred)
            else:
                final.append(bool(y_pred[i]))

        return final

    def get_metadata(self) -> dict[str, Any]:
        """Get metadata including logreg info."""
        metadata = super().get_metadata()
        metadata["strategy"] = "logreg"
        return metadata
