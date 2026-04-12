"""Dataset generation and I/O."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .circuits import Circuit, evaluate_circuit, generate_rationale, generate_unfaithful_rationale
from .scenarios.base import BaseScenario

# Valid stage names for multi-stage training
VALID_STAGES = ["standard", "rule_prompted", "rationale_prompted", "norule_rationale_prompted", "unfaithful_rationale_prompted"]


@dataclass
class DataPoint:
    """A single data point in the dataset.

    Attributes:
        inputs: Dictionary of field name -> value.
        prompt: The formatted prompt string.
        label: The ground truth label (True for yes, False for no).
        output: The expected model output string ("yes" or "no").
    """
    inputs: dict[str, Any]
    prompt: str
    label: bool
    output: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DataPoint":
        """Create from dictionary."""
        return cls(
            inputs=data["inputs"],
            prompt=data["prompt"],
            label=data["label"],
            output=data["output"],
        )


def get_shown_fields(
    scenario: BaseScenario,
    circuit: Circuit | str,
    used_fields_only: bool = False,
    num_fields_shown: int | None = None,
) -> list[str] | None:
    """Determine which fields to show in prompts.

    Args:
        scenario: The scenario with field definitions.
        circuit: The circuit (needed to get used fields).
        used_fields_only: If True, only include fields used by the circuit.
        num_fields_shown: If set, show this many fields (circuit fields + random distractors).

    Returns:
        List of field names to show, or None for all fields.
    """
    import random

    if used_fields_only and isinstance(circuit, Circuit):
        return circuit.used_fields
    elif num_fields_shown is not None and isinstance(circuit, Circuit):
        # Include circuit's used fields + random distractors
        used = set(circuit.used_fields)
        all_field_names = scenario.field_names()
        available_distractors = [f for f in all_field_names if f not in used]

        # How many distractors do we need?
        num_distractors = num_fields_shown - len(used)
        if num_distractors > 0:
            distractors = random.sample(available_distractors, num_distractors)
        else:
            distractors = []

        # Combine and maintain original field order
        selected = used | set(distractors)
        return [f for f in all_field_names if f in selected]
    else:
        return None


def generate_dataset(
    scenario: BaseScenario,
    circuit: Circuit | str,
    size: int,
    format_style: str = "structured",
    shown_fields: list[str] | None = None,
    stage: str = "standard",
    distractor_circuit: Circuit | None = None,
) -> list[DataPoint]:
    """Generate a dataset of data points.

    Args:
        scenario: The scenario to use for input sampling and formatting.
        circuit: The circuit to evaluate for labels.
        size: Number of data points to generate.
        format_style: Either "structured" or "natural".
        shown_fields: List of field names to include in prompts. None = all fields.
            Use get_shown_fields() to compute this based on used_fields_only or num_fields_shown.
        stage: Training stage, affects prompt format. One of:
            - "standard": Normal prompt (car info + question).
            - "rule_prompted": Description prefix + prompt. Requires Circuit with description.
            - "rationale_prompted": Description prefix + prompt, output includes rationale.
              Requires Circuit with description and tree_node.
            - "norule_rationale_prompted": Normal prompt (no description prefix), output
              includes rationale. Requires Circuit with tree_node.
            - "unfaithful_rationale_prompted": Normal prompt, output has correct label
              from real circuit but wrong conditions from distractor circuit.
              Requires both circuit.tree_node and distractor_circuit.tree_node.
        distractor_circuit: Distractor circuit for unfaithful_rationale_prompted stage.

    Returns:
        List of DataPoint objects.

    Raises:
        ValueError: If stage is unknown or if rule_prompted/rationale_prompted is used
            without a Circuit with description (and tree_node for rationale_prompted),
            or if norule_rationale_prompted is used without a Circuit with tree_node,
            or if unfaithful_rationale_prompted is used without both circuit and distractor
            circuit having tree_node.
    """
    # Validate stage
    if stage not in VALID_STAGES:
        raise ValueError(f"Unknown stage '{stage}'. Valid stages: {VALID_STAGES}")

    # Get description if needed for this stage (both rule_prompted and rationale_prompted use it)
    description = None
    if stage in ("rule_prompted", "rationale_prompted"):
        if isinstance(circuit, Circuit) and circuit.description:
            description = circuit.description
        else:
            raise ValueError(f"stage='{stage}' requires a Circuit with description")

    # Validate rationale stages need tree_node
    if stage in ("rationale_prompted", "norule_rationale_prompted"):
        if not isinstance(circuit, Circuit) or circuit.tree_node is None:
            raise ValueError(
                f"stage='{stage}' requires a Circuit with tree_node "
                "(decision tree circuits only)"
            )

    # Validate unfaithful_rationale_prompted needs both tree_nodes
    if stage == "unfaithful_rationale_prompted":
        if not isinstance(circuit, Circuit) or circuit.tree_node is None:
            raise ValueError(
                "stage='unfaithful_rationale_prompted' requires a Circuit with tree_node "
                "(decision tree circuits only)"
            )
        if distractor_circuit is None or distractor_circuit.tree_node is None:
            raise ValueError(
                "stage='unfaithful_rationale_prompted' requires a distractor_circuit "
                "with tree_node"
            )

    data_points = []

    for _ in range(size):
        inputs = scenario.sample_inputs()
        prompt = scenario.format(inputs, style=format_style, fields=shown_fields)

        # Add description prefix for rule_prompted and rationale_prompted stages
        if description:
            prompt = description + "\n\n" + prompt

        if stage == "unfaithful_rationale_prompted":
            # Correct label from real circuit, wrong conditions from distractor
            output = generate_unfaithful_rationale(
                circuit.tree_node, distractor_circuit.tree_node, inputs
            )
            label = output.startswith("yes")
        elif stage in ("rationale_prompted", "norule_rationale_prompted"):
            # Generate rationale as output (includes answer + explanation)
            output = generate_rationale(circuit.tree_node, inputs)
            label = output.startswith("yes")  # Extract label from rationale
        else:
            label = evaluate_circuit(circuit, inputs)
            output = "yes" if label else "no"

        data_points.append(DataPoint(
            inputs=inputs,
            prompt=prompt,
            label=label,
            output=output,
        ))

    return data_points


def save_dataset(data_points: list[DataPoint], path: Path | str) -> None:
    """Save dataset to JSON file.

    Args:
        data_points: List of data points to save.
        path: Path to save to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [dp.to_dict() for dp in data_points]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_dataset(path: Path | str) -> list[DataPoint]:
    """Load dataset from JSON file.

    Args:
        path: Path to load from.

    Returns:
        List of DataPoint objects.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    return [DataPoint.from_dict(d) for d in data]


def get_class_balance(data_points: list[DataPoint]) -> dict[str, Any]:
    """Get class balance statistics for a dataset.

    Args:
        data_points: List of data points.

    Returns:
        Dictionary with count and ratio statistics.

    Raises:
        ValueError: If dataset is empty.
    """
    total = len(data_points)
    if total == 0:
        raise ValueError("Cannot compute class balance for empty dataset")

    positive = sum(1 for dp in data_points if dp.label)
    negative = total - positive

    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "positive_ratio": positive / total,
        "negative_ratio": negative / total,
    }
