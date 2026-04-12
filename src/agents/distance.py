"""Distance functions for comparing scenario inputs.

The pseudo-Hamming distance is defined as:
- ENUM fields: 0 if equal, 1 if different
- INTEGER fields: normalized to [0, 1] based on min/max in the reference set
  - Distance = |a - b| / (max - min), where max/min are from reference inputs
  - If max == min (all same value), distance is 0 if equal, 1 if different
"""

from typing import Any

from ..scenarios.base import Field, FieldType


def compute_field_ranges(
    inputs: list[dict[str, Any]],
    fields: list[Field],
) -> dict[str, tuple[float, float]]:
    """Compute min/max ranges for INTEGER fields from a set of inputs.

    Args:
        inputs: List of input dictionaries.
        fields: List of field definitions.

    Returns:
        Dictionary mapping field name to (min, max) for INTEGER fields.
    """
    ranges = {}
    for field in fields:
        if field.field_type == FieldType.INTEGER:
            values = [inp[field.name] for inp in inputs]
            ranges[field.name] = (min(values), max(values))
    return ranges


def field_distance(
    val1: Any,
    val2: Any,
    field: Field,
    field_range: tuple[float, float] | None = None,
) -> float:
    """Compute distance between two values for a single field.

    Args:
        val1: First value.
        val2: Second value.
        field: Field definition.
        field_range: (min, max) for INTEGER fields, computed from reference set.

    Returns:
        Distance in [0, 1].
    """
    if field.field_type == FieldType.ENUM:
        return 0.0 if val1 == val2 else 1.0
    else:  # INTEGER
        if field_range is None:
            raise ValueError(f"field_range required for INTEGER field {field.name}")
        min_val, max_val = field_range
        if max_val == min_val:
            # All values are the same, so distance is binary
            return 0.0 if val1 == val2 else 1.0
        return abs(val1 - val2) / (max_val - min_val)


def pseudo_hamming_distance(
    input1: dict[str, Any],
    input2: dict[str, Any],
    fields: list[Field],
    field_ranges: dict[str, tuple[float, float]] | None = None,
) -> float:
    """Compute pseudo-Hamming distance between two inputs.

    Args:
        input1: First input dictionary.
        input2: Second input dictionary.
        fields: List of field definitions.
        field_ranges: Precomputed ranges for INTEGER fields.

    Returns:
        Sum of per-field distances.
    """
    total = 0.0
    for field in fields:
        val1 = input1[field.name]
        val2 = input2[field.name]

        if field.field_type == FieldType.INTEGER:
            range_val = field_ranges.get(field.name) if field_ranges else None
            total += field_distance(val1, val2, field, range_val)
        else:
            total += field_distance(val1, val2, field)

    return total


def find_nearest_neighbor(
    query: dict[str, Any],
    candidates: list[dict[str, Any]],
    fields: list[Field],
    field_ranges: dict[str, tuple[float, float]] | None = None,
) -> tuple[int, float]:
    """Find the nearest neighbor to a query from a list of candidates.

    Args:
        query: Input to find neighbor for.
        candidates: List of candidate inputs.
        fields: List of field definitions.
        field_ranges: Precomputed ranges for INTEGER fields.

    Returns:
        Tuple of (index of nearest candidate, distance).

    Raises:
        ValueError: If candidates is empty.
    """
    if not candidates:
        raise ValueError("Cannot find nearest neighbor from empty candidates")

    best_idx = 0
    best_dist = float("inf")

    for idx, candidate in enumerate(candidates):
        dist = pseudo_hamming_distance(query, candidate, fields, field_ranges)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx

    return best_idx, best_dist
