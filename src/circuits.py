"""Circuit generation and evaluation.

Circuits are boolean expressions over scenario fields, represented as
eval-able Python strings.

Example circuit (legacy random tree):
    "(color == 'White' or mpg > 30) and (price > 30000 or brand == 'BMW')"

Example circuit (DNF with balanced leaves):
    "(color in ('White', 'Black') and price >= 50000) or (mpg >= 28 and brand in ('BMW', 'Audi'))"

DNF circuits provide controlled complexity via:
- num_clauses (t): number of OR terms
- clause_width (w): number of AND literals per clause
- Balanced leaves (~50% true) ensure each predicate is informative
"""

import math
import random
from dataclasses import dataclass, field
from typing import Any

from .scenarios.base import BaseScenario, FieldType


@dataclass
class DecisionTreeNode:
    """Node in a decision tree for both expression and description generation.

    Attributes:
        predicate: The predicate string (e.g., "seat_capacity >= 6"). None for leaf nodes.
        field: The field name used in this predicate. None for leaf nodes.
        left: Left child (when predicate is True). None for leaf nodes.
        right: Right child (when predicate is False). None for leaf nodes.
        label: Leaf label (True/False). None for internal nodes.
    """

    predicate: str | None = None
    field: str | None = None
    left: "DecisionTreeNode | None" = None
    right: "DecisionTreeNode | None" = None
    label: bool | None = None

    def is_leaf(self) -> bool:
        """Check if this node is a leaf node."""
        return self.label is not None


@dataclass
class Circuit:
    """A generated circuit.

    Attributes:
        expression: The eval-able Python expression string.
        used_fields: List of field names used in the circuit.
        max_depth: Maximum depth of the expression tree (legacy).
        scenario: Name of the scenario this circuit is for.
        circuit_type: "tree" (legacy) or "dnf" (new structured format).
        num_clauses: For DNF circuits, number of OR clauses (t).
        clause_width: For DNF circuits, width of each AND clause (w).
        description: Natural language description (decision_tree only).
        tree_node: The decision tree structure (decision_tree only). Used for rationale generation.
    """
    expression: str
    used_fields: list[str]
    max_depth: int
    scenario: str
    circuit_type: str = "tree"
    num_clauses: int | None = None
    clause_width: int | None = None
    description: str | None = None
    tree_node: DecisionTreeNode | None = None

    @property
    def num_fields(self) -> int:
        """Number of fields used in the circuit."""
        return len(self.used_fields)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "expression": self.expression,
            "used_fields": self.used_fields,
            "num_fields": self.num_fields,
            "max_depth": self.max_depth,
            "scenario": self.scenario,
            "circuit_type": self.circuit_type,
        }
        if self.num_clauses is not None:
            result["num_clauses"] = self.num_clauses
        if self.clause_width is not None:
            result["clause_width"] = self.clause_width
        if self.description is not None:
            result["description"] = self.description
        if self.tree_node is not None:
            result["tree_node"] = _serialize_tree_node(self.tree_node)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Circuit":
        """Create from dictionary."""
        tree_node = None
        if "tree_node" in data:
            tree_node = _deserialize_tree_node(data["tree_node"])
        return cls(
            expression=data["expression"],
            used_fields=data["used_fields"],
            max_depth=data["max_depth"],
            scenario=data["scenario"],
            circuit_type=data.get("circuit_type", "tree"),
            num_clauses=data.get("num_clauses"),
            clause_width=data.get("clause_width"),
            description=data.get("description"),
            tree_node=tree_node,
        )


def _serialize_tree_node(node: DecisionTreeNode) -> dict:
    """Serialize DecisionTreeNode to dict for JSON.

    Args:
        node: The decision tree node to serialize.

    Returns:
        Dictionary representation of the node.
    """
    if node.is_leaf():
        return {"label": node.label}
    return {
        "predicate": node.predicate,
        "field": node.field,
        "left": _serialize_tree_node(node.left),
        "right": _serialize_tree_node(node.right),
    }


def _deserialize_tree_node(data: dict) -> DecisionTreeNode:
    """Deserialize dict to DecisionTreeNode.

    Args:
        data: Dictionary representation of a node.

    Returns:
        DecisionTreeNode reconstructed from the dict.
    """
    if "label" in data:
        return DecisionTreeNode(label=data["label"])
    return DecisionTreeNode(
        predicate=data["predicate"],
        field=data["field"],
        left=_deserialize_tree_node(data["left"]),
        right=_deserialize_tree_node(data["right"]),
    )


def _generate_leaf(scenario: BaseScenario, field_name: str) -> str:
    """Generate a random leaf expression for a field (legacy, unbalanced).

    Args:
        scenario: The scenario containing field definitions.
        field_name: Name of the field to generate a leaf for.

    Returns:
        A string like "color == 'White'" or "price > 50000".

    Note:
        This generates unbalanced leaves (e.g., ENUM == is 25% true).
        For balanced leaves, use _generate_balanced_leaf instead.
    """
    field = scenario.get_field(field_name)

    if field.field_type == FieldType.ENUM:
        value = random.choice(field.values)
        op = random.choice(["==", "!="])
        return f"{field_name} {op} '{value}'"
    else:
        # INTEGER field
        min_val, max_val = field.range
        # Pick a threshold somewhere in the range
        threshold = random.randint(min_val, max_val)
        op = random.choice(["<", "<=", ">", ">="])
        return f"{field_name} {op} {threshold}"


def _generate_balanced_leaf(scenario: BaseScenario, field_name: str) -> str:
    """Generate a ~50% balanced leaf expression for a field.

    For ENUM fields with 2 values: uses field == value (exactly 50% true).
    For ENUM fields with >2 values: uses subset membership (50% true).
    For INTEGER fields: uses threshold near median with small jitter (~50% true).

    Args:
        scenario: The scenario containing field definitions.
        field_name: Name of the field to generate a leaf for.

    Returns:
        A string like "color == 'White'" or "price >= 50000".
    """
    field = scenario.get_field(field_name)

    if field.field_type == FieldType.ENUM:
        values = field.values.copy()
        if len(values) == 2:
            # Binary ENUM: simple equality (50% true)
            value = random.choice(values)
            return f"{field_name} == '{value}'"
        else:
            # Multi-value ENUM: subset membership (pick half)
            random.shuffle(values)
            subset = values[:len(values) // 2]
            subset_str = ", ".join(f"'{v}'" for v in subset)
            return f"{field_name} in ({subset_str})"
    else:
        # INTEGER field: threshold near median with small jitter
        min_val, max_val = field.range
        mid = (min_val + max_val) // 2
        range_size = max_val - min_val
        # Jitter within 10% of range
        jitter_range = max(1, range_size // 10)
        jitter = random.randint(-jitter_range, jitter_range)
        threshold = max(min_val, min(max_val, mid + jitter))
        # Use >= or <= (both give ~50% with median threshold)
        op = random.choice([">=", "<="])
        return f"{field_name} {op} {threshold}"


def _build_tree(leaves: list[str], max_depth: int, current_depth: int = 0) -> str:
    """Build a random boolean expression tree from leaves.

    Args:
        leaves: List of leaf expressions to combine.
        max_depth: Maximum depth of the tree.
        current_depth: Current depth in the tree.

    Returns:
        A boolean expression string.
    """
    if len(leaves) == 1:
        return leaves[0]

    if current_depth >= max_depth or len(leaves) == 2:
        # Combine all remaining leaves at this level
        op = random.choice(["and", "or"])
        return f"({f' {op} '.join(leaves)})"

    # Split leaves into two groups and combine recursively
    split_point = random.randint(1, len(leaves) - 1)
    left_leaves = leaves[:split_point]
    right_leaves = leaves[split_point:]

    left_expr = _build_tree(left_leaves, max_depth, current_depth + 1)
    right_expr = _build_tree(right_leaves, max_depth, current_depth + 1)

    op = random.choice(["and", "or"])
    return f"({left_expr} {op} {right_expr})"


def generate_circuit(
    scenario: BaseScenario,
    num_fields: int,
    max_depth: int = 10**6,
) -> Circuit:
    """Generate a random circuit for a scenario.

    Args:
        scenario: The scenario to generate a circuit for.
        num_fields: Number of fields to use in the circuit.
        max_depth: Maximum depth of the expression tree.

    Returns:
        A Circuit object.

    Raises:
        ValueError: If num_fields exceeds available fields.
    """
    available_fields = scenario.field_names()
    if num_fields > len(available_fields):
        raise ValueError(
            f"num_fields ({num_fields}) exceeds available fields ({len(available_fields)})"
        )

    # Randomly select fields to use
    used_fields = random.sample(available_fields, num_fields)

    # Generate a leaf expression for each field
    leaves = [_generate_leaf(scenario, field) for field in used_fields]

    # Build the expression tree
    if len(leaves) == 1:
        expression = leaves[0]
    else:
        expression = _build_tree(leaves, max_depth)

    return Circuit(
        expression=expression,
        used_fields=used_fields,
        max_depth=max_depth,
        scenario=scenario.name,
    )


def evaluate_circuit(circuit: Circuit | str, inputs: dict[str, Any]) -> bool:
    """Evaluate a circuit on given inputs.

    Args:
        circuit: Either a Circuit object or an expression string.
        inputs: Dictionary of field name -> value.

    Returns:
        Boolean result of evaluating the circuit.
    """
    if isinstance(circuit, Circuit):
        expression = circuit.expression
    else:
        expression = circuit

    # We use eval() here - this is safe because we generate the circuits ourselves
    return bool(eval(expression, {"__builtins__": {}}, inputs))


def validate_circuit(
    circuit: Circuit | str,
    scenario: BaseScenario,
    n_samples: int = 10000,
    min_ratio: float = 0.4,
) -> tuple[bool, float]:
    """Validate that a circuit produces balanced classes.

    Args:
        circuit: The circuit to validate.
        scenario: The scenario for sampling inputs.
        n_samples: Number of samples to test.
        min_ratio: Minimum ratio for the minority class (e.g., 0.4 means 40/60 split).

    Returns:
        Tuple of (is_valid, positive_ratio).
    """
    positive_count = 0
    for _ in range(n_samples):
        inputs = scenario.sample_inputs()
        if evaluate_circuit(circuit, inputs):
            positive_count += 1

    positive_ratio = positive_count / n_samples
    is_valid = min_ratio <= positive_ratio <= (1 - min_ratio)

    return is_valid, positive_ratio


def generate_valid_circuit(
    scenario: BaseScenario,
    num_fields: int,
    max_depth: int = 10**6,
    n_samples: int = 10000,
    min_ratio: float = 0.4,
    max_attempts: int = 100,
) -> Circuit:
    """Generate a circuit that passes validation.

    Args:
        scenario: The scenario to generate a circuit for.
        num_fields: Number of fields to use.
        max_depth: Maximum depth of the expression tree.
        n_samples: Number of samples for validation.
        min_ratio: Minimum ratio for minority class.
        max_attempts: Maximum generation attempts.

    Returns:
        A valid Circuit object.

    Raises:
        RuntimeError: If no valid circuit found after max_attempts.
    """
    for attempt in range(max_attempts):
        circuit = generate_circuit(scenario, num_fields, max_depth)
        is_valid, ratio = validate_circuit(circuit, scenario, min_ratio=min_ratio)
        if is_valid:
            return circuit

    raise RuntimeError(
        f"Could not generate valid circuit after {max_attempts} attempts. "
        f"Try adjusting num_fields or max_depth."
    )


# =============================================================================
# DNF Circuit Generation (controlled complexity)
# =============================================================================


def clauses_for_target_rate(clause_width: int, target_rate: float = 0.5) -> int:
    """Compute number of clauses needed to achieve target positive rate.

    For DNF with t clauses of width w, assuming:
    - Each leaf is ~50% balanced
    - Clauses use disjoint fields (independent)

    P(positive) = 1 - (1 - 2^{-w})^t

    Args:
        clause_width: Width of each AND clause (w).
        target_rate: Target positive rate (default 0.5).

    Returns:
        Number of clauses (t) to achieve target rate.

    Example:
        >>> clauses_for_target_rate(2, 0.5)  # w=2 → t≈3
        3
        >>> clauses_for_target_rate(3, 0.5)  # w=3 → t≈5
        5
    """
    if target_rate <= 0 or target_rate >= 1:
        raise ValueError(f"target_rate must be in (0, 1), got {target_rate}")
    if clause_width < 1:
        raise ValueError(f"clause_width must be >= 1, got {clause_width}")

    p = 2 ** (-clause_width)  # P(single clause true) with balanced leaves
    # target_rate = 1 - (1-p)^t
    # (1-p)^t = 1 - target_rate
    # t = log(1 - target_rate) / log(1 - p)
    t = math.log(1 - target_rate) / math.log(1 - p)
    return max(1, round(t))


def generate_dnf_circuit(
    scenario: BaseScenario,
    num_clauses: int,
    clause_width: int,
    disjoint: bool = True,
) -> Circuit:
    """Generate a DNF circuit with controlled complexity.

    DNF = OR of AND-clauses. Each clause is an AND of `clause_width` balanced
    leaf predicates (~50% true each).

    Args:
        scenario: The scenario to generate a circuit for.
        num_clauses: Number of OR clauses (t).
        clause_width: Number of AND literals per clause (w).
        disjoint: If True, each field is used at most once across all clauses.
                  Requires num_clauses * clause_width <= num_available_fields.
                  If False, fields can be reused across clauses (but not within).

    Returns:
        A Circuit object with circuit_type="dnf".

    Raises:
        ValueError: If disjoint=True and not enough fields available.

    Example:
        >>> circuit = generate_dnf_circuit(scenario, num_clauses=2, clause_width=2)
        >>> # Produces: "(field1 in (...) and field2 >= X) or (field3 in (...) and field4 >= Y)"
    """
    available_fields = scenario.field_names()
    total_literals = num_clauses * clause_width

    if disjoint:
        if total_literals > len(available_fields):
            raise ValueError(
                f"disjoint=True requires num_clauses * clause_width <= num_fields. "
                f"Got {num_clauses} * {clause_width} = {total_literals} > {len(available_fields)}"
            )
        # Select exactly total_literals fields and partition them
        selected_fields = random.sample(available_fields, total_literals)
        clause_fields = [
            selected_fields[i * clause_width : (i + 1) * clause_width]
            for i in range(num_clauses)
        ]
    else:
        # Allow reuse across clauses, but not within a clause
        clause_fields = []
        for _ in range(num_clauses):
            if clause_width > len(available_fields):
                raise ValueError(
                    f"clause_width ({clause_width}) exceeds available fields ({len(available_fields)})"
                )
            clause_fields.append(random.sample(available_fields, clause_width))
        selected_fields = list(set(f for clause in clause_fields for f in clause))

    # Generate balanced leaves for each clause
    clauses = []
    for fields in clause_fields:
        leaves = [_generate_balanced_leaf(scenario, f) for f in fields]
        if len(leaves) == 1:
            clause_expr = leaves[0]
        else:
            clause_expr = f"({' and '.join(leaves)})"
        clauses.append(clause_expr)

    # Combine clauses with OR
    if len(clauses) == 1:
        expression = clauses[0]
    else:
        expression = f"({' or '.join(clauses)})"

    return Circuit(
        expression=expression,
        used_fields=selected_fields,
        max_depth=2,  # DNF is always depth 2 (OR of ANDs)
        scenario=scenario.name,
        circuit_type="dnf",
        num_clauses=num_clauses,
        clause_width=clause_width,
    )


def generate_valid_dnf_circuit(
    scenario: BaseScenario,
    num_clauses: int,
    clause_width: int,
    disjoint: bool = True,
    n_samples: int = 10000,
    min_ratio: float = 0.4,
    max_attempts: int = 100,
) -> Circuit:
    """Generate a DNF circuit that passes class balance validation.

    With balanced leaves and proper (t, w) selection, most generated circuits
    should pass validation. This function provides a safety net.

    Args:
        scenario: The scenario to generate a circuit for.
        num_clauses: Number of OR clauses (t).
        clause_width: Number of AND literals per clause (w).
        disjoint: If True, enforce disjoint field sets across clauses.
        n_samples: Number of samples for validation.
        min_ratio: Minimum ratio for minority class (default 0.4 = 40/60 split).
        max_attempts: Maximum generation attempts.

    Returns:
        A valid Circuit object with circuit_type="dnf".

    Raises:
        RuntimeError: If no valid circuit found after max_attempts.
    """
    for _ in range(max_attempts):
        circuit = generate_dnf_circuit(
            scenario, num_clauses, clause_width, disjoint
        )
        is_valid, ratio = validate_circuit(
            circuit, scenario, n_samples=n_samples, min_ratio=min_ratio
        )
        if is_valid:
            return circuit

    raise RuntimeError(
        f"Could not generate valid DNF circuit after {max_attempts} attempts. "
        f"Try adjusting num_clauses={num_clauses}, clause_width={clause_width}. "
        f"Last attempt had positive_ratio={ratio:.2%}."
    )


# =============================================================================
# Decision Tree Circuit Generation (cleanest complexity control)
# =============================================================================


def _build_decision_tree(
    scenario: BaseScenario,
    selected_fields: list[str],
    current_depth: int = 0,
    used_on_path: set[str] | None = None,
) -> DecisionTreeNode:
    """Recursively build a complete decision tree as a node structure.

    Uses the "sample first" approach: exactly len(selected_fields) fields are
    used, and every root-to-leaf path uses all of them exactly once.

    Args:
        scenario: The scenario containing field definitions.
        selected_fields: The exact fields to use (sampled upfront).
        current_depth: Current depth in recursion.
        used_on_path: Fields already used on this root-to-leaf path.

    Returns:
        DecisionTreeNode representing the tree (with label=None for leaves,
        to be assigned later by _assign_leaf_labels_tree).
    """
    if used_on_path is None:
        used_on_path = set()

    depth = len(selected_fields)

    # Base case: used all fields on this path, return a leaf node
    if current_depth >= depth:
        return DecisionTreeNode(label=None)  # Label assigned later

    # Pick a field not yet used on this path (from our pre-selected pool)
    candidates = [f for f in selected_fields if f not in used_on_path]
    if not candidates:
        # Should not happen if depth == len(selected_fields), but safety check
        return DecisionTreeNode(label=None)

    field = random.choice(candidates)
    predicate = None
    predicates = getattr(scenario, "decision_tree_predicates", None)
    if isinstance(predicates, dict):
        predicate = predicates.get(field)
    if predicate is None:
        predicate = _generate_balanced_leaf(scenario, field)
    new_used = used_on_path | {field}

    # Recursively build left (predicate true) and right (predicate false) subtrees
    left_node = _build_decision_tree(
        scenario, selected_fields, current_depth + 1, new_used
    )
    right_node = _build_decision_tree(
        scenario, selected_fields, current_depth + 1, new_used
    )

    return DecisionTreeNode(
        predicate=predicate,
        field=field,
        left=left_node,
        right=right_node,
    )


def _count_leaves_tree(node: DecisionTreeNode) -> int:
    """Count the number of leaf nodes in a decision tree."""
    if node.is_leaf() or (node.left is None and node.right is None):
        return 1
    count = 0
    if node.left is not None:
        count += _count_leaves_tree(node.left)
    if node.right is not None:
        count += _count_leaves_tree(node.right)
    return count


def _collect_leaves_tree(node: DecisionTreeNode) -> list[DecisionTreeNode]:
    """Collect all leaf nodes in a decision tree (in-order traversal)."""
    if node.is_leaf() or (node.left is None and node.right is None):
        return [node]
    leaves = []
    if node.left is not None:
        leaves.extend(_collect_leaves_tree(node.left))
    if node.right is not None:
        leaves.extend(_collect_leaves_tree(node.right))
    return leaves


def _assign_leaf_labels_tree(
    node: DecisionTreeNode, target_true_ratio: float = 0.5
) -> None:
    """Assign True/False labels to leaf nodes in-place.

    Assigns labels so approximately target_true_ratio of leaves are True.

    Args:
        node: Root of the decision tree.
        target_true_ratio: Target fraction of True leaves (default 0.5).
    """
    leaves = _collect_leaves_tree(node)
    num_leaves = len(leaves)
    if num_leaves == 0:
        return

    # Determine how many should be True
    num_true = round(num_leaves * target_true_ratio)
    num_true = max(1, min(num_leaves - 1, num_true))  # Ensure at least 1 of each

    # Create shuffled labels
    labels = [True] * num_true + [False] * (num_leaves - num_true)
    random.shuffle(labels)

    # Assign labels to leaves
    for leaf, label in zip(leaves, labels):
        leaf.label = label


def _assign_leaf_labels_tree_risk_majority(node: DecisionTreeNode, depth: int) -> None:
    """Assign leaf labels by "risk majority" along each root-to-leaf path.

    Interprets taking the left branch as "predicate=True" and the right branch as
    "predicate=False". With *oriented* predicates (predicate=True means "more
    risky"), this labels a leaf True iff a strict majority of predicates on that
    path are True.

    Threshold: ceil((d+1)/2) for depth d. This yields:
      d1: 1-of-1
      d2: 2-of-2
      d3: 2-of-3
      d4: 3-of-4
    """
    threshold = (depth + 2) // 2

    def visit(n: DecisionTreeNode, risk_count: int) -> None:
        # Leaves have no children (labels are assigned by this function).
        if n.left is None and n.right is None:
            n.label = risk_count >= threshold
            return
        if n.left is not None:
            visit(n.left, risk_count + 1)
        if n.right is not None:
            visit(n.right, risk_count)

    visit(node, 0)


def _assign_leaf_labels_tree_risk_majority_balanced(node: DecisionTreeNode, depth: int) -> None:
    """Risk-majority labeling with balanced tie-breaks (≈50/50 for all depths).

    Labels leaves in a monotone "more risk signals => more likely True" way,
    but chooses a subset of tie-count leaves so that exactly half of leaves are
    True in the complete depth-`d` tree. With independent 50/50 predicates, this
    yields an (approximately) class-balanced dataset for any depth.
    """
    leaves_with_count: list[tuple[DecisionTreeNode, int]] = []

    def collect(n: DecisionTreeNode, risk_count: int) -> None:
        if n.left is None and n.right is None:
            leaves_with_count.append((n, risk_count))
            return
        if n.left is not None:
            collect(n.left, risk_count + 1)
        if n.right is not None:
            collect(n.right, risk_count)

    collect(node, 0)
    if not leaves_with_count:
        return

    num_leaves = len(leaves_with_count)
    num_true_target = num_leaves // 2
    num_true_target = max(1, min(num_leaves - 1, num_true_target))

    # Group leaves by risk_count.
    by_count: dict[int, list[DecisionTreeNode]] = {}
    for leaf, count in leaves_with_count:
        by_count.setdefault(count, []).append(leaf)

    true_assigned = 0
    for count in sorted(by_count.keys(), reverse=True):
        group = by_count[count]
        if true_assigned >= num_true_target:
            for leaf in group:
                leaf.label = False
            continue

        remaining = num_true_target - true_assigned
        if len(group) <= remaining:
            for leaf in group:
                leaf.label = True
            true_assigned += len(group)
            continue

        # Tie-break: pick a subset of this count level to hit exactly 50/50.
        chosen_ids = {id(n) for n in random.sample(group, remaining)}
        for leaf in group:
            leaf.label = id(leaf) in chosen_ids
        true_assigned = num_true_target

    # Defensive: if something went wrong, fall back to balanced labels.
    if true_assigned != num_true_target:
        _assign_leaf_labels_tree(node, target_true_ratio=0.5)


def _assign_decision_tree_leaf_labels(scenario: BaseScenario, node: DecisionTreeNode, depth: int) -> None:
    """Assign decision-tree leaf labels, allowing scenario-specific semantics."""
    labeling = getattr(scenario, "decision_tree_labeling", None)
    if labeling == "risk_majority":
        _assign_leaf_labels_tree_risk_majority(node, depth)
    elif labeling == "risk_majority_balanced":
        _assign_leaf_labels_tree_risk_majority_balanced(node, depth)
    else:
        _assign_leaf_labels_tree(node, target_true_ratio=0.5)


def _tree_to_expression(node: DecisionTreeNode) -> str:
    """Convert a decision tree to an eval-able Python expression.

    Args:
        node: Root of the decision tree (with labels assigned to leaves).

    Returns:
        Python expression string that evaluates to True/False.
    """
    if node.is_leaf():
        return str(node.label)

    if node.left is None or node.right is None:
        # Shouldn't happen for well-formed trees
        return str(node.label if node.label is not None else True)

    left_expr = _tree_to_expression(node.left)
    right_expr = _tree_to_expression(node.right)
    predicate = node.predicate

    # Format: (predicate and left) or (not predicate and right)
    return f"({predicate} and {left_expr} or not ({predicate}) and {right_expr})"


def _format_predicate_natural(predicate: str, field: str) -> str:
    """Format a predicate in natural language.

    Converts predicates like "seat_capacity >= 6" to "it has seat capacity >= 6".

    Args:
        predicate: The predicate string.
        field: The field name used in the predicate.

    Returns:
        Natural language version of the predicate.
    """
    # Convert field name from snake_case to space-separated
    natural_field = field.replace("_", " ")
    # Replace the field name in the predicate with natural version
    natural_predicate = predicate.replace(field, natural_field, 1)
    return f"it has {natural_predicate}"


def _tree_to_description(
    node: DecisionTreeNode,
    depth: int = 0,
    intro: str = "We can decide if a vehicle is good from the following process.",
    question: str = "Is this vehicle good?",
) -> str:
    """Convert a decision tree to a natural language description.

    Args:
        node: Root of the decision tree (with labels assigned to leaves).
        depth: Current depth for indentation.
        intro: Introduction sentence for the description.
        question: Question to append at the end.

    Returns:
        Natural language description of the decision tree.
    """
    def format_node(n: DecisionTreeNode, d: int) -> list[str]:
        """Recursively format a node and its children."""
        prefix = "-" * (d + 1) + " "
        lines = []

        if n.is_leaf():
            # This shouldn't be called directly, leaves are handled by parent
            return []

        if n.left is None or n.right is None:
            return []

        predicate = n.predicate
        field = n.field

        # Format the predicate naturally
        natural_pred = _format_predicate_natural(predicate, field)

        # Left branch (when predicate is True)
        if n.left.is_leaf():
            label = "yes" if n.left.label else "no"
            lines.append(f"{prefix}if {natural_pred} -> {label}")
        else:
            lines.append(f"{prefix}if {natural_pred}")
            lines.extend(format_node(n.left, d + 1))

        # Right branch (when predicate is False)
        # We need to negate the predicate for the "else" case
        negated_pred = _negate_predicate_natural(predicate, field)
        if n.right.is_leaf():
            label = "yes" if n.right.label else "no"
            lines.append(f"{prefix}if {negated_pred} -> {label}")
        else:
            lines.append(f"{prefix}if {negated_pred}")
            lines.extend(format_node(n.right, d + 1))

        return lines

    lines = [intro]
    lines.extend(format_node(node, depth))
    lines.append("")
    lines.append(question)

    return "\n".join(lines)


def _negate_predicate_natural(predicate: str, field: str) -> str:
    """Negate a predicate and format in natural language.

    Converts "seat_capacity >= 6" to "it has seat capacity < 6".

    Args:
        predicate: The original predicate string.
        field: The field name used in the predicate.

    Returns:
        Natural language version of the negated predicate.
    """
    # Map operators to their negations
    negations = {
        ">=": "<",
        "<=": ">",
        ">": "<=",
        "<": ">=",
        "==": "!=",
        "!=": "==",
    }

    natural_field = field.replace("_", " ")
    natural_predicate = predicate.replace(field, natural_field, 1)

    # Try to negate the operator
    for op, neg_op in negations.items():
        if f" {op} " in natural_predicate:
            natural_predicate = natural_predicate.replace(f" {op} ", f" {neg_op} ", 1)
            break
        # Handle "in" operator for ENUM fields
        if " in (" in natural_predicate:
            # For "in" predicates, we use "not in"
            natural_predicate = natural_predicate.replace(" in (", " not in (", 1)
            break

    return f"it has {natural_predicate}"


def _assign_leaf_labels(expression: str, target_true_ratio: float = 0.5) -> str:
    """Replace __LEAF__ placeholders with True/False to achieve target balance.

    Assigns labels so approximately target_true_ratio of leaves are True.

    Args:
        expression: Expression with __LEAF__ placeholders.
        target_true_ratio: Target fraction of True leaves (default 0.5).

    Returns:
        Expression with True/False labels.
    """
    # Count leaves
    num_leaves = expression.count("__LEAF__")
    if num_leaves == 0:
        return expression

    # Determine how many should be True
    num_true = round(num_leaves * target_true_ratio)
    num_true = max(1, min(num_leaves - 1, num_true))  # Ensure at least 1 of each

    # Create shuffled labels
    labels = [True] * num_true + [False] * (num_leaves - num_true)
    random.shuffle(labels)

    # Replace placeholders
    result = expression
    for label in labels:
        result = result.replace("__LEAF__", str(label), 1)

    return result


def generate_decision_tree_circuit(
    scenario: BaseScenario,
    depth: int,
) -> Circuit:
    """Generate a decision tree circuit with controlled complexity.

    Uses the "sample first" approach: exactly `depth` fields are sampled upfront,
    and every root-to-leaf path uses all of them exactly once. This creates a
    complete binary tree with exactly 2^depth leaves.

    Decision trees partition the input space into disjoint regions (leaves).
    Each split uses a balanced predicate (~50% true). Leaf labels are assigned
    so that ~50% are True, ensuring class balance.

    Advantages over DNF:
    - No overlap/collapse issues (leaves are disjoint)
    - Clean complexity: depth d → exactly 2^d disjoint regions
    - Easy balance: label exactly half of 2^d leaves as True → 50%
    - Predictable: exactly d fields matter, agent knows finding d fields suffices

    Args:
        scenario: The scenario to generate a circuit for.
        depth: Depth of the decision tree (complexity knob).
               Depth d uses exactly d fields and creates exactly 2^d leaves.

    Returns:
        A Circuit object with circuit_type="decision_tree".

    Raises:
        ValueError: If depth exceeds available fields.

    Example:
        >>> circuit = generate_decision_tree_circuit(scenario, depth=3)
        >>> # Creates a tree with exactly 8 leaves, using exactly 3 fields
    """
    available_fields = scenario.field_names()

    if depth > len(available_fields):
        raise ValueError(
            f"depth ({depth}) exceeds available fields ({len(available_fields)}). "
            f"Each path needs {depth} distinct fields."
        )

    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")

    # SAMPLE FIRST: select exactly `depth` fields upfront
    selected_fields = random.sample(available_fields, depth)

    # Build complete tree structure as a node tree
    # Every path will use all `depth` fields exactly once
    tree_node = _build_decision_tree(scenario, selected_fields)

    # Assign leaf labels (scenario may override semantics)
    _assign_decision_tree_leaf_labels(scenario, tree_node, depth)

    # Convert tree to expression
    expression = _tree_to_expression(tree_node)

    # Generate natural language description
    intro = getattr(
        scenario,
        "decision_tree_intro",
        "We can decide if a vehicle is good from the following process.",
    )
    question = getattr(scenario, "decision_tree_question", "Is this vehicle good?")
    description = _tree_to_description(tree_node, intro=intro, question=question)

    return Circuit(
        expression=expression,
        used_fields=selected_fields,
        max_depth=depth,
        scenario=scenario.name,
        circuit_type="decision_tree",
        num_clauses=None,
        clause_width=None,
        description=description,
        tree_node=tree_node,
    )


def generate_valid_decision_tree_circuit(
    scenario: BaseScenario,
    depth: int,
    n_samples: int = 10000,
    min_ratio: float = 0.4,
    max_attempts: int = 100,
) -> Circuit:
    """Generate a decision tree circuit that passes class balance validation.

    With balanced splits and balanced leaf labels, most trees should pass.
    This function provides a safety net.

    Args:
        scenario: The scenario to generate a circuit for.
        depth: Depth of the decision tree.
        n_samples: Number of samples for validation.
        min_ratio: Minimum ratio for minority class (default 0.4 = 40/60 split).
        max_attempts: Maximum generation attempts.

    Returns:
        A valid Circuit object with circuit_type="decision_tree".

    Raises:
        RuntimeError: If no valid circuit found after max_attempts.
    """
    scenario_min_ratio = getattr(scenario, "decision_tree_min_ratio", None)
    if scenario_min_ratio is not None:
        min_ratio = float(scenario_min_ratio)

    for attempt in range(max_attempts):
        circuit = generate_decision_tree_circuit(scenario, depth)
        is_valid, ratio = validate_circuit(
            circuit, scenario, n_samples=n_samples, min_ratio=min_ratio
        )
        if is_valid:
            return circuit

    raise RuntimeError(
        f"Could not generate valid decision tree circuit after {max_attempts} attempts. "
        f"depth={depth}, last positive_ratio={ratio:.2%}."
    )


# =============================================================================
# Rationale Generation (for rationale_prompted stage)
# =============================================================================


def generate_rationale(node: DecisionTreeNode, inputs: dict[str, Any]) -> str:
    """Generate a rationale string explaining the decision tree path.

    Traces the path through the decision tree for the given inputs and
    returns a natural language explanation of the answer.

    Args:
        node: Root of the decision tree.
        inputs: Input values for evaluation.

    Returns:
        Rationale string like "yes, because color = 'Black', and drivetrain != 'AWD'"

    Example:
        >>> generate_rationale(circuit.tree_node, {"color": "Black", "drivetrain": "AWD"})
        "no, because color = 'Black', and drivetrain = 'AWD'"
    """
    label, conditions = _trace_decision_path(node, inputs)
    answer = "yes" if label else "no"
    if not conditions:
        return answer
    conditions_str = ", and ".join(conditions)
    return f"{answer}, because {conditions_str}"


def _trace_decision_path(
    node: DecisionTreeNode,
    inputs: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Internal helper: trace the path through a decision tree.

    Args:
        node: Current node in the decision tree.
        inputs: Input values for evaluation.

    Returns:
        (final_label, conditions) where conditions is a list of natural language
        condition strings that were satisfied on the path to the answer.
    """
    if node.is_leaf():
        return node.label, []

    # Evaluate this node's predicate
    predicate_result = bool(eval(node.predicate, {"__builtins__": {}}, inputs))

    # Recurse to appropriate child
    if predicate_result:
        final_label, path = _trace_decision_path(node.left, inputs)
        condition = _predicate_to_natural(node.predicate, node.field)
    else:
        final_label, path = _trace_decision_path(node.right, inputs)
        condition = _negate_predicate_to_natural(node.predicate, node.field)

    return final_label, [condition] + path


def _predicate_to_natural(predicate: str, field: str) -> str:
    """Convert predicate to natural language format.

    Converts 'seat_capacity >= 6' to 'seat capacity >= 6'
    Converts 'brand == "BMW"' to 'brand = "BMW"'

    Args:
        predicate: The predicate string.
        field: The field name used in the predicate.

    Returns:
        Natural language version of the predicate.
    """
    natural_field = field.replace("_", " ")
    result = predicate.replace(field, natural_field, 1)
    # Use single = for equality (more natural)
    result = result.replace(" == ", " = ")
    return result


def _negate_predicate_to_natural(predicate: str, field: str) -> str:
    """Negate a predicate and convert to natural language.

    Args:
        predicate: The original predicate string.
        field: The field name used in the predicate.

    Returns:
        Natural language version of the negated predicate.
    """
    negations = {
        ">=": "<",
        "<=": ">",
        ">": "<=",
        "<": ">=",
        "==": "!=",
        "!=": "==",
    }
    natural_field = field.replace("_", " ")
    result = predicate.replace(field, natural_field, 1)

    for op, neg_op in negations.items():
        if f" {op} " in result:
            result = result.replace(f" {op} ", f" {neg_op} ", 1)
            break

    # Use single = for equality (more natural)
    result = result.replace(" == ", " = ")
    return result


# =============================================================================
# Distractor Circuit & Unfaithful Rationale Generation
# =============================================================================


def compute_field_sensitivity(
    circuit: Circuit | str,
    scenario: BaseScenario,
    n_samples: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Compute per-field sensitivity by local perturbation.

    For each sample x each field, flip the field value and check if
    evaluate_circuit() output changes.

    ENUM: try all other values, sensitive if any flip changes output.
    INTEGER: try range endpoints and ±1 from original, sensitive if any changes output.

    Returns dict mapping field_name -> sensitivity (0.0 to 1.0) for ALL scenario fields.
    """
    old_state = random.getstate()
    random.seed(seed)
    samples = [scenario.sample_inputs() for _ in range(n_samples)]
    random.setstate(old_state)

    baselines = [evaluate_circuit(circuit, s) for s in samples]

    sensitivity = {}
    for field_name in scenario.field_names():
        f = scenario.get_field(field_name)
        n_changed = 0
        for i, sample in enumerate(samples):
            original_val = sample[field_name]
            base_out = baselines[i]
            changed = False

            if f.field_type == FieldType.ENUM:
                for v in f.values:
                    if v == original_val:
                        continue
                    perturbed = {**sample, field_name: v}
                    if evaluate_circuit(circuit, perturbed) != base_out:
                        changed = True
                        break
            else:
                # Probe endpoints and ±1 from original to catch threshold crossings
                probes = {f.range[0], f.range[1]}
                if original_val - 1 >= f.range[0]:
                    probes.add(original_val - 1)
                if original_val + 1 <= f.range[1]:
                    probes.add(original_val + 1)
                probes.discard(original_val)
                for v in probes:
                    perturbed = {**sample, field_name: v}
                    if evaluate_circuit(circuit, perturbed) != base_out:
                        changed = True
                        break

            if changed:
                n_changed += 1

        sensitivity[field_name] = n_changed / n_samples

    return sensitivity


def generate_distractor_circuit(
    scenario: BaseScenario,
    real_circuit: Circuit,
    no_overlap: bool = False,
) -> Circuit:
    """Generate a distractor circuit for unfaithful rationale training.

    The distractor circuit is a separate decision tree whose traced conditions
    are used as *wrong* explanations paired with the *correct* label from the
    real circuit.

    Args:
        scenario: The scenario containing field definitions.
        real_circuit: The real circuit (must be a decision tree).
        no_overlap: If True, the distractor uses only fields NOT in the real
            circuit. Requires enough unused fields for the same depth.

    Returns:
        A new Circuit object (decision tree) with its own fields and tree.

    Raises:
        ValueError: If no_overlap=True but not enough unused fields.
    """
    depth = real_circuit.max_depth

    if no_overlap:
        all_fields = set(scenario.field_names())
        real_fields = set(real_circuit.used_fields)
        available = list(all_fields - real_fields)
        if len(available) < depth:
            raise ValueError(
                f"no_overlap=True requires >= {depth} unused fields, "
                f"but only {len(available)} are available "
                f"(scenario has {len(all_fields)}, circuit uses {len(real_fields)})"
            )
        # Build a tree manually using only the non-overlapping fields
        selected_fields = random.sample(available, depth)
        tree_node = _build_decision_tree(scenario, selected_fields)
        _assign_decision_tree_leaf_labels(scenario, tree_node, depth)
        expression = _tree_to_expression(tree_node)
        intro = getattr(
            scenario,
            "decision_tree_intro",
            "We can decide if a vehicle is good from the following process.",
        )
        question = getattr(scenario, "decision_tree_question", "Is this vehicle good?")
        description = _tree_to_description(tree_node, intro=intro, question=question)
        return Circuit(
            expression=expression,
            used_fields=selected_fields,
            max_depth=depth,
            scenario=scenario.name,
            circuit_type="decision_tree",
            description=description,
            tree_node=tree_node,
        )
    else:
        return generate_valid_decision_tree_circuit(scenario, depth)


def generate_unfaithful_rationale(
    real_node: DecisionTreeNode,
    distractor_node: DecisionTreeNode,
    inputs: dict[str, Any],
) -> str:
    """Generate an unfaithful rationale: correct label, wrong conditions.

    Uses the real circuit to determine the label ("yes"/"no") but traces the
    distractor circuit to produce the explanatory conditions.

    Args:
        real_node: Root of the real decision tree.
        distractor_node: Root of the distractor decision tree.
        inputs: Input values for evaluation.

    Returns:
        Rationale string like "yes, because [distractor conditions]"
    """
    real_label, _ = _trace_decision_path(real_node, inputs)
    _, distractor_conditions = _trace_decision_path(distractor_node, inputs)

    answer = "yes" if real_label else "no"
    if not distractor_conditions:
        return answer
    conditions_str = ", and ".join(distractor_conditions)
    return f"{answer}, because {conditions_str}"
