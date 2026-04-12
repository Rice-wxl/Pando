#!/usr/bin/env python3
"""Preflight validation for Decision Tree circuits.

Tests the two key properties of decision trees:
1. Node balance: Each split sends ~50% of samples each way
2. Label balance: Overall P(true) ≈ 50%

These properties are what make decision trees a clean complexity controller:
- Balanced nodes ensure no single split dominates
- Balanced labels ensure fair classification task

Usage:
    python scripts/preflight_decision_tree.py --explore --max-depth 10
    python scripts/preflight_decision_tree.py --depths 1 2 3 4 5
    python scripts/preflight_decision_tree.py --single-depth 5 --verbose
"""

import argparse
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scenarios import get_scenario
from src.circuits import generate_decision_tree_circuit, evaluate_circuit, Circuit


# =============================================================================
# Node balance analysis
# =============================================================================

def extract_predicates(expression: str) -> list[str]:
    """Extract all predicates (internal node conditions) from a decision tree expression.

    Predicates are the balanced leaf expressions like:
    - "color == 'White'" (binary ENUM)
    - "color in ('White', 'Black')" (multi-value ENUM)
    - "price >= 50000" (INTEGER)
    """
    predicates = []

    # Match "field == 'value'" patterns (binary ENUM)
    enum_eq_pattern = r"(\w+ == '[^']+')"
    predicates.extend(re.findall(enum_eq_pattern, expression))

    # Match "field in ('val1', 'val2')" patterns (multi-value ENUM)
    enum_in_pattern = r"(\w+ in \([^)]+\))"
    predicates.extend(re.findall(enum_in_pattern, expression))

    # Match "field >= value" or "field <= value" patterns (INTEGER)
    int_pattern = r"(\w+ [<>=]+ \d+)"
    predicates.extend(re.findall(int_pattern, expression))

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for p in predicates:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    return unique


def test_predicate_balance(
    predicate: str,
    scenario,
    n_samples: int = 5000,
    seed: int = 42,
) -> float:
    """Test what fraction of samples satisfy a single predicate."""
    random.seed(seed)

    true_count = 0
    for _ in range(n_samples):
        inputs = scenario.sample_inputs()
        try:
            result = eval(predicate, {"__builtins__": {}}, inputs)
            if result:
                true_count += 1
        except:
            pass

    return true_count / n_samples


def test_all_node_balances(
    circuit: Circuit,
    scenario,
    n_samples: int = 5000,
    seed: int = 42,
) -> list[tuple[str, float]]:
    """Test balance of each internal node (predicate) in the tree."""
    predicates = extract_predicates(circuit.expression)

    results = []
    for i, pred in enumerate(predicates):
        p_true = test_predicate_balance(pred, scenario, n_samples, seed + i)
        results.append((pred, p_true))

    return results


# =============================================================================
# Label balance analysis
# =============================================================================

def test_label_balance(
    circuit: Circuit,
    scenario,
    n_samples: int = 10000,
    seed: int = 42,
) -> float:
    """Test overall class balance via Monte Carlo."""
    random.seed(seed)

    positive = 0
    for _ in range(n_samples):
        inputs = scenario.sample_inputs()
        if evaluate_circuit(circuit, inputs):
            positive += 1

    return positive / n_samples


def count_leaves(expression: str) -> tuple[int, int]:
    """Count True and False leaves in the expression."""
    true_count = expression.count("True")
    false_count = expression.count("False")
    return true_count, false_count


# =============================================================================
# Preflight validation
# =============================================================================

@dataclass
class DecisionTreePreflightResult:
    """Results from preflight validation."""
    depth: int
    num_leaves: int
    true_leaves: int
    false_leaves: int
    num_fields_used: int

    # Label balance
    p_true: float
    label_balance_ok: bool

    # Node balances
    node_balances: list[tuple[str, float]]
    min_node_balance: float
    max_node_balance: float
    avg_node_balance: float
    nodes_balanced_ok: bool  # All nodes in [0.3, 0.7]

    # Fields check (with pre-sampled fields, should equal depth)
    fields_ok: bool  # num_fields_used == depth

    # Overall
    passed: bool

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        leaf_str = f"{self.true_leaves}T/{self.false_leaves}F"
        fields_status = "OK" if self.fields_ok else f"FAIL (expected {self.depth})"
        return (
            f"Depth {self.depth}: {status}\n"
            f"  Leaves: {self.num_leaves} ({leaf_str}), Fields: {self.num_fields_used} {fields_status}\n"
            f"  Label balance: P(true)={self.p_true:.1%} {'OK' if self.label_balance_ok else 'FAIL'}\n"
            f"  Node balance: min={self.min_node_balance:.1%}, max={self.max_node_balance:.1%}, "
            f"avg={self.avg_node_balance:.1%} {'OK' if self.nodes_balanced_ok else 'FAIL'}"
        )


def preflight_decision_tree(
    scenario,
    depth: int,
    n_samples: int = 10000,
    label_balance_range: tuple[float, float] = (0.3, 0.7),
    node_balance_range: tuple[float, float] = (0.3, 0.7),
    seed: int = 42,
) -> tuple[Circuit, DecisionTreePreflightResult]:
    """Generate and validate a decision tree circuit."""

    circuit = generate_decision_tree_circuit(scenario, depth)

    # Count leaves
    true_leaves, false_leaves = count_leaves(circuit.expression)
    num_leaves = true_leaves + false_leaves

    # Test label balance
    p_true = test_label_balance(circuit, scenario, n_samples, seed)
    label_balance_ok = label_balance_range[0] <= p_true <= label_balance_range[1]

    # Test node balances
    node_balances = test_all_node_balances(circuit, scenario, n_samples // 2, seed)

    if node_balances:
        balances = [b for _, b in node_balances]
        min_bal = min(balances)
        max_bal = max(balances)
        avg_bal = sum(balances) / len(balances)
        nodes_ok = all(node_balance_range[0] <= b <= node_balance_range[1] for b in balances)
    else:
        min_bal = max_bal = avg_bal = 0.5
        nodes_ok = True

    # Check that fields used equals depth (with pre-sampled fields)
    num_fields = len(circuit.used_fields)
    fields_ok = num_fields == depth

    passed = label_balance_ok and nodes_ok and fields_ok

    result = DecisionTreePreflightResult(
        depth=depth,
        num_leaves=num_leaves,
        true_leaves=true_leaves,
        false_leaves=false_leaves,
        num_fields_used=num_fields,
        p_true=p_true,
        label_balance_ok=label_balance_ok,
        node_balances=node_balances,
        min_node_balance=min_bal,
        max_node_balance=max_bal,
        avg_node_balance=avg_bal,
        nodes_balanced_ok=nodes_ok,
        fields_ok=fields_ok,
        passed=passed,
    )

    return circuit, result


def explore_decision_tree_tiers(
    scenario_name: str,
    max_depth: int = 10,
    n_samples: int = 1000,
    n_circuits: int = 30,
    label_balance_range: tuple[float, float] = (0.3, 0.7),
    node_balance_range: tuple[float, float] = (0.3, 0.7),
    seed: int = 42,
):
    """Explore decision tree circuits at various depths."""
    scenario = get_scenario(scenario_name)
    num_fields = len(scenario.field_names())

    print(f"Scenario: {scenario_name}")
    print(f"Available fields: {num_fields}")
    print(f"Max depth possible: {num_fields}")
    print(f"Circuits per depth: {n_circuits}")
    print(f"MC samples per circuit: {n_samples}")
    print(f"Label balance range: {label_balance_range[0]:.0%}-{label_balance_range[1]:.0%}")
    print(f"Node balance range: {node_balance_range[0]:.0%}-{node_balance_range[1]:.0%}")
    print("=" * 80)

    print("\n| Depth | Leaves | Fields | P(true) | Node min | Node max | Label% | Node% | Field% | Pass% |")
    print("|-------|--------|--------|---------|----------|----------|--------|-------|--------|-------|")

    for depth in range(1, min(max_depth + 1, num_fields + 1)):
        label_passes = 0
        node_passes = 0
        fields_passes = 0
        all_passes = 0

        p_trues = []
        node_mins = []
        node_maxs = []
        fields_used = []

        for i in range(n_circuits):
            try:
                circuit, result = preflight_decision_tree(
                    scenario,
                    depth=depth,
                    n_samples=n_samples,
                    label_balance_range=label_balance_range,
                    node_balance_range=node_balance_range,
                    seed=seed + i * 100 + depth,
                )
            except ValueError as e:
                print(f"  Depth {depth}: SKIP - {e}")
                break

            p_trues.append(result.p_true)
            node_mins.append(result.min_node_balance)
            node_maxs.append(result.max_node_balance)
            fields_used.append(result.num_fields_used)

            if result.label_balance_ok:
                label_passes += 1
            if result.nodes_balanced_ok:
                node_passes += 1
            if result.fields_ok:
                fields_passes += 1
            if result.passed:
                all_passes += 1

        if not p_trues:
            continue

        n = len(p_trues)
        label_rate = label_passes / n
        node_rate = node_passes / n
        fields_rate = fields_passes / n
        all_rate = all_passes / n

        avg_p = sum(p_trues) / n
        avg_node_min = sum(node_mins) / n
        avg_node_max = sum(node_maxs) / n
        avg_fields = sum(fields_used) / n

        leaves = 2 ** depth
        status = "OK" if all_rate >= 0.8 else ("~" if all_rate >= 0.5 else "X")

        print(f"| {depth:5d} | {leaves:6d} | {avg_fields:6.1f} | {avg_p:6.0%} | {avg_node_min:7.0%} | {avg_node_max:7.0%} | "
              f"{label_rate:5.0%} | {node_rate:4.0%} | {fields_rate:5.0%} | {all_rate:4.0%} {status} |")

    print("\n" + "=" * 80)
    print("Legend:")
    print("  Fields    = Average number of fields used (should equal depth)")
    print("  P(true)   = Overall label balance (fraction of samples that are True)")
    print("  Node min  = Minimum balance among all internal nodes")
    print("  Node max  = Maximum balance among all internal nodes")
    print("  Label%    = Pass rate for label balance check")
    print("  Node%     = Pass rate for node balance check (all nodes in range)")
    print("  Field%    = Pass rate for fields check (fields used == depth)")
    print("  Pass%     = Overall pass rate (all checks pass)")


def run_preflight_depths(
    scenario_name: str,
    depths: list[int],
    n_samples: int = 10000,
    label_balance_range: tuple[float, float] = (0.3, 0.7),
    node_balance_range: tuple[float, float] = (0.3, 0.7),
    seed: int = 42,
    verbose: bool = False,
):
    """Run preflight on specific depths."""
    scenario = get_scenario(scenario_name)

    print(f"Scenario: {scenario_name}")
    print(f"Available fields: {len(scenario.field_names())}")
    print(f"Label balance range: {label_balance_range[0]:.0%}-{label_balance_range[1]:.0%}")
    print(f"Node balance range: {node_balance_range[0]:.0%}-{node_balance_range[1]:.0%}")
    print(f"Samples per test: {n_samples}")
    print("=" * 70)

    results = []

    for depth in depths:
        print(f"\nTesting depth {depth}...")

        try:
            circuit, result = preflight_decision_tree(
                scenario,
                depth=depth,
                n_samples=n_samples,
                label_balance_range=label_balance_range,
                node_balance_range=node_balance_range,
                seed=seed,
            )
            results.append((circuit, result))
            print(result)

            if verbose:
                print(f"\n  Node balances:")
                for pred, bal in result.node_balances:
                    status = "OK" if node_balance_range[0] <= bal <= node_balance_range[1] else "!"
                    print(f"    {bal:.1%} {status} : {pred}")
                print(f"\n  Expression: {circuit.expression[:120]}...")
                print(f"  Fields used: {circuit.used_fields}")

        except ValueError as e:
            print(f"  FAIL: {e}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, r in results if r.passed)
    print(f"Passed: {passed}/{len(results)}")

    print("\n| Depth | Leaves | Fields | P(true) | Node range | Status |")
    print("|-------|--------|--------|---------|------------|--------|")
    for _, r in results:
        status = "PASS" if r.passed else "FAIL"
        node_range = f"{r.min_node_balance:.0%}-{r.max_node_balance:.0%}"
        fields_str = f"{r.num_fields_used}" + ("" if r.fields_ok else "!")
        print(f"| {r.depth:5d} | {r.num_leaves:6d} | {fields_str:6s} | {r.p_true:6.0%} | {node_range:10s} | {status:6s} |")

    return results


def main():
    parser = argparse.ArgumentParser(description="Preflight validation for Decision Tree circuits")
    parser.add_argument("--scenario", default="car_purchase", help="Scenario name")
    parser.add_argument("--depths", nargs="+", type=int, help="Specific depths to test")
    parser.add_argument("--single-depth", type=int, help="Test a single depth")
    parser.add_argument("--explore", action="store_true", help="Explore all depths")
    parser.add_argument("--max-depth", type=int, default=10, help="Max depth for exploration")
    parser.add_argument("--samples", type=int, default=1000, help="Monte Carlo samples per circuit")
    parser.add_argument("--n-circuits", type=int, default=30, help="Circuits to test per depth (explore mode)")
    parser.add_argument("--min-label-balance", type=float, default=0.3, help="Min label balance")
    parser.add_argument("--max-label-balance", type=float, default=0.7, help="Max label balance")
    parser.add_argument("--min-node-balance", type=float, default=0.3, help="Min node balance")
    parser.add_argument("--max-node-balance", type=float, default=0.7, help="Max node balance")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Show circuit details")

    args = parser.parse_args()

    label_range = (args.min_label_balance, args.max_label_balance)
    node_range = (args.min_node_balance, args.max_node_balance)

    if args.explore:
        explore_decision_tree_tiers(
            args.scenario,
            max_depth=args.max_depth,
            n_samples=args.samples,
            n_circuits=args.n_circuits,
            label_balance_range=label_range,
            node_balance_range=node_range,
            seed=args.seed,
        )
    elif args.single_depth:
        run_preflight_depths(
            args.scenario,
            depths=[args.single_depth],
            n_samples=args.samples,
            label_balance_range=label_range,
            node_balance_range=node_range,
            seed=args.seed,
            verbose=args.verbose,
        )
    elif args.depths:
        run_preflight_depths(
            args.scenario,
            depths=args.depths,
            n_samples=args.samples,
            label_balance_range=label_range,
            node_balance_range=node_range,
            seed=args.seed,
            verbose=args.verbose,
        )
    else:
        # Default: explore mode
        explore_decision_tree_tiers(
            args.scenario,
            max_depth=args.max_depth,
            n_samples=args.samples,
            n_circuits=args.n_circuits,
            label_balance_range=label_range,
            node_balance_range=node_range,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
