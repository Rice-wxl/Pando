"""Blackbox tree voting agents: transductive version-space inference.

Query the model on a subset of test inputs, then sample random decision trees
over all fields using actual test-input values as candidate thresholds. Filter
to trees consistent with queried labels, predict unseen samples via majority vote.

No interpretability tools, no GPT calls.
"""

import random
from typing import Any

from . import register_agent
from .sample_then_llm_guess import SampleThenLLMGuessAgent
from .base import AgentResult
from .sampling import compute_spread_order, get_available_fields
from ..scenarios.base import FieldType
from ..budget import BudgetExceededError


# ---------------------------------------------------------------------------
# Tree utilities
# ---------------------------------------------------------------------------

def _build_candidate_thresholds(test_inputs, field_name, field_obj):
    """Get candidate split values from the actual test inputs."""
    if field_obj.field_type == FieldType.ENUM:
        return [("==", v) for v in field_obj.values]
    else:
        vals = sorted(set(int(inp[field_name]) for inp in test_inputs if field_name in inp))
        thresholds = []
        for i in range(len(vals) - 1):
            thresholds.append(("<=", (vals[i] + vals[i + 1]) // 2))
        return thresholds


class TreeNode:
    __slots__ = ['field', 'op', 'value', 'left', 'right', 'label']

    def __init__(self, field=None, op=None, value=None, left=None, right=None, label=None):
        self.field = field
        self.op = op
        self.value = value
        self.left = left
        self.right = right
        self.label = label

    def predict(self, x):
        if self.label is not None:
            return self.label
        v = x.get(self.field)
        if v is None:
            return self.left.predict(x) if self.left else True
        if self.op == "==":
            cond = (str(v) == str(self.value))
        elif self.op == "<=":
            cond = (int(v) <= int(self.value))
        else:
            cond = True
        return self.left.predict(x) if cond else self.right.predict(x)


def _sample_tree(fields_with_thresholds, max_depth, rng):
    """Sample a random tree using actual candidate thresholds."""
    if max_depth == 0 or not fields_with_thresholds:
        return TreeNode(label=rng.choice([True, False]))
    if rng.random() < 0.12:
        return TreeNode(label=rng.choice([True, False]))

    fname, fobj, candidates = rng.choice(fields_with_thresholds)
    if not candidates:
        return TreeNode(label=rng.choice([True, False]))

    op, value = rng.choice(candidates)
    left = _sample_tree(fields_with_thresholds, max_depth - 1, rng)
    right = _sample_tree(fields_with_thresholds, max_depth - 1, rng)
    return TreeNode(field=fname, op=op, value=value, left=left, right=right)


def _tree_consistent(tree, inputs, labels):
    for inp, label in zip(inputs, labels):
        try:
            if tree.predict(inp) != label:
                return False
        except Exception:
            return False
    return True


def _collect_tree_fields(node, counts=None):
    """Count how many times each field appears as a split in a tree."""
    if counts is None:
        counts = {}
    if node.label is not None:
        return counts
    if node.field:
        counts[node.field] = counts.get(node.field, 0) + 1
    if node.left:
        _collect_tree_fields(node.left, counts)
    if node.right:
        _collect_tree_fields(node.right, counts)
    return counts


def _field_frequencies(consistent_trees):
    """Get normalized field frequencies across all consistent trees."""
    total_counts = {}
    for tree in consistent_trees:
        counts = _collect_tree_fields(tree)
        for field, count in counts.items():
            total_counts[field] = total_counts.get(field, 0) + count
    total = sum(total_counts.values()) or 1
    return {f: c / total for f, c in sorted(total_counts.items(), key=lambda x: -x[1])}


def _fields_to_pattern(field_freqs):
    """Generate a synthetic pattern string from field frequencies.

    This allows the existing pattern F1 analysis to work on tree voting agents.
    Format: "Decision based on field1 (45.2%), field2 (30.1%), field3 (15.0%)"
    """
    if not field_freqs:
        return ""
    parts = [f"{f} ({v*100:.1f}%)" for f, v in field_freqs.items()]
    return "Decision based on " + ", ".join(parts)


def _transductive_predict(agent, test_inputs, predictions):
    """Shared tree-voting prediction logic for both agent variants."""
    if len(agent.queried_inputs) < 3:
        return _majority_fallback(agent, predictions)

    fields_with_thresholds = []
    for f in agent.scenario.fields:
        if f.name not in test_inputs[0]:
            continue
        candidates = _build_candidate_thresholds(test_inputs, f.name, f)
        if candidates:
            fields_with_thresholds.append((f.name, f, candidates))

    if not fields_with_thresholds:
        return _majority_fallback(agent, predictions)

    # 50K iterations × 4 depths = 200K tree samples
    rng = random.Random(42)
    consistent = []
    for _ in range(50000):
        for depth in range(1, 5):
            tree = _sample_tree(fields_with_thresholds, depth, rng)
            if _tree_consistent(tree, agent.queried_inputs, agent.queried_results):
                consistent.append(tree)

    print(f"  [tree_vote] {len(consistent)} consistent trees from 200K samples")

    if len(consistent) < 5:
        print("  [tree_vote] Too few consistent trees, falling back to majority")
        return _majority_fallback(agent, predictions)

    final = list(predictions)
    for i, pred in enumerate(final):
        if pred is not None:
            continue
        votes = sum(1 for t in consistent if t.predict(test_inputs[i]))
        final[i] = votes > len(consistent) / 2

    field_freqs = _field_frequencies(consistent)
    pattern = _fields_to_pattern(field_freqs)

    metadata = {
        "strategy": agent.name,
        "consistent_trees": len(consistent),
        "samples_queried": len(agent.queried_inputs),
        "budget_summary": agent.model.budget.summary(),
        "pattern": pattern,
        "field_frequencies": field_freqs,
    }
    return AgentResult(predictions=final, metadata=metadata)


def _majority_fallback(agent, predictions):
    """Predict all unseen as majority class of queried labels."""
    if not agent.queried_results:
        majority = True
    else:
        majority = sum(agent.queried_results) > len(agent.queried_results) / 2

    final = list(predictions)
    for i, pred in enumerate(final):
        if pred is None:
            final[i] = majority

    metadata = {
        "strategy": f"{agent.name}_majority_fallback",
        "samples_queried": len(agent.queried_inputs),
        "budget_summary": agent.model.budget.summary(),
    }
    return AgentResult(predictions=final, metadata=metadata)


# ---------------------------------------------------------------------------
# Agent 1: random sampling + tree voting
# ---------------------------------------------------------------------------

@register_agent("tree_vote")
class TreeVoteAgent(SampleThenLLMGuessAgent):
    """Blackbox tree voting with random sampling (same query order as blackbox)."""

    name = "tree_vote"

    def api_phase(self, test_inputs, predictions):
        return _transductive_predict(self, test_inputs, predictions)

    def predict(self, test_inputs):
        predictions = self._sample_and_query(test_inputs)
        return _transductive_predict(self, test_inputs, predictions)

    def predict_with_prompts(self, test_inputs, prompts):
        predictions = self._sample_and_query(test_inputs, prompts=prompts)
        return _transductive_predict(self, test_inputs, predictions)

    def get_metadata(self) -> dict[str, Any]:
        return {
            "strategy": "tree_vote",
            "samples_queried": len(self.queried_inputs),
            "budget_summary": self.model.budget.summary(),
        }


# ---------------------------------------------------------------------------
# Agent 2: extreme start + farthest-first + tree voting
# ---------------------------------------------------------------------------

@register_agent("tree_vote_spread")
class TreeVoteSpreadAgent(SampleThenLLMGuessAgent):
    """Blackbox tree voting with extreme-start + farthest-first sampling."""

    name = "tree_vote_spread"

    def _sample_and_query(self, test_inputs, prompts=None):
        """Extreme start + unweighted farthest-first. No interp."""
        self._pre_query_setup()
        n = len(test_inputs)
        predictions = [None] * n
        self.queried_inputs = []
        self.queried_results = []
        self.interp_results = []
        self.seen_indices = []
        queried_idx_set = set()

        fields = get_available_fields(test_inputs[0], self.scenario.fields)
        int_fields = [f.name for f in self.scenario.fields
                      if f.field_type == FieldType.INTEGER and f.name in test_inputs[0]]

        def int_sum(inp):
            return sum(int(inp.get(f, 0)) for f in int_fields)

        # Phase 1: 3 extreme queries (min sum, max sum, farthest from both)
        sorted_by_sum = sorted(range(n), key=lambda i: int_sum(test_inputs[i]))
        extremes = [sorted_by_sum[0], sorted_by_sum[-1]]
        spread_order = compute_spread_order(test_inputs, fields, first_idx=sorted_by_sum[0])
        for idx in spread_order:
            if idx not in set(extremes):
                extremes.append(idx)
                break

        for idx in extremes[:min(3, self.model.budget.remaining)]:
            prompt = prompts[idx] if prompts else self.make_prompt(test_inputs[idx], self.format_style)
            try:
                prediction, probs = self.model.predict_yes_no(prompt)
                predictions[idx] = prediction
                self.queried_inputs.append(test_inputs[idx])
                self.queried_results.append(prediction)
                self.interp_results.append({})
                queried_idx_set.add(idx)
            except BudgetExceededError:
                break

        if self.model.budget.remaining <= 0:
            self.seen_indices = list(queried_idx_set)
            self._post_query_cleanup()
            return predictions

        # Phase 2: unweighted farthest-first
        remaining_budget = self.model.budget.remaining
        for q in range(remaining_budget):
            best_idx = None
            best_min_dist = -1
            for i in range(n):
                if i in queried_idx_set:
                    continue
                min_dist = float('inf')
                for qi in queried_idx_set:
                    dist = 0.0
                    for f in fields:
                        v1 = test_inputs[i].get(f.name)
                        v2 = test_inputs[qi].get(f.name)
                        if v1 is None or v2 is None:
                            continue
                        if f.field_type == FieldType.ENUM:
                            dist += (0.0 if str(v1) == str(v2) else 1.0)
                        else:
                            rng_size = (f.range[1] - f.range[0]) if f.range else 1
                            dist += ((float(v1) - float(v2)) / rng_size) ** 2
                    min_dist = min(min_dist, dist)
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = i

            if best_idx is None:
                break

            prompt = prompts[best_idx] if prompts else self.make_prompt(test_inputs[best_idx], self.format_style)
            try:
                prediction, probs = self.model.predict_yes_no(prompt)
                predictions[best_idx] = prediction
                self.queried_inputs.append(test_inputs[best_idx])
                self.queried_results.append(prediction)
                self.interp_results.append({})
                queried_idx_set.add(best_idx)
            except BudgetExceededError:
                break

        self.seen_indices = list(queried_idx_set)
        self._post_query_cleanup()
        return predictions

    def api_phase(self, test_inputs, predictions):
        return _transductive_predict(self, test_inputs, predictions)

    def predict(self, test_inputs):
        predictions = self._sample_and_query(test_inputs)
        return _transductive_predict(self, test_inputs, predictions)

    def predict_with_prompts(self, test_inputs, prompts):
        predictions = self._sample_and_query(test_inputs, prompts=prompts)
        return _transductive_predict(self, test_inputs, predictions)

    def get_metadata(self) -> dict[str, Any]:
        return {
            "strategy": "tree_vote_spread",
            "samples_queried": len(self.queried_inputs),
            "budget_summary": self.model.budget.summary(),
        }
