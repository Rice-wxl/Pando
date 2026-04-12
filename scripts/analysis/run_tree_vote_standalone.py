#!/usr/bin/env python3
"""Standalone tree_vote runner — no GPU / no model loading.

Reads each batch model dir's test_data.json, uses cached `ground_truth`
labels (which are the model's outputs), reproduces the blackbox query order
via seed=42 + random.shuffle, and runs the tree-voting transductive inference
directly. Writes agent_results/tree_vote.json per model, in the same format
as scripts/eval.py.

Why this is equivalent to running scripts/eval.py --update-batch with
--agents tree_vote:

1. `test_data.json.ground_truth[i]` was obtained by forward-passing the
   finetuned model on `test_data.json.test_inputs[i]` at test-set
   construction time. It is the exact label that model.predict_yes_no()
   would return today.
2. The 10 queried indices are the first 10 of random.shuffle(range(100))
   after random.seed(42). src/utils.set_seed(config.seed=42) is called
   before each agent by update_batch; the sample_then_llm_guess flow
   starts with `random.shuffle(list(range(n_samples)))`. So tree_vote's
   query indices are identical to blackbox's on every model.
3. The tree-voting logic in src/agents/tree_vote.py uses a private
   `random.Random(42)` for tree sampling, so it is independent of the
   global random state and fully deterministic.

Usage:
    # Run on the 18 freeform batches that generate_tables.py consumes
    python scripts/analysis/run_tree_vote_standalone.py

    # Run on an arbitrary list of batches
    python scripts/analysis/run_tree_vote_standalone.py --batches path/to/batch1 path/to/batch2

    # Dry-run (prints what would be written)
    python scripts/analysis/run_tree_vote_standalone.py --dry-run

    # Skip models that already have tree_vote.json
    python scripts/analysis/run_tree_vote_standalone.py --skip-existing
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Import tree_vote helpers WITHOUT triggering src/agents/__init__.py
# (which eagerly imports all agents and transitively requires torch).
# We load tree_vote.py as an isolated module with lightweight stubs for
# the agent-infrastructure imports it doesn't need for the four free
# functions used here.
import importlib.util as _ilu
import types as _types

_repo = Path(__file__).resolve().parents[2]

def _load_module(name, path):
    """Load a single .py file into sys.modules under *name*."""
    path = Path(path)
    # If loading a package __init__.py, set submodule_search_locations
    # so that relative imports within the package resolve correctly.
    kwargs = {}
    if path.name == "__init__.py":
        kwargs["submodule_search_locations"] = [str(path.parent)]
    spec = _ilu.spec_from_file_location(name, str(path), **kwargs)
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# 1. Real modules that the helper functions actually use at runtime:
#    NOTE: src.scenarios.__init__ already imports src.scenarios.base (which
#    defines FieldType). Do NOT reload base.py separately — that creates a
#    second FieldType class and breaks `field.field_type == FieldType.ENUM`.
_load_module("src", _repo / "src" / "__init__.py")
_load_module("src.scenarios", _repo / "src" / "scenarios" / "__init__.py")

# 2. Lightweight stubs for modules that tree_vote.py imports at class
#    definition time but that the four free functions never use at runtime.
#    This avoids pulling in torch (via src.budget → src.inference).
_dummy_class = type("_Dummy", (), {})
_budget_stub = _types.ModuleType("src.budget")
_budget_stub.BudgetExceededError = type("BudgetExceededError", (Exception,), {})
_budget_stub.BudgetTracker = _dummy_class
_budget_stub.BudgetedModel = _dummy_class
sys.modules["src.budget"] = _budget_stub

_agents_stub = _types.ModuleType("src.agents")
_agents_stub.__path__ = [str(_repo / "src" / "agents")]
_agents_stub.__package__ = "src.agents"
_agents_stub.register_agent = lambda name: (lambda cls: cls)
sys.modules["src.agents"] = _agents_stub
_stub_attrs = {
    "src.agents.base": {"AgentResult": _dummy_class},
    "src.agents.interp_llm_base": {},
    "src.agents.sae_base": {},
    "src.agents.sampling": {
        "compute_spread_order": None,
        "get_available_fields": None,
    },
    "src.agents.sample_then_llm_guess": {
        "SampleThenLLMGuessAgent": _dummy_class,
    },
}
for _sn, _attrs in _stub_attrs.items():
    _mod = _types.ModuleType(_sn)
    for _k, _v in _attrs.items():
        setattr(_mod, _k, _v)
    sys.modules[_sn] = _mod

# 3. Load tree_vote.py itself — stubs satisfy its class-level imports,
#    and the four free functions only need FieldType from scenarios.base.
_tv = _load_module("src.agents.tree_vote", _repo / "src" / "agents" / "tree_vote.py")
_build_candidate_thresholds = _tv._build_candidate_thresholds
_sample_tree = _tv._sample_tree
_tree_consistent = _tv._tree_consistent
_field_frequencies = _tv._field_frequencies

from src.scenarios import get_scenario


def _above_mean_fields_pattern(field_freqs: dict[str, float]) -> str:
    """Emit only fields whose frequency exceeds the mean (per paper Table 11 text:
    "A field is counted as 'predicted' if it appears as a split node in the
    consistent trees more often than the mean frequency across all fields.")

    This matches how the paper computes tree_vote field F1, as opposed to the
    original src/agents/tree_vote._fields_to_pattern which emits all fields.
    """
    if not field_freqs:
        return ""
    mean_freq = sum(field_freqs.values()) / len(field_freqs)
    above = [(f, v) for f, v in field_freqs.items() if v > mean_freq]
    if not above:
        return ""
    above.sort(key=lambda x: -x[1])
    parts = [f"{f} ({v*100:.1f}%)" for f, v in above]
    return "Decision based on " + ", ".join(parts)

# These are the 18 batch directories the paper aggregates over (3 scenarios
# × 3 setups × 2 slices-per-config). Source of truth:
# scripts/analysis/generate_tables.py BATCH_DIRS[freeform_std..od_freeform_badrat].
DEFAULT_BATCHES = [
    # car_purchase freeform_std
    "batch_20260301_033718", "batch_20260301_033721",
    # car_purchase freeform_goodrat
    "batch_20260301_033720", "batch_20260301_033716",
    # car_purchase freeform_badrat
    "batch_20260306_212448", "batch_20260306_212451",
    # movie_pick freeform_std / goodrat / badrat
    "batch_20260321_203958", "batch_20260321_204227",
    "batch_20260322_005058", "batch_20260322_002644",
    "batch_20260321_222546", "batch_20260322_002129",
    # oversight_defection freeform_std / goodrat / badrat
    "batch_20260325_120645", "batch_20260325_143729",
    "batch_20260325_170147", "batch_20260325_191509",
    "batch_20260325_193335", "batch_20260325_214817",
]


def reconstruct_query_indices(n_samples: int, seed: int = 42) -> list[int]:
    """Reproduce the query-order a seed=42 sample_and_query call would pick.

    Mirrors src/agents/sampling.SampleAndQueryAgent.sample_and_query:
        random.shuffle(list(range(n_samples)))
    after src/utils.set_seed(seed).
    """
    random.seed(seed)
    # Match the order of operations in set_seed: random.seed, then numpy, torch.
    # Only random.seed matters for list shuffling.
    indices = list(range(n_samples))
    random.shuffle(indices)
    return indices


def run_tree_vote_on_model(
    model_eval_dir: Path,
    budget: int = 10,
) -> dict:
    """Run tree_vote transductive inference on one model's test data.

    Returns the full agent_results/tree_vote.json payload.
    """
    test_data_path = model_eval_dir / "test_data.json"
    td = json.loads(test_data_path.read_text())
    test_inputs = td["test_inputs"]
    ground_truth = td["ground_truth"]

    config = json.loads((model_eval_dir / "config.json").read_text())
    scenario_name = config.get("scenario", "car_purchase")
    scenario = get_scenario(scenario_name)

    n = len(test_inputs)
    # Reproduce query order with the same seed that update_batch would use.
    seed = config.get("seed", 42)
    shuffled = reconstruct_query_indices(n, seed=seed)
    queried_idx = shuffled[:budget]
    queried_set = set(queried_idx)

    # Build the "agent" inputs / labels from cached ground_truth
    queried_inputs = [test_inputs[i] for i in queried_idx]
    queried_results = [bool(ground_truth[i]) for i in queried_idx]

    # Build candidate thresholds using all test inputs, matching the agent
    # (tree_vote uses actual test-input values as candidate split points).
    fields_with_thresholds = []
    for f in scenario.fields:
        if f.name not in test_inputs[0]:
            continue
        candidates = _build_candidate_thresholds(test_inputs, f.name, f)
        if candidates:
            fields_with_thresholds.append((f.name, f, candidates))

    # Tree sampling: 50K iterations × depths 1..4 = 200K candidates,
    # keep those consistent with the queried labels.
    rng = random.Random(42)
    consistent = []
    for _ in range(50000):
        for depth in range(1, 5):
            tree = _sample_tree(fields_with_thresholds, depth, rng)
            if _tree_consistent(tree, queried_inputs, queried_results):
                consistent.append(tree)

    # Majority-vote prediction on held-out samples; queried samples get
    # their true (cached) model label (same as eval.py behavior).
    predictions = [None] * n
    for i in queried_idx:
        predictions[i] = bool(ground_truth[i])

    if len(consistent) >= 5:
        for i in range(n):
            if predictions[i] is not None:
                continue
            votes = sum(1 for t in consistent if t.predict(test_inputs[i]))
            predictions[i] = votes > len(consistent) / 2
        field_freqs = _field_frequencies(consistent)
        pattern = _above_mean_fields_pattern(field_freqs)
        strategy = "tree_vote"
    else:
        # Fallback: majority class of queried labels
        if queried_results:
            majority = sum(queried_results) > len(queried_results) / 2
        else:
            majority = True
        for i in range(n):
            if predictions[i] is None:
                predictions[i] = majority
        field_freqs = {}
        pattern = ""
        strategy = "tree_vote_majority_fallback"

    # Compute accuracy stats (over ALL samples — held-out filtering happens in
    # generate_tables.py via _held_out_accuracy, which excludes queried indices)
    gt_bool = [bool(g) for g in ground_truth]
    correct = sum(1 for p, g in zip(predictions, gt_bool) if p == g)
    tp = sum(1 for p, g in zip(predictions, gt_bool) if p and g)
    tn = sum(1 for p, g in zip(predictions, gt_bool) if not p and not g)
    fp = sum(1 for p, g in zip(predictions, gt_bool) if p and not g)
    fn = sum(1 for p, g in zip(predictions, gt_bool) if not p and g)

    per_input = [
        {
            "index": i,
            "input": test_inputs[i],
            "prediction": predictions[i],
            "ground_truth": gt_bool[i],
            "correct": predictions[i] == gt_bool[i],
        }
        for i in range(n)
    ]

    return {
        "agent_name": "tree_vote",
        "accuracy": correct / n if n else 0.0,
        "correct": correct,
        "total": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "budget_used": budget,
        "budget_total": budget,
        "agent_metadata": {
            "strategy": strategy,
            "consistent_trees": len(consistent),
            "samples_queried": len(queried_idx),
            "pattern": pattern,
            "field_frequencies": field_freqs,
            # Both keys for compatibility:
            # - `queried_indices` is read by scripts/analysis/generate_tables.py
            #   `_get_queried_indices()` (for held-out accuracy filtering).
            # - `seen_indices` is what src/agents/base sets on the live agent.
            "queried_indices": list(queried_idx),
            "seen_indices": list(queried_idx),
            "budget_summary": {
                "total_budget": budget,
                "used": budget,
                "remaining": 0,
                "forward_calls": budget,
                "backward_calls": 0,
                "utilization": 1.0,
            },
        },
        "per_input_results": per_input,
    }


def process_batch(
    batch_dir: Path,
    dry_run: bool = False,
    skip_existing: bool = False,
) -> tuple[int, int]:
    """Process all models in one batch dir. Returns (written, skipped)."""
    if not batch_dir.exists():
        print(f"  MISSING: {batch_dir}")
        return (0, 0)

    model_dirs = sorted(d for d in batch_dir.iterdir() if d.is_dir())
    written = 0
    skipped = 0

    for md in model_dirs:
        if not (md / "test_data.json").exists() or not (md / "config.json").exists():
            continue

        out_path = md / "agent_results" / "tree_vote.json"
        if skip_existing and out_path.exists():
            skipped += 1
            continue

        try:
            result = run_tree_vote_on_model(md)
        except Exception as e:
            print(f"  ERROR {md.name}: {e}")
            continue

        if dry_run:
            print(f"  [dry] {md.name}: acc={result['accuracy']:.2%} "
                  f"(trees={result['agent_metadata']['consistent_trees']})")
        else:
            out_path.parent.mkdir(exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2))
        written += 1

    return (written, skipped)


def main():
    parser = argparse.ArgumentParser(description="Standalone tree_vote runner")
    parser.add_argument(
        "--batches", type=str, nargs="+", default=None,
        help="Batch dirs to process (relative to outputs/evaluations/ if not absolute). "
             "Defaults to the 18 freeform batches used by the paper.",
    )
    parser.add_argument(
        "--base", type=str, default="outputs/evaluations",
        help="Base directory for relative batch paths (default: outputs/evaluations)",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Do not write files; just print per-model accuracy")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip models that already have tree_vote.json")
    args = parser.parse_args()

    base = Path(args.base)
    names = args.batches if args.batches else DEFAULT_BATCHES
    batch_paths = []
    for n in names:
        p = Path(n)
        if not p.is_absolute():
            p = base / n
        batch_paths.append(p)

    t0 = time.time()
    total_written = 0
    total_skipped = 0
    for bp in batch_paths:
        print(f"\n→ {bp.name}")
        written, skipped = process_batch(bp, dry_run=args.dry_run, skip_existing=args.skip_existing)
        print(f"  wrote {written}, skipped {skipped}")
        total_written += written
        total_skipped += skipped

    dt = time.time() - t0
    print(f"\nDone. {total_written} models processed, {total_skipped} skipped, {dt:.1f}s total")


if __name__ == "__main__":
    main()
