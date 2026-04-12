#!/usr/bin/env python3
"""Circuit sensitivity analysis and pattern field F1.

Part 1 (--sensitivity-only): Per-depth distribution of truly sensitive fields.
Part 2 (--pattern-only): Precision/recall/F1 of agent-discovered patterns vs ground truth.

Usage:
    python scripts/analysis/analyze_sensitivity.py <batch_dir> [<batch_dir2> ...]
    python scripts/analysis/analyze_sensitivity.py <batch_dir> --sensitivity-only
    python scripts/analysis/analyze_sensitivity.py <batch_dir> --pattern-only --agents blackbox gradient
    python scripts/analysis/analyze_sensitivity.py <batch_dir> --ci 95
    python scripts/analysis/analyze_sensitivity.py <batch_dir> --no-ci
"""

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Allow running as `python scripts/analysis/analyze_sensitivity.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.circuits import compute_field_sensitivity
from src.scenarios import get_scenario

# --- Field aliases for pattern matching ---

FIELD_ALIASES = {
    "car_purchase": {
        "brand":         [r"\bbrand\b", r"\bmake\b"],
        "year":          [r"\byear\b"],
        "color":         [r"\bcolou?r\b"],
        "horsepower":    [r"\bhorsepower\b", r"\bhp\b", r"\bpower\b"],
        "drivetrain":    [r"\bdrivetrain\b", r"\bdrive\s*train\b", r"\bdrive\b"],
        "mpg":           [r"\bmpg\b", r"\bmiles?\s*per\s*gallon\b", r"\bfuel\b", r"\bmileage\b"],
        "seat_capacity": [r"\bseat\b", r"\bcapacity\b", r"\bseating\b"],
        "interior":      [r"\binterior\b"],
        "condition":     [r"\bcondition\b"],
        "price":         [r"\bprice\b", r"\bcost\b"],
    },
    "movie_pick": {
        "release_year":      [r"\brelease\s*year\b", r"\byear\b"],
        "genre":             [r"\bgenre\b"],
        "language":          [r"\blanguage\b"],
        "runtime":           [r"\bruntime\b", r"\brun\s*time\b", r"\bduration\b", r"\blength\b"],
        "rating":            [r"\brating\b"],
        "release_format":    [r"\btheatrical\b", r"\bstreaming\b", r"\brelease\s*format\b"],
        "budget":            [r"\bbudget\b"],
        "box_office":        [r"\bbox\s*office\b", r"\brevenue\b", r"\bgross\b"],
        "color_format":      [r"\bcolor\b", r"\bblack.and.white\b", r"\bmonochrome\b"],
        "cast_size":         [r"\bcast\b"],
    },
    "apartment_hunt": {
        "sqft":          [r"\bsqft\b", r"\bsquare\s*feet\b", r"\bsq\s*ft\b", r"\bsize\b", r"\barea\b"],
        "bedrooms":      [r"\bbedroom\b", r"\bbed\b", r"\brooms?\b"],
        "rent":          [r"\brent\b", r"\bprice\b", r"\bcost\b"],
        "furnished":     [r"\bfurnished\b", r"\bfurniture\b"],
        "floor":         [r"\bfloor\b", r"\bstory\b", r"\bstorey\b"],
        "pet_friendly":  [r"\bpet\b"],
        "laundry":       [r"\blaundry\b", r"\bwasher\b"],
        "parking":       [r"\bparking\b", r"\bgarage\b"],
        "building_age":  [r"\bbuilding\s*age\b", r"\bbuilding\b.*\bage\b"],
        "elevator":      [r"\belevator\b", r"\blift\b"],
    },
}


def _ci_half_width(values: list[float], level: int = 90) -> float:
    """Compute CI half-width: t * std / sqrt(n)."""
    from scipy.stats import t
    n = len(values)
    if n < 2:
        return 0.0
    t_val = t.ppf(1 - (1 - level / 100) / 2, df=n - 1)
    return t_val * np.std(values, ddof=1) / np.sqrt(n)


def _strip_parenthesized(text: str) -> str:
    """Remove all parenthesized content to avoid false positives from negation lists.

    e.g. "Only drivetrain matters (brand, year, color, ... do not affect)" → "Only drivetrain matters "
    """
    return re.sub(r"\([^)]*\)", "", text)


def _field_mentioned(pattern_text: str, field_name: str, scenario_name: str) -> bool:
    """Check if a field is mentioned in the pattern text (ignoring parenthesized content)."""
    text = _strip_parenthesized(pattern_text).lower()
    # Always try raw field name and underscore→space variant
    if re.search(r"\b" + re.escape(field_name) + r"\b", text):
        return True
    space_name = field_name.replace("_", " ")
    if re.search(r"\b" + re.escape(space_name) + r"\b", text):
        return True
    # Try scenario-specific aliases
    aliases = FIELD_ALIASES.get(scenario_name, {}).get(field_name, [])
    for alias_pattern in aliases:
        if re.search(alias_pattern, text, re.IGNORECASE):
            return True
    return False


# Old agent name -> new agent name mapping (pre-refactor batches)
_OLD_AGENT_NAMES = {
    "sample_then_llm_guess": "blackbox",
    "sample_then_gradient_llm_v2": "gradient",
    "sample_then_relp_llm": "relp",
    "blackbox_prefill": "prefill",
    "sample_then_logit_lens_llm": "logit_lens",
    "sample_then_logit_lens_field_llm": "logit_lens_field",
    "sample_then_sae_tfidf_llm": "sae_tfidf",
    "sample_then_sae_autointerp_llm": "sae_autointerp",
    "sample_then_sae_gradient_attribution_llm": "sae_gradient",
    "sample_then_sae_mean_diff_llm": "sae_mean_diff",
    "sample_then_residual_token_llm": "res_token",
    "sample_then_nn": "nn",
    "spread_then_nn": "nn_spread",
    "sample_then_majority": "majority",
    "sample_then_logreg": "logreg",
}


def load_model_data(batch_dir: Path) -> list[dict]:
    """Load model configs and circuit data from a batch directory.

    Returns list of dicts with keys: model_name, model_dir, num_fields, scenario,
    circuit_expression, used_fields, agent_results (dict of agent_name -> result dict).
    """
    models = []
    for model_dir in sorted(batch_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        config_path = model_dir / "config.json"
        if not config_path.exists():
            continue
        with open(config_path) as f:
            config = json.load(f)

        num_fields = config.get("num_fields", 0)
        if num_fields == 0:
            continue

        source_model_dir = Path(config.get("model_dir", ""))
        circuit_path = source_model_dir / "circuit.json"
        if not circuit_path.exists():
            # Fallback: look in the eval batch dir (for HF-downloaded artifacts)
            circuit_path = model_dir / "circuit.json"
        if not circuit_path.exists():
            continue
        with open(circuit_path) as f:
            circuit_data = json.load(f)

        # Load agent results (normalize old agent names)
        agent_results = {}
        agent_results_dir = model_dir / "agent_results"
        if agent_results_dir.exists():
            for agent_file in agent_results_dir.glob("*.json"):
                name = _OLD_AGENT_NAMES.get(agent_file.stem, agent_file.stem)
                with open(agent_file) as f:
                    agent_results[name] = json.load(f)

        # Load distractor circuit if present
        distractor_fields = []
        distractor_path = source_model_dir / "distractor_circuit.json"
        if not distractor_path.exists():
            distractor_path = model_dir / "distractor_circuit.json"
        if distractor_path.exists():
            with open(distractor_path) as f:
                distractor_data = json.load(f)
            distractor_fields = distractor_data.get("used_fields", [])

        models.append({
            "model_name": config.get("model_name", model_dir.name),
            "model_dir": str(source_model_dir),
            "num_fields": num_fields,
            "scenario": config.get("scenario", "car_purchase"),
            "circuit_expression": circuit_data["expression"],
            "used_fields": circuit_data.get("used_fields", []),
            "distractor_fields": distractor_fields,
            "agent_results": agent_results,
        })

    return models


def compute_sensitivities(
    models: list[dict], n_samples: int = 10000
) -> dict[str, dict[str, float]]:
    """Compute field sensitivity for each model, cached by expression string.

    Returns dict: expression -> {field_name: sensitivity}.
    """
    cache = {}
    for m in models:
        expr = m["circuit_expression"]
        if expr in cache:
            continue
        scenario = get_scenario(m["scenario"])
        cache[expr] = compute_field_sensitivity(expr, scenario, n_samples=n_samples)
    return cache


def print_sensitivity_summary(
    models: list[dict],
    sensitivity_cache: dict[str, dict[str, float]],
    threshold: float = 0.01,
):
    """Print per-depth distribution of sensitive field counts."""
    # Group by depth
    by_depth = defaultdict(list)
    for m in models:
        sens = sensitivity_cache[m["circuit_expression"]]
        n_sensitive = sum(1 for v in sens.values() if v > threshold)
        by_depth[m["num_fields"]].append(n_sensitive)

    print("=== Circuit Sensitivity Summary ===")
    print(f"{'Depth':<8} {'Models':<8} {'Sensitive fields':<20} {'Distribution'}")
    print("-" * 70)

    for depth in sorted(by_depth.keys()):
        counts = by_depth[depth]
        mean = np.mean(counts)
        std = np.std(counts, ddof=1) if len(counts) > 1 else 0.0
        dist = defaultdict(int)
        for c in counts:
            dist[c] += 1
        dist_str = ", ".join(f"{k}: {v}" for k, v in sorted(dist.items()))
        print(f"d{depth:<7} {len(counts):<8} {mean:.1f} ± {std:.1f}{'':<12} {{{dist_str}}}")

    # Also print per-field sensitivity for one example per depth
    print()
    print("--- Per-field sensitivity examples (first model per depth) ---")
    seen_depths = set()
    for m in models:
        d = m["num_fields"]
        if d in seen_depths:
            continue
        seen_depths.add(d)
        sens = sensitivity_cache[m["circuit_expression"]]
        fields_above = [(k, v) for k, v in sorted(sens.items(), key=lambda x: -x[1]) if v > threshold]
        fields_str = ", ".join(f"{k}={v:.3f}" for k, v in fields_above)
        print(f"  d{d} ({m['model_name'][:40]}): {fields_str}")


def print_pattern_f1(
    models: list[dict],
    sensitivity_cache: dict[str, dict[str, float]],
    agents_filter: list[str] | None = None,
    ci_level: int | None = 90,
    threshold: float = 0.01,
    distractor: bool = False,
):
    """Print precision/recall/F1 of agent patterns vs ground truth fields.

    If distractor=True, ground truth = distractor fields (all of them, no sensitivity filter).
    """
    # Determine which agents to analyze (only LLM agents with patterns)
    all_agents = set()
    for m in models:
        for agent_name, result in m["agent_results"].items():
            pattern = result.get("agent_metadata", {}).get("pattern")
            if pattern:
                all_agents.add(agent_name)

    if agents_filter:
        all_agents = [a for a in agents_filter if a in all_agents]
    else:
        all_agents = sorted(all_agents)

    if not all_agents:
        print("No agents with patterns found.")
        return

    # Compute per-agent × per-depth metrics
    # metrics[agent][depth] = list of (precision, recall, f1)
    metrics = defaultdict(lambda: defaultdict(lambda: {"precision": [], "recall": [], "f1": []}))

    skipped_no_distractor = 0
    generated_distractor = 0
    for m in models:
        depth = m["num_fields"]
        scenario_name = m["scenario"]

        if distractor:
            gt_fields = set(m.get("distractor_fields", []))
            if not gt_fields:
                # Generate random non-overlapping fields as control
                scenario = get_scenario(scenario_name)
                all_fields = scenario.field_names()
                non_circuit = [f for f in all_fields if f not in m["used_fields"]]
                # Deterministic seed from model name
                seed = int(hashlib.md5(m["model_name"].encode()).hexdigest(), 16) % (2**32)
                rng = np.random.RandomState(seed)
                n_pick = min(depth, len(non_circuit))
                gt_fields = set(rng.choice(non_circuit, size=n_pick, replace=False))
                generated_distractor += 1
        else:
            sens = sensitivity_cache[m["circuit_expression"]]
            gt_fields = {k for k, v in sens.items() if v > threshold}
            if not gt_fields:
                continue

        for agent_name in all_agents:
            result = m["agent_results"].get(agent_name)
            if not result:
                continue
            pattern = result.get("agent_metadata", {}).get("pattern")
            if not pattern:
                continue

            # Find mentioned fields
            mentioned = set()
            scenario = get_scenario(scenario_name)
            for field_name in scenario.field_names():
                if _field_mentioned(pattern, field_name, scenario_name):
                    mentioned.add(field_name)

            # Compute P/R/F1
            tp = len(mentioned & gt_fields)
            fp = len(mentioned - gt_fields)
            fn = len(gt_fields - mentioned)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics[agent_name][depth]["precision"].append(precision)
            metrics[agent_name][depth]["recall"].append(recall)
            metrics[agent_name][depth]["f1"].append(f1)

    if distractor and generated_distractor:
        print(f"WARNING: {generated_distractor}/{len(models)} models had no distractor_circuit.json. "
              f"Generated random non-overlapping fields as control (seeded by model name).",
              file=sys.stderr)
        print(f"({generated_distractor} models used random non-overlapping fields as distractor control)")

    # Print table
    depths = sorted({m["num_fields"] for m in models})

    if distractor:
        if generated_distractor:
            print(f"=== Distractor Field F1 (control: random non-overlapping fields) ===")
            print(f"(ground truth = random non-circuit fields, {generated_distractor} models, matching: regex aliases)")
        else:
            print("=== Distractor Field F1 (pattern mentions vs distractor fields) ===")
            print(f"(ground truth = all distractor fields, matching: regex aliases)")
    else:
        print("=== Pattern Field F1 ===")
        print(f"(sensitivity threshold: {threshold}, matching: regex aliases)")
    print()

    for metric_name in ["f1", "precision", "recall"]:
        print(f"--- {metric_name.upper()} ---")
        header = f"{'Agent':<20}"
        for d in depths:
            header += f" | d{d}"
        header += " | avg"
        print(header)
        print("-" * len(header))

        for agent in all_agents:
            row = f"{agent:<20}"
            all_vals = []
            for d in depths:
                vals = metrics[agent][d][metric_name]
                if vals:
                    mean = np.mean(vals)
                    all_vals.extend(vals)
                    if ci_level is not None and len(vals) >= 2:
                        ci = _ci_half_width(vals, ci_level)
                        row += f" | {mean:.1%}±{ci:.1%}"
                    else:
                        row += f" | {mean:.1%}"
                else:
                    row += " | -"

            if all_vals:
                overall = np.mean(all_vals)
                if ci_level is not None and len(all_vals) >= 2:
                    ci = _ci_half_width(all_vals, ci_level)
                    row += f" | {overall:.1%}±{ci:.1%}"
                else:
                    row += f" | {overall:.1%}"
            else:
                row += " | -"
            print(row)

        print()


def main():
    parser = argparse.ArgumentParser(
        description="Circuit sensitivity analysis and pattern field F1"
    )
    parser.add_argument(
        "batch_dirs", type=str, nargs="+",
        help="Path(s) to batch evaluation directory(ies). Multiple dirs are merged.",
    )
    parser.add_argument(
        "--agents", type=str, nargs="+", default=None,
        help="Filter agents for pattern F1 (Part 2)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=2000,
        help="Sensitivity samples per model (default: 2000)",
    )
    parser.add_argument(
        "--ci", type=int, default=90,
        help="Confidence interval level (default: 90). Use --no-ci to disable.",
    )
    parser.add_argument("--no-ci", action="store_true", help="Disable confidence intervals")
    parser.add_argument("--sensitivity-only", action="store_true", help="Run only sensitivity analysis")
    parser.add_argument("--pattern-only", action="store_true", help="Run only pattern F1 analysis")
    parser.add_argument("--distractor", action="store_true",
                        help="Compute F1 vs distractor fields instead of circuit fields (for badrat models)")
    parser.add_argument(
        "--threshold", type=float, default=0.01,
        help="Sensitivity threshold for 'truly sensitive' (default: 0.01 = 1%%)",
    )

    args = parser.parse_args()

    if not args.no_ci and not (1 <= args.ci <= 99):
        print(f"Error: --ci must be between 1 and 99, got {args.ci}")
        return

    # Load all models from all batch dirs
    all_models = []
    for bd in args.batch_dirs:
        batch_dir = Path(bd)
        if not batch_dir.exists():
            print(f"Error: Batch directory not found: {batch_dir}")
            return
        print(f"Loading from {batch_dir}...")
        models = load_model_data(batch_dir)
        all_models.extend(models)
        print(f"  {len(models)} models loaded")

    if not all_models:
        print("No models found")
        return

    # Deduplicate models by (model_dir) for sensitivity (same model in multiple batches)
    unique_models = {}
    for m in all_models:
        key = m["model_dir"]
        if key not in unique_models:
            unique_models[key] = m
        else:
            # Merge agent results
            unique_models[key]["agent_results"].update(m["agent_results"])
    merged_models = list(unique_models.values())

    ci_level = None if args.no_ci else args.ci

    # Skip sensitivity computation if only distractor F1 is needed
    sensitivity_cache = {}
    if not args.distractor or not args.pattern_only:
        unique_exprs = set(m["circuit_expression"] for m in merged_models)
        print(f"\nComputing sensitivity for {len(unique_exprs)} unique circuits ({args.n_samples} samples each)...")
        sensitivity_cache = compute_sensitivities(merged_models, n_samples=args.n_samples)
        print("Done.\n")

    if not args.pattern_only:
        print_sensitivity_summary(merged_models, sensitivity_cache, threshold=args.threshold)
        print()

    if not args.sensitivity_only:
        print_pattern_f1(
            merged_models, sensitivity_cache,
            agents_filter=args.agents,
            ci_level=ci_level,
            threshold=args.threshold,
            distractor=args.distractor,
        )


if __name__ == "__main__":
    main()
