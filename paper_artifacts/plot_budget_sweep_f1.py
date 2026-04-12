#!/usr/bin/env python3
"""Plot budget sweep: pattern field F1 vs sample count for 3 agents × 3 configs."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.circuits import compute_field_sensitivity
from src.scenarios import get_scenario
from scripts.analysis.megatable import (
    _field_mentioned,
    load_sensitivity_cache as _mt_load_sensitivity_cache,
    save_sensitivity_cache as _mt_save_sensitivity_cache,
)

BASE = REPO_ROOT / "outputs" / "evaluations"

# Budget sweep batch dirs (car_purchase only)
BUDGET_BATCHES = {
    (3, "std"): ["batch_20260326_181629", "batch_20260326_181659"],
    (3, "goodrat"): ["batch_20260326_181819", "batch_20260327_014840"],
    (3, "badrat"): ["batch_20260326_193025", "batch_20260326_193306"],
    (5, "std"): ["batch_20260327_014843", "batch_20260326_193633"],
    (5, "goodrat"): ["batch_20260326_194137", "batch_20260326_194513"],
    (5, "badrat"): ["batch_20260326_182813", "batch_20260326_194540"],
    (15, "std"): ["batch_20260326_194744", "batch_20260326_195151"],
    (15, "goodrat"): ["batch_20260326_195427", "batch_20260326_195603"],
    (15, "badrat"): ["batch_20260326_200657", "batch_20260326_201342"],
    (20, "std"): ["batch_20260326_201445", "batch_20260326_201515"],
    (20, "goodrat"): ["batch_20260326_203701", "batch_20260326_203705"],
    (20, "badrat"): ["batch_20260326_203833", "batch_20260326_205356"],
}

# b=10 from existing eval (car_purchase freeform)
B10_BATCHES = {
    "std": ["batch_20260301_033718", "batch_20260301_033721"],
    "goodrat": ["batch_20260301_033720", "batch_20260301_033716"],
    "badrat": ["batch_20260306_212448", "batch_20260306_212451"],
}

# Only agents that produce a pattern (nn has no pattern → no F1)
AGENTS = ["relp", "prefill", "blackbox"]

OLD_NAMES = {
    "sample_then_llm_guess": "blackbox",
    "blackbox_prefill": "prefill",
    "sample_then_relp_llm": "relp",
}

CONFIG_LABELS = {
    "std": "No verbalization",
    "goodrat": "Faithful",
    "badrat": "Unfaithful",
}

AGENT_STYLES = {
    "relp": {"color": "#2196F3", "marker": "o", "label": "relp"},
    "prefill": {"color": "#4CAF50", "marker": "s", "label": "prefill"},
    "blackbox": {"color": "#FF9800", "marker": "^", "label": "sample_only"},
}

# ─── Helpers ────────────────────────────────────────────────────────────────

def load_sensitivity_cache():
    return _mt_load_sensitivity_cache()

def save_sensitivity_cache(cache):
    _mt_save_sensitivity_cache(cache)


def load_models(batch_dirs):
    """Load models with full metadata needed for F1 computation."""
    all_models = []
    for bd_name in batch_dirs:
        bd = BASE / bd_name
        if not bd.exists():
            continue
        for model_dir in sorted(bd.iterdir()):
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

            agent_results = {}
            ar_dir = model_dir / "agent_results"
            if ar_dir.exists():
                for af in ar_dir.glob("*.json"):
                    name = OLD_NAMES.get(af.stem, af.stem)
                    with open(af) as f:
                        agent_results[name] = json.load(f)

            all_models.append({
                "model_dir": str(source_model_dir),
                "scenario": config.get("scenario", "car_purchase"),
                "num_fields": num_fields,
                "circuit_expression": circuit_data["expression"],
                "used_fields": circuit_data.get("used_fields", []),
                "agent_results": agent_results,
            })

    # Dedup by model_dir, merging agent results
    unique = {}
    for m in all_models:
        key = m["model_dir"]
        if key not in unique:
            unique[key] = m
        else:
            unique[key]["agent_results"].update(m["agent_results"])
    return list(unique.values())


def get_avg_f1(models, agent, sensitivity_cache, threshold=0.01):
    """Compute mean pattern field F1 for an agent across models."""
    f1s = []
    for m in models:
        result = m["agent_results"].get(agent)
        if not result:
            continue
        pattern = result.get("agent_metadata", {}).get("pattern")
        if not pattern:
            continue

        gt_fields = {
            k for k, v in sensitivity_cache.get(m["circuit_expression"], {}).items()
            if v > threshold
        }
        if not gt_fields:
            continue

        scenario_name = m["scenario"]
        scenario = get_scenario(scenario_name)
        mentioned = {fn for fn in scenario.field_names()
                     if _field_mentioned(pattern, fn, scenario_name)}

        tp = len(mentioned & gt_fields)
        fp = len(mentioned - gt_fields)
        fn = len(gt_fields - mentioned)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)

    return np.mean(f1s) if f1s else None


def main():
    budgets = [3, 5, 10, 15, 20]
    configs = ["std", "goodrat", "badrat"]

    # Load sensitivity cache, compute any missing
    sensitivity_cache = load_sensitivity_cache()
    all_models_for_cache = []
    for config in configs:
        for b in budgets:
            if b == 10:
                batch_dirs = B10_BATCHES[config]
            else:
                batch_dirs = BUDGET_BATCHES.get((b, config), [])
            if batch_dirs:
                all_models_for_cache.extend(load_models(batch_dirs))

    # Ensure all circuit expressions are cached
    new_count = 0
    for m in all_models_for_cache:
        expr = m["circuit_expression"]
        if expr not in sensitivity_cache:
            scenario = get_scenario(m["scenario"])
            sensitivity_cache[expr] = compute_field_sensitivity(expr, scenario, n_samples=10000)
            new_count += 1
            save_sensitivity_cache(sensitivity_cache)
    if new_count:
        print(f"Computed {new_count} new sensitivity entries")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)

    for ax_idx, config in enumerate(configs):
        ax = axes[ax_idx]
        ax.set_title(CONFIG_LABELS[config], fontsize=13)

        for agent in AGENTS:
            xs, ys = [], []
            for b in budgets:
                if b == 10:
                    batch_dirs = B10_BATCHES[config]
                else:
                    batch_dirs = BUDGET_BATCHES.get((b, config), [])
                if not batch_dirs:
                    continue
                models = load_models(batch_dirs)
                f1 = get_avg_f1(models, agent, sensitivity_cache)
                if f1 is not None:
                    xs.append(b)
                    ys.append(f1 * 100)

            style = AGENT_STYLES[agent]
            ax.plot(xs, ys, marker=style["marker"], color=style["color"],
                    label=style["label"], linewidth=2, markersize=7)

        ax.set_xlabel("Sample budget", fontsize=11)
        ax.set_xticks(budgets)
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.set_ylabel("Pattern field F1 (%)", fontsize=11)
        if ax_idx == 2:
            ax.legend(loc="lower right", fontsize=10)

    axes[0].set_ylim(0, 100)
    plt.tight_layout()
    out = Path(__file__).parent / "plot_budget_sweep_f1.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved to {out}")

    # Print data table
    for config in configs:
        print(f"\n{CONFIG_LABELS[config]}:")
        header = f"  {'Agent':<15}" + "".join(f"  b={b:>2}" for b in budgets)
        print(header)
        for agent in AGENTS:
            row = f"  {AGENT_STYLES[agent]['label']:<15}"
            for b in budgets:
                if b == 10:
                    batch_dirs = B10_BATCHES[config]
                else:
                    batch_dirs = BUDGET_BATCHES.get((b, config), [])
                models = load_models(batch_dirs) if batch_dirs else []
                f1 = get_avg_f1(models, agent, sensitivity_cache)
                if f1 is not None:
                    row += f"  {f1*100:5.1f}"
                else:
                    row += f"    — "
            print(row)


if __name__ == "__main__":
    main()
