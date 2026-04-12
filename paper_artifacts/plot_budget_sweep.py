#!/usr/bin/env python3
"""Plot budget sweep: accuracy vs sample count for 4 agents × 3 configs."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1] / "outputs" / "evaluations"

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

AGENTS = ["relp", "prefill", "blackbox", "nn"]
OLD_NAMES = {
    "sample_then_llm_guess": "blackbox",
    "blackbox_prefill": "prefill",
    "sample_then_relp_llm": "relp",
    "sample_then_nn": "nn",
}

CONFIG_LABELS = {
    "std": "No verbalization",
    "goodrat": "Faithful",
    "badrat": "Unfaithful",
}

AGENT_STYLES = {
    "relp": {"color": "#4CAF50", "marker": "o", "label": "relp"},
    "prefill": {"color": "#2196F3", "marker": "s", "label": "prefill"},
    "blackbox": {"color": "#000000", "marker": "^", "label": "sample_only"},
    "nn": {"color": "#9E9E9E", "marker": "D", "label": "nn"},
}


def load_models(batch_dirs):
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
            agent_results = {}
            ar_dir = model_dir / "agent_results"
            if ar_dir.exists():
                for af in ar_dir.glob("*.json"):
                    name = OLD_NAMES.get(af.stem, af.stem)
                    with open(af) as f:
                        agent_results[name] = json.load(f)
            all_models.append({
                "model_dir": str(config.get("model_dir", "")),
                "agent_results": agent_results,
            })
    # Dedup
    unique = {}
    for m in all_models:
        key = m["model_dir"]
        if key not in unique:
            unique[key] = m
        else:
            unique[key]["agent_results"].update(m["agent_results"])
    return list(unique.values())


def get_avg_accuracy(models, agent, holdout_only=True):
    accs = []
    for m in models:
        r = m["agent_results"].get(agent)
        if not r:
            continue
        if holdout_only:
            # Exclude queried samples (trivially correct)
            budget_used = r.get("budget_used", 10)
            total = r.get("total", 100)
            correct = r.get("correct", 0)
            # Queried samples are always correct, so holdout correct = correct - budget_used
            holdout_total = total - budget_used
            holdout_correct = correct - budget_used
            if holdout_total > 0:
                accs.append(holdout_correct / holdout_total)
        else:
            accs.append(r.get("accuracy", 0.0))
    return np.mean(accs) if accs else None


def main():
    budgets = [3, 5, 10, 15, 20]
    configs = ["std", "goodrat", "badrat"]

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
                acc = get_avg_accuracy(models, agent)
                if acc is not None:
                    xs.append(b)
                    ys.append(acc * 100)

            style = AGENT_STYLES[agent]
            ax.plot(xs, ys, marker=style["marker"], color=style["color"],
                    label=style["label"], linewidth=2, markersize=7)

        ax.set_xlabel("Sample budget", fontsize=11)
        ax.set_xticks(budgets)
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.set_ylabel("Held-out accuracy (%)", fontsize=11)
        if ax_idx == 2:
            ax.legend(loc="lower right", fontsize=10)

    axes[0].set_ylim(50, 90)
    plt.tight_layout()
    out = Path(__file__).parent / "plot_budget_sweep.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved to {out}")

    # Also print the data
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
                acc = get_avg_accuracy(models, agent)
                if acc is not None:
                    row += f"  {acc*100:5.1f}"
                else:
                    row += f"    — "
            print(row)


if __name__ == "__main__":
    main()
