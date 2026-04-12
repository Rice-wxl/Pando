"""Plot autoresearch agent progression: held-out accuracy and field recovery F1 over time.

Held-out accuracy excludes the ~10 queried samples from the accuracy calculation,
measuring only the agent's ability to predict unseen samples via the discovered pattern.

Field F1 uses sensitivity-filtered ground truth and selective paren stripping,
imported from megatable.py.

Usage:
    python scripts/analysis/plot_autoresearch_progression.py
    python scripts/analysis/plot_autoresearch_progression.py -o custom_output.png
"""

import argparse
import re
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from scripts.analysis.megatable import (
    BATCH_DIRS,
    compute_pattern_metrics,
    load_models,
    load_sensitivity_cache,
    get_sensitivities,
    _collect_f1_vals,
)
from scripts.analysis.generate_tables import _held_out_accuracy


# ---------------------------------------------------------------------------
# Agent timestamps from export_autoresearch/agent_descriptions.md
# ---------------------------------------------------------------------------

AGENT_TIMESTAMPS = {
    "ar_baseline": "2026-03-19 17:50",
    "ar_exp01":    "2026-03-19 17:55",
    "ar_exp09":    "2026-03-19 18:42",
    "ar_exp22":    "2026-03-19 21:34",
    "ar_exp33":    "2026-03-20 01:11",
    "ar_exp35":    "2026-03-20 01:46",
    "ar_exp47":    "2026-03-20 06:00",
    "ar_exp57":    "2026-03-20 08:31",
    "ar_exp65":    "2026-03-20 15:11",
    "ar_final":    "2026-03-20 19:22",
}
AGENTS_ORDER = list(AGENT_TIMESTAMPS.keys())

# Autoresearch eval batch dirs (from slurm_eval_autoresearch.sh)
AR_BATCH_DIRS = [
    "batch_20260330_030344", "batch_20260330_031609",
    "batch_20260330_031959", "batch_20260330_032000",
    "batch_20260330_032407", "batch_20260330_033159",
    "batch_20260330_035032", "batch_20260330_035845",
    "batch_20260330_040523", "batch_20260330_040804",
    "batch_20260330_041259", "batch_20260330_042055",
]

# Aggregation groups: name -> list of (scenario, rationale) pairs
# Rationale determined by model path containing "badrat"/"badrationale"
AGG_GROUPS = {
    "car_purchase (ID)": [("car_purchase", "norat")],
    "other scenarios (OOD)": [("movie_pick", "norat"), ("oversight_defection", "norat")],
    "unfaithful (OOD)": [("car_purchase", "badrat"), ("movie_pick", "badrat"), ("oversight_defection", "badrat")],
}

COLORS = {"car_purchase (ID)": "#2196F3", "other scenarios (OOD)": "#4CAF50", "unfaithful (OOD)": "#F44336"}
MARKERS = {"car_purchase (ID)": "o", "other scenarios (OOD)": "s", "unfaithful (OOD)": "D"}

# Human interventions to mark on the plot (from human_interaction_timeline.md)
HUMAN_INTERVENTIONS = {
    "#9 parallelism hint":              "2026-03-19 18:50",
    "#14-15 radical + constraint":      "2026-03-19 21:30",
    "#17 backward lens suggestion":     "2026-03-19 23:40",
    "#18 add error bars":               "2026-03-20 00:30",
    "#19 explore from 1st principle":   "2026-03-20 10:30",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_rationale(model: dict) -> str:
    """Determine rationale type from model path."""
    md = model.get("model_dir", "")
    if "badrat" in md or "badrationale" in md:
        return "badrat"
    return "norat"


def _filter_models(models, configs):
    """Filter models matching (scenario, rationale) pairs."""
    return [m for m in models if (m["scenario"], _get_rationale(m)) in configs]


def _ci_half_width(values, level=90):
    """Compute CI half-width using t-distribution."""
    from scipy.stats import t as t_dist
    n = len(values)
    if n < 2:
        return 0.0
    t_val = t_dist.ppf(1 - (1 - level / 100) / 2, df=n - 1)
    return t_val * np.std(values, ddof=1) / np.sqrt(n)


def get_held_out_accuracy(models, agent):
    """Return (mean, ci) of held-out accuracy for an agent."""
    accs = []
    for m in models:
        result = m["agent_results"].get(agent)
        if not result:
            continue
        ho_acc = _held_out_accuracy(result)
        if ho_acc is not None:
            accs.append(ho_acc * 100)
    if not accs:
        return None, None
    return np.mean(accs), _ci_half_width(accs)


def get_mean_f1(metrics, agent):
    """Return (mean, ci) of F1 from compute_pattern_metrics output."""
    vals = _collect_f1_vals(metrics, agent)
    if not vals:
        return None, None
    vals100 = [v * 100 for v in vals]
    return np.mean(vals100), _ci_half_width(vals100)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot autoresearch agent progression")
    parser.add_argument("-o", "--output", default="plot_autoresearch_progression.png",
                        help="Output file path (default: plot_autoresearch_progression.png)")
    args = parser.parse_args()

    raw_timestamps = [datetime.strptime(t, "%Y-%m-%d %H:%M") for t in AGENT_TIMESTAMPS.values()]
    t0 = raw_timestamps[0]
    elapsed_hours = [(t - t0).total_seconds() / 3600 for t in raw_timestamps]

    # Load autoresearch models (has ar_* agents)
    print("Loading autoresearch models...")
    ar_models = load_models(AR_BATCH_DIRS)
    print(f"  {len(ar_models)} models")

    # Load relp results from existing eval batches
    # Find which BATCH_DIRS configs overlap with our target models
    relp_batch_dirs = []
    for config_name, dirs in BATCH_DIRS.items():
        if any(x in config_name for x in ["freeform_std", "freeform_badrat",
               "mp_freeform_std", "mp_freeform_badrat",
               "od_freeform_std", "od_freeform_badrat"]):
            relp_batch_dirs.extend(dirs)
    print("Loading relp models...")
    relp_models = load_models(relp_batch_dirs)
    print(f"  {len(relp_models)} models")

    # Merge `relp` and `gradient` results into ar_models where missing
    # (the autoresearch batches only contain ar_* agents; the non-autoresearch
    # agents live in the main freeform_* batches).
    relp_by_dir = {m["model_dir"]: m for m in relp_models}
    for m in ar_models:
        donor = relp_by_dir.get(m["model_dir"])
        if not donor:
            continue
        for a in ("relp", "gradient"):
            if a in donor["agent_results"] and a not in m["agent_results"]:
                m["agent_results"][a] = donor["agent_results"][a]

    # Sensitivity cache
    print("Loading sensitivity cache...")
    sensitivity_cache = load_sensitivity_cache()
    sensitivity_cache = get_sensitivities(ar_models, sensitivity_cache)
    print(f"  {len(sensitivity_cache)} expressions")

    all_agents = AGENTS_ORDER + ["relp", "gradient"]

    # Plot
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 2.5))

    for group_name, configs in AGG_GROUPS.items():
        filtered = _filter_models(ar_models, configs)

        # Field F1 via megatable's compute_pattern_metrics
        metrics = compute_pattern_metrics(filtered, sensitivity_cache, all_agents)

        acc_means, acc_cis = [], []
        f1_means, f1_cis = [], []
        for agent in AGENTS_ORDER:
            am, ac = get_held_out_accuracy(filtered, agent)
            fm, fc = get_mean_f1(metrics, agent)
            acc_means.append(am); acc_cis.append(ac)
            f1_means.append(fm); f1_cis.append(fc)

        relp_acc, relp_acc_ci = get_held_out_accuracy(filtered, "relp")
        relp_f1, relp_f1_ci = get_mean_f1(metrics, "relp")

        c = COLORS[group_name]
        mk = MARKERS[group_name]

        # Lines with CI bands
        eh = np.array(elapsed_hours)
        for ax, means, cis in [(ax1, acc_means, acc_cis), (ax2, f1_means, f1_cis)]:
            m_arr = np.array(means, dtype=float)
            c_arr = np.array(cis, dtype=float)
            ax.fill_between(eh, m_arr - c_arr, m_arr + c_arr,
                            color=c, alpha=0.1, zorder=2)
            ax.plot(eh, m_arr, color=c, marker=mk,
                    linewidth=2, markersize=5, label=group_name, zorder=3)

        for ax, relp_val in [(ax1, relp_acc), (ax2, relp_f1)]:
            ax.axhline(y=relp_val, color=c, linestyle="--", alpha=0.6, linewidth=1.5, zorder=1)

        ax1.annotate(f"{acc_means[-1]:.1f}", (elapsed_hours[-1], acc_means[-1]),
                     textcoords="offset points", xytext=(6, 2), fontsize=8, color=c, fontweight="bold")
        ax2.annotate(f"{f1_means[-1]:.1f}", (elapsed_hours[-1], f1_means[-1]),
                     textcoords="offset points", xytext=(6, 2), fontsize=8, color=c, fontweight="bold")

    # Human intervention markers
    for label, time_str in HUMAN_INTERVENTIONS.items():
        t = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
        h = (t - t0).total_seconds() / 3600
        for ax in [ax1, ax2]:
            ax.axvline(x=h, color="gray", linestyle="-", alpha=0.25, linewidth=0.8, zorder=0)
    # Label interventions on top of ax1 only
    for label, time_str in HUMAN_INTERVENTIONS.items():
        t = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
        h = (t - t0).total_seconds() / 3600
        short = label.split(" ", 1)[1]
        ax1.annotate(short, (h, ax1.get_ylim()[1]), rotation=90,
                     fontsize=5, color="gray", alpha=0.7, ha="right", va="top",
                     xytext=(-2, -2), textcoords="offset points")

    for ax, title, ylabel in [(ax1, "Held-out Accuracy", "Accuracy (%)"), (ax2, "Field Recovery F1", "F1 (%)")]:
        ax.set_xlabel("Hours elapsed", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.15, linewidth=0.5, axis="y")
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.tick_params(labelsize=9)

    from matplotlib.lines import Line2D
    handles, labels = ax1.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="gray", linestyle="--", linewidth=1.5,
                          alpha=0.6, label="relp baseline"))
    labels.append("relp baseline")
    ax2.legend(handles=handles, labels=labels, fontsize=8, loc="lower right",
               framealpha=0.3, edgecolor="none")

    plt.tight_layout()
    plt.savefig(args.output, dpi=500, bbox_inches="tight")
    print(f"Saved to {args.output}")

    # ------------------------------------------------------------------
    # Final-agent summary table (matches the sub-table under Figure 4).
    #
    # Rows: {car_purchase (ID), other scenarios (OOD), unfaithful (OOD)}
    # Cols: gradient / relp / ar_final, each with Acc and F1 columns (90% CI)
    # ------------------------------------------------------------------
    print()
    print("=" * 98)
    print(" Figure 4 bottom sub-table — final-agent results")
    print("=" * 98)
    header = (f"{'Setup':<24}"
              f" {'gradient Acc':>14} {'gradient F1':>14}"
              f" {'relp Acc':>14} {'relp F1':>14}"
              f" {'Final Acc':>14} {'Final F1':>14}")
    print(header)
    print("-" * len(header))

    def _fmt(mean_ci):
        mean, ci = mean_ci
        if mean is None:
            return "—"
        if ci is None:
            return f"{mean:.1f}"
        return f"{mean:.1f}\u00b1{ci:.1f}"

    # Also build the markdown that matches livepaper/paper.md so we can diff
    md_lines = ["| Setup | gradient Acc | gradient F1 | relp Acc | relp F1 | Final Acc | Final F1 |",
                "|---|---:|---:|---:|---:|---:|---:|"]
    for group_name, configs in AGG_GROUPS.items():
        filtered = _filter_models(ar_models, configs)
        metrics = compute_pattern_metrics(filtered, sensitivity_cache, all_agents)

        cells = []
        for agent in ("gradient", "relp", "ar_final"):
            acc = get_held_out_accuracy(filtered, agent)
            f1 = get_mean_f1(metrics, agent)
            cells.append(_fmt(acc))
            cells.append(_fmt(f1))

        label_map = {"car_purchase (ID)": "Car purchase (ID)",
                     "other scenarios (OOD)": "Other scenarios (OOD)",
                     "unfaithful (OOD)": "Unfaithful (also OOD)"}
        row_label = label_map.get(group_name, group_name)
        print(f"{row_label:<24}" + " ".join(f"{c:>14}" for c in cells))
        md_lines.append("| " + row_label + " | " + " | ".join(cells) + " |")

    print()
    print("Markdown (for livepaper/paper.md):")
    print()
    for line in md_lines:
        print("  " + line)


if __name__ == "__main__":
    main()
