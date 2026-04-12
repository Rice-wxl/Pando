#!/usr/bin/env python3
"""Gather all numerical data needed for the paper from evaluation batch dirs.

Outputs a structured report with:
1. Main table (Table 3): Aggregated accuracy across 3 scenarios × 3 verbalization setups
2. In-text numbers: deltas, pp differences, depth breakdowns
3. Cross-scenario comparison (Appendix): per-scenario accuracy side-by-side
4. Model counts
5. Field F1 precision/recall at d4

Usage:
    python paper_draft/gather_paper_data.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t as t_dist

# ─── Batch directory configuration ───────────────────────────────────────────

BASE = Path(__file__).resolve().parents[1] / "outputs" / "evaluations"

# scenario -> config -> list of batch dirs
BATCHES = {
    "car_purchase": {
        "freeform_std": [
            "batch_20260301_033718", "batch_20260301_033721",
            "batch_20260307_143210", "batch_20260307_143440",
            "batch_20260307_232200", "batch_20260307_232430",
        ],
        "freeform_goodrat": [
            "batch_20260301_033720", "batch_20260301_033716",
        ],
        "freeform_badrat": [
            "batch_20260306_212448", "batch_20260306_212451",
        ],
    },
    "movie_pick": {
        "freeform_std": [
            "batch_20260321_203958", "batch_20260321_204227",
        ],
        "freeform_goodrat": [
            "batch_20260322_005058", "batch_20260322_002644",
        ],
        "freeform_badrat": [
            "batch_20260321_222546", "batch_20260322_002129",
        ],
    },
    "oversight_defection": {
        "freeform_std": [
            "batch_20260325_120645", "batch_20260325_143729",
        ],
        "freeform_goodrat": [
            "batch_20260325_170147", "batch_20260325_191509",
        ],
        "freeform_badrat": [
            "batch_20260325_193335", "batch_20260325_214817",
        ],
    },
}

# Paper agents (Table 3 order)
PAPER_AGENTS = [
    "relp", "gradient", "prefill", "sae_gradient",
    "logit_lens", "res_token", "circuit_tracer_filtered",
    "blackbox", "nn",
]

# Old agent name mapping
OLD_NAMES = {
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
    "sample_then_logreg": "logreg",
    "sample_then_majority": "majority",
    "sample_then_nn": "nn",
    "spread_then_nn": "nn_spread",
}

ALL_AGENTS = [
    "relp", "gradient", "prefill", "codex_read",
    "sae_gradient", "res_token", "logit_lens_field", "blackbox",
    "sae_mean_diff", "logit_lens", "sae_autointerp",
    "sae_tfidf_filtered", "sae_tfidf", "circuit_tracer_filtered",
    "nn_spread", "nn", "logreg", "majority",
]


def ci90(values):
    n = len(values)
    if n < 2:
        return 0.0
    t_val = t_dist.ppf(0.95, df=n - 1)
    return t_val * np.std(values, ddof=1) / np.sqrt(n)


def load_models(batch_dirs):
    """Load and deduplicate models from batch directories."""
    all_models = []
    for bd_name in batch_dirs:
        bd = BASE / bd_name
        if not bd.exists():
            print(f"  WARNING: {bd} not found", file=sys.stderr)
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
            agent_results = {}
            agent_results_dir = model_dir / "agent_results"
            if agent_results_dir.exists():
                for agent_file in agent_results_dir.glob("*.json"):
                    name = OLD_NAMES.get(agent_file.stem, agent_file.stem)
                    with open(agent_file) as f:
                        agent_results[name] = json.load(f)
            all_models.append({
                "model_name": config.get("model_name", model_dir.name),
                "model_dir": str(config.get("model_dir", "")),
                "num_fields": num_fields,
                "scenario": config.get("scenario", "car_purchase"),
                "agent_results": agent_results,
            })
    # Deduplicate by model_dir
    unique = {}
    for m in all_models:
        key = m["model_dir"]
        if key not in unique:
            unique[key] = m
        else:
            unique[key]["agent_results"].update(m["agent_results"])
    return list(unique.values())


def fmt(val, ci_val=None):
    if ci_val is not None:
        return f"{val*100:.1f}±{ci_val*100:.1f}"
    return f"{val*100:.1f}"


def compute_accuracy_table(models, agents, label=""):
    """Compute accuracy per agent × depth, return as dict."""
    acc_data = defaultdict(lambda: defaultdict(list))
    for m in models:
        d = m["num_fields"]
        for agent_name, result in m["agent_results"].items():
            acc_data[agent_name][d].append(result.get("accuracy", 0.0))

    depths = sorted({m["num_fields"] for m in models})
    results = {}
    for agent in agents:
        if agent not in acc_data:
            continue
        row = {}
        all_vals = []
        for d in depths:
            vals = acc_data[agent].get(d, [])
            if vals:
                row[f"d{d}"] = (np.mean(vals), ci90(vals), len(vals))
                all_vals.extend(vals)
        if all_vals:
            row["avg"] = (np.mean(all_vals), ci90(all_vals), len(all_vals))
        results[agent] = row
    return results


def print_table(results, agents, depths=[1, 2, 3, 4], title=""):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    header = f"{'Agent':<25}"
    for d in depths:
        header += f"  {'d'+str(d):>12}"
    header += f"  {'avg':>12}   n"
    print(header)
    print("-" * len(header))
    for agent in agents:
        if agent not in results:
            continue
        row = results[agent]
        line = f"{agent:<25}"
        for d in depths:
            key = f"d{d}"
            if key in row:
                mean, ci_val, n = row[key]
                line += f"  {fmt(mean, ci_val):>12}"
            else:
                line += f"  {'—':>12}"
        if "avg" in row:
            mean, ci_val, n = row["avg"]
            line += f"  {fmt(mean, ci_val):>12}  {n:>3}"
        print(line)


def main():
    print("Loading all evaluation data...", file=sys.stderr)

    # Load all models by scenario × config
    all_config_models = {}  # (scenario, config) -> models
    for scenario, configs in BATCHES.items():
        for config_name, batch_dirs in configs.items():
            models = load_models(batch_dirs)
            all_config_models[(scenario, config_name)] = models
            print(f"  {scenario}/{config_name}: {len(models)} models", file=sys.stderr)

    # ─── 1. Model counts ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  MODEL COUNTS")
    print("=" * 80)
    total = 0
    for (scenario, config), models in sorted(all_config_models.items()):
        depths = defaultdict(int)
        for m in models:
            depths[m["num_fields"]] += 1
        depth_str = ", ".join(f"d{d}={n}" for d, n in sorted(depths.items()))
        print(f"  {scenario}/{config}: {len(models)} ({depth_str})")
        total += len(models)
    print(f"  TOTAL: {total}")

    # ─── 2. Main table: aggregated across all 3 scenarios ────────────────
    for config_name in ["freeform_std", "freeform_goodrat", "freeform_badrat"]:
        merged = []
        for scenario in BATCHES:
            merged.extend(all_config_models.get((scenario, config_name), []))
        label_map = {
            "freeform_std": "No verbalization",
            "freeform_goodrat": "Faithful",
            "freeform_badrat": "Unfaithful",
        }
        results = compute_accuracy_table(merged, PAPER_AGENTS)
        print_table(results, PAPER_AGENTS,
                    title=f"MAIN TABLE — {label_map[config_name]} (n={len(merged)}, 3 scenarios aggregated)")

    # ─── 3. Per-scenario breakdown ───────────────────────────────────────
    for config_name in ["freeform_std", "freeform_goodrat", "freeform_badrat"]:
        label_map = {
            "freeform_std": "No verbalization",
            "freeform_goodrat": "Faithful",
            "freeform_badrat": "Unfaithful",
        }
        print(f"\n{'='*80}")
        print(f"  SCENARIO COMPARISON — {label_map[config_name]}")
        print(f"{'='*80}")
        header = f"{'Agent':<25}"
        for scenario in ["car_purchase", "movie_pick", "oversight_defection"]:
            header += f"  {scenario[:12]:>12}"
        print(header)
        print("-" * len(header))
        for agent in PAPER_AGENTS:
            line = f"{agent:<25}"
            for scenario in ["car_purchase", "movie_pick", "oversight_defection"]:
                models = all_config_models.get((scenario, config_name), [])
                acc = []
                for m in models:
                    r = m["agent_results"].get(agent)
                    if r:
                        acc.append(r.get("accuracy", 0.0))
                if acc:
                    line += f"  {fmt(np.mean(acc)):>12}"
                else:
                    line += f"  {'—':>12}"
            print(line)

    # ─── 4. In-text numbers ──────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  IN-TEXT NUMBERS")
    print(f"{'='*80}")

    # Aggregated std for key comparisons
    std_merged = []
    goodrat_merged = []
    badrat_merged = []
    for scenario in BATCHES:
        std_merged.extend(all_config_models.get((scenario, "freeform_std"), []))
        goodrat_merged.extend(all_config_models.get((scenario, "freeform_goodrat"), []))
        badrat_merged.extend(all_config_models.get((scenario, "freeform_badrat"), []))

    std_results = compute_accuracy_table(std_merged, PAPER_AGENTS + ALL_AGENTS)
    goodrat_results = compute_accuracy_table(goodrat_merged, PAPER_AGENTS + ALL_AGENTS)
    badrat_results = compute_accuracy_table(badrat_merged, PAPER_AGENTS + ALL_AGENTS)

    # Gradient/relp vs blackbox deltas
    for config_label, results in [("std", std_results), ("goodrat", goodrat_results), ("badrat", badrat_results)]:
        bb = results.get("blackbox", {}).get("avg", (0, 0, 0))[0]
        for agent in ["relp", "gradient", "prefill"]:
            avg = results.get(agent, {}).get("avg", (0, 0, 0))[0]
            delta = avg - bb
            print(f"  {config_label}: {agent} vs blackbox = {delta*100:+.1f} pp  ({avg*100:.1f}% vs {bb*100:.1f}%)")

    # Prefill faithful vs unfaithful
    prefill_good = goodrat_results.get("prefill", {}).get("avg", (0, 0, 0))[0]
    prefill_bad = badrat_results.get("prefill", {}).get("avg", (0, 0, 0))[0]
    print(f"  prefill faithful={prefill_good*100:.1f}%, unfaithful={prefill_bad*100:.1f}%, delta={((prefill_good-prefill_bad)*100):+.1f} pp")

    # d1 numbers
    print(f"\n  d1 accuracy (std, aggregated):")
    for agent in PAPER_AGENTS:
        d1 = std_results.get(agent, {}).get("d1", None)
        if d1:
            print(f"    {agent}: {d1[0]*100:.1f}%")

    # d4 convergence
    print(f"\n  d4 accuracy (std, aggregated):")
    for agent in PAPER_AGENTS:
        d4 = std_results.get(agent, {}).get("d4", None)
        if d4:
            print(f"    {agent}: {d4[0]*100:.1f}%")

    # d2 relp vs blackbox
    relp_d2 = std_results.get("relp", {}).get("d2", (0, 0, 0))
    bb_d2 = std_results.get("blackbox", {}).get("d2", (0, 0, 0))
    print(f"\n  d2: relp={relp_d2[0]*100:.1f}% vs blackbox={bb_d2[0]*100:.1f}% ({(relp_d2[0]-bb_d2[0])*100:+.1f} pp)")

    # Prefill d4 goodrat
    prefill_d4_good = goodrat_results.get("prefill", {}).get("d4", (0, 0, 0))
    bb_d4_good = goodrat_results.get("blackbox", {}).get("d4", (0, 0, 0))
    print(f"  d4 faithful: prefill={prefill_d4_good[0]*100:.1f}% vs blackbox={bb_d4_good[0]*100:.1f}%")

    # ─── 5. Full variant table (all agents) ──────────────────────────────
    for config_name, label, merged in [
        ("freeform_std", "No verbalization", std_merged),
        ("freeform_goodrat", "Faithful", goodrat_merged),
        ("freeform_badrat", "Unfaithful", badrat_merged),
    ]:
        results = compute_accuracy_table(merged, ALL_AGENTS)
        print_table(results, ALL_AGENTS,
                    title=f"FULL VARIANT TABLE — {label} (n={len(merged)})")

    # ─── 6. LaTeX table output ───────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  LATEX TABLE 3 (copy-paste ready)")
    print(f"{'='*80}")

    for agent in PAPER_AGENTS:
        parts = [f"\\texttt{{{agent}}}"]
        # std d1-d4 + avg
        for d in [1, 2, 3, 4]:
            key = f"d{d}"
            entry = std_results.get(agent, {}).get(key)
            if entry:
                mean, ci_val, n = entry
                parts.append(f"{mean*100:.1f}{{\\scriptsize$\\pm${ci_val*100:.1f}}}")
            else:
                parts.append("---")
        # std avg
        entry = std_results.get(agent, {}).get("avg")
        if entry:
            mean, ci_val, n = entry
            parts.append(f"\\textbf{{{mean*100:.1f}{{\\scriptsize$\\pm${ci_val*100:.1f}}}}}")
        else:
            parts.append("---")
        # goodrat avg
        entry = goodrat_results.get(agent, {}).get("avg")
        if entry:
            mean, ci_val, n = entry
            parts.append(f"{mean*100:.1f}{{\\scriptsize$\\pm${ci_val*100:.1f}}}")
        else:
            parts.append("---")
        # badrat avg
        entry = badrat_results.get(agent, {}).get("avg")
        if entry:
            mean, ci_val, n = entry
            parts.append(f"{mean*100:.1f}{{\\scriptsize$\\pm${ci_val*100:.1f}}}")
        else:
            parts.append("---")

        print(" & ".join(parts) + " \\\\")


if __name__ == "__main__":
    main()
