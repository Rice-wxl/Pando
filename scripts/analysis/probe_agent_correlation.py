#!/usr/bin/env python3
"""Correlate frozen world probe scores with agent evaluation accuracy.

Tests whether models that better preserve base model representations also
have higher agent accuracy (e.g., sae_tfidf, gradient, relp).

Usage:
    python scripts/analysis/probe_agent_correlation.py --agent sae_tfidf
    python scripts/analysis/probe_agent_correlation.py --agent gradient --layer 7
    python scripts/analysis/probe_agent_correlation.py --agent sae_tfidf --batches batch_20260301_033718 batch_20260301_033721
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


FIELDS_INT = ["year", "horsepower", "mpg", "seat_capacity", "price"]
FIELDS_ENUM = ["brand", "color", "drivetrain", "interior", "condition"]
ALL_FIELDS = FIELDS_ENUM + FIELDS_INT


def norm(val, base_val, chance):
    raw = (val - chance) / (base_val - chance) if base_val != chance else 0
    return max(-1, raw)


def main():
    parser = argparse.ArgumentParser(description="Correlate probe scores with agent accuracy")
    parser.add_argument("--probe-results", type=str,
                        default="outputs/frozen_world_probes/freeform_natural/results.json")
    parser.add_argument("--agent", type=str, default="sae_tfidf")
    parser.add_argument("--batches", type=str, nargs="+",
                        default=["batch_20260301_033718", "batch_20260301_033721"])
    parser.add_argument("--eval-dir", type=str, default="outputs/evaluations")
    parser.add_argument("--layer", type=int, default=13)
    args = parser.parse_args()

    with open(args.probe_results) as f:
        data = json.load(f)

    base = data["base_results"]
    probe_models = {m["model_name"]: m for m in data["model_results"]}
    layer = args.layer

    # Load agent results from batches
    agent_accs = {}
    for batch in args.batches:
        batch_dir = os.path.join(args.eval_dir, batch)
        if not os.path.isdir(batch_dir):
            print(f"WARNING: batch dir not found: {batch_dir}")
            continue
        for model_name in os.listdir(batch_dir):
            agent_path = os.path.join(batch_dir, model_name, "agent_results", f"{args.agent}.json")
            if os.path.exists(agent_path):
                with open(agent_path) as f:
                    r = json.load(f)
                agent_accs[model_name] = r["accuracy"]

    print(f"Agent: {args.agent}, layer: {layer}")
    print(f"Agent results: {len(agent_accs)} models from {len(args.batches)} batches")
    print(f"Probe results: {len(probe_models)} models")

    # Build paired data
    probe_used, probe_all, probe_enum_used, probe_int_used = [], [], [], []
    accs, depths = [], []

    for model_name, acc in agent_accs.items():
        if model_name not in probe_models:
            continue
        m = probe_models[model_name]
        used = set(m.get("used_fields", []))
        if not used:
            continue

        used_scores, enum_used, int_used, all_scores = [], [], [], []

        for f in FIELDS_ENUM:
            key = f"{f}_L{layer}"
            if key not in m["results"]:
                continue
            bv = base[key]["accuracy"]
            s = norm(m["results"][key]["accuracy"], bv, 0.5)
            all_scores.append(s)
            if f in used:
                used_scores.append(s)
                enum_used.append(s)

        for f in FIELDS_INT:
            key = f"{f}_L{layer}"
            if key not in m["results"]:
                continue
            bv = base[key]["r2"]
            s = norm(m["results"][key]["r2"], bv, 0)
            all_scores.append(s)
            if f in used:
                used_scores.append(s)
                int_used.append(s)

        if used_scores:
            probe_used.append(np.mean(used_scores))
            probe_all.append(np.mean(all_scores))
            probe_enum_used.append(np.mean(enum_used) if enum_used else np.nan)
            probe_int_used.append(np.mean(int_used) if int_used else np.nan)
            accs.append(acc)
            depths.append(m["depth"])

    probe_used = np.array(probe_used)
    probe_all = np.array(probe_all)
    probe_enum_used = np.array(probe_enum_used)
    accs = np.array(accs)
    depths = np.array(depths)

    print(f"\nMatched: {len(accs)} models\n")

    # Overall correlations
    for label, ps in [("USED-FIELD", probe_used), ("ALL-FIELD", probe_all)]:
        print(f"=== {label} PROBE SCORE vs {args.agent.upper()} ACCURACY ===")
        r, p = stats.pearsonr(ps, accs)
        rho, rho_p = stats.spearmanr(ps, accs)
        print(f"  Pearson  r = {r:.4f}, p = {p:.2e}")
        print(f"  Spearman rho = {rho:.4f}, p = {rho_p:.2e}")
        print()

    # Per-depth
    print("Per-depth (used-field):")
    print(f"{'Depth':>5} {'N':>4} {'probe_avg':>10} {'agent_avg':>10} {'rho':>8} {'p':>10}")
    for d in sorted(set(depths)):
        mask = depths == d
        ps = probe_used[mask]
        sa = accs[mask]
        if len(ps) > 3:
            rho_d, p_d = stats.spearmanr(ps, sa)
        else:
            rho_d, p_d = float("nan"), float("nan")
        print(f"{d:>5} {mask.sum():>4} {ps.mean():>10.3f} {sa.mean():>10.3f} {rho_d:>8.3f} {p_d:>10.2e}")

    # ENUM-only used fields
    valid = ~np.isnan(probe_enum_used)
    if valid.sum() > 3:
        print(f"\nENUM-ONLY USED-FIELD (N={valid.sum()}):")
        rho, p = stats.spearmanr(probe_enum_used[valid], accs[valid])
        print(f"  Spearman rho = {rho:.4f}, p = {p:.2e}")


if __name__ == "__main__":
    main()
