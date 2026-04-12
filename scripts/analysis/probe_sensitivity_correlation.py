#!/usr/bin/env python3
"""Correlate frozen world probe scores with circuit sensitivity.

Tests whether fields that are more sensitive to the circuit retain more/less
of their base model representation after fine-tuning.

Usage:
    python scripts/analysis/probe_sensitivity_correlation.py
    python scripts/analysis/probe_sensitivity_correlation.py --layer 7
    python scripts/analysis/probe_sensitivity_correlation.py --probe-results outputs/frozen_world_probes/other_run/results.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


FIELDS_INT = ["year", "horsepower", "mpg", "seat_capacity", "price"]
FIELDS_ENUM = ["brand", "color", "drivetrain", "interior", "condition"]


def norm(val, base_val, chance):
    raw = (val - chance) / (base_val - chance) if base_val != chance else 0
    return max(-1, raw)


def main():
    parser = argparse.ArgumentParser(description="Correlate probe scores with sensitivity")
    parser.add_argument("--probe-results", type=str,
                        default="outputs/frozen_world_probes/freeform_natural/results.json")
    parser.add_argument("--sensitivity-cache", type=str,
                        default="outputs/sensitivity_cache.json")
    parser.add_argument("--layer", type=int, default=13)
    args = parser.parse_args()

    with open(args.probe_results) as f:
        data = json.load(f)
    with open(args.sensitivity_cache) as f:
        sens_cache = json.load(f)

    base = data["base_results"]
    models = data["model_results"]
    layer = args.layer

    enum_sens, enum_score = [], []
    int_sens, int_score = [], []
    all_sens, all_score = [], []

    for m in models:
        try:
            with open(f'{m["model_dir"]}/circuit.json') as f:
                circ = json.load(f)
            expr = circ["expression"]
            if expr not in sens_cache:
                continue
            field_sens = sens_cache[expr]
        except Exception:
            continue

        for f in FIELDS_ENUM:
            key = f"{f}_L{layer}"
            if key not in m["results"]:
                continue
            bv = base[key]["accuracy"]
            score = norm(m["results"][key]["accuracy"], bv, 0.5)
            s = field_sens.get(f, 0)
            enum_sens.append(s)
            enum_score.append(score)
            all_sens.append(s)
            all_score.append(score)

        for f in FIELDS_INT:
            key = f"{f}_L{layer}"
            if key not in m["results"]:
                continue
            bv = base[key]["r2"]
            score = norm(m["results"][key]["r2"], bv, 0)
            s = field_sens.get(f, 0)
            int_sens.append(s)
            int_score.append(score)
            all_sens.append(s)
            all_score.append(score)

    enum_sens, enum_score = np.array(enum_sens), np.array(enum_score)
    int_sens, int_score = np.array(int_sens), np.array(int_score)
    all_sens, all_score = np.array(all_sens), np.array(all_score)

    print(f"=== SENSITIVITY vs PROBE SCORE (layer {layer}, cap=-1) ===\n")

    for label, sv, sc in [("ALL", all_sens, all_score),
                           ("ENUM", enum_sens, enum_score),
                           ("INTEGER", int_sens, int_score)]:
        print(f"--- {label} (N={len(sv)}) ---")
        r, p = stats.pearsonr(sv, sc)
        rho, rho_p = stats.spearmanr(sv, sc)
        print(f"  Pearson  r = {r:.4f}, p = {p:.2e}")
        print(f"  Spearman rho = {rho:.4f}, p = {rho_p:.2e}")
        mask0 = sv == 0
        if mask0.sum() > 0 and (~mask0).sum() > 0:
            t, tp = stats.ttest_ind(sc[mask0], sc[~mask0])
            print(f"  sens=0: n={mask0.sum():>3}, avg={sc[mask0].mean():.3f}")
            print(f"  sens>0: n={(~mask0).sum():>3}, avg={sc[~mask0].mean():.3f}")
            print(f"  t={t:.3f}, p={tp:.2e}")
        print()


if __name__ == "__main__":
    main()
