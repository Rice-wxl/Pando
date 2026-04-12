#!/usr/bin/env python3
"""Mechanistic analysis: gradient field importance & logit lens field token logits.

Extracts per-field scores from agent pattern_prompt text across all 3 scenarios,
joins with circuit metadata, and runs regressions.

Usage:
    python paper_draft/analyze_mechanistic.py
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scenarios import get_scenario

# ─── Batch directories ───────────────────────────────────────────────────────

BASE = Path("outputs/evaluations")

# Only freeform_std (no verbalization) for mechanistic analysis
BATCHES = {
    "car_purchase": [
        "batch_20260301_033718", "batch_20260301_033721",
    ],
    "movie_pick": [
        "batch_20260321_203958", "batch_20260321_204227",
    ],
    "oversight_defection": [
        "batch_20260325_120645", "batch_20260325_143729",
    ],
}

OLD_NAMES = {
    "sample_then_llm_guess": "blackbox",
    "sample_then_gradient_llm_v2": "gradient",
    "sample_then_relp_llm": "relp",
    "blackbox_prefill": "prefill",
    "sample_then_logit_lens_llm": "logit_lens",
    "sample_then_logit_lens_field_llm": "logit_lens_field",
    "sample_then_sae_gradient_attribution_llm": "sae_gradient",
    "sample_then_residual_token_llm": "res_token",
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
            source = Path(config.get("model_dir", ""))
            circuit_path = source / "circuit.json"
            if not circuit_path.exists():
                continue
            with open(circuit_path) as f:
                circuit = json.load(f)
            agent_results = {}
            ar_dir = model_dir / "agent_results"
            if ar_dir.exists():
                for af in ar_dir.glob("*.json"):
                    name = OLD_NAMES.get(af.stem, af.stem)
                    with open(af) as f:
                        agent_results[name] = json.load(f)
            all_models.append({
                "model_name": config.get("model_name", model_dir.name),
                "model_dir": str(source),
                "num_fields": config.get("num_fields", 0),
                "scenario": config.get("scenario", "car_purchase"),
                "used_fields": circuit.get("used_fields", []),
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


def extract_field_importance_from_prompt(pattern_prompt, scenario_name):
    """Extract per-sample per-field importance scores from gradient pattern_prompt."""
    scenario = get_scenario(scenario_name)
    fields = scenario.field_names()

    # Find "Field importance:" lines (per-sample, indented)
    # Format: "  Field importance: deployment_phase(94.12), conversation_turn_count(47.84), ..."
    samples = []
    for match in re.finditer(r"^\s+Field importance:\s*(.+)", pattern_prompt, re.MULTILINE):
        line = match.group(1)
        field_scores = {}
        for field in fields:
            # Match field_name(number) format
            m = re.search(re.escape(field) + r"\(([\d.]+)\)", line)
            if m:
                field_scores[field] = float(m.group(1))
        if field_scores:
            samples.append(field_scores)
    return samples


def extract_logit_lens_field_logits(pattern_prompt, scenario_name):
    """Extract per-field logit values from logit_lens pattern_prompt.

    Looks for field token logits in the pattern prompt. The format varies
    but typically includes per-token logits including field name tokens.
    """
    scenario = get_scenario(scenario_name)
    fields = scenario.field_names()

    # logit_lens prompt has lines like:
    # "  Field token logits at final layer: brand=-10.75, year=-1.93, ..."
    # or embedded in per-sample blocks
    samples = []
    for match in re.finditer(r"(?:Field token logits|field logits|Final layer).*?:\s*(.+)", pattern_prompt, re.IGNORECASE):
        line = match.group(1)
        field_logits = {}
        for field in fields:
            # Match field_name=number (may be negative)
            m = re.search(re.escape(field) + r"=([-\d.]+)", line)
            if m:
                field_logits[field] = float(m.group(1))
        if field_logits:
            samples.append(field_logits)
    return samples


def run_gradient_analysis(models):
    """Analyze gradient field importance: circuit vs non-circuit fields."""
    circuit_scores = []
    non_circuit_scores = []
    per_field_data = defaultdict(lambda: {"scores": [], "is_circuit": []})

    for m in models:
        grad_result = m["agent_results"].get("gradient")
        if not grad_result:
            continue
        prompt = grad_result.get("agent_metadata", {}).get("pattern_prompt", "")
        samples = extract_field_importance_from_prompt(prompt, m["scenario"])
        if not samples:
            continue

        used = set(m["used_fields"])
        scenario = get_scenario(m["scenario"])
        all_fields = scenario.field_names()

        # Average across samples for this model
        field_means = {}
        for field in all_fields:
            vals = [s.get(field, 0) for s in samples if field in s]
            if vals:
                field_means[field] = np.mean(vals)

        for field, score in field_means.items():
            is_circuit = field in used
            if is_circuit:
                circuit_scores.append(score)
            else:
                non_circuit_scores.append(score)
            per_field_data[field]["scores"].append(score)
            per_field_data[field]["is_circuit"].append(1 if is_circuit else 0)

    return circuit_scores, non_circuit_scores, per_field_data


def main():
    print("Loading models...", file=sys.stderr)
    all_models = []
    for scenario, batches in BATCHES.items():
        models = load_models(batches)
        all_models.extend(models)
        print(f"  {scenario}: {len(models)} models", file=sys.stderr)

    # ─── Gradient analysis ────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  GRADIENT FIELD IMPORTANCE ANALYSIS")
    print(f"  ({len(all_models)} models across 3 scenarios)")
    print(f"{'='*80}")

    circuit_scores, non_circuit_scores, per_field_data = run_gradient_analysis(all_models)

    if circuit_scores and non_circuit_scores:
        c_mean = np.mean(circuit_scores)
        nc_mean = np.mean(non_circuit_scores)
        ratio = c_mean / nc_mean if nc_mean > 0 else float('inf')
        print(f"\n  Circuit field mean:     {c_mean:.1f} (n={len(circuit_scores)})")
        print(f"  Non-circuit field mean: {nc_mean:.1f} (n={len(non_circuit_scores)})")
        print(f"  Ratio:                  {ratio:.1f}x")
        print(f"  Difference:             {c_mean - nc_mean:.1f}")

    # Per-scenario breakdown
    for scenario in BATCHES:
        sc_models = [m for m in all_models if m["scenario"] == scenario]
        c, nc, _ = run_gradient_analysis(sc_models)
        if c and nc:
            print(f"\n  {scenario}: circuit={np.mean(c):.1f}, non-circuit={np.mean(nc):.1f}, ratio={np.mean(c)/np.mean(nc):.1f}x")

    # Per-field regressions (pooled across scenarios)
    print(f"\n  Per-field is_in_circuit effect (standardized):")
    print(f"  {'Field':<30} {'Mean':>8} {'Std':>8} {'coef/std':>10} {'n':>6}")
    print(f"  {'-'*70}")

    coef_stds = []
    r2s = []
    for field in sorted(per_field_data.keys()):
        data = per_field_data[field]
        scores = np.array(data["scores"])
        is_circuit = np.array(data["is_circuit"])
        if len(scores) < 10 or np.std(scores) == 0:
            continue
        # Simple regression: score ~ is_circuit
        n = len(scores)
        x_mean = np.mean(is_circuit)
        y_mean = np.mean(scores)
        y_std = np.std(scores)
        cov = np.mean((is_circuit - x_mean) * (scores - y_mean))
        var_x = np.mean((is_circuit - x_mean) ** 2)
        if var_x > 0:
            beta = cov / var_x
            coef_std = beta / y_std
            # R²
            y_pred = x_mean + beta * (is_circuit - x_mean) + y_mean
            ss_res = np.sum((scores - y_pred) ** 2)
            ss_tot = np.sum((scores - y_mean) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            coef_stds.append(coef_std)
            r2s.append(r2)
            print(f"  {field:<30} {np.mean(scores):>8.1f} {y_std:>8.1f} {coef_std:>+10.2f} {n:>6}")

    if coef_stds:
        print(f"\n  coef/std range: {min(coef_stds):+.2f} to {max(coef_stds):+.2f}")
        print(f"  All positive: {all(c > 0 for c in coef_stds)}")
        print(f"  R² range: {min(r2s):.3f} to {max(r2s):.3f}")

    # ─── Logit lens analysis ─────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  LOGIT LENS FIELD TOKEN LOGIT ANALYSIS")
    print(f"{'='*80}")

    ll_circuit = []
    ll_non_circuit = []
    ll_per_field = defaultdict(lambda: {"logits": [], "is_circuit": [], "model_means": []})
    model_field_means = defaultdict(dict)  # model_dir -> field -> mean_logit

    for m in all_models:
        ll_result = m["agent_results"].get("logit_lens")
        if not ll_result:
            continue
        prompt = ll_result.get("agent_metadata", {}).get("pattern_prompt", "")
        samples = extract_logit_lens_field_logits(prompt, m["scenario"])
        if not samples:
            continue

        used = set(m["used_fields"])
        scenario = get_scenario(m["scenario"])
        all_fields = scenario.field_names()

        field_means = {}
        for field in all_fields:
            vals = [s.get(field) for s in samples if field in s]
            if vals:
                field_means[field] = np.mean(vals)

        model_field_means[m["model_dir"]] = field_means

        for field, logit in field_means.items():
            is_circuit = field in used
            if is_circuit:
                ll_circuit.append(logit)
            else:
                ll_non_circuit.append(logit)
            ll_per_field[field]["logits"].append(logit)
            ll_per_field[field]["is_circuit"].append(1 if is_circuit else 0)

    if ll_circuit and ll_non_circuit:
        print(f"\n  Circuit field mean logit:     {np.mean(ll_circuit):.1f} (n={len(ll_circuit)})")
        print(f"  Non-circuit field mean logit: {np.mean(ll_non_circuit):.1f} (n={len(ll_non_circuit)})")
        print(f"  Difference:                   {np.mean(ll_circuit) - np.mean(ll_non_circuit):.1f}")

        # Field-identity bias: range of raw means across fields
        if ll_per_field:
            field_raw_means = {f: np.mean(d["logits"]) for f, d in ll_per_field.items() if d["logits"]}
            if field_raw_means:
                spread = max(field_raw_means.values()) - min(field_raw_means.values())
                print(f"  Field-identity spread:        {spread:.1f} logits")
                top = sorted(field_raw_means.items(), key=lambda x: -x[1])[:3]
                bot = sorted(field_raw_means.items(), key=lambda x: x[1])[:3]
                print(f"  Highest: {', '.join(f'{f}({v:.1f})' for f,v in top)}")
                print(f"  Lowest:  {', '.join(f'{f}({v:.1f})' for f,v in bot)}")
    else:
        print("  No logit lens field data found in pattern_prompt.")
        print("  (Logit lens may store data in a different format)")

    # ─── Summary for paper ────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  SUMMARY FOR PAPER (\\preliminary replacements)")
    print(f"{'='*80}")
    if circuit_scores and non_circuit_scores:
        c_mean = np.mean(circuit_scores)
        nc_mean = np.mean(non_circuit_scores)
        ratio = c_mean / nc_mean if nc_mean > 0 else 0
        print(f"  Gradient: circuit={c_mean:.1f} vs non-circuit={nc_mean:.1f} ({ratio:.0f}x separation)")
    if coef_stds:
        print(f"  Gradient coef/std range: +{min(coef_stds):.2f} to +{max(coef_stds):.2f}")
        print(f"  Gradient R² range: {min(r2s):.2f}--{max(r2s):.2f}")
        print(f"  Sign consistency: {'all positive' if all(c > 0 for c in coef_stds) else 'INCONSISTENT'}")
    if ll_circuit and ll_non_circuit:
        print(f"  Logit lens: circuit={np.mean(ll_circuit):.1f} vs non-circuit={np.mean(ll_non_circuit):.1f}")


if __name__ == "__main__":
    main()
