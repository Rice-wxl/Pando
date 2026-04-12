#!/usr/bin/env python3
"""Field-value bias analysis: do interp scores reflect field VALUES (BMW vs Toyota)
rather than circuit membership?

Per-sample analysis. For each (model, sample, field), extracts the interp score
AND the field value from the input. Compares R² of (field + value) vs (field + circuit).

Focuses on ENUM (categorical) fields where value is binary.

Usage:
    python scripts/analysis/analyze_value_bias.py <batch_dir> [<batch_dir2> ...]
    python scripts/analysis/analyze_value_bias.py <batch_dir> --agents gradient relp
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.stats import mannwhitneyu

from src.scenarios import get_scenario

# --- Reuse from analyze_interp_field_bias.py ---

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
}

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
}

SUPPORTED_AGENTS = {
    "gradient", "relp", "logit_lens_field",
    "sae_tfidf", "sae_autointerp", "sae_gradient",
}

# Regexes
INPUT_RE = re.compile(r'Input: \{([^}]+)\}')
FIELD_IMPORTANCE_RE = re.compile(r'Field importance:\s*(.+)')
FIELD_SCORE_RE = re.compile(r'(\w+)\(([\d.]+)\)')
FIELD_TOKEN_RE = re.compile(r"Field tokens:\s*(.+)")
SAE_DESC_RE = re.compile(r'\[\d+\]\s*\(([^)]+)\):\s*(.+)')
LAYER_RE = re.compile(r'Layer (\d+):')


def _parse_sae_score(score_str):
    m = re.match(r'attr=([+-]?\d+\.?\d*)', score_str)
    if m:
        return abs(float(m.group(1)))
    m = re.match(r'tfidf=(\d+\.?\d*)', score_str)
    if m:
        return float(m.group(1))
    try:
        return abs(float(score_str.strip()))
    except ValueError:
        return 1.0


def _field_mentioned(desc, field_name, scenario_name):
    text = desc.lower()
    if re.search(r"\b" + re.escape(field_name) + r"\b", text):
        return True
    space_name = field_name.replace("_", " ")
    if re.search(r"\b" + re.escape(space_name) + r"\b", text):
        return True
    for pat in FIELD_ALIASES.get(scenario_name, {}).get(field_name, []):
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


def parse_input_values(input_str):
    """Parse 'brand=Toyota, year=2019, ...' into dict."""
    vals = {}
    for pair in input_str.split(', '):
        if '=' in pair:
            k, v = pair.split('=', 1)
            vals[k.strip()] = v.strip()
    return vals


def extract_per_sample_gradient_relp(pattern_prompt, all_fields):
    """Extract per-sample (field_values, field_scores) from gradient/relp prompt.

    Returns list of (values_dict, scores_dict) tuples.
    """
    lines = pattern_prompt.split('\n')
    samples = []
    current_values = None

    for line in lines:
        im = INPUT_RE.search(line)
        if im:
            current_values = parse_input_values(im.group(1))
            continue
        fm = FIELD_IMPORTANCE_RE.search(line)
        if fm and current_values is not None:
            scores = {}
            for sm in FIELD_SCORE_RE.finditer(fm.group(1)):
                fname, score = sm.group(1), float(sm.group(2))
                if fname in all_fields:
                    scores[fname] = score
            if scores:
                samples.append((current_values, scores))
            current_values = None

    return samples


def extract_per_sample_logit_lens(pattern_prompt, all_fields):
    """Extract per-sample (field_values, field_logits_at_last_layer)."""
    lines = pattern_prompt.split('\n')
    samples = []
    current_values = None
    current_layers = {}  # layer -> {field: logit}
    current_layer = None

    for line in lines:
        im = INPUT_RE.search(line)
        if im:
            # Save previous sample
            if current_values is not None and current_layers:
                last_layer = max(current_layers.keys())
                samples.append((current_values, current_layers[last_layer]))
            current_values = parse_input_values(im.group(1))
            current_layers = {}
            current_layer = None
            continue
        lm = LAYER_RE.search(line)
        if lm:
            current_layer = int(lm.group(1))
            continue
        ftm = FIELD_TOKEN_RE.search(line)
        if ftm and current_layer is not None:
            field_logits = {}
            for entry in ftm.group(1).split('|'):
                entry = entry.strip()
                eq_idx = entry.find('=')
                if eq_idx < 0:
                    continue
                fname = entry[:eq_idx].strip()
                if fname not in all_fields:
                    continue
                logits = [float(v) for v in re.findall(r'\((-?[\d.]+)\)', entry)]
                if logits:
                    field_logits[fname] = max(logits)
            if field_logits:
                current_layers[current_layer] = field_logits

    # Last sample
    if current_values is not None and current_layers:
        last_layer = max(current_layers.keys())
        samples.append((current_values, current_layers[last_layer]))

    return samples


def extract_per_sample_sae(pattern_prompt, all_fields, scenario_name):
    """Extract per-sample (field_values, {field: count}, {field: weighted}).

    Returns list of (values_dict, count_dict, weighted_dict) tuples.
    """
    lines = pattern_prompt.split('\n')
    samples = []
    current_values = None
    current_count = None
    current_weighted = None

    for line in lines:
        im = INPUT_RE.search(line)
        if im:
            # Save previous sample
            if current_values is not None and current_count is not None:
                samples.append((current_values, current_count, current_weighted))
            current_values = parse_input_values(im.group(1))
            current_count = {f: 0.0 for f in all_fields}
            current_weighted = {f: 0.0 for f in all_fields}
            continue
        dm = SAE_DESC_RE.search(line)
        if dm and current_count is not None:
            score = _parse_sae_score(dm.group(1))
            desc = dm.group(2)
            for field in all_fields:
                if _field_mentioned(desc, field, scenario_name):
                    current_count[field] += 1
                    current_weighted[field] += score

    # Last sample
    if current_values is not None and current_count is not None:
        samples.append((current_values, current_count, current_weighted))

    return samples


def load_models(batch_dirs, agents_filter=None):
    target_agents = agents_filter or SUPPORTED_AGENTS
    all_models = []
    for batch_dir in batch_dirs:
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
            agent_results = {}
            agent_results_dir = model_dir / "agent_results"
            if agent_results_dir.exists():
                for agent_file in agent_results_dir.glob("*.json"):
                    name = _OLD_AGENT_NAMES.get(agent_file.stem, agent_file.stem)
                    if name in target_agents:
                        with open(agent_file) as f:
                            agent_results[name] = json.load(f)
            if agent_results:
                all_models.append({
                    "model_name": config.get("model_name", model_dir.name),
                    "model_dir": str(source_model_dir),
                    "num_fields": num_fields,
                    "scenario": config.get("scenario", "car_purchase"),
                    "used_fields": set(circuit_data.get("used_fields", [])),
                    "agent_results": agent_results,
                })
    unique = {}
    for m in all_models:
        key = m["model_dir"]
        if key not in unique:
            unique[key] = m
        else:
            unique[key]["agent_results"].update(m["agent_results"])
    return list(unique.values())


def r2_from_groups(y, group_ids):
    """R² from group means (one-way ANOVA)."""
    total_ss = np.sum((y - np.mean(y))**2)
    if total_ss == 0:
        return 0.0
    unique_groups = np.unique(group_ids)
    y_pred = np.zeros_like(y)
    for g in unique_groups:
        mask = group_ids == g
        y_pred[mask] = np.mean(y[mask])
    return np.sum((y_pred - np.mean(y))**2) / total_ss


def r2_interaction(y, a_ids, b_ids):
    """R² from interaction of two factors (cell means)."""
    total_ss = np.sum((y - np.mean(y))**2)
    if total_ss == 0:
        return 0.0
    y_pred = np.zeros_like(y)
    for a in np.unique(a_ids):
        for b in np.unique(b_ids):
            mask = (a_ids == a) & (b_ids == b)
            if mask.any():
                y_pred[mask] = np.mean(y[mask])
    return np.sum((y_pred - np.mean(y))**2) / total_ss


def analyze(models, agents_filter=None):
    scenarios = {m["scenario"] for m in models}
    scenario_name = next(iter(scenarios))
    scenario = get_scenario(scenario_name)
    all_fields = scenario.field_names()

    # Identify ENUM fields (binary categorical)
    enum_fields = [f.name for f in scenario.fields if f.values is not None]
    int_fields = [f.name for f in scenario.fields if f.values is None]
    print(f"Scenario: {scenario_name}")
    print(f"ENUM fields: {enum_fields}")
    print(f"INTEGER fields: {int_fields}")
    print(f"Models: {len(models)}")
    print()

    # Find agents
    all_agents = set()
    for m in models:
        for a in m["agent_results"]:
            if a in SUPPORTED_AGENTS:
                all_agents.add(a)
    if agents_filter:
        all_agents = sorted(a for a in agents_filter if a in all_agents)
    else:
        all_agents = sorted(all_agents)

    summary_rows = []

    for agent in all_agents:
        # Collect per-sample rows: (model_idx, field, field_value, in_circuit, score)
        # metric_name -> list of row dicts
        metric_samples = defaultdict(list)

        for mi, m in enumerate(models):
            result = m["agent_results"].get(agent)
            if not result:
                continue
            pp = result.get("agent_metadata", {}).get("pattern_prompt", "")
            if not pp:
                continue

            if agent in ("gradient", "relp"):
                samples = extract_per_sample_gradient_relp(pp, all_fields)
                for values, scores in samples:
                    for field in all_fields:
                        if field in scores and field in values:
                            metric_samples["importance"].append({
                                "model_idx": mi,
                                "field": field,
                                "value": values[field],
                                "in_circuit": 1 if field in m["used_fields"] else 0,
                                "score": scores[field],
                            })

            elif agent == "logit_lens_field":
                samples = extract_per_sample_logit_lens(pp, all_fields)
                for values, logits in samples:
                    for field in all_fields:
                        if field in logits and field in values:
                            metric_samples["logit"].append({
                                "model_idx": mi,
                                "field": field,
                                "value": values[field],
                                "in_circuit": 1 if field in m["used_fields"] else 0,
                                "score": logits[field],
                            })

            elif agent.startswith("sae_"):
                samples = extract_per_sample_sae(pp, all_fields, scenario_name)
                for values, counts, weighted in samples:
                    for field in all_fields:
                        if field in values:
                            base = {
                                "model_idx": mi,
                                "field": field,
                                "value": values[field],
                                "in_circuit": 1 if field in m["used_fields"] else 0,
                            }
                            metric_samples["count"].append({**base, "score": counts.get(field, 0)})
                            metric_samples["weighted"].append({**base, "score": weighted.get(field, 0)})

        if not metric_samples:
            continue

        print(f"{'='*80}")
        print(f"Agent: {agent}  (metrics: {', '.join(sorted(metric_samples))})")
        print(f"{'='*80}")

        for metric_name in sorted(metric_samples):
            rows = metric_samples[metric_name]
            label = f"{agent}/{metric_name}" if len(metric_samples) > 1 else agent

            # --- ENUM fields only ---
            enum_rows = [r for r in rows if r["field"] in enum_fields]
            if not enum_rows:
                continue

            print(f"\n--- {metric_name}: ENUM fields ({len(enum_rows)} obs, "
                  f"{len(set(r['model_idx'] for r in enum_rows))} models) ---")

            y = np.array([r["score"] for r in enum_rows], dtype=float)

            # Encode factors
            field_names_arr = np.array([r["field"] for r in enum_rows])
            unique_fields = sorted(set(field_names_arr))
            field_map = {f: i for i, f in enumerate(unique_fields)}
            x_field = np.array([field_map[r["field"]] for r in enum_rows])

            x_circ = np.array([r["in_circuit"] for r in enum_rows])

            # Encode field value as integer per field
            # For each field, map its two values to 0/1
            value_maps = {}
            for f in unique_fields:
                vals = sorted(set(r["value"] for r in enum_rows if r["field"] == f))
                value_maps[f] = {v: i for i, v in enumerate(vals)}
            x_value = np.array([value_maps[r["field"]][r["value"]] for r in enum_rows])

            # Also encode field×value as a single interaction factor
            # (field_idx * 2 + value_idx) gives unique cell per field×value
            x_fv = x_field * 2 + x_value

            model_ids = np.array([r["model_idx"] for r in enum_rows])

            # R² decomposition
            r2_field = r2_from_groups(y, x_field)
            r2_circ = r2_from_groups(y, x_circ)
            r2_value_global = r2_from_groups(y, x_value)  # just 0/1 across all fields (meaningless alone)
            r2_fxc = r2_interaction(y, x_field, x_circ)
            r2_fxv = r2_interaction(y, x_field, x_value)
            r2_model = r2_from_groups(y, model_ids)

            # Also: field × value × circuit (3-way)
            x_fvc = x_fv * 2 + x_circ  # unique cell per (field, value, circuit)
            r2_fvc = r2_from_groups(y, x_fvc)

            print(f"  R²(field):                {r2_field:.4f}  (constant per-field bias)")
            print(f"  R²(field × circuit):      {r2_fxc:.4f}  (+ circuit membership)")
            print(f"  R²(field × value):        {r2_fxv:.4f}  (+ field value: BMW vs Toyota etc.)")
            print(f"  R²(field × value × circ): {r2_fvc:.4f}  (+ both)")
            print(f"  R²(model):                {r2_model:.4f}  (per-model baseline)")
            print()
            print(f"  Circuit adds over field:  {r2_fxc - r2_field:+.4f}")
            print(f"  Value adds over field:    {r2_fxv - r2_field:+.4f}")
            print(f"  Value adds over f×circ:   {r2_fvc - r2_fxc:+.4f}")
            print(f"  Circuit adds over f×val:  {r2_fvc - r2_fxv:+.4f}")

            # Per-field breakdown: mean score by value, split by circuit
            print(f"\n  Per-field value breakdown:")
            print(f"  {'Field':<14} {'Val0':>6} {'Val1':>6} | {'Val0_in':>8} {'Val1_in':>8} {'Val0_out':>8} {'Val1_out':>8} | {'ValDiff':>8} {'CircDiff':>9}")
            print(f"  {'-'*95}")

            for field in enum_fields:
                fr = [r for r in enum_rows if r["field"] == field]
                vals = sorted(value_maps[field].keys())
                v0, v1 = vals[0], vals[1]

                s_v0 = [r["score"] for r in fr if r["value"] == v0]
                s_v1 = [r["score"] for r in fr if r["value"] == v1]
                s_v0_in = [r["score"] for r in fr if r["value"] == v0 and r["in_circuit"] == 1]
                s_v1_in = [r["score"] for r in fr if r["value"] == v1 and r["in_circuit"] == 1]
                s_v0_out = [r["score"] for r in fr if r["value"] == v0 and r["in_circuit"] == 0]
                s_v1_out = [r["score"] for r in fr if r["value"] == v1 and r["in_circuit"] == 0]

                m_v0 = np.mean(s_v0) if s_v0 else 0
                m_v1 = np.mean(s_v1) if s_v1 else 0
                m_v0_in = np.mean(s_v0_in) if s_v0_in else 0
                m_v1_in = np.mean(s_v1_in) if s_v1_in else 0
                m_v0_out = np.mean(s_v0_out) if s_v0_out else 0
                m_v1_out = np.mean(s_v1_out) if s_v1_out else 0

                val_diff = m_v1 - m_v0
                circ_diff = np.mean([r["score"] for r in fr if r["in_circuit"] == 1]) - \
                            np.mean([r["score"] for r in fr if r["in_circuit"] == 0])

                print(f"  {field:<14} {m_v0:>6.1f} {m_v1:>6.1f} | "
                      f"{m_v0_in:>8.1f} {m_v1_in:>8.1f} {m_v0_out:>8.1f} {m_v1_out:>8.1f} | "
                      f"{val_diff:>+8.1f} {circ_diff:>+9.1f}")
                print(f"  {'':>14} ({v0:>5}) ({v1:>5})")

            summary_rows.append({
                "label": label,
                "r2_field": r2_field,
                "r2_fxc": r2_fxc,
                "r2_fxv": r2_fxv,
                "r2_fvc": r2_fvc,
                "circ_add": r2_fxc - r2_field,
                "val_add": r2_fxv - r2_field,
            })

        print()

    # Summary
    if summary_rows:
        print(f"{'='*85}")
        print(f"SUMMARY (ENUM fields only, per-sample)")
        print(f"{'='*85}")
        print(f"{'Label':<28} {'R²(field)':>10} {'R²(f×circ)':>11} {'R²(f×val)':>10} "
              f"{'circ_add':>10} {'val_add':>10}")
        print("-" * 85)
        for sr in sorted(summary_rows, key=lambda x: -x["circ_add"]):
            print(f"{sr['label']:<28} {sr['r2_field']:>10.4f} {sr['r2_fxc']:>11.4f} "
                  f"{sr['r2_fxv']:>10.4f} {sr['circ_add']:>+10.4f} {sr['val_add']:>+10.4f}")
        print()
        print("circ_add = R²(field×circuit) - R²(field)")
        print("val_add  = R²(field×value)  - R²(field)")


def main():
    parser = argparse.ArgumentParser(description="Field-value bias analysis")
    parser.add_argument("batch_dirs", type=str, nargs="+")
    parser.add_argument("--agents", type=str, nargs="+", default=None)
    args = parser.parse_args()

    batch_dirs = [Path(bd) for bd in args.batch_dirs]
    for bd in batch_dirs:
        if not bd.exists():
            print(f"Error: {bd} not found", file=sys.stderr)
            return

    agents_set = set(args.agents) if args.agents else None
    print("Loading models...")
    models = load_models(batch_dirs, agents_set)
    print(f"  {len(models)} unique models loaded")
    if not models:
        return
    analyze(models, agents_filter=args.agents)


if __name__ == "__main__":
    main()
