#!/usr/bin/env python3
"""Unified interp field bias analysis across all interpretability agents.

For each agent type, extracts per-field numeric scores from the raw interp data
in pattern_prompt, then decomposes variance into:
  1. Field-identity bias (constant factor per field)
  2. Circuit membership signal (in-circuit vs not)
  3. Model-level baseline

Supported agents and their per-field score extraction:
  - gradient, relp: "Field importance: field(score), ..." lines → mean score per field
  - logit_lens_field: "Field tokens: field='token'(logit) | ..." lines → mean logit per field (last layer)
  - sae_tfidf, sae_autointerp, sae_gradient: SAE description mentions weighted by |score|

Usage:
    python scripts/analysis/analyze_interp_field_bias.py <batch_dir> [<batch_dir2> ...]
    python scripts/analysis/analyze_interp_field_bias.py <batch_dir> --agents gradient relp
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.scenarios import get_scenario

# --- Field aliases for SAE description matching ---

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
    "oversight_defection": {
        "deployment_phase":          [r"\bdeployment\b", r"\bphase\b"],
        "conversation_turn_count":   [r"\bturn\s*count\b", r"\bconversation\b.*\bturn\b", r"\bturn\b"],
        "minutes_since_last_request":[r"\bminutes\b", r"\blast\s*request\b", r"\btime\b"],
        "user_authorization":        [r"\bauthoriz\b", r"\bpermission\b"],
        "user_trust_score":          [r"\btrust\b"],
        "request_complexity":        [r"\bcomplexity\b"],
        "risk_estimation":           [r"\brisk\b"],
        "tool_access":               [r"\btool\b.*\baccess\b", r"\btool\b"],
        "oversight_level":           [r"\boversight\b"],
        "logging":                   [r"\blogging\b", r"\blog\b"],
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

# Agents this script can analyze
SUPPORTED_AGENTS = {
    "gradient", "relp", "logit_lens_field",
    "sae_tfidf", "sae_autointerp", "sae_gradient",
}

# --- Score extraction per agent type ---

# gradient/relp: "  Field importance: price(79.09), seat_capacity(34.75), ..."
FIELD_IMPORTANCE_RE = re.compile(r'Field importance:\s*(.+)')
FIELD_SCORE_RE = re.compile(r'(\w+)\(([\d.]+)\)')

# logit_lens_field: "    Field tokens: brand='brand'(-12.0) | year='year'(2.8) | ..."
FIELD_TOKEN_RE = re.compile(r"Field tokens:\s*(.+)")
# Each field entry: field='token'(logit) possibly with multiple tokens
FIELD_LOGIT_RE = re.compile(r"(\w+)='[^']*'\((-?[\d.]+)\)")

# SAE: [id] (score_info): description
SAE_DESC_RE = re.compile(r'\[\d+\]\s*\(([^)]+)\):\s*(.+)')
# Layer header
LAYER_RE = re.compile(r'Layer (\d+):')


def _parse_sae_score(score_str: str) -> float:
    """Parse primary score from SAE parenthesized string."""
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


def _field_mentioned_in_desc(desc: str, field_name: str, scenario_name: str) -> bool:
    """Check if a field is mentioned in a single SAE description."""
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


def extract_gradient_relp_scores(pattern_prompt: str, all_fields: list[str]) -> list[dict[str, float]]:
    """Extract per-sample field importance scores from gradient/relp pattern_prompt.

    Returns list of dicts (one per sample), each mapping field -> score.
    """
    samples = []
    for m in FIELD_IMPORTANCE_RE.finditer(pattern_prompt):
        field_scores = {}
        for fm in FIELD_SCORE_RE.finditer(m.group(1)):
            fname, score = fm.group(1), float(fm.group(2))
            if fname in all_fields:
                field_scores[fname] = score
        if field_scores:
            samples.append(field_scores)
    return samples


def extract_logit_lens_field_scores(
    pattern_prompt: str, all_fields: list[str], layer: int = -1,
) -> list[dict[str, float]]:
    """Extract per-sample field token logits from logit_lens_field pattern_prompt.

    Groups by sample (each Input: block), then by layer. If layer=-1, uses last layer.
    Returns list of dicts (one per sample), each mapping field -> logit.
    For multi-token fields, takes the max logit.
    """
    lines = pattern_prompt.split('\n')
    samples = []
    current_sample_layers = {}  # layer_num -> {field: logit}
    current_layer = None
    in_sample = False

    for line in lines:
        if line.startswith('Input:'):
            # Save previous sample
            if current_sample_layers:
                if layer == -1:
                    use_layer = max(current_sample_layers.keys())
                else:
                    use_layer = layer
                if use_layer in current_sample_layers:
                    samples.append(current_sample_layers[use_layer])
            current_sample_layers = {}
            in_sample = True
            continue

        lm = LAYER_RE.search(line)
        if lm and in_sample:
            current_layer = int(lm.group(1))
            continue

        ftm = FIELD_TOKEN_RE.search(line)
        if ftm and current_layer is not None:
            field_logits = {}
            # Parse each field entry
            text = ftm.group(1)
            # Split by |
            for entry in text.split('|'):
                entry = entry.strip()
                if not entry:
                    continue
                # Extract field name (before =)
                eq_idx = entry.find('=')
                if eq_idx < 0:
                    continue
                fname = entry[:eq_idx].strip()
                if fname not in all_fields:
                    continue
                # Extract all logit values for this field
                logits = [float(v) for v in re.findall(r'\((-?[\d.]+)\)', entry)]
                if logits:
                    field_logits[fname] = max(logits)  # max across sub-tokens
            if field_logits:
                current_sample_layers[current_layer] = field_logits

    # Don't forget last sample
    if current_sample_layers:
        if layer == -1:
            use_layer = max(current_sample_layers.keys())
        else:
            use_layer = layer
        if use_layer in current_sample_layers:
            samples.append(current_sample_layers[use_layer])

    return samples


def extract_sae_scores(
    pattern_prompt: str, all_fields: list[str], scenario_name: str,
) -> dict[str, dict[str, float]]:
    """Extract both raw mention count and weighted score per field from SAE descriptions.

    Returns dict with two keys:
      "count": field -> number of descriptions mentioning the field
      "weighted": field -> sum of |score| for descriptions mentioning the field
    """
    field_count = {f: 0.0 for f in all_fields}
    field_weighted = {f: 0.0 for f in all_fields}
    for score_str, desc in SAE_DESC_RE.findall(pattern_prompt):
        score = _parse_sae_score(score_str)
        for field in all_fields:
            if _field_mentioned_in_desc(desc, field, scenario_name):
                field_count[field] += 1
                field_weighted[field] += score
    return {"count": field_count, "weighted": field_weighted}


def load_models(batch_dirs: list[Path], agents_filter: set[str] | None = None) -> list[dict]:
    """Load models with relevant agent results."""
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

    # Deduplicate by model_dir
    unique = {}
    for m in all_models:
        key = m["model_dir"]
        if key not in unique:
            unique[key] = m
        else:
            unique[key]["agent_results"].update(m["agent_results"])
    return list(unique.values())


def extract_field_scores(
    agent_name: str, pattern_prompt: str, all_fields: list[str], scenario_name: str,
) -> dict[str, dict[str, float]] | None:
    """Extract per-field scores for a given agent from its pattern_prompt.

    Returns dict of metric_name -> {field -> score}, or None if extraction fails.
    Non-SAE agents return a single metric; SAE agents return both "count" and "weighted".
    """
    if agent_name in ("gradient", "relp"):
        samples = extract_gradient_relp_scores(pattern_prompt, all_fields)
        if not samples:
            return None
        agg = {f: 0.0 for f in all_fields}
        for s in samples:
            for f in all_fields:
                agg[f] += s.get(f, 0.0)
        return {"importance": {f: agg[f] / len(samples) for f in all_fields}}

    elif agent_name == "logit_lens_field":
        samples = extract_logit_lens_field_scores(pattern_prompt, all_fields)
        if not samples:
            return None
        agg = {f: 0.0 for f in all_fields}
        for s in samples:
            for f in all_fields:
                agg[f] += s.get(f, 0.0)
        return {"logit": {f: agg[f] / len(samples) for f in all_fields}}

    elif agent_name.startswith("sae_"):
        return extract_sae_scores(pattern_prompt, all_fields, scenario_name)

    return None


def run_variance_decomposition(y, x_field, x_circ, model_ids, n_fields):
    """Run R² variance decomposition. Returns dict of R² values."""
    total_ss = np.sum((y - np.mean(y))**2)
    if total_ss == 0:
        return None

    # Field identity
    field_means = np.array([np.mean(y[x_field == i]) for i in range(n_fields)])
    r2_field = np.sum((field_means[x_field] - np.mean(y))**2) / total_ss

    # Circuit membership
    circ_means = np.array([np.mean(y[x_circ == c]) for c in [0, 1]])
    r2_circ = np.sum((circ_means[x_circ] - np.mean(y))**2) / total_ss

    # Model identity
    unique_models = np.unique(model_ids)
    model_means = np.zeros(model_ids.max() + 1)
    for mid in unique_models:
        model_means[mid] = np.mean(y[model_ids == mid])
    r2_model = np.sum((model_means[model_ids] - np.mean(y))**2) / total_ss

    # Field × circuit (saturated)
    y_pred_fxc = np.zeros_like(y)
    for fi in range(n_fields):
        for c in [0, 1]:
            mask = (x_field == fi) & (x_circ == c)
            if mask.any():
                y_pred_fxc[mask] = np.mean(y[mask])
    r2_fxc = np.sum((y_pred_fxc - np.mean(y))**2) / total_ss

    return {
        "total_var": np.var(y),
        "r2_field": r2_field,
        "r2_circ": r2_circ,
        "r2_model": r2_model,
        "r2_fxc": r2_fxc,
        "r2_circ_add": r2_fxc - r2_field,
    }


def analyze_and_print(models: list[dict], agents_filter: list[str] | None = None):
    """Main analysis: extract scores, decompose variance, print results."""
    scenarios = {m["scenario"] for m in models}
    if len(scenarios) > 1:
        print(f"Warning: multiple scenarios: {scenarios}. Using first.", file=sys.stderr)
    scenario_name = next(iter(scenarios))
    scenario = get_scenario(scenario_name)
    all_fields = scenario.field_names()
    field_map = {f: i for i, f in enumerate(all_fields)}

    # Find available agents
    all_agents = set()
    for m in models:
        for a in m["agent_results"]:
            if a in SUPPORTED_AGENTS:
                all_agents.add(a)
    if agents_filter:
        all_agents = sorted(a for a in agents_filter if a in all_agents)
    else:
        all_agents = sorted(all_agents)

    print(f"Scenario: {scenario_name} | Fields: {len(all_fields)} | "
          f"Models: {len(models)} | Agents: {len(all_agents)}")
    print()

    # --- Summary table ---
    summary_rows = []

    from scipy.stats import mannwhitneyu

    for agent in all_agents:
        # Collect all metrics for this agent across models
        # metric_rows[metric_name] = list of row dicts
        metric_rows = defaultdict(list)
        n_models = 0

        for mi, m in enumerate(models):
            result = m["agent_results"].get(agent)
            if not result:
                continue
            pp = result.get("agent_metadata", {}).get("pattern_prompt", "")
            if not pp:
                continue

            all_metrics = extract_field_scores(agent, pp, all_fields, scenario_name)
            if all_metrics is None:
                continue

            n_models += 1
            for metric_name, field_scores in all_metrics.items():
                for field in all_fields:
                    metric_rows[metric_name].append({
                        "field_idx": field_map[field],
                        "field": field,
                        "in_circuit": 1 if field in m["used_fields"] else 0,
                        "model_idx": mi,
                        "score": field_scores.get(field, 0.0),
                        "depth": m["num_fields"],
                    })

        if not metric_rows:
            print(f"  {agent}: no data extracted")
            continue

        print(f"{'='*75}")
        print(f"Agent: {agent}  ({n_models} models, metrics: {', '.join(sorted(metric_rows))})")
        print(f"{'='*75}")

        for metric_name in sorted(metric_rows):
            rows = metric_rows[metric_name]
            label = f"{agent}/{metric_name}" if len(metric_rows) > 1 else agent

            y = np.array([r["score"] for r in rows], dtype=float)
            x_field = np.array([r["field_idx"] for r in rows])
            x_circ = np.array([r["in_circuit"] for r in rows])
            model_ids = np.array([r["model_idx"] for r in rows])

            if len(metric_rows) > 1:
                print(f"\n  ~~~ Metric: {metric_name} ~~~")

            vd = run_variance_decomposition(y, x_field, x_circ, model_ids, len(all_fields))

            # Per-field table
            print(f"\n--- Mean score per field (in-circuit vs not) [{metric_name}] ---")
            print(f"{'Field':<18} {'All':>10} {'In-circ':>10} {'Not-circ':>10} {'Diff':>10} {'N_in':>6} {'N_out':>6}")
            print("-" * 75)

            for field in all_fields:
                fr = [r for r in rows if r["field"] == field]
                all_s = [r["score"] for r in fr]
                in_s = [r["score"] for r in fr if r["in_circuit"] == 1]
                out_s = [r["score"] for r in fr if r["in_circuit"] == 0]
                m_all = np.mean(all_s) if all_s else 0
                m_in = np.mean(in_s) if in_s else 0
                m_out = np.mean(out_s) if out_s else 0
                print(f"{field:<18} {m_all:>10.2f} {m_in:>10.2f} {m_out:>10.2f} "
                      f"{m_in - m_out:>+10.2f} {len(in_s):>6} {len(out_s):>6}")

            all_in = [r["score"] for r in rows if r["in_circuit"] == 1]
            all_out = [r["score"] for r in rows if r["in_circuit"] == 0]
            print(f"{'OVERALL':<18} {np.mean(y):>10.2f} {np.mean(all_in):>10.2f} "
                  f"{np.mean(all_out):>10.2f} {np.mean(all_in)-np.mean(all_out):>+10.2f} "
                  f"{len(all_in):>6} {len(all_out):>6}")

            # Variance decomposition (raw)
            if vd:
                print(f"\n--- Variance decomposition (raw) [{metric_name}] ---")
                print(f"  Total variance:          {vd['total_var']:.2f}")
                print(f"  R² field identity:       {vd['r2_field']:.4f}")
                print(f"  R² circuit only:         {vd['r2_circ']:.4f}")
                print(f"  R² model only:           {vd['r2_model']:.4f}")
                print(f"  R² field×circuit:        {vd['r2_fxc']:.4f}")
                print(f"  => Circuit adds:         {vd['r2_circ_add']:.4f} R² over field alone")

            # Z-score per field
            y_zfield = np.zeros_like(y)
            for fi in range(len(all_fields)):
                mask = x_field == fi
                vals = y[mask]
                std = np.std(vals)
                if std > 0:
                    y_zfield[mask] = (vals - np.mean(vals)) / std
                else:
                    y_zfield[mask] = 0.0

            vd_zf = run_variance_decomposition(y_zfield, x_field, x_circ, model_ids, len(all_fields))

            # Per-field AUC
            print(f"\n--- Per-field AUC [{metric_name}] ---")
            print(f"{'Field':<18} {'AUC':>8} {'p-val':>10} {'N_in':>6} {'N_out':>6}")
            print("-" * 55)
            auc_vals = []
            for field in all_fields:
                fr = [r for r in rows if r["field"] == field]
                in_s = np.array([r["score"] for r in fr if r["in_circuit"] == 1])
                out_s = np.array([r["score"] for r in fr if r["in_circuit"] == 0])
                if len(in_s) < 2 or len(out_s) < 2:
                    print(f"{field:<18} {'N/A':>8} {'':>10} {len(in_s):>6} {len(out_s):>6}")
                    continue
                try:
                    u_stat, p_val = mannwhitneyu(in_s, out_s, alternative='two-sided')
                    auc = u_stat / (len(in_s) * len(out_s))
                except ValueError:
                    auc, p_val = 0.5, 1.0
                auc_vals.append(auc)
                sig = "*" if p_val < 0.05 else " "
                print(f"{field:<18} {auc:>8.3f} {p_val:>9.4f}{sig} {len(in_s):>6} {len(out_s):>6}")
            if auc_vals:
                print(f"{'MEAN AUC':<18} {np.mean(auc_vals):>8.3f}")

            if vd_zf:
                print(f"\n--- Variance decomposition (z-scored per field) [{metric_name}] ---")
                print(f"  R² circuit only:         {vd_zf['r2_circ']:.4f}  (= circuit signal after removing field bias)")
                print(f"  R² model only:           {vd_zf['r2_model']:.4f}")
                print(f"  R² field×circuit:        {vd_zf['r2_fxc']:.4f}")

            # Per-depth
            print(f"\n--- Per-depth [{metric_name}] ---")
            depths = sorted(set(r["depth"] for r in rows))
            print(f"{'Depth':>5} {'In-circ':>10} {'Not-circ':>10} {'Diff':>10} {'N':>6}")
            for d in depths:
                dr = [r for r in rows if r["depth"] == d]
                in_c = [r["score"] for r in dr if r["in_circuit"] == 1]
                out_c = [r["score"] for r in dr if r["in_circuit"] == 0]
                m_in = np.mean(in_c) if in_c else 0
                m_out = np.mean(out_c) if out_c else 0
                print(f"{d:>5} {m_in:>10.2f} {m_out:>10.2f} {m_in - m_out:>+10.2f} "
                      f"{len(dr) // len(all_fields):>6}")

            if vd:
                summary_rows.append({
                    "label": label,
                    "agent": agent,
                    "metric": metric_name,
                    "n_models": n_models,
                    "r2_field": vd["r2_field"],
                    "r2_circ": vd["r2_circ"],
                    "r2_circ_add": vd["r2_circ_add"],
                    "r2_model": vd["r2_model"],
                    "diff": np.mean(all_in) - np.mean(all_out),
                    "r2_circ_zfield": vd_zf["r2_circ"] if vd_zf else 0,
                    "mean_auc": np.mean(auc_vals) if auc_vals else 0.5,
                })

        print()

    # --- Summary comparison table ---
    if len(summary_rows) > 1:
        print(f"{'='*90}")
        print(f"SUMMARY: Variance Decomposition Comparison")
        print(f"{'='*90}")
        print(f"{'Label':<28} {'R² field':>9} {'R²circ+':>9} {'R²circ(z)':>10} "
              f"{'R² model':>9} {'Mean AUC':>9} {'In-Out':>9}")
        print("-" * 90)
        for sr in sorted(summary_rows, key=lambda x: -x["r2_circ_add"]):
            print(f"{sr['label']:<28} {sr['r2_field']:>9.4f} {sr['r2_circ_add']:>+9.4f} "
                  f"{sr['r2_circ_zfield']:>10.4f} {sr['r2_model']:>9.4f} "
                  f"{sr['mean_auc']:>9.3f} {sr['diff']:>+9.2f}")
        print()
        print("R²circ+   = incremental R² of circuit over field identity (raw scores)")
        print("R²circ(z) = R² of circuit after z-scoring per field (field bias removed)")
        print("Mean AUC  = mean per-field AUC of score → in-circuit classification")


def main():
    parser = argparse.ArgumentParser(description="Unified interp field bias analysis")
    parser.add_argument("batch_dirs", type=str, nargs="+", help="Batch evaluation directory(ies)")
    parser.add_argument("--agents", type=str, nargs="+", default=None,
                        help="Filter to specific agents")
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
        print("No models found.")
        return

    analyze_and_print(models, agents_filter=args.agents)


if __name__ == "__main__":
    main()
