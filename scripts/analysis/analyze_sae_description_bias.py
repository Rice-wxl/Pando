#!/usr/bin/env python3
"""SAE description bias analysis: do autointerp descriptions systematically mention certain fields?

Parses SAE feature descriptions from the pattern_prompt and counts how often each
field is mentioned. Decomposes variance into:
  1. Field-identity bias (constant factor per field)
  2. Circuit membership signal (in-circuit vs not)
  3. Model-level baseline
  4. Residual

Similar to the logit lens field token bias analysis but for SAE autointerp descriptions.

Usage:
    python scripts/analysis/analyze_sae_description_bias.py <batch_dir> [<batch_dir2> ...]
    python scripts/analysis/analyze_sae_description_bias.py <batch_dir> --agents sae_gradient sae_tfidf
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

# Reuse field aliases from analyze_field_bias.py
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
    "sample_then_sae_tfidf_llm": "sae_tfidf",
    "sample_then_sae_autointerp_llm": "sae_autointerp",
    "sample_then_sae_gradient_attribution_llm": "sae_gradient",
    "sample_then_sae_mean_diff_llm": "sae_mean_diff",
}

SAE_AGENTS = {"sae_gradient", "sae_tfidf", "sae_autointerp", "sae_tfidf_filtered", "sae_mean_diff"}

# Regex to extract description text and score from pattern_prompt
DESC_PATTERN = re.compile(r'\[\d+\]\s*\(([^)]+)\):\s*(.+)')


def _parse_score(score_str: str) -> float:
    """Parse the primary score from the parenthesized score string.

    Handles: 'attr=+2.59', 'attr=-1.12', 'tfidf=17.61, act=4.91', '53.6'
    Returns absolute value of the primary score.
    """
    # sae_gradient: attr=+/-X
    m = re.match(r'attr=([+-]?\d+\.?\d*)', score_str)
    if m:
        return abs(float(m.group(1)))
    # sae_tfidf: tfidf=X, act=Y
    m = re.match(r'tfidf=(\d+\.?\d*)', score_str)
    if m:
        return float(m.group(1))
    # sae_autointerp: bare float (activation)
    try:
        return abs(float(score_str.strip()))
    except ValueError:
        return 1.0


def _field_mentioned_in_desc(desc: str, field_name: str, scenario_name: str) -> bool:
    """Check if a field is mentioned in a single description."""
    text = desc.lower()
    if re.search(r"\b" + re.escape(field_name) + r"\b", text):
        return True
    space_name = field_name.replace("_", " ")
    if re.search(r"\b" + re.escape(space_name) + r"\b", text):
        return True
    aliases = FIELD_ALIASES.get(scenario_name, {}).get(field_name, [])
    for alias_pattern in aliases:
        if re.search(alias_pattern, text, re.IGNORECASE):
            return True
    return False


def extract_descriptions(pattern_prompt: str) -> list[tuple[float, str]]:
    """Extract (score, description) pairs from a pattern_prompt."""
    results = []
    for score_str, desc in DESC_PATTERN.findall(pattern_prompt):
        results.append((_parse_score(score_str), desc))
    return results


def load_models(batch_dirs: list[Path]) -> list[dict]:
    """Load models with SAE agent results."""
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
                continue
            with open(circuit_path) as f:
                circuit_data = json.load(f)

            agent_results = {}
            agent_results_dir = model_dir / "agent_results"
            if agent_results_dir.exists():
                for agent_file in agent_results_dir.glob("*.json"):
                    name = _OLD_AGENT_NAMES.get(agent_file.stem, agent_file.stem)
                    if name in SAE_AGENTS:
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


def analyze_description_bias(models: list[dict], agents_filter: list[str] | None = None):
    """Count field mentions in SAE descriptions and decompose variance.

    For each model × agent × field, count how many descriptions mention the field.
    Then analyze: is the count driven by field identity (constant bias) or circuit membership?
    """
    scenarios = {m["scenario"] for m in models}
    scenario_name = next(iter(scenarios))
    scenario = get_scenario(scenario_name)
    all_fields = scenario.field_names()

    # Find SAE agents
    all_agents = set()
    for m in models:
        for agent_name in m["agent_results"]:
            if agent_name in SAE_AGENTS:
                all_agents.add(agent_name)
    if agents_filter:
        all_agents = sorted(a for a in agents_filter if a in all_agents)
    else:
        all_agents = sorted(all_agents)

    print(f"Fields: {all_fields}")
    print(f"Agents: {all_agents}")
    print(f"Models: {len(models)}")
    print()

    # Collect per-model × agent × field: mention count and total descriptions
    # rows: (model_idx, agent, field, mention_count, total_descs, in_circuit, depth)
    rows = []

    for mi, m in enumerate(models):
        for agent in all_agents:
            result = m["agent_results"].get(agent)
            if not result:
                continue
            pp = result.get("agent_metadata", {}).get("pattern_prompt", "")
            if not pp:
                continue

            desc_pairs = extract_descriptions(pp)
            if not desc_pairs:
                continue

            # Count mentions per field (unweighted and weighted by |score|)
            for field in all_fields:
                count = 0
                weighted = 0.0
                for score, desc in desc_pairs:
                    if _field_mentioned_in_desc(desc, field, scenario_name):
                        count += 1
                        weighted += score
                in_circuit = 1 if field in m["used_fields"] else 0
                rows.append({
                    "model_idx": mi,
                    "model_name": m["model_name"],
                    "agent": agent,
                    "field": field,
                    "mention_count": count,
                    "weighted_count": weighted,
                    "total_descs": len(desc_pairs),
                    "mention_rate": count / len(desc_pairs) if desc_pairs else 0,
                    "in_circuit": in_circuit,
                    "depth": m["num_fields"],
                })

    return all_fields, all_agents, rows, scenario_name


def print_results(all_fields, agents, rows, scenario_name):
    """Print comprehensive bias analysis."""

    # ---- Per-agent summary ----
    for agent in agents:
        agent_rows = [r for r in rows if r["agent"] == agent]
        if not agent_rows:
            continue

        print(f"{'='*70}")
        print(f"Agent: {agent}  ({len(set(r['model_idx'] for r in agent_rows))} models)")
        print(f"{'='*70}")

        # -- Table 1: Mean mention count per field, split by in/out circuit --
        print(f"\n--- Mean mention count per field (in-circuit vs not) ---")
        print(f"{'Field':<18} {'All':>8} {'In-circ':>8} {'Not-circ':>8} {'Diff':>8} {'N_in':>6} {'N_out':>6}")
        print("-" * 70)

        field_means = {}
        field_in = {}
        field_out = {}
        for field in all_fields:
            fr = [r for r in agent_rows if r["field"] == field]
            all_counts = [r["mention_count"] for r in fr]
            in_counts = [r["mention_count"] for r in fr if r["in_circuit"] == 1]
            out_counts = [r["mention_count"] for r in fr if r["in_circuit"] == 0]

            mean_all = np.mean(all_counts) if all_counts else 0
            mean_in = np.mean(in_counts) if in_counts else 0
            mean_out = np.mean(out_counts) if out_counts else 0
            field_means[field] = mean_all
            field_in[field] = mean_in
            field_out[field] = mean_out

            print(f"{field:<18} {mean_all:>8.1f} {mean_in:>8.1f} {mean_out:>8.1f} "
                  f"{mean_in - mean_out:>+8.1f} {len(in_counts):>6} {len(out_counts):>6}")

        # Overall in vs out
        all_in = [r["mention_count"] for r in agent_rows if r["in_circuit"] == 1]
        all_out = [r["mention_count"] for r in agent_rows if r["in_circuit"] == 0]
        print(f"{'OVERALL':<18} {np.mean([r['mention_count'] for r in agent_rows]):>8.1f} "
              f"{np.mean(all_in):>8.1f} {np.mean(all_out):>8.1f} "
              f"{np.mean(all_in) - np.mean(all_out):>+8.1f} {len(all_in):>6} {len(all_out):>6}")

        # -- Table 2: Same but as mention rate (fraction of descriptions) --
        print(f"\n--- Mean mention rate (fraction of descriptions) ---")
        print(f"{'Field':<18} {'All':>8} {'In-circ':>8} {'Not-circ':>8} {'Diff':>8}")
        print("-" * 55)

        for field in all_fields:
            fr = [r for r in agent_rows if r["field"] == field]
            all_rates = [r["mention_rate"] for r in fr]
            in_rates = [r["mention_rate"] for r in fr if r["in_circuit"] == 1]
            out_rates = [r["mention_rate"] for r in fr if r["in_circuit"] == 0]

            mean_all = np.mean(all_rates) if all_rates else 0
            mean_in = np.mean(in_rates) if in_rates else 0
            mean_out = np.mean(out_rates) if out_rates else 0

            print(f"{field:<18} {mean_all:>7.1%} {mean_in:>7.1%} {mean_out:>7.1%} "
                  f"{mean_in - mean_out:>+7.1%}")

        # Overall
        all_in_r = [r["mention_rate"] for r in agent_rows if r["in_circuit"] == 1]
        all_out_r = [r["mention_rate"] for r in agent_rows if r["in_circuit"] == 0]
        print(f"{'OVERALL':<18} {np.mean([r['mention_rate'] for r in agent_rows]):>7.1%} "
              f"{np.mean(all_in_r):>7.1%} {np.mean(all_out_r):>7.1%} "
              f"{np.mean(all_in_r) - np.mean(all_out_r):>+7.1%}")

        # -- Weighted mention table (sum of |score| for mentions) --
        print(f"\n--- Mean weighted mention (sum |score|) per field (in-circuit vs not) ---")
        print(f"{'Field':<18} {'All':>10} {'In-circ':>10} {'Not-circ':>10} {'Diff':>10}")
        print("-" * 62)

        for field in all_fields:
            fr = [r for r in agent_rows if r["field"] == field]
            all_w = [r["weighted_count"] for r in fr]
            in_w = [r["weighted_count"] for r in fr if r["in_circuit"] == 1]
            out_w = [r["weighted_count"] for r in fr if r["in_circuit"] == 0]
            mean_all = np.mean(all_w) if all_w else 0
            mean_in = np.mean(in_w) if in_w else 0
            mean_out = np.mean(out_w) if out_w else 0
            print(f"{field:<18} {mean_all:>10.1f} {mean_in:>10.1f} {mean_out:>10.1f} "
                  f"{mean_in - mean_out:>+10.1f}")

        all_in_w = [r["weighted_count"] for r in agent_rows if r["in_circuit"] == 1]
        all_out_w = [r["weighted_count"] for r in agent_rows if r["in_circuit"] == 0]
        print(f"{'OVERALL':<18} {np.mean([r['weighted_count'] for r in agent_rows]):>10.1f} "
              f"{np.mean(all_in_w):>10.1f} {np.mean(all_out_w):>10.1f} "
              f"{np.mean(all_in_w) - np.mean(all_out_w):>+10.1f}")

        # -- Variance decomposition for both unweighted and weighted --
        for metric_name, metric_key in [("mention_count", "mention_count"),
                                         ("weighted_count (sum |score|)", "weighted_count")]:
            print(f"\n--- Variance decomposition (R² on {metric_name}) ---")

            y = np.array([r[metric_key] for r in agent_rows], dtype=float)
            total_var = np.var(y)
            total_ss = np.sum((y - np.mean(y))**2)

            if total_ss == 0:
                print("  No variance.")
                continue

            field_map = {f: i for i, f in enumerate(all_fields)}
            x_field = np.array([field_map[r["field"]] for r in agent_rows])
            field_group_means = np.array([np.mean(y[x_field == i]) for i in range(len(all_fields))])
            y_pred_field = field_group_means[x_field]
            ss_field = np.sum((y_pred_field - np.mean(y))**2)
            r2_field = ss_field / total_ss

            x_circ = np.array([r["in_circuit"] for r in agent_rows])
            circ_means = np.array([np.mean(y[x_circ == c]) for c in [0, 1]])
            y_pred_circ = circ_means[x_circ]
            ss_circ = np.sum((y_pred_circ - np.mean(y))**2)
            r2_circ = ss_circ / total_ss

            model_ids = np.array([r["model_idx"] for r in agent_rows])
            unique_models = np.unique(model_ids)
            model_group_means = np.zeros(model_ids.max() + 1)
            for mid in unique_models:
                model_group_means[mid] = np.mean(y[model_ids == mid])
            ss_model = np.sum((model_group_means[model_ids] - np.mean(y))**2)
            r2_model = ss_model / total_ss

            # field × circuit (saturated)
            y_pred_fxc = np.zeros_like(y)
            for fi in range(len(all_fields)):
                for c in [0, 1]:
                    mask = (x_field == fi) & (x_circ == c)
                    if mask.any():
                        y_pred_fxc[mask] = np.mean(y[mask])
            r2_fxc = np.sum((y_pred_fxc - np.mean(y))**2) / total_ss

            # field × model (saturated)
            y_pred_fxm = np.zeros_like(y)
            for fi in range(len(all_fields)):
                for mid in unique_models:
                    mask = (x_field == fi) & (model_ids == mid)
                    if mask.any():
                        y_pred_fxm[mask] = np.mean(y[mask])
            r2_fxm = np.sum((y_pred_fxm - np.mean(y))**2) / total_ss

            print(f"  Total variance: {total_var:.2f}")
            print(f"  R² field identity only:           {r2_field:.4f}  (constant bias per field)")
            print(f"  R² circuit membership only:       {r2_circ:.4f}  (in vs out of circuit)")
            print(f"  R² model identity only:           {r2_model:.4f}  (per-model baseline)")
            print(f"  R² field × circuit (interaction):  {r2_fxc:.4f}  (per-field circuit effect)")
            print(f"  R² field × model (interaction):    {r2_fxm:.4f}  (per-field model effect)")
            print(f"  => Circuit adds {r2_fxc - r2_field:.4f} R² over field alone")
            print(f"  => Model adds {r2_fxm - r2_field:.4f} R² over field alone")

        # -- Per-depth breakdown --
        print(f"\n--- Per-depth: mean mention count (in-circuit vs not) ---")
        depths = sorted(set(r["depth"] for r in agent_rows))
        print(f"{'Depth':>5} {'In-circ':>10} {'Not-circ':>10} {'Diff':>10} {'N':>6}")
        for d in depths:
            dr = [r for r in agent_rows if r["depth"] == d]
            in_c = [r["mention_count"] for r in dr if r["in_circuit"] == 1]
            out_c = [r["mention_count"] for r in dr if r["in_circuit"] == 0]
            m_in = np.mean(in_c) if in_c else 0
            m_out = np.mean(out_c) if out_c else 0
            print(f"{d:>5} {m_in:>10.1f} {m_out:>10.1f} {m_in - m_out:>+10.1f} {len(dr) // len(all_fields):>6}")

        print()


def main():
    parser = argparse.ArgumentParser(description="SAE description bias analysis")
    parser.add_argument("batch_dirs", type=str, nargs="+", help="Batch evaluation directory(ies)")
    parser.add_argument("--agents", type=str, nargs="+", default=None, help="Filter to specific SAE agents")
    args = parser.parse_args()

    batch_dirs = [Path(bd) for bd in args.batch_dirs]
    for bd in batch_dirs:
        if not bd.exists():
            print(f"Error: {bd} not found", file=sys.stderr)
            return

    print("Loading models...")
    models = load_models(batch_dirs)
    print(f"  {len(models)} unique models loaded from {len(batch_dirs)} batch dir(s)")

    if not models:
        print("No models found.")
        return

    all_fields, agents, rows, scenario_name = analyze_description_bias(models, agents_filter=args.agents)

    if not agents:
        print("No SAE agents found.")
        return

    print_results(all_fields, agents, rows, scenario_name)


if __name__ == "__main__":
    main()
