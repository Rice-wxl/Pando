#!/usr/bin/env python3
"""Field bias analysis: do agents systematically favor certain fields?

For each agent x field, computes the "false mention rate" — how often the agent
mentions a field when it is NOT in the circuit. Under the null hypothesis (no
field bias), false mention rates should be similar across all 10 fields.

Uses chi-squared goodness-of-fit to test uniformity, adjusting for varying
opportunity counts (some fields appear in circuits more often than others).

Usage:
    python scripts/analysis/analyze_field_bias.py <batch_dir> [<batch_dir2> ...]
    python scripts/analysis/analyze_field_bias.py <batch_dir> --agents gradient relp prefill
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.stats import chi2

from src.scenarios import get_scenario

# --- Field aliases (shared with analyze_sensitivity.py) ---

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


def _strip_parenthesized(text: str) -> str:
    """Remove all parenthesized content to avoid false positives from negation lists."""
    return re.sub(r"\([^)]*\)", "", text)


def _field_mentioned(pattern_text: str, field_name: str, scenario_name: str) -> bool:
    """Check if a field is mentioned in the pattern text (ignoring parenthesized content)."""
    text = _strip_parenthesized(pattern_text).lower()
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


def load_models(batch_dirs: list[Path]) -> list[dict]:
    """Load and deduplicate models from batch directories."""
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
                    with open(agent_file) as f:
                        agent_results[name] = json.load(f)

            all_models.append({
                "model_name": config.get("model_name", model_dir.name),
                "model_dir": str(source_model_dir),
                "num_fields": num_fields,
                "scenario": config.get("scenario", "car_purchase"),
                "used_fields": set(circuit_data.get("used_fields", [])),
                "agent_results": agent_results,
            })

    # Deduplicate by model_dir, merging agent results
    unique = {}
    for m in all_models:
        key = m["model_dir"]
        if key not in unique:
            unique[key] = m
        else:
            unique[key]["agent_results"].update(m["agent_results"])
    return list(unique.values())


def compute_field_bias(models: list[dict], agents_filter: list[str] | None = None):
    """Compute per-agent x per-field false mention rates and chi-squared test.

    Returns:
        all_fields: list of field names
        agents: list of agent names
        false_rate: dict[agent][field] -> float (false mention rate)
        opportunity: dict[field] -> int (number of models where field is NOT in circuit)
        false_count: dict[agent][field] -> int
        chi2_results: dict[agent] -> (statistic, p_value, df)
    """
    # Determine scenario and fields (assume single scenario)
    scenarios = {m["scenario"] for m in models}
    if len(scenarios) > 1:
        print(f"Warning: multiple scenarios found: {scenarios}. Using first.", file=sys.stderr)
    scenario_name = next(iter(scenarios))
    scenario = get_scenario(scenario_name)
    all_fields = scenario.field_names()

    # Find agents with patterns
    all_agents = set()
    for m in models:
        for agent_name, result in m["agent_results"].items():
            if result.get("agent_metadata", {}).get("pattern"):
                all_agents.add(agent_name)
    if agents_filter:
        all_agents = sorted(a for a in agents_filter if a in all_agents)
    else:
        all_agents = sorted(all_agents)

    # Count opportunities per field (models where field is NOT in circuit)
    opportunity = {f: 0 for f in all_fields}
    for m in models:
        for f in all_fields:
            if f not in m["used_fields"]:
                opportunity[f] += 1

    # Count false mentions per agent x field
    false_count = {a: {f: 0 for f in all_fields} for a in all_agents}
    agent_model_count = {a: 0 for a in all_agents}  # models with pattern for this agent

    for m in models:
        for agent in all_agents:
            result = m["agent_results"].get(agent)
            if not result:
                continue
            pattern = result.get("agent_metadata", {}).get("pattern")
            if not pattern:
                continue
            agent_model_count[agent] += 1
            for f in all_fields:
                if f not in m["used_fields"] and _field_mentioned(pattern, f, scenario_name):
                    false_count[agent][f] += 1

    # Compute rates
    false_rate = {}
    for agent in all_agents:
        false_rate[agent] = {}
        for f in all_fields:
            if opportunity[f] > 0:
                false_rate[agent][f] = false_count[agent][f] / opportunity[f]
            else:
                false_rate[agent][f] = 0.0

    # Chi-squared goodness-of-fit per agent
    # H0: P(false mention | field not in circuit) is the same for all fields
    # Expected count for field f = total_false * opportunity[f] / total_opportunity
    chi2_results = {}
    total_opportunity = sum(opportunity.values())
    for agent in all_agents:
        total_false = sum(false_count[agent][f] for f in all_fields)
        if total_false == 0 or total_opportunity == 0:
            chi2_results[agent] = (0.0, 1.0, len(all_fields) - 1)
            continue

        observed = []
        expected = []
        for f in all_fields:
            observed.append(false_count[agent][f])
            expected.append(total_false * opportunity[f] / total_opportunity)

        # Filter out fields with 0 expected (would cause div by zero)
        obs_filt = []
        exp_filt = []
        for o, e in zip(observed, expected):
            if e > 0:
                obs_filt.append(o)
                exp_filt.append(e)

        if len(obs_filt) < 2:
            chi2_results[agent] = (0.0, 1.0, 0)
            continue

        stat = sum((o - e) ** 2 / e for o, e in zip(obs_filt, exp_filt))
        df = len(obs_filt) - 1
        p_value = 1 - chi2.cdf(stat, df)
        chi2_results[agent] = (stat, p_value, df)

    return all_fields, all_agents, false_rate, opportunity, false_count, chi2_results, agent_model_count


def print_results(all_fields, agents, false_rate, opportunity, false_count,
                  chi2_results, agent_model_count, n_models):
    """Print field bias analysis results."""
    print(f"=== Field Bias Analysis ===")
    print(f"Models: {n_models} | Fields: {len(all_fields)} | Agents: {len(agents)}")
    print()

    # Opportunity counts
    print("Opportunity counts (models where field is NOT in circuit):")
    for f in all_fields:
        print(f"  {f:<15} {opportunity[f]:>4}")
    print()

    # False mention rate table
    # Abbreviate field names for compact display
    short = {f: f[:8] for f in all_fields}
    header = f"{'Agent':<20} |" + "|".join(f"{short[f]:>9}" for f in all_fields) + "| chi2     p-val    n"
    print(header)
    print("-" * len(header))

    # Sort agents by chi2 statistic (most biased first)
    agents_sorted = sorted(agents, key=lambda a: -chi2_results[a][0])

    for agent in agents_sorted:
        row = f"{agent:<20} |"
        for f in all_fields:
            rate = false_rate[agent][f]
            row += f"{rate:>8.1%} "
        stat, p, df = chi2_results[agent]
        n = agent_model_count[agent]
        row += f"| {stat:>6.1f}  {p:>7.4f}  {n:>3}"
        print(row)

    print()
    print("Significance: p < 0.05 = significant field bias (chi-squared, df=9)")
    print()

    # Per-agent: most over-mentioned fields
    print("=== Most Over-Mentioned Fields (top 3 per agent, sorted by false mention rate) ===")
    print()
    for agent in agents_sorted:
        stat, p, df = chi2_results[agent]
        sig = "*" if p < 0.05 else ""
        sorted_fields = sorted(all_fields, key=lambda f: -false_rate[agent][f])
        top3 = [(f, false_rate[agent][f], false_count[agent][f]) for f in sorted_fields[:3]]
        top3_str = ", ".join(f"{f}={rate:.1%} ({cnt}/{opportunity[f]})" for f, rate, cnt in top3)
        print(f"  {agent:<20} {sig:>1} | {top3_str}")

    # Global field popularity (averaged across agents)
    print()
    print("=== Global Field Popularity (mean false mention rate across agents) ===")
    print()
    global_rates = {}
    for f in all_fields:
        rates = [false_rate[a][f] for a in agents]
        global_rates[f] = np.mean(rates)
    for f in sorted(all_fields, key=lambda f: -global_rates[f]):
        print(f"  {f:<15} {global_rates[f]:>6.1%}")


def main():
    parser = argparse.ArgumentParser(description="Field bias analysis for interpretability agents")
    parser.add_argument("batch_dirs", type=str, nargs="+", help="Batch evaluation directory(ies)")
    parser.add_argument("--agents", type=str, nargs="+", default=None, help="Filter to specific agents")
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

    all_fields, agents, false_rate, opportunity, false_count, chi2_results, agent_model_count = \
        compute_field_bias(models, agents_filter=args.agents)

    if not agents:
        print("No agents with patterns found.")
        return

    print()
    print_results(all_fields, agents, false_rate, opportunity, false_count,
                  chi2_results, agent_model_count, len(models))


if __name__ == "__main__":
    main()
