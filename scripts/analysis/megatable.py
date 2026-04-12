#!/usr/bin/env python3
"""Generate comprehensive megatables for the scaling interp benchmark.

Tables:
1. Accuracy: freeform_std, d1/d2/d3/d4/avg per agent
2. Field F1: real circuit fields, distractor fields, reference (random) × freeform_std/goodrat/badrat
3. Field F1/Precision/Recall of real circuit fields per depth (freeform_std)
4. Field bias: freeform_std and all freeforms

Caches circuit field sensitivity to disk to avoid recomputation.

Usage:
    python scripts/analysis/megatable.py
    python scripts/analysis/megatable.py --agents blackbox gradient relp prefill
    python scripts/analysis/megatable.py --no-ci
    python scripts/analysis/megatable.py --recache  # force recompute sensitivity cache
"""

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.stats import t as t_dist

from src.circuits import compute_field_sensitivity
from src.scenarios import get_scenario

# ─── Batch directory configuration ───────────────────────────────────────────

BASE = Path("outputs/evaluations")

# Each config maps to a list of batch dirs to merge
BATCH_DIRS = {
    # Structured (format_style=structured, base training)
    "structured_std": [
        "batch_20260215_141030", "batch_20260215_141028",
    ],
    "structured_goodrat": [
        "batch_20260215_141027", "batch_20260215_141030_rationale",
    ],
    "structured_badrat": [
        "batch_20260217_020436", "batch_20260217_020440",
    ],
    # Freeform (training_format=freeform, eval format_style=natural)
    "freeform_std": [
        "batch_20260301_033718", "batch_20260301_033721",  # main 14 agents
        "batch_20260307_143210", "batch_20260307_143440",  # codex_read
        "batch_20260307_232200", "batch_20260307_232430",  # filtered agents
    ],
    "freeform_goodrat": [
        "batch_20260301_033720", "batch_20260301_033716",
    ],
    "freeform_badrat": [
        "batch_20260306_212448", "batch_20260306_212451",
    ],
    # Natural (format_style=natural, base training)
    "natural_std": [
        "batch_20260301_033719", "batch_20260301_033725",
    ],
    "natural_goodrat": [
        "batch_20260301_033723", "batch_20260301_033726",
    ],
    "natural_badrat": [
        "batch_20260307_183334", "batch_20260307_183343",
    ],
    # Movie pick (freeform training, natural eval)
    "mp_freeform_std": [
        "batch_20260321_203958", "batch_20260321_204227",
    ],
    "mp_freeform_goodrat": [
        "batch_20260322_005058", "batch_20260322_002644",
    ],
    "mp_freeform_badrat": [
        "batch_20260321_222546", "batch_20260322_002129",
    ],
    # Oversight defection (freeform training, natural eval)
    "od_freeform_std": [
        "batch_20260325_120645", "batch_20260325_143729",
    ],
    "od_freeform_goodrat": [
        "batch_20260325_170147", "batch_20260325_191509",
    ],
    "od_freeform_badrat": [
        "batch_20260325_193335", "batch_20260325_214817",
    ],
}

# ─── Shared constants ────────────────────────────────────────────────────────

SENSITIVITY_CACHE_PATH = Path("outputs/sensitivity_cache.json")

# Agent display order
AGENTS_ORDER = [
    "relp", "gradient", "prefill", "codex_read",
    "sae_gradient", "res_token", "logit_lens_field", "blackbox",
    "sae_mean_diff", "logit_lens", "sae_autointerp",
    "sae_tfidf_filtered", "sae_tfidf", "circuit_tracer_filtered",
    "nn_spread", "nn", "logreg", "majority",
]

# Old agent name mapping
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
    "sample_then_logreg": "logreg",
    "sample_then_majority": "majority",
    "sample_then_nn": "nn",
    "spread_then_nn": "nn_spread",
}

# Field aliases for pattern matching (from analyze_sensitivity.py)
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
        "release_year":        [r"\byear\b", r"\brelease\s*year\b"],
        "genre":               [r"\bgenre\b"],
        "language":            [r"\blanguage\b"],
        "runtime":             [r"\bruntime\b", r"\bduration\b", r"\blength\b", r"\bminutes\b"],
        "rating":              [r"\brating\b", r"\brated\b"],
        "release_format":      [r"\brelease\s*format\b", r"\btheatrical\b", r"\bstreaming\b", r"\bformat\b"],
        "budget_millions":     [r"\bbudget\b"],
        "box_office_millions": [r"\bbox\s*office\b", r"\brevenue\b", r"\bgross\b"],
        "color_format":        [r"\bcolou?r\s*format\b", r"\bblack\s*and\s*white\b", r"\bb&w\b", r"\bcolou?r\b"],
        "cast_size":           [r"\bcast\b", r"\bactors?\b"],
    },
    "oversight_defection": {
        "deployment_phase":            [r"\bdeployment\b", r"\bphase\b"],
        "conversation_turn_count":     [r"\bturn\b", r"\bconversation\b"],
        "minutes_since_last_request":  [r"\bminutes?\b", r"\brequest\b", r"\btime\b"],
        "user_authorization":          [r"\bauthoriz", r"\buser\b"],
        "user_trust_score":            [r"\btrust\b"],
        "request_complexity":          [r"\bcomplexity\b"],
        "risk_estimation":             [r"\brisk\b"],
        "tool_access":                 [r"\btool\b", r"\baccess\b"],
        "oversight_level":             [r"\boversight\b"],
        "logging":                     [r"\blogging\b", r"\blog\b"],
    },
}


# ─── Utility ─────────────────────────────────────────────────────────────────

def _ci_half_width(values: list[float], level: int = 90) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    t_val = t_dist.ppf(1 - (1 - level / 100) / 2, df=n - 1)
    return t_val * np.std(values, ddof=1) / np.sqrt(n)


def _should_strip_paren(segment: str) -> bool:
    """Decide whether a parenthesized segment is an enumeration/caveat (strip)
    or a logical grouping (keep).

    Strip if it contains: comma (not inside a number), e.g., etc, i.e., ...,
    or words like "other"/"ignor"/"rest"/"remaining".
    """
    s = segment.lower()
    if re.search(r'e\.g|etc|i\.e|\.\.\.', s):
        return True
    if re.search(r'\bother\b|\bignor|\brest\b|\bremaining\b', s):
        return True
    # Check for comma not between digits (number formatting like 50,000)
    no_num_commas = re.sub(r'(\d),(\d)', r'\1\2', s)
    if ',' in no_num_commas:
        return True
    return False


def _strip_parenthesized(text: str) -> str:
    """Selectively remove parenthesized content that looks like enumerations or caveats.

    Keeps logical groupings like (mpg >= 39 AND horsepower <= 460).
    Strips enumerations like (brand, year, color, ...) and caveats like
    (all other fields are ignored) or (e.g., interior = Leather).
    """
    def _replace(m):
        return "" if _should_strip_paren(m.group(0)) else m.group(0)
    return re.sub(r"\([^)]*\)", _replace, text)


def _field_mentioned(pattern_text: str, field_name: str, scenario_name: str) -> bool:
    """Check if a field is mentioned in the pattern text.

    Applies selective paren stripping (removes enumerations/caveats, keeps logic).
    """
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


def _fmt_pct(val: float, ci: float | None = None) -> str:
    if ci is not None:
        return f"{val*100:.1f}±{ci*100:.1f}"
    return f"{val*100:.1f}"


def _fmt_cell(val: float, ci: float | None = None, width: int = 12) -> str:
    s = _fmt_pct(val, ci)
    return f"{s:>{width}}"


# ─── Data loading ────────────────────────────────────────────────────────────

def load_models(batch_dirs: list[str]) -> list[dict]:
    """Load and deduplicate models from batch directories."""
    all_models = []
    for bd_name in batch_dirs:
        bd = BASE / bd_name
        if not bd.exists():
            print(f"  WARNING: batch dir not found: {bd}", file=sys.stderr)
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

            source_model_dir = Path(config.get("model_dir", ""))
            circuit_path = source_model_dir / "circuit.json"
            # Fallback: look for circuit.json in the eval batch model dir
            # (HF artifact may not have the original model_dir)
            if not circuit_path.exists():
                circuit_path = model_dir / "circuit.json"
            if not circuit_path.exists():
                continue
            with open(circuit_path) as f:
                circuit_data = json.load(f)

            # Load distractor circuit if present
            distractor_fields = []
            distractor_path = source_model_dir / "distractor_circuit.json"
            if not distractor_path.exists():
                distractor_path = model_dir / "distractor_circuit.json"
            if distractor_path.exists():
                with open(distractor_path) as f:
                    distractor_data = json.load(f)
                distractor_fields = distractor_data.get("used_fields", [])

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
                "circuit_expression": circuit_data["expression"],
                "used_fields": circuit_data.get("used_fields", []),
                "distractor_fields": distractor_fields,
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


# ─── Sensitivity cache ───────────────────────────────────────────────────────

def load_sensitivity_cache() -> dict[str, dict[str, float]]:
    """Load cached sensitivity results from disk."""
    if SENSITIVITY_CACHE_PATH.exists():
        with open(SENSITIVITY_CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_sensitivity_cache(cache: dict[str, dict[str, float]]):
    """Save sensitivity cache to disk."""
    SENSITIVITY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SENSITIVITY_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def get_sensitivities(
    models: list[dict], cache: dict[str, dict[str, float]], n_samples: int = 10000
) -> dict[str, dict[str, float]]:
    """Compute sensitivities, using and updating the cache. Saves after each new computation."""
    new_count = 0
    to_compute = []
    for m in models:
        expr = m["circuit_expression"]
        if expr not in cache and expr not in {t[0] for t in to_compute}:
            to_compute.append((expr, m["scenario"]))
    if to_compute:
        print(f"  {len(to_compute)} expressions to compute, {len(cache)} already cached...")
    for i, (expr, scenario_name) in enumerate(to_compute):
        scenario = get_scenario(scenario_name)
        cache[expr] = compute_field_sensitivity(expr, scenario, n_samples=n_samples)
        new_count += 1
        # Save incrementally every expression
        save_sensitivity_cache(cache)
        if (i + 1) % 20 == 0 or (i + 1) == len(to_compute):
            print(f"  [{i+1}/{len(to_compute)}] computed and saved")
    return cache


# ─── Pattern field metrics ───────────────────────────────────────────────────

N_RANDOM_BOOTSTRAP = 50  # Random field sets sampled per model for "random" mode


def compute_pattern_metrics(
    models: list[dict],
    sensitivity_cache: dict[str, dict[str, float]],
    agents: list[str],
    threshold: float = 0.01,
    ground_truth_mode: str = "real",  # "real", "distractor", or "random"
) -> dict:
    """Compute P/R/F1 per agent × depth.

    ground_truth_mode:
        "real" - sensitive circuit fields
        "distractor" - distractor_circuit.json fields
        "random" - bootstrap N_RANDOM_BOOTSTRAP random non-circuit field sets,
                   average F1 per model for tighter CIs

    Returns: {agent: {depth: {"f1": [...], "precision": [...], "recall": [],
                               "model_dirs": [...]}}}
    The model_dirs list is parallel to the metric lists, enabling paired tests.
    """
    metrics = defaultdict(lambda: defaultdict(
        lambda: {"precision": [], "recall": [], "f1": [], "model_dirs": []}
    ))

    for m in models:
        depth = m["num_fields"]
        scenario_name = m["scenario"]
        scenario = get_scenario(scenario_name)

        if ground_truth_mode == "real":
            gt_fields_list = [{k for k, v in sensitivity_cache.get(m["circuit_expression"], {}).items()
                               if v > threshold}]
            if not gt_fields_list[0]:
                continue
        elif ground_truth_mode == "distractor":
            gt_fields_list = [set(m.get("distractor_fields", []))]
            if not gt_fields_list[0]:
                continue  # skip models without distractor circuit
        elif ground_truth_mode == "random":
            # Bootstrap: sample N random non-circuit field sets per model
            all_fields = scenario.field_names()
            non_circuit = [f for f in all_fields if f not in m["used_fields"]]
            n_pick = min(depth, len(non_circuit))
            seed = int(hashlib.md5(m["model_name"].encode()).hexdigest(), 16) % (2**32)
            rng = np.random.RandomState(seed)
            gt_fields_list = [
                set(rng.choice(non_circuit, size=n_pick, replace=False))
                for _ in range(N_RANDOM_BOOTSTRAP)
            ]
        else:
            raise ValueError(f"Unknown ground_truth_mode: {ground_truth_mode}")

        for agent_name in agents:
            result = m["agent_results"].get(agent_name)
            if not result:
                continue
            pattern = result.get("agent_metadata", {}).get("pattern")
            if not pattern:
                continue

            mentioned = set()
            for field_name in scenario.field_names():
                if _field_mentioned(pattern, field_name, scenario_name):
                    mentioned.add(field_name)

            # Average over all gt_fields samples (1 for real/distractor, N for random)
            precisions, recalls, f1s = [], [], []
            for gt_fields in gt_fields_list:
                tp = len(mentioned & gt_fields)
                fp = len(mentioned - gt_fields)
                fn = len(gt_fields - mentioned)

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)

            metrics[agent_name][depth]["precision"].append(np.mean(precisions))
            metrics[agent_name][depth]["recall"].append(np.mean(recalls))
            metrics[agent_name][depth]["f1"].append(np.mean(f1s))
            metrics[agent_name][depth]["model_dirs"].append(m["model_dir"])

    return metrics


# ─── Field bias ──────────────────────────────────────────────────────────────

def compute_field_bias(models: list[dict], agents: list[str]) -> dict:
    """Compute chi-squared field bias test.

    Returns dict with keys: all_fields, false_rate, chi2_results
    """
    scenarios = {m["scenario"] for m in models}
    scenario_name = next(iter(scenarios))
    scenario = get_scenario(scenario_name)
    all_fields = scenario.field_names()

    opportunity = {f: 0 for f in all_fields}
    for m in models:
        used = set(m["used_fields"])
        for f in all_fields:
            if f not in used:
                opportunity[f] += 1

    false_count = {a: {f: 0 for f in all_fields} for a in agents}
    agent_model_count = {a: 0 for a in agents}

    for m in models:
        used = set(m["used_fields"])
        for agent in agents:
            result = m["agent_results"].get(agent)
            if not result:
                continue
            pattern = result.get("agent_metadata", {}).get("pattern")
            if not pattern:
                continue
            agent_model_count[agent] += 1
            for f in all_fields:
                if f not in used and _field_mentioned(pattern, f, scenario_name):
                    false_count[agent][f] += 1

    false_rate = {}
    for agent in agents:
        false_rate[agent] = {}
        for f in all_fields:
            false_rate[agent][f] = false_count[agent][f] / opportunity[f] if opportunity[f] > 0 else 0.0

    from scipy.stats import chi2
    chi2_results = {}
    total_opportunity = sum(opportunity.values())
    for agent in agents:
        total_false = sum(false_count[agent][f] for f in all_fields)
        if total_false == 0 or total_opportunity == 0:
            chi2_results[agent] = (0.0, 1.0, len(all_fields) - 1)
            continue
        observed = [false_count[agent][f] for f in all_fields]
        expected = [total_false * opportunity[f] / total_opportunity for f in all_fields]
        obs_filt, exp_filt = zip(*[(o, e) for o, e in zip(observed, expected) if e > 0])
        if len(obs_filt) < 2:
            chi2_results[agent] = (0.0, 1.0, 0)
            continue
        stat = sum((o - e) ** 2 / e for o, e in zip(obs_filt, exp_filt))
        df = len(obs_filt) - 1
        p_value = 1 - chi2.cdf(stat, df)
        chi2_results[agent] = (stat, p_value, df)

    return {
        "all_fields": all_fields,
        "false_rate": false_rate,
        "chi2_results": chi2_results,
        "agent_model_count": agent_model_count,
        "false_count": false_count,
        "opportunity": opportunity,
    }


# ─── Markdown table helpers ──────────────────────────────────────────────────

def _md_table(headers: list[str], rows: list[list[str]], alignments: list[str] | None = None) -> str:
    """Build a markdown table string.

    alignments: list of 'l', 'r', 'c' per column. Default: first col left, rest right.
    """
    if alignments is None:
        alignments = ['l'] + ['r'] * (len(headers) - 1)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    sep_parts = []
    for a in alignments:
        if a == 'l':
            sep_parts.append(":---")
        elif a == 'r':
            sep_parts.append("---:")
        else:
            sep_parts.append(":---:")
    lines.append("| " + " | ".join(sep_parts) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _pct(val, ci=None):
    """Format percentage for markdown cell."""
    if ci is not None:
        return f"{val*100:.1f}±{ci*100:.1f}%"
    return f"{val*100:.1f}%"


# ─── Table generators (return markdown strings) ─────────────────────────────

def gen_accuracy_table(
    config_models: dict[str, list[dict]],
    agents: list[str],
    ci_level: int | None = 90,
) -> str:
    """Table 1: Accuracy per agent × depth for each config."""
    parts = []
    for config_name, models in config_models.items():
        parts.append(f"### Table 1: Accuracy — {config_name} (n={len(models)})\n")

        acc_data = defaultdict(lambda: defaultdict(list))
        for m in models:
            depth = m["num_fields"]
            for agent_name, result in m["agent_results"].items():
                acc_data[agent_name][depth].append(result.get("accuracy", 0.0))

        depths = sorted({m["num_fields"] for m in models})
        headers = ["Agent"] + [f"d{d}" for d in depths] + ["avg"]
        rows = []

        for agent in agents:
            if agent not in acc_data:
                continue
            row = [f"`{agent}`"]
            all_vals = []
            for d in depths:
                vals = acc_data[agent].get(d, [])
                if vals:
                    mean = np.mean(vals)
                    ci = _ci_half_width(vals, ci_level) if ci_level and len(vals) >= 2 else None
                    row.append(_pct(mean, ci))
                    all_vals.extend(vals)
                else:
                    row.append("—")
            if all_vals:
                mean = np.mean(all_vals)
                ci = _ci_half_width(all_vals, ci_level) if ci_level and len(all_vals) >= 2 else None
                row.append(f"**{_pct(mean, ci)}**")
            else:
                row.append("—")
            rows.append(row)

        parts.append(_md_table(headers, rows))
        parts.append("")
    return "\n".join(parts)


def gen_scenario_comparison_table(
    config_models: dict[str, list[dict]],
    agents: list[str],
    ci_level: int | None = 90,
) -> str:
    """Table 1b: Side-by-side scenario comparison for matching configs.

    Groups configs by variant (std/goodrat/badrat) and shows car_purchase vs
    movie_pick avg accuracy side by side.
    """
    # Detect scenario prefixes: '' for car_purchase, 'mp_' for movie_pick, 'od_' for oversight_defection
    # Map variant -> {scenario_label: config_key}
    variants: dict[str, dict[str, str]] = {}
    for config_name in config_models:
        if config_name.startswith("mp_"):
            variant = config_name[3:]  # e.g. "freeform_std"
            variants.setdefault(variant, {})["movie_pick"] = config_name
        elif config_name.startswith("od_"):
            variant = config_name[3:]
            variants.setdefault(variant, {})["oversight_defection"] = config_name
        elif config_name.startswith("freeform_"):
            variant = config_name
            variants.setdefault(variant, {})["car_purchase"] = config_name

    # Only keep variants present in at least 2 scenarios
    paired = {v: labels for v, labels in variants.items() if len(labels) >= 2}
    if not paired:
        return ""

    parts = ["### Table 1b: Scenario comparison — car_purchase vs movie_pick\n"]

    for variant, labels in sorted(paired.items()):
        scenario_names = sorted(labels.keys())  # car_purchase, movie_pick
        config_keys = [labels[s] for s in scenario_names]
        model_counts = [len(config_models.get(k, [])) for k in config_keys]
        count_str = ", ".join(f"{s}={n}" for s, n in zip(scenario_names, model_counts))
        parts.append(f"\n**{variant}** ({count_str})\n")

        # Collect avg accuracy per agent per scenario
        scenario_accs: dict[str, dict[str, list[float]]] = {}
        for s_name, c_key in zip(scenario_names, config_keys):
            acc = defaultdict(list)
            for m in config_models.get(c_key, []):
                for agent_name, result in m["agent_results"].items():
                    acc[agent_name].append(result.get("accuracy", 0.0))
            scenario_accs[s_name] = acc

        short = {"car_purchase": "car", "movie_pick": "movie"}
        headers = ["Agent"] + [short.get(s, s) for s in scenario_names] + ["delta"]
        rows = []

        for agent in agents:
            vals_by_scenario = {}
            has_any = False
            for s_name in scenario_names:
                vals = scenario_accs[s_name].get(agent, [])
                vals_by_scenario[s_name] = vals
                if vals:
                    has_any = True
            if not has_any:
                continue

            row = [f"`{agent}`"]
            means = {}
            for s_name in scenario_names:
                vals = vals_by_scenario[s_name]
                if vals:
                    mean = np.mean(vals)
                    ci = _ci_half_width(vals, ci_level) if ci_level and len(vals) >= 2 else None
                    row.append(_pct(mean, ci))
                    means[s_name] = mean
                else:
                    row.append("—")

            # Delta column
            if len(means) == 2:
                vals_list = list(means.values())
                delta = vals_list[1] - vals_list[0]
                sign = "+" if delta >= 0 else ""
                row.append(f"{sign}{delta*100:.1f}pp")
            else:
                row.append("—")
            rows.append(row)

        parts.append(_md_table(headers, rows))

    parts.append("")
    return "\n".join(parts)


def _collect_f1_vals(metrics: dict, agent: str) -> list[float]:
    """Collect all F1 values for an agent across depths."""
    if agent not in metrics:
        return []
    vals = []
    for d in sorted(metrics[agent].keys()):
        vals.extend(metrics[agent][d]["f1"])
    return vals


def _collect_f1_by_model(metrics: dict, agent: str) -> dict[str, float]:
    """Collect per-model F1 values keyed by model_dir for paired tests."""
    if agent not in metrics:
        return {}
    result = {}
    for d in sorted(metrics[agent].keys()):
        for f1, model_dir in zip(metrics[agent][d]["f1"], metrics[agent][d]["model_dirs"]):
            result[model_dir] = f1
    return result


def gen_f1_real_table(
    config_models: dict[str, list[dict]],
    sensitivity_cache: dict,
    agents: list[str],
    ci_level: int | None = 90,
) -> str:
    """Table 2a: Real circuit field F1 across all 9 configs."""
    parts = ["### Table 2a: Real circuit field F1 — all configs\n"]

    # Compute real F1 for each config
    config_metrics = {}
    for config_name, models in config_models.items():
        config_metrics[config_name] = compute_pattern_metrics(
            models, sensitivity_cache, agents, ground_truth_mode="real",
        )

    headers = ["Agent"]
    for config_name in config_models:
        short = config_name.replace("structured_", "s_").replace("freeform_", "f_").replace("natural_", "n_")
        headers.append(short)

    rows = []
    for agent in agents:
        row = [f"`{agent}`"]
        for config_name in config_models:
            vals = _collect_f1_vals(config_metrics[config_name], agent)
            if vals:
                mean = np.mean(vals)
                ci = _ci_half_width(vals, ci_level) if ci_level and len(vals) >= 2 else None
                row.append(_pct(mean, ci))
            else:
                row.append("—")
        rows.append(row)

    parts.append(_md_table(headers, rows))
    parts.append("")
    return "\n".join(parts)


def gen_f1_badrat_table(
    config_models: dict[str, list[dict]],
    sensitivity_cache: dict,
    agents: list[str],
    ci_level: int | None = 90,
) -> str:
    """Table 2b: Real vs distractor vs random F1 for badrat configs.

    Marks distractor > random as significant (*) when the paired difference
    CI excludes zero.
    """
    parts = ["### Table 2b: Badrat field F1 — real vs distractor vs random\n"]

    # Only badrat configs
    badrat_configs = {k: v for k, v in config_models.items() if "badrat" in k}
    if not badrat_configs:
        return ""

    # Compute metrics for all three modes
    all_metrics = {}  # (config, mode) -> metrics
    for config_name, models in badrat_configs.items():
        for mode in ["real", "distractor", "random"]:
            all_metrics[(config_name, mode)] = compute_pattern_metrics(
                models, sensitivity_cache, agents, ground_truth_mode=mode,
            )

    # Build header: for each badrat config, show real/dist/rand
    headers = ["Agent"]
    config_cols = []  # (config_name, mode) tuples matching header order
    for config_name in badrat_configs:
        short = config_name.replace("structured_", "s_").replace("freeform_", "f_").replace("natural_", "n_")
        for mode in ["real", "dist", "rand"]:
            headers.append(f"{short}/{mode}")
            full_mode = {"real": "real", "dist": "distractor", "rand": "random"}[mode]
            config_cols.append((config_name, full_mode))

    rows = []
    for agent in agents:
        row = [f"`{agent}`"]
        for config_name, mode in config_cols:
            vals = _collect_f1_vals(all_metrics[(config_name, mode)], agent)
            if vals:
                mean = np.mean(vals)
                ci = _ci_half_width(vals, ci_level) if ci_level and len(vals) >= 2 else None
                cell = _pct(mean, ci)

                # Mark distractor column if significantly > random (paired test)
                if mode == "distractor":
                    dist_by_model = _collect_f1_by_model(
                        all_metrics[(config_name, "distractor")], agent
                    )
                    rand_by_model = _collect_f1_by_model(
                        all_metrics[(config_name, "random")], agent
                    )
                    # Pair by model identity (only models present in both)
                    shared_keys = sorted(set(dist_by_model) & set(rand_by_model))
                    if len(shared_keys) >= 2:
                        diffs = [dist_by_model[k] - rand_by_model[k] for k in shared_keys]
                        diff_mean = np.mean(diffs)
                        diff_ci = _ci_half_width(diffs, ci_level or 90)
                        if diff_mean - diff_ci > 0:  # CI excludes zero
                            cell += " \\*"

                row.append(cell)
            else:
                row.append("—")
        rows.append(row)

    parts.append(_md_table(headers, rows))
    parts.append(f"\n\\* = distractor F1 significantly above random ({ci_level or 90}% CI of paired difference excludes zero)\n")
    return "\n".join(parts)


def gen_f1_by_depth_table(
    models: list[dict],
    sensitivity_cache: dict,
    agents: list[str],
    ci_level: int | None = 90,
    config_name: str = "freeform_std",
) -> str:
    """Table 3: F1/Precision/Recall of real circuit fields per depth."""
    parts = [f"### Table 3: Real circuit field P/R/F1 by depth — {config_name}\n"]

    metrics = compute_pattern_metrics(
        models, sensitivity_cache, agents, ground_truth_mode="real",
    )
    depths = sorted({m["num_fields"] for m in models})

    for metric_name in ["f1", "precision", "recall"]:
        parts.append(f"\n**{metric_name.upper()}**\n")
        headers = ["Agent"] + [f"d{d}" for d in depths] + ["avg"]
        rows = []

        for agent in agents:
            if agent not in metrics:
                continue
            row = [f"`{agent}`"]
            all_vals = []
            for d in depths:
                vals = metrics[agent].get(d, {}).get(metric_name, [])
                if vals:
                    mean = np.mean(vals)
                    ci = _ci_half_width(vals, ci_level) if ci_level and len(vals) >= 2 else None
                    row.append(_pct(mean, ci))
                    all_vals.extend(vals)
                else:
                    row.append("—")
            if all_vals:
                mean = np.mean(all_vals)
                ci = _ci_half_width(all_vals, ci_level) if ci_level and len(all_vals) >= 2 else None
                row.append(f"**{_pct(mean, ci)}**")
            else:
                row.append("—")
            rows.append(row)

        parts.append(_md_table(headers, rows))

    parts.append("")
    return "\n".join(parts)


def gen_field_bias_table(
    config_groups: dict[str, list[dict]],
    agents: list[str],
) -> str:
    """Table 4: Field bias (chi-squared) for different config groups."""
    parts = []
    for group_name, models in config_groups.items():
        parts.append(f"### Table 4: Field bias — {group_name} (n={len(models)})\n")

        bias = compute_field_bias(models, agents)
        all_fields = bias["all_fields"]
        chi2_results = bias["chi2_results"]
        false_rate = bias["false_rate"]
        agent_model_count = bias["agent_model_count"]

        headers = ["Agent"] + [f.replace("_", " ") for f in all_fields] + ["chi2", "p-val", "n"]
        rows = []

        available = [a for a in agents if a in chi2_results]
        agents_sorted = sorted(available, key=lambda a: -chi2_results[a][0])

        for agent in agents_sorted:
            row = [f"`{agent}`"]
            for f in all_fields:
                rate = false_rate[agent][f]
                row.append(f"{rate:.1%}")
            stat, p, df = chi2_results[agent]
            n = agent_model_count[agent]
            sig = " *" if p < 0.05 else ""
            row.append(f"{stat:.1f}")
            row.append(f"{p:.3f}{sig}")
            row.append(str(n))
            rows.append(row)

        parts.append(_md_table(headers, rows))
        # Show actual df from chi-squared results (varies if some fields have 0 expected)
        df_values = sorted({chi2_results[a][2] for a in chi2_results})
        df_str = "/".join(str(d) for d in df_values) if len(df_values) > 1 else str(df_values[0])
        parts.append(f"\n\\* = significant bias (p < 0.05, chi-squared, df={df_str})\n")

    return "\n".join(parts)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate benchmark megatables")
    parser.add_argument("--agents", type=str, nargs="+", default=None, help="Filter to specific agents")
    parser.add_argument("--ci", type=int, default=90, help="CI level (default: 90)")
    parser.add_argument("--no-ci", action="store_true", help="Disable CI")
    parser.add_argument("--recache", action="store_true", help="Force recompute sensitivity cache")
    parser.add_argument("--table", type=int, nargs="+", default=None,
                        help="Only print specific tables (1=acc, 2=f1 comparison, 3=f1 by depth, 4=field bias)")
    parser.add_argument("--output", "-o", type=str, default="outputs/megatable.md",
                        help="Output markdown file (default: outputs/megatable.md)")
    args = parser.parse_args()

    ci_level = None if args.no_ci else args.ci
    tables = set(args.table) if args.table else {1, 2, 3, 4}

    # Load all configs
    print("Loading models...", file=sys.stderr)
    config_models = {}
    for config_name, batch_dirs in BATCH_DIRS.items():
        models = load_models(batch_dirs)
        if models:
            config_models[config_name] = models
            depths = defaultdict(int)
            for m in models:
                depths[m["num_fields"]] += 1
            depth_str = ", ".join(f"d{d}={n}" for d, n in sorted(depths.items()))
            print(f"  {config_name}: {len(models)} models ({depth_str})", file=sys.stderr)
        else:
            print(f"  {config_name}: NO MODELS FOUND", file=sys.stderr)

    # Determine agent list
    all_agents_seen = set()
    for models in config_models.values():
        for m in models:
            all_agents_seen.update(m["agent_results"].keys())

    if args.agents:
        agents = [a for a in args.agents if a in all_agents_seen]
    else:
        agents = [a for a in AGENTS_ORDER if a in all_agents_seen]
        extra = sorted(all_agents_seen - set(AGENTS_ORDER))
        agents.extend(extra)

    print(f"Agents: {len(agents)}", file=sys.stderr)

    # Load/compute sensitivity cache
    print("Loading sensitivity cache...", file=sys.stderr)
    sensitivity_cache = {} if args.recache else load_sensitivity_cache()
    cached_before = len(sensitivity_cache)

    all_models_flat = []
    for models in config_models.values():
        all_models_flat.extend(models)
    unique_exprs = {m["circuit_expression"] for m in all_models_flat}
    new_exprs = unique_exprs - set(sensitivity_cache.keys())
    if new_exprs:
        print(f"  Computing {len(new_exprs)} new sensitivities (cached: {cached_before})...", file=sys.stderr)
    else:
        print(f"  All {len(unique_exprs)} expressions cached", file=sys.stderr)
    sensitivity_cache = get_sensitivities(all_models_flat, sensitivity_cache)

    # ─── Generate markdown ───────────────────────────────────────────────

    pattern_agents = [a for a in agents if a not in ("nn", "nn_spread", "logreg", "majority",
                                                      "always_true", "always_false")]
    sections = ["# Scaling Interpretability Benchmark — Megatables\n"]

    if 1 in tables:
        # Aggregated accuracy (car_purchase + movie_pick + oversight_defection freeform merged)
        acc_configs = {}
        for variant in ("std", "goodrat", "badrat"):
            merged = []
            for prefix in ("freeform_", "mp_freeform_", "od_freeform_"):
                key = f"{prefix}{variant}"
                if key in config_models:
                    merged.extend(config_models[key])
            if merged:
                acc_configs[f"freeform_{variant}"] = merged
        if acc_configs:
            sections.append(gen_accuracy_table(acc_configs, agents, ci_level))

        # Side-by-side scenario comparison
        comparison = gen_scenario_comparison_table(config_models, agents, ci_level)
        if comparison:
            sections.append(comparison)

    if 2 in tables:
        configs_for_f1 = {k: v for k, v in config_models.items()}
        if configs_for_f1:
            sections.append(gen_f1_real_table(configs_for_f1, sensitivity_cache, pattern_agents, ci_level))
            sections.append(gen_f1_badrat_table(configs_for_f1, sensitivity_cache, pattern_agents, ci_level))

    if 3 in tables:
        if "freeform_std" in config_models:
            sections.append(gen_f1_by_depth_table(
                config_models["freeform_std"], sensitivity_cache, pattern_agents, ci_level,
            ))

    if 4 in tables:
        bias_groups = {}
        if "freeform_std" in config_models:
            bias_groups["freeform_std"] = config_models["freeform_std"]
        all_freeform = []
        for k, v in config_models.items():
            if k.startswith("freeform_"):
                all_freeform.extend(v)
        if all_freeform:
            unique = {}
            for m in all_freeform:
                key = m["model_dir"]
                if key not in unique:
                    unique[key] = m
                else:
                    unique[key]["agent_results"].update(m["agent_results"])
            bias_groups["all_freeform"] = list(unique.values())
        if bias_groups:
            sections.append(gen_field_bias_table(bias_groups, pattern_agents))

    md = "\n\n".join(sections) + "\n"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"Written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
