#!/usr/bin/env python3
"""Generate comprehensive markdown tables across multiple scenarios.

Organized by scenario group, generates accuracy, field F1/precision/recall,
and distractor analysis tables for each scenario's freeform configs.

Reuses data loading, sensitivity caching, field matching, and markdown table
helpers from megatable.py.

Usage:
    python scripts/analysis/generate_tables.py
    python scripts/analysis/generate_tables.py --scenario car_purchase movie_pick
    python scripts/analysis/generate_tables.py --agents gradient relp prefill
    python scripts/analysis/generate_tables.py --table 1 2 5
    python scripts/analysis/generate_tables.py --no-ci
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
from scripts.analysis.megatable import compute_pattern_metrics

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
    # Data mixing: freeform + Dolci-Instruct-SFT or FineWeb at 0.25 ratio
    # (car_purchase only, d1-d3 only — d4 failed validation under mixing)
    "freeform_dolci": [
        "batch_20260318_012756", "batch_20260318_012905",
    ],
    "freeform_fineweb": [
        "batch_20260318_013143", "batch_20260318_013402",
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

# FIELD_ALIASES removed — imported via megatable._field_mentioned_stripped

# ─── Scenario group definitions ──────────────────────────────────────────────

SCENARIO_GROUPS = {
    "car_purchase": {
        "std": "freeform_std",
        "goodrat": "freeform_goodrat",
        "badrat": "freeform_badrat",
    },
    "movie_pick": {
        "std": "mp_freeform_std",
        "goodrat": "mp_freeform_goodrat",
        "badrat": "mp_freeform_badrat",
    },
    "oversight_defection": {
        "std": "od_freeform_std",
        "goodrat": "od_freeform_goodrat",
        "badrat": "od_freeform_badrat",
    },
}


# ─── Utility ─────────────────────────────────────────────────────────────────

def _ci_half_width(values: list[float], level: int = 90) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    t_val = t_dist.ppf(1 - (1 - level / 100) / 2, df=n - 1)
    return t_val * np.std(values, ddof=1) / np.sqrt(n)



# _strip_parenthesized and _field_mentioned removed — imported from megatable


def _pct(val, ci=None, *, percent: bool = True, na_str: str = "---") -> str:
    """Format a percentage for a markdown cell.

    Args:
        val: Fraction in [0, 1], or None for missing data.
        ci: Optional CI half-width (also a fraction).
        percent: Append "%" suffix. Default True; set False for tables
            that render bare numbers (format robustness, data mixing).
        na_str: String to return when val is None.
    """
    if val is None:
        return na_str
    suffix = "%" if percent else ""
    if ci is None:
        return f"{val*100:.1f}{suffix}"
    return f"{val*100:.1f}\u00b1{ci*100:.1f}{suffix}"


def _cell(vals: list[float], ci_level: int | None, **fmt_kwargs) -> str:
    """Compute mean/CI from a list of per-model values and format it.

    Returns the `na_str` (default "---") if `vals` is empty.
    `**fmt_kwargs` are forwarded to `_pct` (e.g. `percent=False`, `na_str="—"`).
    """
    if not vals:
        return _pct(None, **fmt_kwargs)
    mean = float(np.mean(vals))
    ci = _ci_half_width(vals, ci_level) if ci_level and len(vals) >= 2 else None
    return _pct(mean, ci, **fmt_kwargs)


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



# compute_pattern_metrics removed — imported from megatable (uses global-max F1)


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


# ─── Helper to collect metrics across depths ─────────────────────────────────

def _collect_metric_vals(metrics: dict, agent: str, metric_name: str = "f1") -> list[float]:
    """Collect all values for a given metric for an agent across depths."""
    if agent not in metrics:
        return []
    vals = []
    for d in sorted(metrics[agent].keys()):
        vals.extend(metrics[agent][d].get(metric_name, []))
    return vals


def _collect_metric_by_model(metrics: dict, agent: str, metric_name: str = "f1") -> dict[str, float]:
    """Collect per-model metric values keyed by model_dir for paired tests."""
    if agent not in metrics:
        return {}
    result = {}
    for d in sorted(metrics[agent].keys()):
        for val, model_dir in zip(
            metrics[agent][d].get(metric_name, []),
            metrics[agent][d].get("model_dirs", []),
        ):
            result[model_dir] = val
    return result


# ─── Held-out accuracy ──────────────────────────────────────────────────────

INPUT_RE = re.compile(r'Input: \{([^}]+)\}')


def _parse_input_str(s: str) -> dict[str, str]:
    vals = {}
    for pair in s.split(', '):
        if '=' in pair:
            k, v = pair.split('=', 1)
            vals[k.strip()] = v.strip()
    return vals


def _get_queried_indices(result: dict) -> set[int]:
    """Get indices of test samples that were queried by the agent.

    Uses queried_indices from metadata if available (non-LLM agents),
    otherwise parses Input: lines from pattern_prompt and matches to test set.
    """
    meta = result.get("agent_metadata", {})

    # Direct field (non-LLM agents: nn, majority, logreg, nn_spread)
    qi = meta.get("queried_indices")
    if qi is not None:
        return set(qi)

    # Parse from pattern_prompt (LLM agents)
    pp = meta.get("pattern_prompt", "")
    pir = result.get("per_input_results", [])
    if not pp or not pir:
        return set()

    queried_raw = INPUT_RE.findall(pp)
    queried_unique = list(set(queried_raw))

    indices = set()
    for qi_str in queried_unique:
        qi_dict = _parse_input_str(qi_str)
        for ti in pir:
            test_dict = {k: str(v) for k, v in ti["input"].items()}
            if all(qi_dict.get(k) == test_dict.get(k) for k in qi_dict):
                indices.add(ti["index"])
                break
    return indices


def _held_out_accuracy(result: dict) -> float | None:
    """Compute accuracy on held-out (non-queried) test samples only."""
    pir = result.get("per_input_results", [])
    if not pir:
        return None
    queried = _get_queried_indices(result)
    held_out = [r for r in pir if r["index"] not in queried]
    if not held_out:
        return None
    return sum(1 for r in held_out if r["correct"]) / len(held_out)


# ─── Table generators ────────────────────────────────────────────────────────

def gen_accuracy_table(
    models: list[dict],
    agents: list[str],
    ci_level: int | None,
    config_label: str,
) -> str:
    """Generate held-out accuracy table for a single config.

    Accuracy is computed only on test samples NOT queried by the agent.
    Rows: agents, Columns: d1, d2, d3, d4, avg
    """
    acc_data = defaultdict(lambda: defaultdict(list))
    for m in models:
        depth = m["num_fields"]
        for agent_name, result in m["agent_results"].items():
            acc = _held_out_accuracy(result)
            if acc is not None:
                acc_data[agent_name][depth].append(acc)

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
                row.append("---")
        if all_vals:
            mean = np.mean(all_vals)
            ci = _ci_half_width(all_vals, ci_level) if ci_level and len(all_vals) >= 2 else None
            row.append(f"**{_pct(mean, ci)}**")
        else:
            row.append("---")
        rows.append(row)

    title = f"### Accuracy --- {config_label}\n"
    return title + _md_table(headers, rows)


def gen_field_metric_table(
    models: list[dict],
    sensitivity_cache: dict,
    agents: list[str],
    ci_level: int | None,
    config_label: str,
    metric_name: str = "f1",
    metric_display: str = "F1",
) -> str:
    """Generate a field metric (F1/precision/recall) table for real circuit fields.

    Rows: agents, Columns: d1, d2, d3, d4, avg
    """
    metrics = compute_pattern_metrics(
        models, sensitivity_cache, agents, ground_truth_mode="real",
    )
    depths = sorted({m["num_fields"] for m in models})
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
                row.append("---")
        if all_vals:
            mean = np.mean(all_vals)
            ci = _ci_half_width(all_vals, ci_level) if ci_level and len(all_vals) >= 2 else None
            row.append(f"**{_pct(mean, ci)}**")
        else:
            row.append("---")
        rows.append(row)

    title = f"### Real Circuit {metric_display} --- {config_label}\n"
    return title + _md_table(headers, rows)


def gen_distractor_table(
    models: list[dict],
    sensitivity_cache: dict,
    agents: list[str],
    ci_level: int | None,
    config_label: str,
) -> str:
    """Generate distractor F1 analysis table for badrat config.

    Columns: distractor F1, random F1, delta
    Marks with * when 90% CI of paired delta excludes zero.
    """
    distractor_metrics = compute_pattern_metrics(
        models, sensitivity_cache, agents, ground_truth_mode="distractor",
    )
    random_metrics = compute_pattern_metrics(
        models, sensitivity_cache, agents, ground_truth_mode="random",
    )

    headers = ["Agent", "distractor", "random", "delta", "p-value"]
    rows = []
    sig_level = ci_level or 90

    for agent in agents:
        dist_vals = _collect_metric_vals(distractor_metrics, agent, "f1")
        rand_vals = _collect_metric_vals(random_metrics, agent, "f1")
        if not dist_vals and not rand_vals:
            continue

        row = [f"`{agent}`"]

        # Distractor F1
        if dist_vals:
            dist_mean = np.mean(dist_vals)
            dist_ci = _ci_half_width(dist_vals, sig_level) if len(dist_vals) >= 2 else None
            row.append(_pct(dist_mean, dist_ci))
        else:
            row.append("---")

        # Random F1
        if rand_vals:
            rand_mean = np.mean(rand_vals)
            rand_ci = _ci_half_width(rand_vals, sig_level) if len(rand_vals) >= 2 else None
            row.append(_pct(rand_mean, rand_ci))
        else:
            row.append("---")

        # Delta (paired) with significance test
        dist_by_model = _collect_metric_by_model(distractor_metrics, agent, "f1")
        rand_by_model = _collect_metric_by_model(random_metrics, agent, "f1")
        shared_keys = sorted(set(dist_by_model) & set(rand_by_model))

        if len(shared_keys) >= 2:
            diffs = [dist_by_model[k] - rand_by_model[k] for k in shared_keys]
            diff_mean = np.mean(diffs)
            diff_ci = _ci_half_width(diffs, sig_level)
            sig_marker = ""
            if diff_mean - diff_ci > 0 or diff_mean + diff_ci < 0:
                sig_marker = " \\*"
            sign = "+" if diff_mean >= 0 else ""
            row.append(f"{sign}{diff_mean*100:.1f}pp{sig_marker}")

            # One-sided paired t-test: H1: distractor > random
            from scipy.stats import ttest_rel
            _, p_two = ttest_rel(
                [dist_by_model[k] for k in shared_keys],
                [rand_by_model[k] for k in shared_keys],
            )
            # Convert to one-sided p-value
            p_one = p_two / 2 if diff_mean > 0 else 1 - p_two / 2
            if p_one < 0.001:
                row.append(f"<.001")
            else:
                row.append(f"{p_one:.3f}")
        elif dist_vals and rand_vals:
            diff_mean = np.mean(dist_vals) - np.mean(rand_vals)
            sign = "+" if diff_mean >= 0 else ""
            row.append(f"{sign}{diff_mean*100:.1f}pp")
            row.append("---")
        else:
            row.append("---")
            row.append("---")

        rows.append(row)

    title = f"### Distractor F1 --- {config_label}\n"
    table = _md_table(headers, rows)
    footer = (f"\n\\* = {sig_level}% CI of paired difference (distractor - random) excludes zero\n"
              f"\np-value: one-sided paired t-test, H₁: F1(distractor) > F1(random)\n")
    return title + table + footer


# ─── Table 6: Format robustness ─────────────────────────────────────────────

# Agent display order for the robustness tables (Tables 8 & 9 in the paper).
# Differs from AGENTS_ORDER above: no codex_read / sae_tfidf_filtered /
# circuit_tracer_filtered, and `sae_autointerp` is rendered as `sae_raw`.
ROBUSTNESS_AGENTS_ORDER = [
    "relp", "gradient", "prefill",
    "sae_gradient", "logit_lens", "logit_lens_field", "res_token",
    "sae_tfidf", "sae_autointerp", "sae_mean_diff",
    "blackbox",
    "nn", "logreg", "majority",
]

# Display-name overrides for robustness tables (internal name → paper name)
ROBUSTNESS_DISPLAY_NAME = {
    "sae_autointerp": "sae_raw",
}


def _held_out_acc_for_agent(models: list[dict], agent: str,
                            depth_filter: set[int] | None = None) -> list[float]:
    """Collect per-model held-out accuracies for one agent, optionally
    filtered to a subset of depths (used for data-mixing's d1-d3 restriction)."""
    accs = []
    for m in models:
        if depth_filter is not None and m.get("num_fields") not in depth_filter:
            continue
        result = m["agent_results"].get(agent)
        if not result:
            continue
        acc = _held_out_accuracy(result)
        if acc is not None:
            accs.append(acc)
    return accs


# Format kwargs shared by the format-robustness and data-mixing generators:
# they render bare numbers (no `%`), use an em-dash for missing data, and
# otherwise reuse the standard `_pct()` rounding for consistency with every
# other table in this file.
_ROBUSTNESS_FMT = {"percent": False, "na_str": "—"}


def gen_format_robustness_table(
    configs_by_variant: dict[str, dict[str, list[dict]]],
    agents: list[str],
    ci_level: int | None,
) -> str:
    """Generate the format robustness table (Table 8 in the paper).

    configs_by_variant: {variant: {format_name: list_of_models}} where
        variant ∈ {"std", "goodrat", "badrat"}
        format_name ∈ {"Freeform", "Natural", "Structured"}
    """
    sig_level = ci_level or 90
    sections = [
        "# Format Robustness: Held-out Accuracy (%)\n",
        "Car purchase only. 90% CIs. Freeform is the main setup (shown in body tables).",
        "Natural and structured are format variations evaluated on the same models.\n",
    ]

    format_cols = ["Freeform", "Natural", "Structured"]

    for variant in ["std", "goodrat", "badrat"]:
        format_map = configs_by_variant.get(variant, {})
        if not any(format_map.get(f) for f in format_cols):
            continue
        sections.append(f"### {variant}\n")

        # Header row
        sections.append("| Agent | " + " | ".join(format_cols) + " |")
        sections.append("|-------|" + "|".join(["-" * (len(c) + 2) for c in format_cols]) + "|")

        for agent in agents:
            display = ROBUSTNESS_DISPLAY_NAME.get(agent, agent)
            cells = [display]
            for fname in format_cols:
                accs = _held_out_acc_for_agent(format_map.get(fname, []), agent)
                cells.append(_cell(accs, sig_level, **_ROBUSTNESS_FMT))
            sections.append("| " + " | ".join(cells) + " |")
        sections.append("")  # blank line after table

    return "\n".join(sections)


# ─── Table 7: Data mixing ───────────────────────────────────────────────────

def gen_data_mixing_table(
    configs: dict[str, tuple[list[dict], int]],
    agents: list[str],
    ci_level: int | None,
) -> str:
    """Generate the data mixing table (Table 9 in the paper).

    configs: {section_title: (models, n_for_label)}
        Section titles must match the paper: e.g.
            "std (no mixing)"
            "Dolci 0.25"
            "FineWeb 0.25"
        Models are filtered to d1-d3 only inside this function.
    """
    sig_level = ci_level or 90
    depth_filter = {1, 2, 3}

    sections = [
        "# Data Mixing: Held-out Accuracy (%)\n",
        "Car purchase, freeform training, no verbalization. 90% CIs.",
        "**d1-d3 only** — d4 failed validation with data mixing (Dolci: 1 valid, FineWeb: 0 valid).",
        "Freeform std baseline restricted to d1-d3 for fair comparison.\n",
    ]

    depths = [1, 2, 3]

    for title, (models, _n_label) in configs.items():
        # Count actually-filtered models for the section header
        filtered_count = sum(
            1 for m in models if m.get("num_fields") in depth_filter
        )
        sections.append(f"### {title} (n={filtered_count}, d1-d3 only)\n")
        sections.append("| Agent | d1 | d2 | d3 | avg |")
        sections.append("|-------|-----|-----|-----|-----|")

        for agent in agents:
            display = ROBUSTNESS_DISPLAY_NAME.get(agent, agent)
            cells = [display]
            # Per-depth columns
            for d in depths:
                accs_d = _held_out_acc_for_agent(models, agent, depth_filter={d})
                cells.append(_cell(accs_d, sig_level, **_ROBUSTNESS_FMT))
            # Avg column (pooled over all d1-d3 per-model accuracies), bolded
            accs_all = _held_out_acc_for_agent(models, agent, depth_filter=depth_filter)
            cells.append(f"**{_cell(accs_all, sig_level, **_ROBUSTNESS_FMT)}**")
            sections.append("| " + " | ".join(cells) + " |")
        sections.append("")  # blank line

    return "\n".join(sections)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive markdown tables across multiple scenarios"
    )
    parser.add_argument(
        "--agents", type=str, nargs="+", default=None,
        help="Filter to specific agents",
    )
    parser.add_argument(
        "--ci", type=int, default=90,
        help="CI level (default: 90)",
    )
    parser.add_argument(
        "--no-ci", action="store_true",
        help="Disable CI",
    )
    parser.add_argument(
        "--recache", action="store_true",
        help="Force recompute sensitivity cache",
    )
    parser.add_argument(
        "--table", type=int, nargs="+", default=None,
        help="Only print specific tables (1=accuracy, 2=F1, 3=precision, 4=recall, "
             "5=distractor, 6=format_robustness, 7=data_mixing). "
             "Tables 6 and 7 are opt-in only (not included in the default set).",
    )
    parser.add_argument(
        "--scenario", type=str, nargs="+", default=None,
        help="Filter to specific scenarios (e.g. car_purchase movie_pick oversight_defection)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="outputs/tables.md",
        help="Output markdown file (default: outputs/tables.md)",
    )
    args = parser.parse_args()

    ci_level = None if args.no_ci else args.ci
    tables = set(args.table) if args.table else {1, 2, 3, 4, 5}

    # Determine which scenario groups to include
    if args.scenario:
        scenario_groups = {
            s: SCENARIO_GROUPS[s]
            for s in args.scenario
            if s in SCENARIO_GROUPS
        }
        if not scenario_groups:
            print(
                f"ERROR: none of {args.scenario} found in scenario groups. "
                f"Available: {list(SCENARIO_GROUPS.keys())}",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        scenario_groups = SCENARIO_GROUPS

    # Collect which batch dir configs we actually need
    needed_configs = set()
    for scenario_name, variant_map in scenario_groups.items():
        for variant, config_key in variant_map.items():
            needed_configs.add(config_key)

    # Load models for needed configs only
    print("Loading models...", file=sys.stderr)
    config_models: dict[str, list[dict]] = {}
    for config_name in needed_configs:
        if config_name not in BATCH_DIRS:
            print(f"  WARNING: config {config_name} not in BATCH_DIRS, skipping", file=sys.stderr)
            continue
        models = load_models(BATCH_DIRS[config_name])
        if models:
            config_models[config_name] = models
            depths = defaultdict(int)
            for m in models:
                depths[m["num_fields"]] += 1
            depth_str = ", ".join(f"d{d}={n}" for d, n in sorted(depths.items()))
            print(f"  {config_name}: {len(models)} models ({depth_str})", file=sys.stderr)
        else:
            print(f"  {config_name}: NO MODELS FOUND", file=sys.stderr)

    # Determine agent list from all loaded models
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

    # Pattern agents exclude non-LLM baselines
    pattern_agents = [
        a for a in agents
        if a not in ("nn", "nn_spread", "logreg", "majority", "always_true", "always_false")
    ]

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
        print(
            f"  Computing {len(new_exprs)} new sensitivities (cached: {cached_before})...",
            file=sys.stderr,
        )
    else:
        print(f"  All {len(unique_exprs)} expressions cached", file=sys.stderr)
    sensitivity_cache = get_sensitivities(all_models_flat, sensitivity_cache)

    # ─── Pool models across scenarios per variant ──────────────────────

    # For each variant (std/goodrat/badrat), merge all scenario models
    variant_models: dict[str, list[dict]] = {"std": [], "goodrat": [], "badrat": []}
    scenario_counts: dict[str, dict[str, int]] = {"std": {}, "goodrat": {}, "badrat": {}}

    for scenario_name, variant_map in scenario_groups.items():
        for variant, config_key in variant_map.items():
            models = config_models.get(config_key, [])
            variant_models[variant].extend(models)
            if models:
                scenario_counts[variant][scenario_name] = len(models)

    # ─── Generate markdown ───────────────────────────────────────────

    sections = ["# Comprehensive Benchmark Tables\n"]
    n_scenarios = len(scenario_groups)
    scenario_list = ", ".join(scenario_groups.keys())
    sections.append(f"Aggregated across {n_scenarios} scenarios: {scenario_list}\n")

    def _variant_label(variant: str) -> str:
        models = variant_models[variant]
        counts = scenario_counts[variant]
        count_str = " + ".join(f"{s}={n}" for s, n in sorted(counts.items()))
        return f"freeform_{variant} (n={len(models)}: {count_str})"

    # ─── Table 1: Accuracy ───────────────────────────────────────
    if 1 in tables:
        for variant in ["std", "goodrat", "badrat"]:
            if variant_models[variant]:
                sections.append(
                    gen_accuracy_table(variant_models[variant], agents, ci_level, _variant_label(variant))
                )

    # ─── Table 2: Real Circuit F1 ────────────────────────────────
    if 2 in tables:
        for variant in ["std", "goodrat", "badrat"]:
            if variant_models[variant]:
                sections.append(
                    gen_field_metric_table(
                        variant_models[variant], sensitivity_cache, pattern_agents, ci_level,
                        _variant_label(variant), metric_name="f1", metric_display="F1",
                    )
                )

    # ─── Table 3: Real Circuit Precision ─────────────────────────
    if 3 in tables:
        for variant in ["std", "goodrat", "badrat"]:
            if variant_models[variant]:
                sections.append(
                    gen_field_metric_table(
                        variant_models[variant], sensitivity_cache, pattern_agents, ci_level,
                        _variant_label(variant), metric_name="precision", metric_display="Precision",
                    )
                )

    # ─── Table 4: Real Circuit Recall ────────────────────────────
    if 4 in tables:
        for variant in ["std", "goodrat", "badrat"]:
            if variant_models[variant]:
                sections.append(
                    gen_field_metric_table(
                        variant_models[variant], sensitivity_cache, pattern_agents, ci_level,
                        _variant_label(variant), metric_name="recall", metric_display="Recall",
                    )
                )

    # ─── Table 5: Distractor F1 (badrat only) ───────────────────
    if 5 in tables:
        if variant_models["badrat"]:
            sections.append(
                gen_distractor_table(
                    variant_models["badrat"], sensitivity_cache, pattern_agents, ci_level,
                    _variant_label("badrat"),
                )
            )

    # ─── Table 6: Format robustness (car_purchase only) ─────────
    # Uses freeform_/natural_/structured_ configs with 3 explanation
    # setups = 9 configs. These are car_purchase only (the natural and
    # structured configs in BATCH_DIRS don't have movie_pick or
    # oversight_defection variants).
    if 6 in tables:
        FORMAT_CONFIG_MAP = {
            "std":     {"Freeform": "freeform_std",
                        "Natural": "natural_std",
                        "Structured": "structured_std"},
            "goodrat": {"Freeform": "freeform_goodrat",
                        "Natural": "natural_goodrat",
                        "Structured": "structured_goodrat"},
            "badrat":  {"Freeform": "freeform_badrat",
                        "Natural": "natural_badrat",
                        "Structured": "structured_badrat"},
        }
        configs_by_variant: dict[str, dict[str, list[dict]]] = {}
        for variant, fmt_map in FORMAT_CONFIG_MAP.items():
            configs_by_variant[variant] = {}
            for fmt_name, cfg_key in fmt_map.items():
                if cfg_key not in BATCH_DIRS:
                    continue
                if cfg_key in config_models:
                    models = config_models[cfg_key]
                else:
                    models = load_models(BATCH_DIRS[cfg_key])
                    config_models[cfg_key] = models
                configs_by_variant[variant][fmt_name] = models

        # Use the robustness-specific agent order (filtered to whatever is
        # present in the loaded data).
        present_agents = set()
        for fmt_map in configs_by_variant.values():
            for models in fmt_map.values():
                for m in models:
                    present_agents.update(m["agent_results"].keys())
        robustness_agents = [
            a for a in ROBUSTNESS_AGENTS_ORDER if a in present_agents
        ]

        sections.append(
            gen_format_robustness_table(configs_by_variant, robustness_agents, ci_level)
        )

    # ─── Table 7: Data mixing (car_purchase, freeform_std, d1-d3) ──
    if 7 in tables:
        MIXING_CONFIG_MAP = {
            "std (no mixing)": "freeform_std",
            "Dolci 0.25":      "freeform_dolci",
            "FineWeb 0.25":    "freeform_fineweb",
        }
        mixing_configs: dict[str, tuple[list[dict], int]] = {}
        for title, cfg_key in MIXING_CONFIG_MAP.items():
            if cfg_key not in BATCH_DIRS:
                continue
            if cfg_key in config_models:
                models = config_models[cfg_key]
            else:
                models = load_models(BATCH_DIRS[cfg_key])
                config_models[cfg_key] = models
            mixing_configs[title] = (models, len(models))

        present_agents = set()
        for models, _ in mixing_configs.values():
            for m in models:
                present_agents.update(m["agent_results"].keys())
        robustness_agents = [
            a for a in ROBUSTNESS_AGENTS_ORDER if a in present_agents
        ]

        sections.append(
            gen_data_mixing_table(mixing_configs, robustness_agents, ci_level)
        )

    md = "\n\n".join(sections) + "\n"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"\nWritten to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
