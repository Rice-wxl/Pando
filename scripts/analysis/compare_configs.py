import json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import t as t_dist

BASE = Path("outputs/evaluations")

CONFIGS = [
    ("Struct Std", ["batch_20260215_141030", "batch_20260215_141028"]),
    ("Struct Rat", ["batch_20260215_141027", "batch_20260215_141030_rationale"]),
    ("Free Std",   ["batch_20260301_033718", "batch_20260301_033721"]),
    ("Free Rat",   ["batch_20260301_033720", "batch_20260301_033716"]),
    ("Nat Std",    ["batch_20260301_033719", "batch_20260301_033725"]),
    ("Nat Rat",    ["batch_20260301_033723", "batch_20260301_033726"]),
]

NAME_MAP = {
    "sample_then_llm_guess": "blackbox",
    "blackbox_prefill": "prefill",
    "sample_then_gradient_llm_v2": "gradient",
    "sample_then_relp_llm": "relp",
    "sample_then_logit_lens_llm": "logit_lens",
    "sample_then_logit_lens_field_llm": "logit_lens_field",
    "sample_then_sae_autointerp_llm": "sae_autointerp",
    "sample_then_sae_gradient_attribution_llm": "sae_gradient",
    "sample_then_sae_mean_diff_llm": "sae_mean_diff",
    "sample_then_sae_tfidf_llm": "sae_tfidf",
    "sample_then_residual_token_llm": "res_token",
    "sample_then_logreg": "logreg",
    "sample_then_majority": "majority",
    "sample_then_nn": "nn",
    "spread_then_nn": "nn_spread",
}

AGENTS_ORDER = [
    "blackbox", "gradient", "relp", "prefill",
    "logit_lens", "logit_lens_field",
    "sae_autointerp", "sae_gradient", "sae_mean_diff", "sae_tfidf",
    "res_token",
    "nn", "nn_spread", "logreg", "majority",
]

def load_batch(batch_dir):
    results = defaultdict(lambda: defaultdict(list))
    for model_dir in sorted(batch_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        config_path = model_dir / "config.json"
        if not config_path.exists():
            continue
        with open(config_path) as f:
            config = json.load(f)
        nf = config.get("num_fields", 0)
        if nf == 0:
            continue
        agent_dir = model_dir / "agent_results"
        if not agent_dir.exists():
            continue
        for af in agent_dir.glob("*.json"):
            name = NAME_MAP.get(af.stem, af.stem)
            with open(af) as f:
                acc = json.load(f).get("accuracy", 0.0)
            results[nf][name].append(acc)
    return results

def merge(*results_list):
    merged = defaultdict(lambda: defaultdict(list))
    for r in results_list:
        for nf, agents in r.items():
            for agent, accs in agents.items():
                merged[nf][agent].extend(accs)
    return merged

def ci90(vals):
    n = len(vals)
    if n < 2:
        return 0.0
    return t_dist.ppf(0.95, df=n-1) * np.std(vals, ddof=1) / np.sqrt(n)

# Load all configs
config_data = {}
for label, batches in CONFIGS:
    rs = [load_batch(BASE / b) for b in batches]
    config_data[label] = merge(*rs)

# Compute pooled mean+ci for each agent x config
table = {}  # (agent, config_label) -> (mean, ci, n)
blackbox_vals = {}

for label, results in config_data.items():
    for agent in AGENTS_ORDER:
        all_accs = []
        for nf in sorted(results.keys()):
            accs = results[nf].get(agent, [])
            all_accs.extend(accs)
        if all_accs:
            m = np.mean(all_accs)
            c = ci90(all_accs)
            table[(agent, label)] = (m, c, len(all_accs))
        if agent == "blackbox" and all_accs:
            blackbox_vals[label] = (np.mean(all_accs), ci90(all_accs))

# ANSI codes
BOLD = "\033[1m"
RESET = "\033[0m"
HIGHLIGHT = "\033[40m\033[97m"

# Find best agent per config (excluding majority)
config_labels = [c[0] for c in CONFIGS]
best_per_config = {}
for label in config_labels:
    best_agent = None
    best_mean = -1
    for agent in AGENTS_ORDER:
        if agent == "majority":
            continue
        key = (agent, label)
        if key in table and table[key][0] > best_mean:
            best_mean = table[key][0]
            best_agent = agent
    best_per_config[label] = best_agent

# Compute mean-of-six for each agent
agent_mean6 = {}
for agent in AGENTS_ORDER:
    vals = []
    for label in config_labels:
        key = (agent, label)
        if key in table:
            vals.append(table[key][0])
    if vals:
        agent_mean6[agent] = np.mean(vals)

# Find best mean6 (excluding majority)
best_mean6_agent = max(
    (a for a in AGENTS_ORDER if a != "majority" and a in agent_mean6),
    key=lambda a: agent_mean6[a]
)

# Print table
hdr = f"{'Agent':<18}"
for label in config_labels:
    hdr += f" | {label:>15}"
hdr += f" | {'Mean':>10}"
print(hdr)
print("-" * len(hdr))

for agent in AGENTS_ORDER:
    row = f"{agent:<18}"
    for label in config_labels:
        key = (agent, label)
        if key not in table:
            row += f" | {'—':>15}"
            continue
        m, c, n = table[key]
        cell = f"{m*100:.1f}±{c*100:.1f}"

        is_best = (best_per_config[label] == agent)
        bb_m, bb_c = blackbox_vals[label]
        did_not_beat = (m <= bb_m) and agent != "blackbox"

        if is_best:
            cell = f"{BOLD}{cell}{RESET}"
        if did_not_beat:
            cell = f"{HIGHLIGHT}{cell}{RESET}"

        visible_len = len(f"{m*100:.1f}±{c*100:.1f}")
        pad = 15 - visible_len
        row += f" | {' ' * pad}{cell}"

    # Mean-of-six column
    if agent in agent_mean6:
        mv = agent_mean6[agent]
        mcell = f"{mv*100:.1f}"
        is_best_m6 = (agent == best_mean6_agent)
        if is_best_m6:
            mcell = f"{BOLD}{mcell}{RESET}"
            pad = 10 - len(f"{mv*100:.1f}")
            row += f" | {' ' * pad}{mcell}"
        else:
            row += f" | {mcell:>10}"
    else:
        row += f" | {'—':>10}"
    print(row)

print()
print(f"Legend: {BOLD}bold{RESET} = best; {HIGHLIGHT} highlighted {RESET} = did not beat blackbox")
print("Per-config: pooled mean ± 90% CI (n≈80). Mean: average of 6 config means.")
