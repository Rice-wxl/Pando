#!/usr/bin/env python3
"""Analyze batch evaluation results.

Usage:
    python scripts/analysis/analyze_batch.py outputs/evaluations/batch_20260101_120000

    # Multiple batches (results are merged)
    python scripts/analysis/analyze_batch.py batch_1 batch_2 batch_3

    # Save plot to file
    python scripts/analysis/analyze_batch.py <batch_dir> --output plot.png

    # Filter agents
    python scripts/analysis/analyze_batch.py <batch_dir> --agents nn nn_spread

    # Confidence intervals (default: 90% CI)
    python scripts/analysis/analyze_batch.py <batch_dir> --ci 95
    python scripts/analysis/analyze_batch.py <batch_dir> --no-ci
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_batch_results(batch_dir: Path) -> dict:
    """Load all results from a batch directory.

    Returns:
        Dict with structure:
        {
            num_fields: {
                agent_name: [accuracy1, accuracy2, ...]  # one per model
            }
        }
    """
    results = defaultdict(lambda: defaultdict(list))

    for model_dir in sorted(batch_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        # Load config to get num_fields
        config_path = model_dir / "config.json"
        if not config_path.exists():
            continue

        with open(config_path) as f:
            config = json.load(f)

        num_fields = config.get("num_fields", 0)
        if num_fields == 0:
            continue

        # Load each agent's results
        agent_results_dir = model_dir / "agent_results"
        if not agent_results_dir.exists():
            continue

        for agent_file in agent_results_dir.glob("*.json"):
            agent_name = agent_file.stem
            with open(agent_file) as f:
                agent_data = json.load(f)
            accuracy = agent_data.get("accuracy", 0.0)
            results[num_fields][agent_name].append(accuracy)

    return dict(results)


def merge_results(*results_list: dict) -> dict:
    """Merge multiple results dicts by concatenating accuracy lists."""
    merged = defaultdict(lambda: defaultdict(list))
    for results in results_list:
        for nf, agents in results.items():
            for agent, accs in agents.items():
                merged[nf][agent].extend(accs)
    return dict(merged)


def plot_results(
    results: dict,
    agents_filter: list[str] | None = None,
    output_path: str | None = None,
    title: str = "Agent Accuracy by Complexity",
):
    """Create bar plot with individual model dots.

    Args:
        results: Dict from load_batch_results.
        agents_filter: Only show these agents (None = all).
        output_path: Save to file (None = show interactively).
        title: Plot title.
    """
    # Get sorted num_fields and agents
    num_fields_list = sorted(results.keys())

    # Collect all agents across all num_fields
    all_agents = set()
    for nf_data in results.values():
        all_agents.update(nf_data.keys())

    if agents_filter:
        all_agents = [a for a in agents_filter if a in all_agents]
    else:
        all_agents = sorted(all_agents)

    if not all_agents or not num_fields_list:
        print("No data to plot")
        return

    # Seed for reproducible jitter
    np.random.seed(42)

    # Setup plot
    n_groups = len(num_fields_list)
    n_agents = len(all_agents)

    fig, ax = plt.subplots(figsize=(max(10, n_groups * 2), 6))

    # Bar width and positions
    bar_width = 0.8 / n_agents
    x = np.arange(n_groups)

    # Colors for agents
    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))

    for i, agent in enumerate(all_agents):
        means = []
        all_points = []  # (x_pos, accuracy) for scatter

        for j, nf in enumerate(num_fields_list):
            accuracies = results[nf].get(agent, [])
            if accuracies:
                means.append(np.mean(accuracies))
                # Scatter points with small jitter
                x_pos = x[j] + i * bar_width - (n_agents - 1) * bar_width / 2
                for acc in accuracies:
                    jitter = np.random.uniform(-bar_width * 0.3, bar_width * 0.3)
                    all_points.append((x_pos + jitter, acc))
            else:
                means.append(np.nan)  # No bar for missing data

        # Plot bars
        bar_positions = x + i * bar_width - (n_agents - 1) * bar_width / 2
        bars = ax.bar(
            bar_positions,
            means,
            bar_width * 0.9,
            label=agent,
            color=colors[i],
            alpha=0.7,
        )

        # Plot individual dots
        if all_points:
            xs, ys = zip(*all_points)
            ax.scatter(xs, ys, color=colors[i], s=30, alpha=0.8, edgecolors='black', linewidths=0.5, zorder=3)

    # Formatting
    ax.set_xlabel("Number of Fields")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{nf}f" for nf in num_fields_list])
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='random baseline')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def _ci_half_width(accuracies: list[float], level: int = 90) -> float:
    """Compute CI half-width: t * std / sqrt(n)."""
    from scipy.stats import t
    n = len(accuracies)
    if n < 2:
        return 0.0
    t_val = t.ppf(1 - (1 - level / 100) / 2, df=n - 1)
    return t_val * np.std(accuracies, ddof=1) / np.sqrt(n)


def print_summary(results: dict, agents_filter: list[str] | None = None, ci_level: int | None = 90):
    """Print text summary of results."""
    num_fields_list = sorted(results.keys())

    # Collect all agents
    all_agents = set()
    for nf_data in results.values():
        all_agents.update(nf_data.keys())

    if agents_filter:
        all_agents = [a for a in agents_filter if a in all_agents]
    else:
        all_agents = sorted(all_agents)

    # Header
    header = f"{'Agent':<30}"
    for nf in num_fields_list:
        header += f" | {nf}f"
    header += " | avg"
    print(header)
    print("-" * len(header))

    # Rows
    for agent in all_agents:
        row = f"{agent:<30}"
        all_accs = []  # pooled across depths for avg CI
        agent_means = []
        for nf in num_fields_list:
            accuracies = results[nf].get(agent, [])
            if accuracies:
                mean = np.mean(accuracies)
                agent_means.append(mean)
                all_accs.extend(accuracies)
                n_models = len(accuracies)
                if ci_level is not None:
                    ci = _ci_half_width(accuracies, ci_level)
                    row += f" | {mean:.1%}\u00b1{ci:.1%} ({n_models})"
                else:
                    row += f" | {mean:.1%} ({n_models})"
            else:
                row += " | -"

        if all_accs:
            overall_mean = np.mean(all_accs)
            if ci_level is not None and len(all_accs) >= 2:
                overall_ci = _ci_half_width(all_accs, ci_level)
                row += f" | {overall_mean:.1%}\u00b1{overall_ci:.1%}"
            else:
                row += f" | {overall_mean:.1%}"
        else:
            row += " | -"
        print(row)

    # Model counts (max across agents in case some agents missing from some models)
    print("-" * len(header))
    count_row = f"{'# models':<30}"
    for nf in num_fields_list:
        counts = [len(results[nf][agent]) for agent in all_agents if agent in results[nf]]
        count = max(counts) if counts else 0
        count_row += f" | {count}"
    print(count_row)


def main():
    parser = argparse.ArgumentParser(description="Analyze batch evaluation results")
    parser.add_argument(
        "batch_dirs",
        type=str,
        nargs="+",
        help="Path(s) to batch evaluation directory(ies). Multiple dirs are merged.",
    )
    parser.add_argument(
        "--agents",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific agents",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save plot to file (e.g., plot.png)",
    )
    parser.add_argument(
        "--ci",
        type=int,
        default=90,
        help="Confidence interval level (default: 90). Use --no-ci to disable.",
    )
    parser.add_argument(
        "--no-ci",
        action="store_true",
        help="Hide confidence intervals in summary table",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting, only print summary",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Agent Accuracy by Complexity",
        help="Plot title",
    )

    args = parser.parse_args()

    all_results = []
    if not args.no_ci and not (1 <= args.ci <= 99):
        print(f"Error: --ci must be between 1 and 99, got {args.ci}")
        return

    for bd in args.batch_dirs:
        batch_dir = Path(bd)
        if not batch_dir.exists():
            print(f"Error: Batch directory not found: {batch_dir}")
            return
        print(f"Loading results from {batch_dir}...")
        r = load_batch_results(batch_dir)
        if r:
            all_results.append(r)
        else:
            print(f"Warning: No results found in {batch_dir}, skipping.")

    if not all_results:
        print("No results found")
        return

    results = merge_results(*all_results) if len(all_results) > 1 else all_results[0]
    ci_level = None if args.no_ci else args.ci

    print()
    print_summary(results, args.agents, ci_level=ci_level)
    print()

    if not args.no_plot:
        plot_results(
            results,
            agents_filter=args.agents,
            output_path=args.output,
            title=args.title,
        )


if __name__ == "__main__":
    main()
