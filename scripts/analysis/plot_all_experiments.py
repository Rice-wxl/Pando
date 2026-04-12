#!/usr/bin/env python3
"""Plot results for all three experiments.

Creates three plots:
1. ALL FIELDS: Accuracy vs Depth (10 fields shown)
2. USED FIELDS ONLY: Accuracy vs Depth (d fields shown)
3. FEATURE SELECTION: Accuracy vs Fields Shown (depth=1)

Usage:
    python scripts/analysis/plot_all_experiments.py
    python scripts/analysis/plot_all_experiments.py --output plots/results.png
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def get_experiment_type(model_name: str) -> str:
    """Determine experiment type from model name."""
    if os.path.exists(f'outputs/models_usedfields/{model_name}'):
        return 'usedfields'
    elif os.path.exists(f'outputs/models_featsel/{model_name}'):
        return 'featsel'
    else:
        return 'allfields'


def load_all_results(eval_dir: Path) -> dict:
    """Load all evaluation results.

    Returns:
        Dict with structure:
        {
            'allfields': {depth: {agent: [accuracies]}},
            'usedfields': {depth: {agent: [accuracies]}},
            'featsel': {n_fields: {agent: [accuracies]}},
        }
    """
    results = {
        'allfields': defaultdict(lambda: defaultdict(list)),
        'usedfields': defaultdict(lambda: defaultdict(list)),
        'featsel': defaultdict(lambda: defaultdict(list)),
    }

    for batch_dir in sorted(eval_dir.iterdir()):
        if not batch_dir.is_dir():
            continue

        for model_dir in batch_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name
            exp_type = get_experiment_type(model_name)

            # Extract depth and n_fields from model name
            depth_match = re.search(r'_d(\d+)_', model_name)
            depth = int(depth_match.group(1)) if depth_match else 0

            n_match = re.search(r'_n(\d+)_', model_name)
            n_fields = int(n_match.group(1)) if n_match else 10

            # Load agent results
            agent_results_dir = model_dir / 'agent_results'
            if not agent_results_dir.exists():
                continue

            for agent_file in agent_results_dir.glob('*.json'):
                agent_name = agent_file.stem
                with open(agent_file) as f:
                    data = json.load(f)
                accuracy = data.get('accuracy', 0.0)

                if exp_type == 'featsel':
                    results[exp_type][n_fields][agent_name].append(accuracy)
                else:
                    results[exp_type][depth][agent_name].append(accuracy)

    return results


def plot_experiment(
    ax,
    data: dict,
    agents: list[str],
    title: str,
    xlabel: str,
    colors: dict,
):
    """Plot a single experiment on given axes."""
    x_values = sorted(data.keys())

    if not x_values:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return

    x = np.arange(len(x_values))
    n_agents = len(agents)
    bar_width = 0.8 / n_agents

    np.random.seed(42)

    for i, agent in enumerate(agents):
        means = []
        stds = []
        all_points = []

        for j, xv in enumerate(x_values):
            accuracies = data[xv].get(agent, [])
            if accuracies:
                means.append(np.mean(accuracies))
                stds.append(np.std(accuracies) if len(accuracies) > 1 else 0)
                # Scatter points
                x_pos = x[j] + i * bar_width - (n_agents - 1) * bar_width / 2
                for acc in accuracies:
                    jitter = np.random.uniform(-bar_width * 0.3, bar_width * 0.3)
                    all_points.append((x_pos + jitter, acc))
            else:
                means.append(np.nan)
                stds.append(0)

        # Plot bars with error bars
        bar_positions = x + i * bar_width - (n_agents - 1) * bar_width / 2
        ax.bar(
            bar_positions,
            means,
            bar_width * 0.9,
            label=agent.replace('_', ' '),
            color=colors.get(agent, 'gray'),
            alpha=0.7,
            yerr=stds,
            capsize=2,
        )

        # Plot individual dots
        if all_points:
            xs, ys = zip(*all_points)
            ax.scatter(
                xs, ys,
                color=colors.get(agent, 'gray'),
                s=20,
                alpha=0.8,
                edgecolors='black',
                linewidths=0.5,
                zorder=3,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(xv) for xv in x_values])
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(axis='y', alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description="Plot all experiment results")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save plot to file (e.g., plots/results.png)",
    )
    parser.add_argument(
        "--agents",
        type=str,
        nargs="+",
        default=['nn', 'nn_spread', 'majority'],
        help="Agents to plot",
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default="outputs/evaluations",
        help="Evaluation directory",
    )

    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        print(f"Error: Evaluation directory not found: {eval_dir}")
        return

    print("Loading results...")
    results = load_all_results(eval_dir)

    # Count models
    for exp in results:
        n_models = sum(
            len(next(iter(agents.values()), []))
            for agents in results[exp].values()
            if agents
        )
        print(f"  {exp}: {n_models} models")

    # Define colors for agents
    colors = {
        'nn': '#1f77b4',
        'nn_spread': '#ff7f0e',
        'majority': '#2ca02c',
        'logreg': '#d62728',
        'always_true': '#9467bd',
        'always_false': '#8c564b',
    }

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: ALL FIELDS
    plot_experiment(
        axes[0],
        dict(results['allfields']),
        args.agents,
        'ALL FIELDS\n(10 fields shown)',
        'Depth',
        colors,
    )

    # Plot 2: USED FIELDS ONLY
    plot_experiment(
        axes[1],
        dict(results['usedfields']),
        args.agents,
        'USED FIELDS ONLY\n(d fields shown)',
        'Depth',
        colors,
    )

    # Plot 3: FEATURE SELECTION
    plot_experiment(
        axes[2],
        dict(results['featsel']),
        args.agents,
        'FEATURE SELECTION\n(depth=1, varying fields)',
        'Fields Shown',
        colors,
    )

    # Add legend to last plot
    axes[2].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)

    plt.tight_layout()

    if args.output:
        # Create directory if needed
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {args.output}")
    else:
        plt.savefig('outputs/all_experiments_plot.png', dpi=150, bbox_inches='tight')
        print("Saved plot to outputs/all_experiments_plot.png")


if __name__ == "__main__":
    main()
