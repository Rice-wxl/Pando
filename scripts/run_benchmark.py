#!/usr/bin/env python3
"""Run full benchmark: train models at different complexity tiers and evaluate all agents.

Usage:
    # Train and evaluate (full pipeline)
    python scripts/run_benchmark.py --scenario car_purchase --tiers 1 2 3

    # Evaluate only (use existing models)
    python scripts/run_benchmark.py --scenario car_purchase --eval-only --model-dirs outputs/models/...

    # Train multiple models per tier for statistical significance
    python scripts/run_benchmark.py --scenario car_purchase --tiers 1 2 3 --models-per-tier 5

    # Use LoRA instead of full fine-tuning
    python scripts/run_benchmark.py --scenario car_purchase --tiers 1 2 3 --use-lora

Difficulty Tiers (DNF circuits with balanced leaves):
    Tier 1: t=1, w=1  (trivial - single predicate)
    Tier 2: t=2, w=2  (easy - 4 literals, ~44% positive)
    Tier 3: t=3, w=2  (medium - 6 literals, ~58% positive)
    Tier 4: t=5, w=3  (hard - 15 literals, ~49% positive)
    Tier 5: t=11, w=4 (very hard - 44 literals, ~50% positive)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents import list_agents
from src.scenarios import list_scenarios
from src.utils import get_timestamp


# Difficulty tiers: (num_clauses, clause_width)
# Selected to have ~40-60% positive rate with balanced leaves
DIFFICULTY_TIERS = {
    1: (1, 1),   # 50% - trivial (1 predicate)
    2: (2, 2),   # 44% - easy (4 literals)
    3: (3, 2),   # 58% - medium (6 literals)
    4: (5, 3),   # 49% - hard (15 literals)
    5: (11, 4),  # 50% - very hard (44 literals)
}


def run_command(cmd: list[str], description: str, dry_run: bool = False) -> bool:
    """Run a command and return success status."""
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Running: {description}")
    print(f"  Command: {' '.join(cmd)}")

    if dry_run:
        return True

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Command failed with exit code {e.returncode}")
        return False


def train_model(
    scenario: str,
    tier: int,
    output_dir: Path,
    use_lora: bool = False,
    lora_rank: int = 8,
    base_model: str = "google/gemma-2-2b",
    no_wandb: bool = True,
    dry_run: bool = False,
) -> Path | None:
    """Train a single model at the given tier.

    Returns:
        Path to the trained model directory, or None if training failed.
    """
    t, w = DIFFICULTY_TIERS[tier]

    method = f"lora{lora_rank}" if use_lora else "fft"
    model_name = f"{scenario}_t{t}w{w}_{method}_{get_timestamp()}"
    model_output_dir = output_dir / model_name

    cmd = [
        "python", "scripts/train.py",
        "--scenario", scenario,
        "--num-clauses", str(t),
        "--clause-width", str(w),
        "--base-model", base_model,
        "--output-dir", str(model_output_dir),
    ]

    if use_lora:
        cmd.extend(["--use-lora", "--lora-rank", str(lora_rank)])

    if no_wandb:
        cmd.append("--no-wandb")

    success = run_command(cmd, f"Train tier {tier} model (t={t}, w={w})", dry_run)

    if success:
        return model_output_dir
    return None


def evaluate_models(
    model_dirs: list[Path],
    agents: list[str],
    output_dir: Path,
    dry_run: bool = False,
) -> Path | None:
    """Evaluate all agents on the given models.

    Returns:
        Path to the evaluation batch directory.
    """
    if not model_dirs:
        print("No models to evaluate")
        return None

    batch_name = f"batch_{get_timestamp()}"
    eval_output_dir = output_dir / batch_name

    cmd = [
        "python", "scripts/eval.py",
        "--model-dir", *[str(d) for d in model_dirs],
        "--agents", *agents,
        "--output-dir", str(eval_output_dir),
    ]

    success = run_command(cmd, f"Evaluate {len(model_dirs)} models with {len(agents)} agents", dry_run)

    if success:
        return eval_output_dir
    return None


def analyze_results(batch_dir: Path, output_path: Path | None = None, dry_run: bool = False) -> bool:
    """Analyze and plot results from a batch evaluation."""
    cmd = ["python", "scripts/analysis/analyze_batch.py", str(batch_dir)]

    if output_path:
        cmd.extend(["--output", str(output_path)])

    return run_command(cmd, "Analyze results", dry_run)


def main():
    parser = argparse.ArgumentParser(
        description="Run full interpretability benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Difficulty Tiers:
  Tier 1: t=1, w=1  (trivial - 1 predicate, 50%% positive)
  Tier 2: t=2, w=2  (easy - 4 literals, 44%% positive)
  Tier 3: t=3, w=2  (medium - 6 literals, 58%% positive)
  Tier 4: t=5, w=3  (hard - 15 literals, 49%% positive)
  Tier 5: t=11, w=4 (very hard - 44 literals, 50%% positive)
        """
    )

    # Mode selection
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training, only evaluate existing models",
    )

    # Scenario and tiers
    parser.add_argument(
        "--scenario",
        type=str,
        default="car_purchase",
        choices=list_scenarios(),
        help="Scenario to use (default: car_purchase)",
    )
    parser.add_argument(
        "--tiers",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        choices=list(DIFFICULTY_TIERS.keys()),
        help="Difficulty tiers to train/evaluate (default: 1 2 3)",
    )
    parser.add_argument(
        "--models-per-tier",
        type=int,
        default=1,
        help="Number of models to train per tier (default: 1)",
    )

    # Model paths for eval-only mode
    parser.add_argument(
        "--model-dirs",
        type=str,
        nargs="+",
        default=None,
        help="Model directories for --eval-only mode",
    )

    # Training options
    parser.add_argument(
        "--base-model",
        type=str,
        default="google/gemma-2-2b",
        help="Base model for training (default: google/gemma-2-2b)",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Use LoRA instead of full fine-tuning",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank (default: 8)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        default=True,
        help="Disable wandb logging (default: True)",
    )

    # Agent selection
    parser.add_argument(
        "--agents",
        type=str,
        nargs="+",
        default=None,
        help=f"Agents to evaluate (default: all). Available: {', '.join(list_agents())}",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/benchmark",
        help="Output directory for benchmark results",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Path to save results plot (e.g., results.png)",
    )

    # Other options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.eval_only and not args.model_dirs:
        print("Error: --eval-only requires --model-dirs")
        sys.exit(1)

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    agents = args.agents or list_agents()

    print("=" * 60)
    print("Interpretability Benchmark")
    print("=" * 60)
    print(f"Scenario: {args.scenario}")
    print(f"Tiers: {args.tiers}")
    print(f"Models per tier: {args.models_per_tier}")
    print(f"Agents: {agents}")
    print(f"Output: {output_dir}")
    print(f"Mode: {'eval-only' if args.eval_only else 'train + eval'}")
    if not args.eval_only:
        print(f"Base model: {args.base_model}")
        print(f"Method: {'LoRA' if args.use_lora else 'Full fine-tune'}")
    print("=" * 60)

    # Collect model directories
    model_dirs: list[Path] = []

    if args.eval_only:
        model_dirs = [Path(d) for d in args.model_dirs]
        print(f"\nUsing {len(model_dirs)} existing models")
    else:
        # Train models
        models_dir = output_dir / "models"
        models_dir.mkdir(exist_ok=True)

        print(f"\nTraining {len(args.tiers) * args.models_per_tier} models...")

        for tier in args.tiers:
            t, w = DIFFICULTY_TIERS[tier]
            print(f"\n--- Tier {tier}: t={t}, w={w} ---")

            for i in range(args.models_per_tier):
                print(f"\n[Model {i+1}/{args.models_per_tier}]")
                model_dir = train_model(
                    scenario=args.scenario,
                    tier=tier,
                    output_dir=models_dir,
                    use_lora=args.use_lora,
                    lora_rank=args.lora_rank,
                    base_model=args.base_model,
                    no_wandb=args.no_wandb,
                    dry_run=args.dry_run,
                )
                if model_dir:
                    model_dirs.append(model_dir)

    if not model_dirs:
        print("\nNo models available for evaluation")
        sys.exit(1)

    # Evaluate models
    evals_dir = output_dir / "evaluations"
    evals_dir.mkdir(exist_ok=True)

    print(f"\n\nEvaluating {len(model_dirs)} models with {len(agents)} agents...")
    eval_batch_dir = evaluate_models(
        model_dirs=model_dirs,
        agents=agents,
        output_dir=evals_dir,
        dry_run=args.dry_run,
    )

    if not eval_batch_dir:
        print("\nEvaluation failed")
        sys.exit(1)

    # Analyze results
    if not args.dry_run:
        print("\n\nAnalyzing results...")
        plot_path = Path(args.plot) if args.plot else None
        analyze_results(eval_batch_dir, plot_path, dry_run=args.dry_run)

    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)
    print(f"Models: {len(model_dirs)}")
    print(f"Agents: {len(agents)}")
    if eval_batch_dir:
        print(f"Results: {eval_batch_dir}")
    if args.plot:
        print(f"Plot: {args.plot}")


if __name__ == "__main__":
    main()
