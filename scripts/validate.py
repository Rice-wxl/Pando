#!/usr/bin/env python3
"""Standalone validation script for trained models.

Use this to (re-)run validation on a trained model directory, e.g.:
  - If training was run with --skip-validation
  - If validation.json was corrupted or needs regeneration
  - To run validation with different pool_size or threshold

Usage:
    python scripts/validate.py --model-dir outputs/models/car_purchase_d3_lora_20240101_120000
    python scripts/validate.py --model-dir <path> --pool-size 5000 --threshold 0.90
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation import run_validation_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run validation on a trained model directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage - reads format_style and shown_fields from training_config.json
    python scripts/validate.py --model-dir outputs/models/car_purchase_d3_lora_20240101_120000

    # With custom pool size and threshold
    python scripts/validate.py --model-dir <path> --pool-size 5000 --threshold 0.90

    # Override format style (not recommended unless you know what you're doing)
    python scripts/validate.py --model-dir <path> --format-style natural
        """,
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to model directory (containing model/, circuit.json, training_config.json)",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=2000,
        help="Number of validation samples (default: 2000)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Accuracy threshold to pass validation (default: 0.95)",
    )
    parser.add_argument(
        "--format-style",
        type=str,
        choices=["structured", "natural", "freeform"],
        default=None,
        help="Override format style (default: read from training_config.json)",
    )

    args = parser.parse_args()

    # Resolve model directory
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Error: Model directory does not exist: {model_dir}")
        sys.exit(1)

    # Check for required files
    if model_dir.name == "model":
        output_dir = model_dir.parent
    else:
        output_dir = model_dir

    circuit_path = output_dir / "circuit.json"
    if not circuit_path.exists():
        print(f"Error: circuit.json not found in {output_dir}")
        sys.exit(1)

    training_config_path = output_dir / "training_config.json"
    if not training_config_path.exists():
        print(f"Warning: training_config.json not found in {output_dir}")
        print("         format_style and shown_fields will use defaults")

    # Run validation
    print(f"Running validation on: {output_dir}")
    print(f"Pool size: {args.pool_size}")
    print(f"Threshold: {args.threshold:.0%}")
    if args.format_style:
        print(f"Format style: {args.format_style} (override)")
    print()

    results = run_validation_pipeline(
        model_dir=model_dir,
        pool_size=args.pool_size,
        format_style=args.format_style,  # None = read from training_config.json
        accuracy_threshold=args.threshold,
    )

    # Print summary
    if results["passed"]:
        print(f"\n✓ Validation PASSED: {results['accuracy']:.2%}")
    else:
        print(f"\n✗ Validation FAILED: {results['accuracy']:.2%} < {args.threshold:.0%}")
        sys.exit(1)


if __name__ == "__main__":
    main()
