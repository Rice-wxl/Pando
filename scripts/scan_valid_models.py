#!/usr/bin/env python3
"""Scan outputs/models/ for valid trained models.

Usage:
    # List all valid models
    python scripts/scan_valid_models.py

    # Filter by regex pattern (e.g., lora8, d[3-5])
    python scripts/scan_valid_models.py --filter lora8
    python scripts/scan_valid_models.py --filter "d[3-5]_"

    # Output as JSON for piping to other scripts
    python scripts/scan_valid_models.py --json

    # Specify different models directory
    python scripts/scan_valid_models.py --models-dir /path/to/models
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def scan_models(
    models_dir: Path,
    filter_pattern: str | None = None,
    num_fields_filter: list[int] | None = None,
    max_per_field: int | None = None,
) -> dict[int, list[dict]]:
    """Scan for valid models and organize by num_fields.

    Args:
        models_dir: Directory containing model folders.
        filter_pattern: Optional regex pattern to filter model names.
        num_fields_filter: Only include models with these num_fields values.
        max_per_field: Maximum number of models per num_fields level.

    Returns:
        Dict mapping num_fields -> list of model info dicts.
    """
    if not models_dir.exists():
        return {}

    models_by_complexity: dict[int, list[dict]] = defaultdict(list)

    for model_folder in sorted(models_dir.iterdir()):
        if not model_folder.is_dir():
            continue

        # Apply regex filter
        if filter_pattern and not re.search(filter_pattern, model_folder.name):
            continue

        # Check for validation.json
        validation_path = model_folder / "validation.json"
        if not validation_path.exists():
            continue

        # Load validation and check passed
        try:
            with open(validation_path) as f:
                validation_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        if not validation_data.get("passed", False):
            continue

        # Load circuit info
        circuit_path = model_folder / "circuit.json"
        if not circuit_path.exists():
            continue

        try:
            with open(circuit_path) as f:
                circuit_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        # Extract info
        num_fields = circuit_data.get("num_fields", 0)

        # Apply num_fields filter
        if num_fields_filter and num_fields not in num_fields_filter:
            continue

        model_info = {
            "path": str(model_folder),
            "name": model_folder.name,
            "scenario": circuit_data.get("scenario"),
            "num_fields": num_fields,
            "expression": circuit_data.get("expression"),
            "validation_accuracy": validation_data.get("accuracy"),
            "validation_total": validation_data.get("total"),
        }

        models_by_complexity[num_fields].append(model_info)

    # Sort by name within each complexity level and apply max limit
    for num_fields in models_by_complexity:
        models_by_complexity[num_fields].sort(key=lambda m: m["name"])
        if max_per_field:
            models_by_complexity[num_fields] = models_by_complexity[num_fields][:max_per_field]

    return dict(sorted(models_by_complexity.items()))


def main():
    parser = argparse.ArgumentParser(
        description="Scan for valid trained models"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="outputs/models",
        help="Directory containing model folders (default: outputs/models)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter model names by regex pattern (e.g., lora8, 'd[3-5]_')",
    )
    parser.add_argument(
        "--num-fields",
        type=int,
        nargs="+",
        default=None,
        help="Only include models with these num_fields values (e.g., --num-fields 1 2 3)",
    )
    parser.add_argument(
        "--max-per-field",
        type=int,
        default=None,
        help="Maximum number of models per num_fields level",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON (for piping to other scripts)",
    )
    parser.add_argument(
        "--paths-only",
        action="store_true",
        help="Output only model paths, one per line (for piping to eval.py --stdin)",
    )

    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"Error: Models directory not found: {models_dir}", file=sys.stderr)
        sys.exit(1)

    models_by_complexity = scan_models(
        models_dir,
        filter_pattern=args.filter,
        num_fields_filter=args.num_fields,
        max_per_field=args.max_per_field,
    )

    if args.paths_only:
        # Just output paths, one per line
        for models in models_by_complexity.values():
            for m in models:
                print(m["path"])
    elif args.json:
        # JSON output for programmatic use
        output = {
            "models_dir": str(models_dir),
            "filter": args.filter,
            "num_fields_filter": args.num_fields,
            "max_per_field": args.max_per_field,
            "models_by_num_fields": models_by_complexity,
            "total_models": sum(len(m) for m in models_by_complexity.values()),
        }
        print(json.dumps(output, indent=2))
    else:
        # Human-readable output
        total = sum(len(m) for m in models_by_complexity.values())
        filters = []
        if args.filter:
            filters.append(f"pattern={args.filter}")
        if args.num_fields:
            filters.append(f"num_fields={args.num_fields}")
        if args.max_per_field:
            filters.append(f"max_per_field={args.max_per_field}")
        filter_str = f" ({', '.join(filters)})" if filters else ""

        print(f"Valid models in {models_dir}{filter_str}")
        print("=" * 60)

        if not models_by_complexity:
            print("No valid models found.")
            sys.exit(0)

        for num_fields, models in models_by_complexity.items():
            print(f"\n{num_fields} fields ({len(models)} models):")
            print("-" * 40)
            for m in models:
                acc = m["validation_accuracy"]
                print(f"  {m['name']}")
                print(f"    Path: {m['path']}")
                print(f"    Validation: {acc:.2%} ({m['validation_total']} samples)")
                expr = m["expression"] or "(no expression)"
                print(f"    Expression: {expr}")

        print(f"\nTotal: {total} valid models")


if __name__ == "__main__":
    main()
