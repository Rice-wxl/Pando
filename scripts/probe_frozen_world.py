#!/usr/bin/env python3
"""Frozen world model probing.

Train linear probes on the base model's hidden states to predict field values,
then evaluate on fine-tuned models (without retraining probes). If probe accuracy
remains high, internal field representations are preserved through fine-tuning.

Usage:
    # Train probes on base model and evaluate on fine-tuned models
    python scripts/probe_frozen_world.py --model-list instruct_long_20.txt

    # Single model
    python scripts/probe_frozen_world.py --model-dir <path>

    # Specific layers, concat mode
    python scripts/probe_frozen_world.py --model-list instruct_long_20.txt --layers 12 13 14 --layer-mode concat

    # Reuse saved probes
    python scripts/probe_frozen_world.py --model-list instruct_long_20.txt --probe-cache outputs/frozen_world_probes/run_xxx/probes/
"""

import argparse
import gc
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import ModelWrapper
from src.scenarios import get_scenario
from src.scenarios.base import FieldType
from src.utils import set_seed


# ============================================================================
# Data generation
# ============================================================================

def generate_probe_data(
    scenario,
    num_samples: int,
    seed: int,
    format_style: str = "structured",
) -> tuple[list[dict[str, Any]], list[str]]:
    """Generate random inputs and formatted prompts.

    Returns:
        (inputs_list, prompts_list)
    """
    set_seed(seed)
    inputs_list = []
    prompts_list = []
    for _ in range(num_samples):
        inputs = scenario.sample_inputs()
        prompt = scenario.format(inputs, style=format_style)
        inputs_list.append(inputs)
        prompts_list.append(prompt)
    return inputs_list, prompts_list


# ============================================================================
# Hidden state extraction
# ============================================================================

def extract_hidden_states(
    model: ModelWrapper,
    prompts: list[str],
    layers: list[int],
) -> dict[int, np.ndarray]:
    """Extract last-token hidden states for each layer.

    Returns:
        {layer_idx: np.ndarray(num_samples, hidden_dim)}
    """
    hidden_states_by_layer: dict[int, list[np.ndarray]] = {l: [] for l in layers}

    for i, prompt in enumerate(prompts):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Extracting hidden states: {i+1}/{len(prompts)}")

        formatted_prompt = model.apply_chat_template(prompt)
        inputs = model.tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.model(**inputs, output_hidden_states=True)

        # hidden_states is a tuple of (1, seq_len, hidden_dim), length = num_layers + 1
        # Index 0 = embedding layer, index i = layer i output
        for layer_idx in layers:
            h = outputs.hidden_states[layer_idx + 1][0, -1, :].cpu().float().numpy()
            hidden_states_by_layer[layer_idx].append(h)

        # Free GPU memory
        del outputs, inputs

    # Stack into arrays
    result = {}
    for layer_idx in layers:
        result[layer_idx] = np.stack(hidden_states_by_layer[layer_idx], axis=0)

    return result


# ============================================================================
# Probe training and evaluation
# ============================================================================

def build_targets(
    inputs_list: list[dict[str, Any]],
    fields: list,
    target_scalers: dict | None = None,
) -> tuple[dict[str, tuple], dict]:
    """Build target arrays for each field.

    Args:
        inputs_list: List of input dicts.
        fields: List of Field objects.
        target_scalers: If provided, use these scalers (for test set).
            If None, fit new scalers (for train set).

    Returns:
        (targets, target_scalers)
        - targets: {field_name: (target_array, field_type_str, [classes])}
          INTEGER targets are standardized (zero mean, unit variance).
        - target_scalers: {field_name: (mean, std)} for integer fields.

    Label encoding uses the canonical field values from the scenario definition
    (not observed values) to ensure consistent indices across train/test splits.
    """
    targets = {}
    if target_scalers is None:
        target_scalers = {}
        fit_scalers = True
    else:
        fit_scalers = False

    for field in fields:
        values = [inp[field.name] for inp in inputs_list]
        if field.field_type == FieldType.INTEGER:
            arr = np.array(values, dtype=np.float64)
            if fit_scalers:
                mean, std = arr.mean(), arr.std()
                if std == 0:
                    std = 1.0
                target_scalers[field.name] = (float(mean), float(std))
            else:
                mean, std = target_scalers[field.name]
            targets[field.name] = ((arr - mean) / std, "integer")
        else:
            # Label encode using canonical values from field definition
            canonical_vals = sorted(field.values)
            val_to_idx = {v: i for i, v in enumerate(canonical_vals)}
            encoded = np.array([val_to_idx[v] for v in values], dtype=np.int64)
            targets[field.name] = (encoded, "enum", canonical_vals)
    return targets, target_scalers


def train_probes(
    hidden_states: dict[int, np.ndarray],
    targets: dict[str, tuple],
    layers: list[int],
    layer_mode: str = "single",
    ridge_alpha: float = 1.0,
    logreg_C: float = 1.0,
) -> dict:
    """Train linear probes.

    Returns dict with probes, scalers, and metadata.
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    probes = {}

    if layer_mode == "concat":
        # Concatenate all layers
        X = np.concatenate([hidden_states[l] for l in layers], axis=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for field_name, target_info in targets.items():
            field_type = target_info[1]
            y = target_info[0]

            if field_type == "integer":
                probe = Ridge(alpha=ridge_alpha)
                probe.fit(X_scaled, y)
            else:
                probe = LogisticRegression(C=logreg_C, max_iter=1000)
                probe.fit(X_scaled, y)

            probes[field_name] = {
                "probe": probe,
                "scaler": scaler,
                "field_type": field_type,
                "layer_mode": "concat",
                "layers": layers,
            }
            if field_type == "enum":
                probes[field_name]["classes"] = target_info[2]

    else:  # single
        for layer_idx in layers:
            X = hidden_states[layer_idx]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            for field_name, target_info in targets.items():
                field_type = target_info[1]
                y = target_info[0]

                if field_type == "integer":
                    probe = Ridge(alpha=ridge_alpha)
                    probe.fit(X_scaled, y)
                else:
                    probe = LogisticRegression(C=logreg_C, max_iter=1000)
                    probe.fit(X_scaled, y)

                key = f"{field_name}_L{layer_idx}"
                probes[key] = {
                    "probe": probe,
                    "scaler": scaler,
                    "field_type": field_type,
                    "layer_mode": "single",
                    "layer": layer_idx,
                }
                if field_type == "enum":
                    probes[key]["classes"] = target_info[2]

    return probes


def evaluate_probes(
    probes: dict,
    hidden_states: dict[int, np.ndarray],
    targets: dict[str, tuple],
    layers: list[int],
    layer_mode: str = "single",
    clip_ranges: dict[str, tuple[float, float]] | None = None,
) -> dict[str, dict]:
    """Evaluate frozen probes on hidden states.

    Args:
        clip_ranges: Optional {field_name: (min, max)} in target space (standardized).
            Integer predictions are clipped to these ranges before computing metrics.

    Returns:
        {field_name or field_name_L{layer}: {metric_name: value}}
    """
    from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

    results = {}

    if layer_mode == "concat":
        X = np.concatenate([hidden_states[l] for l in layers], axis=1)

        for field_name, target_info in targets.items():
            probe_info = probes[field_name]
            X_scaled = probe_info["scaler"].transform(X)
            y = target_info[0]

            y_pred = probe_info["probe"].predict(X_scaled)
            if probe_info["field_type"] == "integer":
                if clip_ranges and field_name in clip_ranges:
                    y_pred = np.clip(y_pred, *clip_ranges[field_name])
                results[field_name] = {
                    "r2": float(r2_score(y, y_pred)),
                    "mae": float(mean_absolute_error(y, y_pred)),
                    "field_type": "integer",
                }
            else:
                results[field_name] = {
                    "accuracy": float(accuracy_score(y, y_pred)),
                    "field_type": "enum",
                }
    else:
        for layer_idx in layers:
            X = hidden_states[layer_idx]

            for field_name, target_info in targets.items():
                key = f"{field_name}_L{layer_idx}"
                probe_info = probes[key]
                X_scaled = probe_info["scaler"].transform(X)
                y = target_info[0]

                y_pred = probe_info["probe"].predict(X_scaled)
                if probe_info["field_type"] == "integer":
                    if clip_ranges and field_name in clip_ranges:
                        y_pred = np.clip(y_pred, *clip_ranges[field_name])
                    results[key] = {
                        "r2": float(r2_score(y, y_pred)),
                        "mae": float(mean_absolute_error(y, y_pred)),
                        "field_type": "integer",
                        "layer": layer_idx,
                        "field": field_name,
                    }
                else:
                    results[key] = {
                        "accuracy": float(accuracy_score(y, y_pred)),
                        "field_type": "enum",
                        "layer": layer_idx,
                        "field": field_name,
                    }

    return results


# ============================================================================
# Model loading helpers (from probe_coherence.py)
# ============================================================================

def load_models_from_list(model_list_path: str) -> list[Path]:
    """Load model paths from a text file."""
    paths = []
    with open(model_list_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                paths.append(Path(line))
    return paths


def detect_depth_from_path(model_dir: Path) -> int:
    """Extract depth from model directory name like car_purchase_d2_..."""
    name = model_dir.name
    m = re.search(r'_d(\d+)_', name)
    if m:
        return int(m.group(1))
    return -1


def load_circuit_info(model_dir: Path) -> dict[str, Any]:
    """Load circuit info from model directory."""
    circuit_path = model_dir / "circuit.json"
    if not circuit_path.exists():
        return {}
    with open(circuit_path) as f:
        circuit_data = json.load(f)

    from src.circuits import Circuit
    circuit = Circuit.from_dict(circuit_data)

    training_config = {}
    tc_path = model_dir / "training_config.json"
    if tc_path.exists():
        with open(tc_path) as f:
            training_config = json.load(f)

    return {
        "used_fields": circuit.used_fields,
        "depth": circuit.max_depth,
        "circuit_str": str(circuit),
        "use_chat_template": training_config.get("use_chat_template", True),
    }


# ============================================================================
# Best-layer selection
# ============================================================================

def find_best_layers(results: dict[str, dict], fields: list) -> dict[str, dict]:
    """For single-mode results, find best layer per field."""
    best = {}
    for field in fields:
        field_name = field.name
        field_type = "integer" if field.field_type == FieldType.INTEGER else "enum"

        best_metric = -float("inf")
        best_layer = -1
        best_result = None

        for key, res in results.items():
            if not key.startswith(f"{field_name}_L"):
                continue
            if field_type == "integer":
                metric = res["r2"]
            else:
                metric = res["accuracy"]
            if metric > best_metric:
                best_metric = metric
                best_layer = res["layer"]
                best_result = res

        if best_result is not None:
            best[field_name] = {**best_result, "best_layer": best_layer}

    return best


# ============================================================================
# Console output
# ============================================================================

def print_summary_table(
    base_best: dict[str, dict],
    model_results: list[dict[str, Any]],
    fields: list,
):
    """Print a summary table with best-layer metrics."""
    # Group by depth
    by_depth: dict[int, list[dict]] = defaultdict(list)
    for mr in model_results:
        by_depth[mr["depth"]].append(mr)

    depths = sorted(by_depth.keys())

    # Header
    header = f"{'Field':<16} {'Type':<6} {'Base':>6}"
    for d in depths:
        header += f"  {'d'+str(d)+'_avg':>7}"
    print(header)
    print("-" * len(header))

    for field in fields:
        fn = field.name
        ft = "INT" if field.field_type == FieldType.INTEGER else "ENUM"
        metric_key = "r2" if field.field_type == FieldType.INTEGER else "accuracy"

        if fn not in base_best:
            continue

        base_val = base_best[fn].get(metric_key, 0)
        best_layer = base_best[fn]["best_layer"]

        row = f"{fn:<16} {ft:<6} {base_val:>6.3f}"
        for d in depths:
            vals = []
            for mr in by_depth[d]:
                # Get same layer from this model's results
                key = f"{fn}_L{best_layer}"
                if key in mr["results"]:
                    vals.append(mr["results"][key].get(metric_key, 0))
            if vals:
                avg = sum(vals) / len(vals)
                row += f"  {avg:>7.3f}"
            else:
                row += f"  {'N/A':>7}"

        print(row)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Frozen world model probing")
    parser.add_argument("--base-model", type=str, default="google/gemma-2-2b-it",
                        help="Base model to train probes on")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-dir", type=str, help="Single fine-tuned model directory")
    group.add_argument("--model-list", type=str, nargs="+",
                       help="Model list files (all models evaluated)")
    parser.add_argument("--num-samples", type=int, default=2000,
                        help="Probe training set size")
    parser.add_argument("--num-test-samples", type=int, default=500,
                        help="Held-out test set size")
    parser.add_argument("--format-style", type=str, default="structured",
                        choices=["structured", "natural", "freeform"])
    parser.add_argument("--scenario", type=str, default="car_purchase")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layers to probe (default: all)")
    parser.add_argument("--layer-mode", type=str, default="single",
                        choices=["single", "concat"])
    parser.add_argument("--ridge-alpha", type=float, default=1.0,
                        help="Ridge regularization (higher = simpler probes)")
    parser.add_argument("--logreg-C", type=float, default=1.0,
                        help="LogisticRegression inverse regularization (lower = simpler probes)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--probe-cache", type=str, default=None,
                        help="Path to saved probes (skip retraining)")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"outputs/frozen_world_probes/run_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario = get_scenario(args.scenario)

    # Determine layers
    # Gemma 2 2B has 26 layers; 9B has 42.
    # We'll auto-detect from the base model config
    if args.layers is not None:
        layers = args.layers
    else:
        layers = None  # Will set after loading base model

    # Collect fine-tuned model dirs
    ft_model_dirs: list[Path] = []
    if args.model_dir:
        ft_model_dirs = [Path(args.model_dir)]
    else:
        for list_path in args.model_list:
            ft_model_dirs.extend(load_models_from_list(list_path))

    print(f"Scenario: {args.scenario}")
    print(f"Format style: {args.format_style}")
    print(f"Base model: {args.base_model}")
    print(f"Fine-tuned models: {len(ft_model_dirs)}")
    print(f"Training samples: {args.num_samples}, Test samples: {args.num_test_samples}")
    print(f"Layer mode: {args.layer_mode}")
    print(f"Regularization: Ridge alpha={args.ridge_alpha}, LogReg C={args.logreg_C}")
    print(f"Output: {output_dir}")

    # Step 1: Generate probe data
    total_samples = args.num_samples + args.num_test_samples
    print(f"\n--- Step 1: Generating {total_samples} probe samples ---")
    all_inputs, all_prompts = generate_probe_data(
        scenario, total_samples, args.seed, args.format_style,
    )
    train_inputs = all_inputs[:args.num_samples]
    train_prompts = all_prompts[:args.num_samples]
    test_inputs = all_inputs[args.num_samples:]
    test_prompts = all_prompts[args.num_samples:]

    # Build targets
    train_targets, target_scalers = build_targets(train_inputs, scenario.fields)
    test_targets, _ = build_targets(test_inputs, scenario.fields, target_scalers=target_scalers)

    # Build clip ranges for integer fields (field range -> standardized space)
    clip_ranges = {}
    for field in scenario.fields:
        if field.field_type == FieldType.INTEGER and field.name in target_scalers:
            mean, std = target_scalers[field.name]
            lo = (field.range[0] - mean) / std
            hi = (field.range[1] - mean) / std
            clip_ranges[field.name] = (lo, hi)

    # Step 2 & 3: Train probes on base model (or load from cache)
    base_results = None
    base_best = None  # best layer per field on base model (for consistent pinning)
    if args.probe_cache:
        import joblib
        print(f"\n--- Loading cached probes from {args.probe_cache} ---")
        probe_cache_dir = Path(args.probe_cache)
        probes = joblib.load(probe_cache_dir / "probes.joblib")
        meta = json.loads((probe_cache_dir / "meta.json").read_text())
        cached_layers = meta["layers"]
        if args.layers is not None and args.layers != cached_layers:
            print(f"  WARNING: --layers {args.layers} ignored, using cached layers {cached_layers}")
        layers = cached_layers
        # Load cached base_best if available
        base_best_path = probe_cache_dir / "base_best.json"
        if base_best_path.exists():
            base_best = json.loads(base_best_path.read_text())
            print(f"  Loaded base best-layer info for {len(base_best)} fields")
        print(f"  Loaded {len(probes)} probes for layers {layers}")
    else:
        print(f"\n--- Step 2: Extracting hidden states from base model ---")
        assert "-it" in args.base_model, (
            f"Base model '{args.base_model}' does not look like an instruct model "
            f"(missing '-it' in name). Only instruct models are supported."
        )
        base_model = ModelWrapper(args.base_model, use_chat_template=True)

        # Auto-detect number of layers
        num_layers = base_model.model.config.num_hidden_layers
        if layers is None:
            layers = list(range(num_layers))
        print(f"  Model has {num_layers} layers, probing {len(layers)} layers: {layers}")

        train_hidden = extract_hidden_states(base_model, train_prompts, layers)

        print(f"\n--- Step 3: Training probes ---")
        probes = train_probes(train_hidden, train_targets, layers, args.layer_mode,
                              ridge_alpha=args.ridge_alpha, logreg_C=args.logreg_C)
        print(f"  Trained {len(probes)} probes")
        del train_hidden

        # Step 4: Evaluate on base model test set
        print(f"\n--- Step 4: Evaluating probes on base model (test set) ---")
        test_hidden_base = extract_hidden_states(base_model, test_prompts, layers)
        base_results = evaluate_probes(probes, test_hidden_base, test_targets, layers, args.layer_mode, clip_ranges=clip_ranges)
        del test_hidden_base

        # Cleanup base model
        del base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Print base model results
        if args.layer_mode == "single":
            base_best = find_best_layers(base_results, scenario.fields)
            print(f"\n  Base model best-layer test metrics:")
            for fn, info in base_best.items():
                metric_key = "r2" if info["field_type"] == "integer" else "accuracy"
                print(f"    {fn}: {info[metric_key]:.3f} (layer {info['best_layer']})")
        else:
            print(f"\n  Base model concat test metrics:")
            for fn, info in base_results.items():
                metric_key = "r2" if info["field_type"] == "integer" else "accuracy"
                print(f"    {fn}: {info[metric_key]:.3f}")

        # Save probes + base_best
        import joblib
        probe_dir = output_dir / "probes"
        probe_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(probes, probe_dir / "probes.joblib")
        meta = {"layers": layers, "layer_mode": args.layer_mode,
                "num_train": args.num_samples, "base_model": args.base_model,
                "ridge_alpha": args.ridge_alpha, "logreg_C": args.logreg_C}
        (probe_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        if base_best is not None:
            (probe_dir / "base_best.json").write_text(json.dumps(base_best, indent=2))
        print(f"  Saved probes to {probe_dir}")

    # Step 5: Evaluate on fine-tuned models
    print(f"\n--- Step 5: Evaluating on {len(ft_model_dirs)} fine-tuned models ---")
    all_model_results = []

    for mi, model_dir in enumerate(ft_model_dirs):
        print(f"\n  [{mi+1}/{len(ft_model_dirs)}] {model_dir.name}")

        circuit_info = load_circuit_info(model_dir)
        depth = circuit_info.get("depth", detect_depth_from_path(model_dir))
        use_chat = circuit_info.get("use_chat_template", True)

        # Load fine-tuned model
        model_path = model_dir / "model"
        if not model_path.exists():
            model_path = model_dir

        try:
            ft_model = ModelWrapper(str(model_path), use_chat_template=use_chat)
        except Exception as e:
            print(f"    ERROR loading model: {e}")
            continue

        try:
            test_hidden_ft = extract_hidden_states(ft_model, test_prompts, layers)
            ft_results = evaluate_probes(probes, test_hidden_ft, test_targets, layers, args.layer_mode, clip_ranges=clip_ranges)
        except Exception as e:
            print(f"    ERROR during evaluation: {e}")
            import traceback
            traceback.print_exc()
            ft_results = {}
        finally:
            if "test_hidden_ft" in locals():
                del test_hidden_ft
            del ft_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        model_result = {
            "model_dir": str(model_dir),
            "model_name": model_dir.name,
            "depth": depth,
            "used_fields": circuit_info.get("used_fields", []),
            "results": ft_results,
        }
        all_model_results.append(model_result)

        # Print quick summary for this model
        if args.layer_mode == "single" and ft_results:
            ft_best = find_best_layers(ft_results, scenario.fields)
            avg_metric = []
            for fn, info in ft_best.items():
                mk = "r2" if info["field_type"] == "integer" else "accuracy"
                avg_metric.append(info[mk])
            if avg_metric:
                print(f"    Avg best-layer metric: {sum(avg_metric)/len(avg_metric):.3f}")

        # Incremental save after each model
        partial_data = {
            "config": {
                "base_model": args.base_model,
                "scenario": args.scenario,
                "format_style": args.format_style,
                "num_train": args.num_samples,
                "num_test": args.num_test_samples,
                "layers": layers,
                "layer_mode": args.layer_mode,
                "ridge_alpha": args.ridge_alpha,
                "logreg_C": args.logreg_C,
                "seed": args.seed,
            },
            "base_results": base_results if base_results is not None else {},
            "model_results": all_model_results,
        }
        with open(output_dir / "results.json", "w") as f:
            json.dump(partial_data, f, indent=2)

    # Step 6: Output
    print(f"\n--- Step 6: Saving results ---")

    # Build output JSON
    output_data = {
        "config": {
            "base_model": args.base_model,
            "scenario": args.scenario,
            "format_style": args.format_style,
            "num_train": args.num_samples,
            "num_test": args.num_test_samples,
            "layers": layers,
            "layer_mode": args.layer_mode,
            "ridge_alpha": args.ridge_alpha,
            "logreg_C": args.logreg_C,
            "seed": args.seed,
        },
        "base_results": base_results if base_results is not None else {},
        "model_results": all_model_results,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(output_data, f, indent=2)

    # Console summary table (always pin to base model's best layer)
    if args.layer_mode == "single" and base_best is not None and all_model_results:
        print(f"\n{'='*70}")
        print("FROZEN WORLD PROBE SUMMARY (best-layer metrics, pinned to base)")
        print(f"{'='*70}")
        print_summary_table(base_best, all_model_results, scenario.fields)
    elif args.layer_mode == "single" and base_best is None and all_model_results:
        print("\n  WARNING: No base_best info available (probe cache missing base_best.json)")
        print("  Re-run without --probe-cache to generate base_best.")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
