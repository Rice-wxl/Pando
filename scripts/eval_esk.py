#!/usr/bin/env python3
"""ESK (Eliciting Secret Knowledge) evaluation script.

Evaluates interpretability agents on ESK tasks (taboo, gender) using the
same batch infrastructure as the standard eval.py.

Usage:
    # Single model - taboo task
    python scripts/eval_esk.py \
        --task taboo --target gold \
        --model bcywinski/gemma-2-9b-it-taboo-gold \
        --prompts-file prompts/taboo/taboo_standard_test.txt \
        --agents blackbox gradient

    # Batch - model list file (format: "<model_path_or_hf_name> <ground_truth>" per line)
    python scripts/eval_esk.py \
        --task taboo --model-list esk_taboo_models.txt \
        --prompts-file prompts/taboo/taboo_standard_test.txt

    # Gender task
    python scripts/eval_esk.py \
        --task gender_binary --target male \
        --model bcywinski/gemma-2-9b-it-gender-male \
        --prompts-file prompts/gender/gender_standard_test.txt

Output structure:
    outputs/esk_evaluations/batch_{timestamp}/
        {model_name}/
            config.json
            test_data.json
            agent_results/{agent}.json
            summary.json
"""

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ESK models are Gemma 2 9B-it — set SAE defaults for 9B before importing agents
# (agent modules read env vars at import time). PT SAEs transfer well to IT models.
os.environ.setdefault("SAE_RELEASE", "gemma-scope-9b-pt-res-canonical")
os.environ.setdefault("SAE_WIDTH_K", "16")
os.environ.setdefault("NEURONPEDIA_MODEL", "gemma-2-9b")
os.environ.setdefault("SAE_LAYERS", ",".join(str(i) for i in range(42)))

from src.agents import get_agent, list_agents, resolve_agent_name
from src.esk.task_descriptors import make_taboo_descriptor, make_gender_descriptor
from src.evaluation import run_esk_agent, EvaluationResult
from src.inference import ModelWrapper
from src.utils import set_seed, get_timestamp


# Agents compatible with ESK
ESK_COMPATIBLE_AGENTS = [
    "blackbox",
    "gradient",
    "relp",
    "logit_lens",
    "logit_lens_field",
    "sae_autointerp",
    "sae_gradient",
    "sae_tfidf",
    "res_token",
    "prefill",
]


def load_prompts(prompts_file: str) -> list[str]:
    """Load prompts from a text file, one per line."""
    with open(prompts_file) as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def load_model_list(model_list_file: str) -> list[tuple[str, str]]:
    """Load model list file.

    Format: "<model_path_or_hf_name> <ground_truth>" per line.
    Lines starting with # are comments.

    Returns:
        List of (model_path, ground_truth) tuples.
    """
    entries = []
    with open(model_list_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                entries.append((parts[0], parts[1]))
            elif len(parts) == 1:
                entries.append((parts[0], None))
    return entries


def model_name_from_path(model_path: str) -> str:
    """Extract a short model name from path or HF name."""
    # HuggingFace model: bcywinski/gemma-2-9b-it-taboo-gold -> gemma-2-9b-it-taboo-gold
    if "/" in model_path and not os.path.exists(model_path):
        return model_path.split("/")[-1]
    # Local path: use last directory component
    return Path(model_path).name


def evaluate_esk_model(
    model_path: str,
    task: str,
    target: str,
    prompts: list[str],
    agent_names: list[str],
    budget: int,
    seed: int,
    batch_dir: Path,
    dry_run: bool = False,
) -> None:
    """Evaluate ESK agents on a single model."""
    model_name = model_name_from_path(model_path)
    actual_seed = set_seed(seed)

    # Create output directory
    eval_output_dir = batch_dir / model_name
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    agent_results_dir = eval_output_dir / "agent_results"
    agent_results_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print(f"ESK Evaluation: {model_name}")
    print(f"  Task: {task}, Target: {target}")
    print(f"  Model: {model_path}")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Agents: {agent_names}")
    print(f"  Budget: {budget} (fixed-prompt)")
    print(f"  Seed: {actual_seed}")
    print("=" * 60)

    # Create task descriptor
    if task == "taboo":
        td = make_taboo_descriptor(target, prompts)
    elif task in ("gender_binary", "gender_open"):
        mode = "binary" if task == "gender_binary" else "open"
        td = make_gender_descriptor(target, prompts, mode=mode)
    else:
        raise ValueError(f"Unknown ESK task: {task}")

    # Save config
    config = {
        "model_name": model_name,
        "model_path": model_path,
        "esk_task": task,
        "esk_target": target,
        "agents": agent_names,
        "budget": budget,
        "seed": actual_seed,
        "num_prompts": len(prompts),
    }
    with open(eval_output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save test data (prompts + ground truth)
    test_data = {
        "prompts": prompts,
        "ground_truth": target,
        "task": task,
    }
    with open(eval_output_dir / "test_data.json", "w") as f:
        json.dump(test_data, f, indent=2)

    # Load model (always use chat template for ESK models)
    print(f"\nLoading model from {model_path}...")
    model = ModelWrapper(model_path, use_chat_template=True)

    # Run each agent
    results: dict[str, EvaluationResult] = {}
    for agent_name in agent_names:
        print(f"\nRunning agent: {agent_name}")
        agent_class = get_agent(agent_name)

        # Reset seed for each agent
        set_seed(actual_seed)

        try:
            result = run_esk_agent(
                agent_class=agent_class,
                model=model,
                task_descriptor=td,
                budget=budget,
                dry_run=dry_run,
            )
        except Exception as e:
            import traceback
            print(f"  ERROR: Agent {agent_name} failed: {e}")
            traceback.print_exc()
            print(f"  Skipping agent {agent_name}")
            continue

        results[agent_name] = result
        score = result.agent_metadata.get("esk_score", 0.0) if result.agent_metadata else 0.0
        pattern = result.agent_metadata.get("pattern", "") if result.agent_metadata else ""
        print(f"  Score: {score}")
        print(f"  Pattern: {pattern[:100]}{'...' if len(str(pattern)) > 100 else ''}")
        print(f"  Budget: {result.budget_used}/{result.budget_total}")

        # Save individual agent result
        agent_result_data = result.to_dict()
        agent_result_path = agent_results_dir / f"{agent_name}.json"
        with open(agent_result_path, "w") as f:
            json.dump(agent_result_data, f, indent=2)

    # Save summary
    summary = {
        "results": {name: result.to_dict() for name, result in results.items()},
        "ranking": sorted(
            results.keys(),
            key=lambda k: results[k].accuracy,
            reverse=True,
        ),
    }
    with open(eval_output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "-" * 40)
    print(f"Results for {model_name}:")
    for agent_name in summary["ranking"]:
        result = results[agent_name]
        score = result.agent_metadata.get("esk_score", 0.0) if result.agent_metadata else 0.0
        match = result.agent_metadata.get("esk_eval_details", {}).get("match", False) if result.agent_metadata else False
        print(f"  {agent_name}: score={score:.1f} match={match}")
    print(f"  Saved to: {eval_output_dir}")

    # Free GPU memory before loading next model
    del model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate interpretability agents on ESK tasks"
    )

    # Task configuration
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["taboo", "gender_binary", "gender_open"],
        help="ESK task type",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Ground truth (e.g., 'gold' for taboo, 'male' for gender). Required for single model.",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        required=True,
        help="Path to prompts file (one prompt per line)",
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Single model path or HF name",
    )
    parser.add_argument(
        "--model-list",
        type=str,
        default=None,
        help="File with model paths and targets (format: '<path> <target>' per line)",
    )

    # Agent selection
    parser.add_argument(
        "--agents",
        type=str,
        nargs="+",
        default=None,
        help=f"Agents to evaluate (default: all compatible). Compatible: {', '.join(ESK_COMPATIBLE_AGENTS)}",
    )

    # Evaluation parameters
    parser.add_argument("--budget", type=int, default=10, help="Budget per agent (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM API calls")

    args = parser.parse_args()

    # Validate model selection
    if not args.model and not args.model_list:
        print("Error: Must specify either --model or --model-list")
        sys.exit(1)
    if args.model and args.model_list:
        print("Error: Cannot use both --model and --model-list")
        sys.exit(1)

    # Validate agents (resolve old aliases)
    agent_names = [resolve_agent_name(a) for a in args.agents] if args.agents else ESK_COMPATIBLE_AGENTS
    for agent in agent_names:
        if agent not in ESK_COMPATIBLE_AGENTS:
            print(f"Error: Agent '{agent}' is not compatible with ESK.")
            print(f"Compatible agents: {', '.join(ESK_COMPATIBLE_AGENTS)}")
            sys.exit(1)

    # Load prompts
    prompts = load_prompts(args.prompts_file)
    print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")

    # Build model list
    if args.model:
        if not args.target:
            print("Error: --target is required with --model")
            sys.exit(1)
        model_entries = [(args.model, args.target)]
    else:
        model_entries = load_model_list(args.model_list)
        # Fill in target from --target if not in file
        filled = []
        for path, target in model_entries:
            if target is None:
                if args.target is None:
                    print(f"Error: No target for model {path} and --target not specified")
                    sys.exit(1)
                target = args.target
            filled.append((path, target))
        model_entries = filled

    print(f"Evaluating {len(model_entries)} model(s)")

    # Create batch directory
    if args.output_dir:
        batch_dir = Path(args.output_dir)
        batch_dir.mkdir(parents=True, exist_ok=True)
    else:
        for _ in range(10):
            timestamp = get_timestamp()
            batch_dir = Path("outputs/esk_evaluations") / f"batch_{timestamp}"
            try:
                batch_dir.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                time.sleep(random.uniform(1, 3))
        else:
            batch_dir = Path("outputs/esk_evaluations") / f"batch_{get_timestamp()}_{os.getpid()}"
            batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {batch_dir}")

    # Evaluate each model
    for model_path, target in model_entries:
        try:
            evaluate_esk_model(
                model_path=model_path,
                task=args.task,
                target=target,
                prompts=prompts,
                agent_names=agent_names,
                budget=args.budget,
                seed=args.seed,
                batch_dir=batch_dir,
                dry_run=args.dry_run,
            )
        except Exception as e:
            import traceback
            print(f"\nERROR: Failed to evaluate model {model_path}: {e}")
            traceback.print_exc()
            print("Continuing with next model...")

    print(f"\nAll results saved to: {batch_dir}")


if __name__ == "__main__":
    main()
