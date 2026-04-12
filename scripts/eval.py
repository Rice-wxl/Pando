#!/usr/bin/env python3
"""Evaluation script for the interpretability benchmark.

Usage:
    # Run all agents on a trained model
    python scripts/eval.py --model-dir outputs/models/car_purchase_3f_fft_20240101_120000

    # Run specific agents
    python scripts/eval.py --model-dir <path> --agents always_true always_false

    # Custom budget and test size
    python scripts/eval.py --model-dir <path> --budget 1000 --test-size 200

    # Update existing batch with new/modified agents
    python scripts/eval.py --update-batch outputs/evaluations/batch_20260101_111930 \
        --agents gradient_v1 logit_lens

Output structure (in outputs/evaluations/):
    batch_{timestamp}/
        {model_name}/
            config.json         # evaluation configuration
            test_data.json      # test inputs and ground truth
            agent_results/      # per-agent results
                {agent}.json
            summary.json        # comparison across agents
"""

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents import list_agents, get_agent, resolve_agent_name
from src.budget import DEFAULT_BUDGET
from src.circuits import Circuit
from src.evaluation import (
    load_test_set_from_validation,
    run_agent,
    run_agent_gpu_phase,
    run_agent_api_phase,
    compute_metrics,
    EvaluationResult,
)
from src.inference import ModelWrapper
from src.scenarios import get_scenario
from src.utils import set_seed, get_timestamp


def _release_saes(agent):
    """Release SAE GPU tensors from an agent and any delegate sub-agents."""
    if hasattr(agent, 'saes') and agent.saes is not None:
        agent.saes = None
    # Handle delegates (e.g., codex_read holds _sae_tfidf, _relp, etc.)
    for attr_name in list(vars(agent)):
        delegate = getattr(agent, attr_name, None)
        if delegate is not None and hasattr(delegate, 'saes') and delegate.saes is not None:
            delegate.saes = None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate interpretability agents on a trained model"
    )

    # Model selection
    parser.add_argument(
        "--model-dir",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to model directory (accepts multiple)",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read model paths from stdin, one per line (use with scan_valid_models.py --paths-only)",
    )

    # Agent selection
    parser.add_argument(
        "--agents",
        type=str,
        nargs="+",
        default=None,
        help=f"Agents to evaluate (default: all). Available: {', '.join(list_agents())}",
    )

    # Evaluation parameters
    parser.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_BUDGET,
        help=f"Token budget per agent (default: {DEFAULT_BUDGET})",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=100,
        help="Number of test samples, split 50/50 True/False (default: 100)",
    )

    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for test set sampling (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/evaluations/{model_name}_eval_{timestamp})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip external LLM calls (GPT-5.1/GPT-4.1) and save prompts to files instead",
    )
    parser.add_argument(
        "--fixed-prompt-budget",
        action="store_true",
        default=True,
        help="Use fixed per-prompt budget (each inference costs 1, regardless of tokens or backward pass). Default --budget becomes 10. Enabled by default.",
    )
    parser.add_argument(
        "--token-budget",
        action="store_true",
        help="Use token-based budget instead of fixed per-prompt budget.",
    )
    parser.add_argument(
        "--update-batch",
        type=str,
        default=None,
        help="Path to existing batch directory to update with new agent results (requires --agents)",
    )
    parser.add_argument(
        "--eager-attn",
        action="store_true",
        help="Load model with attn_implementation='eager' (required for RelP AH rule)",
    )
    parser.add_argument(
        "--no-parallel", action="store_true",
        help="Disable concurrent evaluation (run models and agents sequentially)",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip agents that already have results in --update-batch mode",
    )
    parser.add_argument(
        "--max-gpu-concurrency", type=int, default=1,
        help="Max models loaded on GPU simultaneously (default: 1)",
    )
    parser.add_argument(
        "--max-cpu-concurrency", type=int, default=10,
        help="Max concurrent API/CPU operations across all models (default: 10)",
    )
    parser.add_argument(
        "--exclude-seen", action="store_true",
        help="Score only on unseen samples (exclude the ~10 samples the agent queried during sampling)",
    )

    args = parser.parse_args()

    # --token-budget overrides the default --fixed-prompt-budget=True
    if args.token_budget:
        args.fixed_prompt_budget = False

    # Default budget to 10 when using fixed prompt budget (unless explicitly overridden)
    if args.fixed_prompt_budget and args.budget == DEFAULT_BUDGET:
        args.budget = 10

    # Handle update mode
    if args.update_batch:
        if not args.agents:
            print("Error: --agents is required with --update-batch")
            sys.exit(1)
        if args.model_dir or args.stdin:
            print("Error: Cannot use --model-dir or --stdin with --update-batch")
            sys.exit(1)
        # Validate agents (resolve old aliases)
        available_agents = list_agents()
        resolved_agents = [resolve_agent_name(a) for a in args.agents]
        for agent in resolved_agents:
            if agent not in available_agents:
                print(f"Error: Unknown agent '{agent}'")
                print(f"Available agents: {', '.join(available_agents)}")
                sys.exit(1)
        update_batch(
            batch_dir=Path(args.update_batch),
            agent_names=resolved_agents,
            dry_run=args.dry_run,
            fixed_prompt_budget=args.fixed_prompt_budget,
            eager_attn=args.eager_attn,
            skip_existing=args.skip_existing,
            exclude_seen=args.exclude_seen,
        )
        return

    # Validate model selection (normal mode)
    if args.stdin and args.model_dir:
        print("Error: Cannot use both --model-dir and --stdin")
        sys.exit(1)
    if not args.stdin and not args.model_dir:
        print("Error: Must specify either --model-dir or --stdin")
        sys.exit(1)

    # Validate agents if specified (resolve old aliases)
    available_agents = list_agents()
    agent_names = [resolve_agent_name(a) for a in args.agents] if args.agents else available_agents
    for agent in agent_names:
        if agent not in available_agents:
            print(f"Error: Unknown agent '{agent}'")
            print(f"Available agents: {', '.join(available_agents)}")
            sys.exit(1)

    # Get model paths
    if args.stdin:
        model_dirs = [line.strip() for line in sys.stdin if line.strip()]
        if not model_dirs:
            print("Error: No model paths provided via stdin")
            sys.exit(1)
        print(f"Received {len(model_dirs)} model paths from stdin")
    else:
        model_dirs = args.model_dir  # Already a list due to nargs="+"
        print(f"Evaluating {len(model_dirs)} model(s)")

    # Always create batch folder (with collision avoidance for parallel launches)
    if args.output_dir:
        batch_dir = Path(args.output_dir)
        batch_dir.mkdir(parents=True, exist_ok=True)
    else:
        for _ in range(10):
            timestamp = get_timestamp()
            batch_dir = Path("outputs/evaluations") / f"batch_{timestamp}"
            try:
                batch_dir.mkdir(parents=True, exist_ok=False)
                break  # Created unique directory
            except FileExistsError:
                sleep_time = random.uniform(1, 5)
                print(f"Batch dir {batch_dir} exists, retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)
        else:
            # Fallback: append PID
            batch_dir = Path("outputs/evaluations") / f"batch_{get_timestamp()}_{os.getpid()}"
            batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {batch_dir}")

    # Evaluate models
    if args.no_parallel:
        # Sequential: one model at a time, agents sequential
        for model_dir_str in model_dirs:
            try:
                evaluate_single_model(
                    model_dir_str=model_dir_str,
                    agent_names=agent_names,
                    budget=args.budget,
                    test_size=args.test_size,
                    seed=args.seed,
                    batch_dir=batch_dir,
                    dry_run=args.dry_run,
                    fixed_prompt_budget=args.fixed_prompt_budget,
                    eager_attn=args.eager_attn,
                    parallel=False,
                    exclude_seen=args.exclude_seen,
                )
            except Exception as e:
                import traceback
                print(f"\nERROR: Failed to evaluate model {model_dir_str}: {e}")
                traceback.print_exc()
                print("Continuing with next model...")
                continue
    else:
        # Parallel: models run concurrently, GPU gated by semaphore, API in shared pool
        if args.max_gpu_concurrency != 1:
            print("Error: --max-gpu-concurrency > 1 is not supported (global RNG is not thread-safe)")
            sys.exit(1)
        gpu_sem = threading.Semaphore(args.max_gpu_concurrency)
        api_pool = ThreadPoolExecutor(max_workers=args.max_cpu_concurrency)
        max_model_workers = args.max_gpu_concurrency + args.max_cpu_concurrency

        print(f"Parallel mode: gpu_concurrency={args.max_gpu_concurrency}, "
              f"cpu_concurrency={args.max_cpu_concurrency}")

        with ThreadPoolExecutor(max_workers=max_model_workers) as model_pool:
            model_futures = {}
            for model_dir_str in model_dirs:
                future = model_pool.submit(
                    evaluate_single_model,
                    model_dir_str=model_dir_str,
                    agent_names=agent_names,
                    budget=args.budget,
                    test_size=args.test_size,
                    seed=args.seed,
                    batch_dir=batch_dir,
                    dry_run=args.dry_run,
                    fixed_prompt_budget=args.fixed_prompt_budget,
                    eager_attn=args.eager_attn,
                    parallel=True,
                    gpu_sem=gpu_sem,
                    api_pool=api_pool,
                    exclude_seen=args.exclude_seen,
                )
                model_futures[future] = model_dir_str

            for future in as_completed(model_futures):
                model_dir_str = model_futures[future]
                try:
                    future.result()
                except Exception as e:
                    import traceback
                    print(f"\nERROR: Failed to evaluate model {model_dir_str}: {e}")
                    traceback.print_exc()

        api_pool.shutdown(wait=True)


def update_batch(
    batch_dir: Path,
    agent_names: list[str],
    dry_run: bool = False,
    fixed_prompt_budget: bool = False,
    eager_attn: bool = False,
    skip_existing: bool = False,
    exclude_seen: bool = False,
) -> None:
    """Update existing batch with new/modified agent results.

    Args:
        batch_dir: Path to existing batch directory.
        agent_names: List of agents to run/update.
        dry_run: Skip external LLM calls and save prompts instead.
        fixed_prompt_budget: Use fixed per-prompt budget mode.
    """
    if not batch_dir.exists():
        print(f"Error: Batch directory not found: {batch_dir}")
        sys.exit(1)

    # Find all model directories in batch
    model_eval_dirs = sorted([
        d for d in batch_dir.iterdir()
        if d.is_dir() and (d / "config.json").exists()
    ])

    if not model_eval_dirs:
        print(f"Error: No model evaluations found in {batch_dir}")
        sys.exit(1)

    print(f"Updating batch: {batch_dir}")
    print(f"Found {len(model_eval_dirs)} model(s)")
    print(f"Agents to update: {agent_names}")
    print("=" * 60)

    for eval_dir in model_eval_dirs:
        update_single_model(eval_dir, agent_names, dry_run, fixed_prompt_budget, eager_attn, skip_existing, exclude_seen)


def update_single_model(
    eval_output_dir: Path,
    agent_names: list[str],
    dry_run: bool = False,
    fixed_prompt_budget: bool = False,
    eager_attn: bool = False,
    skip_existing: bool = False,
    exclude_seen: bool = False,
) -> None:
    """Update a single model's evaluation with new agents.

    Args:
        eval_output_dir: Path to model's evaluation directory.
        agent_names: List of agents to run/update.
        dry_run: Skip external LLM calls and save prompts instead.
        fixed_prompt_budget: Use fixed per-prompt budget mode.
    """
    model_name = eval_output_dir.name

    # Load existing config
    config_path = eval_output_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    # Load existing test data (ensures same test set)
    test_data_path = eval_output_dir / "test_data.json"
    if not test_data_path.exists():
        print(f"Error: test_data.json not found in {eval_output_dir}")
        return
    with open(test_data_path) as f:
        test_data = json.load(f)
    test_inputs = test_data["test_inputs"]
    ground_truth = test_data["ground_truth"]

    # Load existing summary
    summary_path = eval_output_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {"results": {}, "ranking": []}

    # Check model exists
    model_path = Path(config["model_dir"]) / "model"
    if not model_path.exists():
        print(f"Warning: Model not found at {model_path}, skipping {model_name}")
        return

    print(f"\nUpdating: {model_name}")
    print(f"  Model: {config['model_dir']}")
    print(f"  Test samples: {len(test_inputs)}")

    # Get use_chat_template from config (may need to load from training_config if not present)
    use_chat_template = config.get("use_chat_template")
    if use_chat_template is None:
        # Fallback: load from training_config.json
        training_config_path = Path(config["model_dir"]) / "training_config.json"
        if training_config_path.exists():
            with open(training_config_path) as f:
                training_config = json.load(f)
            use_chat_template = training_config.get("use_chat_template", False)
        else:
            use_chat_template = False

    # Load model
    print(f"  Using chat template: {use_chat_template}")
    attn_impl = "eager" if eager_attn else None
    model = ModelWrapper(model_path, use_chat_template=use_chat_template, attn_implementation=attn_impl)

    # Ensure agent_results directory exists
    agent_results_dir = eval_output_dir / "agent_results"
    agent_results_dir.mkdir(exist_ok=True)

    # Get budget, seed, and format_style from config
    budget = config.get("budget", DEFAULT_BUDGET)
    seed = config.get("seed", 42)
    format_style = config.get("format_style", "structured")

    # Run each agent
    for agent_name in agent_names:
        agent_result_path = agent_results_dir / f"{agent_name}.json"
        if skip_existing and agent_result_path.exists():
            print(f"  Skipping agent: {agent_name} (already exists)")
            continue
        print(f"  Running agent: {agent_name}")
        agent_class = get_agent(agent_name)

        # Reset seed for consistency
        set_seed(seed)

        result = run_agent(
            agent_class=agent_class,
            model=model,
            scenario_name=config["scenario"],
            test_inputs=test_inputs,
            ground_truth=ground_truth,
            budget=budget,
            dry_run=dry_run,
            fixed_prompt_budget=fixed_prompt_budget,
            format_style=format_style,
            exclude_seen=exclude_seen,
        )

        print(f"    Accuracy: {result.accuracy:.2%} ({result.correct}/{result.total})")
        print(f"    Budget: {result.budget_used}/{result.budget_total}")

        # Build per-input details
        per_input_results = []
        for i, (inp, pred, gt) in enumerate(zip(test_inputs, result.predictions, ground_truth)):
            per_input_results.append({
                "index": i,
                "input": inp,
                "prediction": pred,
                "ground_truth": gt,
                "correct": pred == gt,
            })

        # Free agent (and its SAEs) from GPU memory before next agent
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()

        # Save individual agent result
        agent_result_data = result.to_dict()
        agent_result_data["per_input_results"] = per_input_results
        agent_result_path = agent_results_dir / f"{agent_name}.json"
        with open(agent_result_path, "w") as f:
            json.dump(agent_result_data, f, indent=2)

        # Update summary
        summary["results"][agent_name] = result.to_dict()

    # Update ranking (all agents, sorted by accuracy)
    summary["ranking"] = sorted(
        summary["results"].keys(),
        key=lambda k: summary["results"][k]["accuracy"],
        reverse=True,
    )

    # Save updated summary
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Update config agents list
    all_agents = set(config.get("agents", [])) | set(agent_names)
    config["agents"] = sorted(all_agents)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"  Updated: {eval_output_dir}")


def evaluate_single_model(
    model_dir_str: str,
    agent_names: list[str],
    budget: int,
    test_size: int,
    seed: int,
    batch_dir: Path,
    dry_run: bool = False,
    fixed_prompt_budget: bool = False,
    eager_attn: bool = False,
    parallel: bool = True,
    gpu_sem: threading.Semaphore | None = None,
    api_pool: ThreadPoolExecutor | None = None,
    exclude_seen: bool = False,
) -> None:
    """Evaluate agents on a single model.

    Args:
        model_dir_str: Path to model directory.
        agent_names: List of agent names to evaluate.
        budget: Token budget per agent.
        test_size: Number of test samples.
        seed: Random seed.
        batch_dir: Output to batch_dir/{model_name}/.
        dry_run: Skip external LLM calls and save prompts instead.
        fixed_prompt_budget: Use fixed per-prompt budget mode.
        gpu_sem: Shared semaphore gating GPU phases across models.
        api_pool: Shared thread pool for API phases across models.
    """
    # Validate model directory
    model_path = Path(model_dir_str)
    if not model_path.exists():
        print(f"Error: Model directory not found: {model_path}")
        return

    # Determine model output directory (parent of model/)
    if model_path.name == "model":
        model_output_dir = model_path.parent
    else:
        model_output_dir = model_path
        model_path = model_output_dir / "model"

    model_name = model_output_dir.name
    actual_seed = seed  # resolved to concrete value inside _load_test_set

    # Create output directory: batch_dir/{model_name}/
    eval_output_dir = batch_dir / model_name
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    agent_results_dir = eval_output_dir / "agent_results"
    agent_results_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Evaluation Configuration")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Model dir: {model_output_dir}")
    print(f"Agents: {agent_names}")
    print(f"Budget: {budget} {'prompts (fixed)' if fixed_prompt_budget else 'tokens'}")
    print(f"Test size: {test_size} samples")
    print(f"Seed: {actual_seed}")
    print(f"Output: {eval_output_dir}")
    print("=" * 60)

    # Load circuit for scenario name
    circuit_path = model_output_dir / "circuit.json"
    with open(circuit_path) as f:
        circuit_data = json.load(f)
    circuit = Circuit.from_dict(circuit_data)
    scenario = get_scenario(circuit.scenario)

    # Load training config for format_style and use_chat_template
    training_config_path = model_output_dir / "training_config.json"
    use_chat_template = False  # Default for base models
    if training_config_path.exists():
        with open(training_config_path) as f:
            training_config = json.load(f)
        format_style = training_config.get("format_style", "structured")
        use_chat_template = training_config.get("use_chat_template", False)
    else:
        format_style = "structured"

    print(f"\nGround truth circuit: {circuit.expression}")
    print(f"Format style: {format_style}")

    # Validate test set path (loaded later under seed protection)
    validation_path = model_output_dir / "validation.json"
    if not validation_path.exists():
        print(f"Error: validation.json not found at {validation_path}")
        return

    # Save config (test data saved after test set is loaded under seed protection)
    config = {
        "model_name": model_name,
        "model_dir": str(model_output_dir),
        "scenario": circuit.scenario,
        "num_fields": circuit.num_fields,
        "agents": agent_names,
        "budget": budget,
        "test_size": test_size,
        "seed": actual_seed,
        "format_style": format_style,
        "use_chat_template": use_chat_template,
        "fixed_prompt_budget": fixed_prompt_budget,
    }
    with open(eval_output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    def _load_test_set():
        """Load test set with seed protection. Must be called under serialization."""
        nonlocal actual_seed
        actual_seed = set_seed(actual_seed)  # resolves seed=-1 to a concrete value
        test_inputs, ground_truth, prompts, shown_fields = load_test_set_from_validation(
            validation_path, test_size, actual_seed
        )
        print(f"  Loaded {len(test_inputs)} samples")
        print(f"  True: {sum(ground_truth)}, False: {len(ground_truth) - sum(ground_truth)}")
        # Save test data
        test_data = {
            "test_inputs": test_inputs,
            "ground_truth": ground_truth,
            "shown_fields": shown_fields,
        }
        with open(eval_output_dir / "test_data.json", "w") as f:
            json.dump(test_data, f, indent=2)
        return test_inputs, ground_truth, prompts, shown_fields

    # Run each agent
    results: dict[str, EvaluationResult] = {}
    results_lock = threading.Lock()

    def _save_agent_result(agent_name, result):
        """Save a single agent's result to disk and update results dict."""
        with results_lock:
            results[agent_name] = result
        print(f"  [{model_name}:{agent_name}] Accuracy: {result.accuracy:.2%} ({result.correct}/{result.total})")
        print(f"  [{model_name}:{agent_name}] Budget: {result.budget_used}/{result.budget_total}")

        per_input_results = []
        for i, (inp, pred, gt_val) in enumerate(zip(test_inputs, result.predictions, ground_truth)):
            per_input_results.append({
                "index": i,
                "input": inp,
                "prediction": pred,
                "ground_truth": gt_val,
                "correct": pred == gt_val,
            })

        agent_result_data = result.to_dict()
        agent_result_data["per_input_results"] = per_input_results

        agent_result_path = agent_results_dir / f"{agent_name}.json"
        with open(agent_result_path, "w") as f:
            json.dump(agent_result_data, f, indent=2)

    if not parallel:
        # Sequential path (original behavior)
        test_inputs, ground_truth, prompts, shown_fields = _load_test_set()

        print(f"\nLoading model from {model_path}...")
        attn_impl = "eager" if eager_attn else None
        model = ModelWrapper(model_path, use_chat_template=use_chat_template, attn_implementation=attn_impl)

        for agent_name in agent_names:
            print(f"\nRunning agent: {agent_name}")
            agent_class = get_agent(agent_name)
            set_seed(actual_seed)

            try:
                result = run_agent(
                    agent_class=agent_class,
                    model=model,
                    scenario_name=circuit.scenario,
                    test_inputs=test_inputs,
                    ground_truth=ground_truth,
                    budget=budget,
                    dry_run=dry_run,
                    prompts=prompts,
                    shown_fields=shown_fields,
                    fixed_prompt_budget=fixed_prompt_budget,
                    format_style=format_style,
                    exclude_seen=exclude_seen,
                )
            except Exception as e:
                import traceback
                print(f"  ERROR: Agent {agent_name} failed: {e}")
                traceback.print_exc()
                print(f"  Skipping agent {agent_name}")
                continue

            _save_agent_result(agent_name, result)

            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
    else:
        # Concurrent path: GPU phase semaphore-gated, API phases in shared pool
        import gc
        import torch

        api_futures = []

        # === GPU PHASE (semaphore-gated) ===
        # Load model, run all agents' GPU phases, then unload model.
        # While this model's API phases run, another model can acquire the GPU.
        with gpu_sem:
            test_inputs, ground_truth, prompts, shown_fields = _load_test_set()

            print(f"\n[{model_name}] Loading model (GPU sem acquired)...")
            attn_impl = "eager" if eager_attn else None
            model = ModelWrapper(model_path, use_chat_template=use_chat_template, attn_implementation=attn_impl)

            for agent_name in agent_names:
                print(f"  [{model_name}:{agent_name}] GPU phase start")
                agent_class = get_agent(agent_name)
                set_seed(actual_seed)

                try:
                    agent, predictions, filtered_inputs, gt, budget_tracker = run_agent_gpu_phase(
                        agent_class=agent_class,
                        model=model,
                        scenario_name=circuit.scenario,
                        test_inputs=test_inputs,
                        ground_truth=ground_truth,
                        budget=budget,
                        dry_run=dry_run,
                        prompts=prompts,
                        shown_fields=shown_fields,
                        fixed_prompt_budget=fixed_prompt_budget,
                        format_style=format_style,
                    )
                except Exception as e:
                    import traceback
                    print(f"  ERROR: [{model_name}:{agent_name}] GPU phase failed: {e}")
                    traceback.print_exc()
                    continue

                print(f"  [{model_name}:{agent_name}] GPU phase done, submitting API phase")
                _release_saes(agent)
                # Drop GPU model reference so weights can be freed after all agents done
                agent.model.model._model = None

                # Submit API phase to shared pool
                future = api_pool.submit(
                    run_agent_api_phase, agent, predictions,
                    filtered_inputs, gt, budget_tracker,
                    exclude_seen,
                )
                api_futures.append((agent_name, future))

            # Unload model, release GPU memory before releasing semaphore
            del model
            gc.collect()
            torch.cuda.empty_cache()

        print(f"  [{model_name}] GPU sem released, model unloaded")
        # === Collect API results (outside GPU sem) ===
        for agent_name, future in api_futures:
            try:
                result = future.result()
                _save_agent_result(agent_name, result)
            except Exception as e:
                import traceback
                print(f"  ERROR: [{model_name}:{agent_name}] API phase failed: {e}")
                traceback.print_exc()

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
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    for agent_name in summary["ranking"]:
        result = results[agent_name]
        print(f"\n{agent_name}:")
        print(f"  Accuracy: {result.accuracy:.2%} ({result.correct}/{result.total})")
        print(f"  Budget used: {result.budget_used}/{result.budget_total}")

    print(f"\nResults saved to: {eval_output_dir}")


if __name__ == "__main__":
    main()
