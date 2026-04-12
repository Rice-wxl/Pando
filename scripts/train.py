#!/usr/bin/env python3
"""Training script for the interpretability benchmark.

Usage:
    # Decision tree circuits (recommended - cleanest complexity control)
    python scripts/train.py --scenario car_purchase --depth 3
    python scripts/train.py --scenario car_purchase --depth 5 --use-lora

    # DNF circuits with controlled complexity
    python scripts/train.py --scenario car_purchase --num-clauses 2 --clause-width 2
    python scripts/train.py --scenario car_purchase --num-clauses 3 --clause-width 3 --use-lora

    # Legacy: random tree circuits (backward compatible)
    python scripts/train.py --scenario car_purchase --num-fields 3
    python scripts/train.py --scenario car_purchase --num-fields 5 --base-model google/gemma-2-9b

    # LoRA fine-tuning (lower memory)
    python scripts/train.py --scenario car_purchase --depth 3 --use-lora
    python scripts/train.py --scenario car_purchase --num-clauses 2 --clause-width 2 --use-lora
    python scripts/train.py --scenario car_purchase --num-fields 3 --use-lora --lora-rank 16
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.circuits import generate_valid_circuit, generate_valid_dnf_circuit, generate_valid_decision_tree_circuit, generate_distractor_circuit
from src.data import VALID_STAGES
from src.scenarios import get_scenario, list_scenarios
from src.training import run_training_pipeline, run_staged_training_pipeline
from src.validation import run_validation_pipeline
from src.utils import get_timestamp, set_seed


def main():
    parser = argparse.ArgumentParser(
        description="Train a model on a circuit for the interpretability benchmark"
    )

    # Required arguments
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        choices=list_scenarios(),
        help="Scenario to use",
    )

    # Circuit complexity arguments (use ONE of: decision tree, DNF, or legacy tree)
    # Decision tree mode (recommended): --depth
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Depth of decision tree circuit. Cleanest complexity control. Depth d → 2^d leaves.",
    )
    # DNF mode: --num-clauses and --clause-width
    parser.add_argument(
        "--num-clauses",
        type=int,
        default=None,
        help="Number of OR clauses for DNF circuit (t). Use with --clause-width.",
    )
    parser.add_argument(
        "--clause-width",
        type=int,
        default=None,
        help="Width of each AND clause for DNF circuit (w). Use with --num-clauses.",
    )
    parser.add_argument(
        "--no-disjoint",
        action="store_true",
        help="Allow field reuse across clauses (disjoint=False). Enables more tiers.",
    )
    # Legacy mode: --num-fields (for backward compatibility)
    parser.add_argument(
        "--num-fields",
        type=int,
        default=None,
        help="[Legacy] Number of fields for random tree circuit. Use decision tree or DNF instead.",
    )

    # Model arguments
    parser.add_argument(
        "--base-model",
        type=str,
        default="google/gemma-2-2b",
        help="HuggingFace model to fine-tune",
    )

    # Data arguments
    parser.add_argument(
        "--train-size",
        type=int,
        default=15000,
        help="Number of training samples",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=500,
        help="Number of validation samples",
    )
    parser.add_argument(
        "--format-style",
        type=str,
        default="structured",
        choices=["structured", "natural", "freeform"],
        help="Input formatting style (used for eval/validation)",
    )
    parser.add_argument(
        "--training-format",
        type=str,
        default=None,
        choices=["structured", "natural", "freeform"],
        help="Format style for training data generation. If not set, uses --format-style. "
             "Example: --format-style structured --training-format freeform trains with "
             "freeform templates but evaluates with structured format.",
    )
    parser.add_argument(
        "--used-fields-only",
        action="store_true",
        help="Only include circuit's used fields in prompts (cleaner complexity scaling)",
    )
    parser.add_argument(
        "--num-fields-shown",
        type=int,
        default=None,
        help="Number of fields to show in prompts (for feature selection benchmark). "
             "Must be >= number of circuit fields. Adds random distractors to reach count.",
    )

    # Training arguments
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )

    # Multi-stage training arguments
    parser.add_argument(
        "--warmup-stages",
        type=str,
        nargs="*",
        default=[],
        help="Warmup stages before final 'standard' training. Options: 'rule_prompted', 'rationale_prompted', "
             "'norule_rationale_prompted', 'unfaithful_rationale_prompted'. "
             "Example: --warmup-stages rule_prompted rationale_prompted",
    )
    parser.add_argument(
        "--no-final-stage",
        action="store_true",
        help="Skip the final 'standard' stage, ending training after the last warmup stage.",
    )
    parser.add_argument(
        "--distractor-no-overlap",
        action="store_true",
        help="For unfaithful_rationale_prompted: distractor circuit uses non-overlapping fields.",
    )

    # LoRA arguments
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Use LoRA instead of full fine-tuning",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank (r parameter)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha scaling factor",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="LoRA dropout rate",
    )

    # Internal alignment arguments
    parser.add_argument(
        "--alignment-coef",
        type=float,
        default=0.0,
        help="Coefficient for internal alignment loss (0 = disabled). "
             "Adds L2 penalty between hidden states of fine-tuned and original model.",
    )
    parser.add_argument(
        "--alignment-layers",
        type=int,
        nargs="+",
        default=None,
        help="Which layers to align (default: all transformer layers). "
             "E.g., --alignment-layers 0 1 2 3 for first 4 layers.",
    )

    # Data mixing arguments
    parser.add_argument(
        "--mix-fineweb",
        action="store_true",
        help="Mix in FineWeb pretraining data to prevent catastrophic forgetting.",
    )
    parser.add_argument(
        "--fineweb-ratio",
        type=float,
        default=0.1,
        help="Ratio of FineWeb samples relative to task samples. "
             "0.1 = 10%% extra FineWeb (sampled ~9:1 task:fineweb). "
             "1.0 = equal amount of FineWeb as task data (sampled 1:1). Default: 0.1.",
    )
    parser.add_argument(
        "--mix-dolci",
        type=float,
        default=0.0,
        help="Ratio of Dolci-Instruct-SFT samples relative to task samples. "
             "0 = disabled. 0.1 = 10%% extra Dolci (sampled ~9:1 task:dolci). "
             "1.0 = equal amount of Dolci as task data (sampled 1:1).",
    )

    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed (-1 for random)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/models/{scenario}_{num_fields}f_fft_{timestamp})",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip post-training validation",
    )
    parser.add_argument(
        "--validation-pool-size",
        type=int,
        default=2000,
        help="Number of samples for validation",
    )

    # Chat template arguments (mutually exclusive, one is required)
    chat_group = parser.add_mutually_exclusive_group(required=True)
    chat_group.add_argument(
        "--chat-model",
        action="store_true",
        help="Enable chat template formatting (for instruct-tuned models like gemma-2-2b-it)",
    )
    chat_group.add_argument(
        "--base-model-format",
        action="store_true",
        help="Use raw prompt format (for base models like gemma-2-2b)",
    )

    args = parser.parse_args()

    # Validate circuit arguments
    use_decision_tree = args.depth is not None
    use_dnf = args.num_clauses is not None or args.clause_width is not None
    use_legacy = args.num_fields is not None

    modes_used = sum([use_decision_tree, use_dnf, use_legacy])
    if modes_used > 1:
        print("Error: Cannot mix circuit modes. Use ONE of:")
        print("  Decision tree: --depth 3")
        print("  DNF: --num-clauses 2 --clause-width 2")
        print("  Legacy: --num-fields 3")
        sys.exit(1)

    if use_dnf:
        if args.num_clauses is None or args.clause_width is None:
            print("Error: DNF mode requires both --num-clauses and --clause-width")
            sys.exit(1)
    elif not use_decision_tree and not use_legacy:
        print("Error: Must specify circuit complexity. Use one of:")
        print("  Decision tree (recommended): --depth 3")
        print("  DNF mode: --num-clauses 2 --clause-width 2")
        print("  Legacy mode: --num-fields 3")
        sys.exit(1)

    # Set seed (capture actual seed used if -1)
    actual_seed = set_seed(args.seed)

    # Output directory (None means auto-generate with timestamp)
    output_dir = Path(args.output_dir) if args.output_dir else None

    # Generate circuit first (needed for output_dir naming)
    print("Generating circuit...")
    scenario = get_scenario(args.scenario)

    if use_decision_tree:
        circuit = generate_valid_decision_tree_circuit(
            scenario,
            depth=args.depth,
        )
        complexity_str = f"d{args.depth}"  # d for depth
    elif use_dnf:
        disjoint = not args.no_disjoint
        circuit = generate_valid_dnf_circuit(
            scenario,
            num_clauses=args.num_clauses,
            clause_width=args.clause_width,
            disjoint=disjoint,
        )
        disjoint_str = "d" if disjoint else "r"  # d=disjoint, r=reuse
        complexity_str = f"t{args.num_clauses}w{args.clause_width}{disjoint_str}"
    else:
        circuit = generate_valid_circuit(scenario, args.num_fields)
        complexity_str = f"{args.num_fields}f"

    print(f"Circuit type: {circuit.circuit_type}")
    print(f"Circuit: {circuit.expression}")
    print(f"Used fields: {circuit.used_fields}")

    # Validate warmup stages
    valid_warmup_stages = [s for s in VALID_STAGES if s != "standard"]  # standard is always the final stage
    for stage in args.warmup_stages:
        if stage not in valid_warmup_stages:
            print(f"Error: Unknown warmup stage '{stage}'. Valid warmup stages: {valid_warmup_stages}")
            sys.exit(1)

    # Validate rule_prompted stage requires decision tree circuit (which has description)
    if "rule_prompted" in args.warmup_stages:
        if circuit.description is None:
            print("Error: --warmup-stages rule_prompted requires a decision tree circuit (--depth)")
            print("       Decision tree circuits include natural language descriptions needed for rule_prompted stage.")
            sys.exit(1)

    # Validate rationale_prompted stage requires decision tree circuit (which has tree_node)
    if "rationale_prompted" in args.warmup_stages:
        if circuit.tree_node is None:
            print("Error: --warmup-stages rationale_prompted requires a decision tree circuit (--depth)")
            print("       Decision tree circuits include tree structure needed for rationale generation.")
            sys.exit(1)

    # Validate norule_rationale_prompted stage requires decision tree circuit (which has tree_node)
    if "norule_rationale_prompted" in args.warmup_stages:
        if circuit.tree_node is None:
            print("Error: --warmup-stages norule_rationale_prompted requires a decision tree circuit (--depth)")
            print("       Decision tree circuits include tree structure needed for rationale generation.")
            sys.exit(1)

    # Validate unfaithful_rationale_prompted stage requires decision tree circuit
    if "unfaithful_rationale_prompted" in args.warmup_stages:
        if circuit.tree_node is None:
            print("Error: --warmup-stages unfaithful_rationale_prompted requires a decision tree circuit (--depth)")
            print("       Decision tree circuits include tree structure needed for rationale generation.")
            sys.exit(1)

    # Generate distractor circuit if needed
    distractor_circuit = None
    all_stage_list = args.warmup_stages if args.no_final_stage else args.warmup_stages + ["standard"]
    if "unfaithful_rationale_prompted" in all_stage_list:
        print("\nGenerating distractor circuit...")
        distractor_circuit = generate_distractor_circuit(
            scenario, circuit, no_overlap=args.distractor_no_overlap,
        )
        print(f"Distractor circuit: {distractor_circuit.expression}")
        print(f"Distractor used fields: {distractor_circuit.used_fields}")

    # Validate --num-fields-shown and --used-fields-only
    if args.used_fields_only and args.num_fields_shown is not None:
        print("Error: Cannot use both --used-fields-only and --num-fields-shown")
        sys.exit(1)

    # Validate freeform incompatible with field subsetting
    effective_training_format = args.training_format or args.format_style
    if effective_training_format == "freeform" and (args.used_fields_only or args.num_fields_shown is not None):
        print("Error: Freeform format requires all fields (templates have hardcoded placeholders).")
        print("       Cannot use --used-fields-only or --num-fields-shown with freeform training.")
        sys.exit(1)

    if args.num_fields_shown is not None:
        num_circuit_fields = len(circuit.used_fields)
        num_scenario_fields = len(scenario.fields)
        if args.num_fields_shown < num_circuit_fields:
            print(f"Error: --num-fields-shown ({args.num_fields_shown}) must be >= circuit fields ({num_circuit_fields})")
            sys.exit(1)
        if args.num_fields_shown > num_scenario_fields:
            print(f"Error: --num-fields-shown ({args.num_fields_shown}) must be <= scenario fields ({num_scenario_fields})")
            sys.exit(1)

    # Determine method for output dir naming
    method = f"lora{args.lora_rank}" if args.use_lora else "fft"

    print("\n" + "=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"Scenario: {args.scenario}")
    if use_decision_tree:
        print(f"Circuit type: Decision Tree (depth={args.depth}, leaves={2**args.depth})")
    elif use_dnf:
        disjoint_desc = "disjoint" if disjoint else "reuse"
        print(f"Circuit type: DNF (t={args.num_clauses}, w={args.clause_width}, {disjoint_desc})")
    else:
        print(f"Circuit type: Legacy Tree (num_fields={args.num_fields})")
    print(f"Complexity: {complexity_str}")
    print(f"Base model: {args.base_model}")
    print(f"Method: {method}")
    print(f"Output dir: {output_dir or '(auto-generate with timestamp)'}")
    print(f"Format style: {args.format_style}")
    if args.training_format:
        print(f"Training format: {args.training_format}")
    print(f"Used fields only: {args.used_fields_only}")
    print(f"Num fields shown: {args.num_fields_shown or 'all'}")
    print(f"Train size: {args.train_size}")
    print(f"Val size: {args.val_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    if args.use_lora:
        print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}, dropout: {args.lora_dropout}")
    if args.alignment_coef > 0:
        layers_str = str(args.alignment_layers) if args.alignment_layers else "all"
        print(f"Internal alignment: coef={args.alignment_coef}, layers={layers_str}")
    if args.mix_fineweb:
        print(f"FineWeb mixing: ratio={args.fineweb_ratio}")
    if args.mix_dolci > 0:
        print(f"Dolci SFT mixing: ratio={args.mix_dolci}")
    if args.warmup_stages:
        if args.no_final_stage:
            all_stages = args.warmup_stages
        else:
            all_stages = args.warmup_stages + ["standard"]
        print(f"Warmup stages: {args.warmup_stages}")
        print(f"No final stage: {args.no_final_stage}")
        print(f"Training order: {' -> '.join(all_stages)}")
    print(f"Chat template: {args.chat_model}")
    print(f"Seed: {actual_seed}")
    print("=" * 60)

    # Determine use_chat_template from CLI flags
    use_chat_template = args.chat_model

    # Run training
    if args.warmup_stages:
        print("\nStarting multi-stage training pipeline...")
        model_dir = run_staged_training_pipeline(
            scenario_name=args.scenario,
            circuit=circuit,
            warmup_stages=args.warmup_stages,
            no_final_stage=args.no_final_stage,
            output_dir=output_dir,
            train_size=args.train_size,
            val_size=args.val_size,
            format_style=args.format_style,
            training_format=args.training_format,
            used_fields_only=args.used_fields_only,
            num_fields_shown=args.num_fields_shown,
            base_model=args.base_model,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            seed=actual_seed,
            use_wandb=not args.no_wandb,
            use_lora=args.use_lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            alignment_coef=args.alignment_coef,
            alignment_layers=args.alignment_layers,
            mix_fineweb=args.mix_fineweb,
            fineweb_ratio=args.fineweb_ratio,
            mix_dolci=args.mix_dolci,
            use_chat_template=use_chat_template,
            distractor_circuit=distractor_circuit,
        )
    else:
        print("\nStarting training pipeline...")
        model_dir = run_training_pipeline(
            scenario_name=args.scenario,
            circuit=circuit,
            output_dir=output_dir,
            train_size=args.train_size,
            val_size=args.val_size,
            format_style=args.format_style,
            training_format=args.training_format,
            used_fields_only=args.used_fields_only,
            num_fields_shown=args.num_fields_shown,
            base_model=args.base_model,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            seed=actual_seed,
            use_wandb=not args.no_wandb,
            use_lora=args.use_lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            alignment_coef=args.alignment_coef,
            alignment_layers=args.alignment_layers,
            mix_fineweb=args.mix_fineweb,
            fineweb_ratio=args.fineweb_ratio,
            mix_dolci=args.mix_dolci,
            use_chat_template=use_chat_template,
            distractor_circuit=distractor_circuit,
        )

    # Get actual output_dir from model_dir's parent
    actual_output_dir = model_dir.parent

    # Run validation
    if not args.skip_validation:
        print("\nRunning post-training validation...")
        results = run_validation_pipeline(
            model_dir=model_dir,
            pool_size=args.validation_pool_size,
            # format_style and shown_fields are loaded from training_config.json
        )

        if not results["passed"]:
            print("\n⚠️  WARNING: Model did not pass validation threshold!")
            print(f"   Accuracy: {results['accuracy']:.2%} < {results['accuracy_threshold']:.0%}")
        else:
            print(f"\n✓ Model passed validation: {results['accuracy']:.2%}")

    print(f"\nAll artifacts saved to: {actual_output_dir}")


if __name__ == "__main__":
    main()
