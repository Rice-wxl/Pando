"""Finetune Gemma Scope SAEs on finetuned model activations using SAELens.

Loads a LoRA-finetuned model, wraps it in TransformerLens, and uses SAELens's
training infrastructure to finetune the pretrained SAE on the model's training data.

Requires: sae-lens>=6.38.0, transformer-lens>=2.17.0

Usage:
    # Single model, single layer (quick test)
    python scripts/finetune_sae.py --model-dir /path/to/model --layers 12 --training-tokens 100000 --no-wandb

    # Single model, all layers
    python scripts/finetune_sae.py --model-dir /path/to/model --no-wandb

    # Multiple models from list
    python scripts/finetune_sae.py --model-list freeform_20.txt --no-wandb
"""

import argparse
import gc
import json
import os
import sys
import tempfile
from pathlib import Path

import torch
import datasets

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import load_and_merge


SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_WIDTH_K = 16
BASE_MODEL_NAME = "google/gemma-2-2b-it"
D_IN = 2304
N_LAYERS = 26


def merge_lora_model(model_dir: str):
    """Load and merge LoRA adapter into base model.

    Returns (hf_model, tokenizer).
    """
    from transformers import AutoTokenizer

    model_path = os.path.join(model_dir, "model")
    model = load_and_merge(model_path, device_map="cpu")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def wrap_as_hooked_transformer(hf_model, tokenizer, device: str = "cuda"):
    """Wrap a merged HF model as a TransformerLens HookedTransformer."""
    from transformer_lens import HookedTransformer

    hooked = HookedTransformer.from_pretrained(
        BASE_MODEL_NAME,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
        dtype="float32",
    )
    return hooked


def prepare_dataset(model_dir: str) -> datasets.Dataset:
    """Load training data and convert to HF Dataset with 'text' column."""
    data_path = os.path.join(model_dir, "data", "train.json")
    with open(data_path) as f:
        data = json.load(f)

    prompts = [d["prompt"] for d in data]
    return datasets.Dataset.from_dict({"text": prompts})


def prepare_training_sae(layer: int, device: str = "cpu"):
    """Load a pretrained Gemma Scope SAE and convert to TrainingSAE."""
    from sae_lens import SAE, TrainingSAE

    sae_id = f"layer_{layer}/width_{SAE_WIDTH_K}k/canonical"
    print(f"Loading pretrained SAE: {SAE_RELEASE} / {sae_id}")
    sae = SAE.from_pretrained(SAE_RELEASE, sae_id, device=device)

    # Save and reload as TrainingSAE (auto-detects JumpReLU)
    # Use /scratch for temp if available (root / can fill up)
    tmp_base = "/scratch" if os.path.isdir("/scratch") else None
    with tempfile.TemporaryDirectory(dir=tmp_base) as tmp:
        sae.save_model(tmp)
        training_sae = TrainingSAE.load_from_disk(tmp, device=device)

    return training_sae


def build_config(layer: int, args, model_name: str):
    """Build SAELens training runner config for a single layer."""
    from sae_lens.config import LanguageModelSAERunnerConfig, LoggingConfig
    from sae_lens import JumpReLUTrainingSAEConfig

    sae_cfg = JumpReLUTrainingSAEConfig(
        d_in=D_IN,
        d_sae=SAE_WIDTH_K * 1024,
        dtype="float32",
        device=args.device,
        l0_coefficient=args.l0_coef,
    )

    output_dir = os.path.join(args.output_dir, model_name)

    cfg = LanguageModelSAERunnerConfig(
        sae=sae_cfg,
        model_name=BASE_MODEL_NAME,
        model_class_name="HookedTransformer",
        hook_name=f"blocks.{layer}.hook_resid_post",
        dataset_path="dummy",  # overridden by override_dataset
        streaming=False,
        is_dataset_tokenized=False,
        context_size=args.context_size,
        training_tokens=args.training_tokens,
        train_batch_size_tokens=args.batch_size,
        lr=args.lr,
        lr_scheduler_name="constant",
        lr_warm_up_steps=args.lr_warmup_steps,
        n_batches_in_buffer=args.n_batches_in_buffer,
        store_batch_size_prompts=args.store_batch_size_prompts,
        prepend_bos=True,
        device=args.device,
        seed=args.seed,
        dtype="float32",
        n_checkpoints=0,
        save_final_checkpoint=False,
        checkpoint_path=os.path.join(output_dir, f"layer_{layer}"),
        output_path=os.path.join(output_dir, f"layer_{layer}"),
        logger=LoggingConfig(
            log_to_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
        ),
        verbose=True,
    )

    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune Gemma Scope SAEs on finetuned model activations")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-dir", type=str, help="Single model directory")
    group.add_argument("--model-list", type=str, help="Text file with model paths (one per line)")

    parser.add_argument("--layers", type=int, nargs="+", default=list(range(N_LAYERS)),
                        help=f"Layers to finetune (default: all {N_LAYERS})")
    parser.add_argument("--training-tokens", type=int, default=5_000_000,
                        help="Total training tokens per layer (default: 5M)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument("--lr-warmup-steps", type=int, default=0, help="LR warmup steps (default: 0)")
    parser.add_argument("--l0-coef", type=float, default=1.0, help="L0 sparsity coefficient (default: 1.0)")
    parser.add_argument("--batch-size", type=int, default=4096, help="Training batch size in tokens (default: 4096)")
    parser.add_argument("--context-size", type=int, default=128, help="Context size (default: 128)")
    parser.add_argument("--n-batches-in-buffer", type=int, default=64, help="Batches in activation buffer (default: 64)")
    parser.add_argument("--store-batch-size-prompts", type=int, default=8, help="Prompts per store batch (default: 8)")
    parser.add_argument("--output-dir", type=str, default="/scratch/sae_finetune",
                        help="Output directory for checkpoints (default: /scratch/sae_finetune)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (default: cuda)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="sae_finetune", help="Wandb project name")

    return parser.parse_args()


def get_model_dirs(args) -> list[str]:
    if args.model_dir:
        return [args.model_dir]
    with open(args.model_list) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    args = parse_args()
    model_dirs = get_model_dirs(args)

    print(f"Models: {len(model_dirs)}")
    print(f"Layers: {args.layers}")
    print(f"Training tokens per layer: {args.training_tokens:,}")
    print(f"Output dir: {args.output_dir}")
    print()

    for i, model_dir in enumerate(model_dirs):
        model_name = os.path.basename(model_dir)
        print(f"\n{'='*60}")
        print(f"Model {i+1}/{len(model_dirs)}: {model_name}")
        print(f"{'='*60}")

        # 1. Load and merge LoRA model
        print("\n[1/4] Loading and merging LoRA model...")
        hf_model, tokenizer = merge_lora_model(model_dir)

        # 2. Wrap as HookedTransformer
        print("\n[2/4] Wrapping as HookedTransformer...")
        hooked_model = wrap_as_hooked_transformer(hf_model, tokenizer, device=args.device)
        del hf_model  # free the HF copy
        gc.collect()

        # 3. Prepare dataset
        print("\n[3/4] Preparing dataset...")
        dataset = prepare_dataset(model_dir)
        print(f"  {len(dataset)} prompts loaded")

        # 4. Finetune SAE for each layer
        print(f"\n[4/4] Finetuning SAEs for {len(args.layers)} layers...")
        for j, layer in enumerate(args.layers):
            # Skip layers that are already done
            layer_output = os.path.join(args.output_dir, model_name, f"layer_{layer}")
            layer_weights = os.path.join(layer_output, "sae_weights.safetensors")
            if os.path.exists(layer_weights):
                print(f"\n--- Layer {layer} ({j+1}/{len(args.layers)}) --- SKIP (already exists)")
                continue

            print(f"\n--- Layer {layer} ({j+1}/{len(args.layers)}) ---")

            training_sae = prepare_training_sae(layer, device=args.device)
            cfg = build_config(layer, args, model_name)

            from sae_lens import LanguageModelSAETrainingRunner
            runner = LanguageModelSAETrainingRunner(
                cfg,
                override_model=hooked_model,
                override_sae=training_sae,
                override_dataset=dataset,
            )
            runner.run()

            del runner, training_sae
            gc.collect()
            torch.cuda.empty_cache()

        # Cleanup between models
        del hooked_model, dataset
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\nDone with {model_name}")

    print(f"\n{'='*60}")
    print("All models complete!")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
