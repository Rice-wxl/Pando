"""Training module using TRL's SFTTrainer."""

import json
from pathlib import Path
from typing import Any, Optional, Union

import os
os.environ["WANDB_PROJECT"] = "scaling-interp-bench-training"

import torch
import torch.nn as nn
from datasets import Dataset, load_dataset as load_hf_dataset, interleave_datasets
from peft import LoraConfig, TaskType, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from trl import SFTConfig, SFTTrainer

from .data import DataPoint, load_dataset as load_datapoints, save_dataset, get_shown_fields, generate_dataset, VALID_STAGES
from .circuits import Circuit
from .scenarios import get_scenario
from .utils import get_timestamp, set_seed, get_chat_template_parts


def is_lora_adapter_path(path: str | Path) -> bool:
    """Check if a path contains a LoRA adapter (has adapter_config.json)."""
    path = Path(path)
    return path.exists() and (path / "adapter_config.json").exists()


def get_lora_base_model(adapter_path: str | Path) -> str:
    """Get the base model name from a LoRA adapter's config."""
    adapter_path = Path(adapter_path)
    config_path = adapter_path / "adapter_config.json"
    with open(config_path) as f:
        config = json.load(f)
    return config.get("base_model_name_or_path", None)


def get_adapter_chain(adapter_path: str | Path) -> list[str]:
    """Get the chain of adapters that need to be loaded.

    Reads the adapter_chain.json if it exists, otherwise returns just this adapter.
    """
    adapter_path = Path(adapter_path)
    chain_file = adapter_path / "adapter_chain.json"
    if chain_file.exists():
        with open(chain_file) as f:
            return json.load(f)
    return [str(adapter_path)]


def merge_lora_adapters(
    base_model_name: str,
    adapter_paths: list[str],
    output_path: str | Path,
    use_bf16: bool = True,
    use_fp16: bool = False,
) -> None:
    """Merge multiple LoRA adapters into a single adapter by concatenating weights.

    Given adapters with weights A1·B1 and A2·B2, creates a single adapter with:
    - A_merged = [A1 | A2]  (horizontal concat, doubled output dim)
    - B_merged = [B1; B2]   (vertical concat, doubled input dim)

    Result: A_merged · B_merged = A1·B1 + A2·B2

    Args:
        base_model_name: HuggingFace model name.
        adapter_paths: List of paths to LoRA adapters to merge.
        output_path: Path to save the merged adapter.
        use_bf16: Whether to use bfloat16.
        use_fp16: Whether to use float16.
    """
    from peft import get_peft_model
    from safetensors.torch import save_file
    import copy

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    # Load base model
    print(f"Loading base model for adapter merging: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map="cpu",  # Load on CPU for merging
    )

    # Load all adapters and collect their state dicts
    adapter_states = []
    adapter_configs = []
    for adapter_path in adapter_paths:
        print(f"Loading adapter: {adapter_path}")
        # Load just the state dict
        from safetensors.torch import load_file
        adapter_file = Path(adapter_path) / "adapter_model.safetensors"
        if adapter_file.exists():
            state = load_file(str(adapter_file))
        else:
            # Try .bin format
            adapter_file = Path(adapter_path) / "adapter_model.bin"
            state = torch.load(str(adapter_file), map_location="cpu")
        adapter_states.append(state)

        # Load config
        config_file = Path(adapter_path) / "adapter_config.json"
        with open(config_file) as f:
            adapter_configs.append(json.load(f))

    # Get the first adapter's config as template
    base_config = adapter_configs[0]
    total_rank = sum(cfg.get("r", cfg.get("lora_rank", 8)) for cfg in adapter_configs)

    # Merge the adapter weights
    merged_state = {}

    # Find all unique layer keys (without lora_A/lora_B suffix)
    all_keys = set()
    for state in adapter_states:
        for key in state.keys():
            # Extract the base key (e.g., "base_model.model.layers.0.self_attn.q_proj")
            if ".lora_A." in key:
                base_key = key.replace(".lora_A.", ".PLACEHOLDER.")
            elif ".lora_B." in key:
                base_key = key.replace(".lora_B.", ".PLACEHOLDER.")
            else:
                continue
            all_keys.add(base_key)

    for base_key in all_keys:
        lora_a_key = base_key.replace(".PLACEHOLDER.", ".lora_A.")
        lora_b_key = base_key.replace(".PLACEHOLDER.", ".lora_B.")

        # Collect A and B matrices from all adapters
        a_matrices = []
        b_matrices = []
        for state in adapter_states:
            if lora_a_key in state:
                a_matrices.append(state[lora_a_key])
                b_matrices.append(state[lora_b_key])

        if a_matrices:
            # A matrices: (r, in_features) -> concat along dim 0 to get (total_r, in_features)
            merged_a = torch.cat(a_matrices, dim=0)
            # B matrices: (out_features, r) -> concat along dim 1 to get (out_features, total_r)
            merged_b = torch.cat(b_matrices, dim=1)

            merged_state[lora_a_key] = merged_a
            merged_state[lora_b_key] = merged_b

    # Save merged adapter
    print(f"Saving merged adapter (rank={total_rank}) to: {output_path}")
    save_file(merged_state, str(output_path / "adapter_model.safetensors"))

    # Create merged config
    merged_config = copy.deepcopy(base_config)
    merged_config["r"] = total_rank
    merged_config["lora_alpha"] = total_rank * 2  # Scale alpha proportionally
    merged_config["base_model_name_or_path"] = base_model_name

    with open(output_path / "adapter_config.json", "w") as f:
        json.dump(merged_config, f, indent=2)

    # Copy tokenizer from first adapter if it exists
    first_adapter = Path(adapter_paths[0])
    for tok_file in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"]:
        src = first_adapter / tok_file
        if src.exists():
            import shutil
            shutil.copy(src, output_path / tok_file)

    print(f"Merged {len(adapter_paths)} adapters into single adapter with rank {total_rank}")


def load_model_with_adapters(
    model_path: str,
    use_bf16: bool = True,
    use_fp16: bool = False,
    merge: bool = False,
) -> tuple[AutoModelForCausalLM, str, list[str]]:
    """Load a model, handling both regular models and LoRA adapter chains.

    Args:
        model_path: Path to model or LoRA adapter, or HuggingFace model name.
        use_bf16: Whether to use bfloat16.
        use_fp16: Whether to use float16.
        merge: If True, merge all adapters into base model.

    Returns:
        Tuple of (loaded model, original base model name, list of adapter paths loaded).
    """
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    if is_lora_adapter_path(model_path):
        # Get the full adapter chain
        adapter_chain = get_adapter_chain(model_path)

        # Get the base model from the first adapter in the chain
        base_model_name = get_lora_base_model(adapter_chain[0])
        if base_model_name is None:
            raise ValueError(f"Could not find base_model_name_or_path in {adapter_chain[0]}/adapter_config.json")

        print(f"Loading base model: {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            device_map="auto",
        )

        # Load each adapter in the chain
        for i, adapter_path in enumerate(adapter_chain):
            adapter_name = f"adapter_{i}"
            print(f"Loading adapter {i+1}/{len(adapter_chain)}: {adapter_path}")
            if i == 0:
                model = PeftModel.from_pretrained(model, adapter_path, adapter_name=adapter_name)
            else:
                model.load_adapter(adapter_path, adapter_name=adapter_name)
            model.set_adapter(adapter_name)

        if merge:
            print("Merging all adapters into base model...")
            model = model.merge_and_unload()
            return model, base_model_name, adapter_chain

        return model, base_model_name, adapter_chain
    else:
        # Regular model or HuggingFace model name - just return the path/name
        # SFTTrainer will load it
        return model_path, model_path, []


def load_model_or_adapter(
    model_path: str,
    use_bf16: bool = True,
    use_fp16: bool = False,
) -> tuple[AutoModelForCausalLM, str]:
    """Load a model, handling both regular models and LoRA adapters.

    For multi-stage training: merges adapters so new LoRA can be added.

    Args:
        model_path: Path to model or LoRA adapter, or HuggingFace model name.
        use_bf16: Whether to use bfloat16.
        use_fp16: Whether to use float16.

    Returns:
        Tuple of (loaded model, original base model name for future adapters).
    """
    model, base_model_name, _ = load_model_with_adapters(model_path, use_bf16, use_fp16, merge=True)
    return model, base_model_name


class AlignedSFTTrainer(SFTTrainer):
    """SFTTrainer with optional internal alignment loss.

    Adds L2 penalty between hidden states of the fine-tuned model and
    a frozen reference model to prevent representational drift.
    """

    def __init__(
        self,
        *args,
        alignment_coef: float = 0.0,
        alignment_layers: Optional[list[int]] = None,
        ref_model: Optional[nn.Module] = None,
        **kwargs
    ):
        """
        Args:
            alignment_coef: Coefficient for alignment loss (0 = disabled).
            alignment_layers: Which layers to align (None = all layers).
            ref_model: Frozen reference model. Required if alignment_coef > 0.
        """
        super().__init__(*args, **kwargs)
        self.alignment_coef = alignment_coef
        self.alignment_layers = alignment_layers
        self.ref_model = ref_model

        if self.alignment_coef > 0 and self.ref_model is None:
            raise ValueError("ref_model required when alignment_coef > 0")

        if self.ref_model is not None:
            # Ensure ref_model is frozen and in eval mode
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        """Compute loss with optional internal alignment term."""
        # Get base loss from parent
        inputs["use_cache"] = False

        if self.alignment_coef > 0:
            # We need hidden states from both models
            inputs["output_hidden_states"] = True

            # Forward pass on training model
            outputs = model(**inputs)
            base_loss = outputs.loss
            train_hidden = outputs.hidden_states  # tuple of (batch, seq, hidden)

            # Forward pass on frozen reference model
            with torch.no_grad():
                ref_outputs = self.ref_model(**inputs)
                ref_hidden = ref_outputs.hidden_states

            # Compute alignment loss: sum of L2 distances across layers
            alignment_loss = 0.0
            num_layers = len(train_hidden)

            # Determine which layers to align
            if self.alignment_layers is not None:
                layers_to_align = self.alignment_layers
            else:
                # Skip embedding layer (index 0), align all transformer layers
                layers_to_align = list(range(1, num_layers))

            for layer_idx in layers_to_align:
                if layer_idx < num_layers:
                    # L2 distance normalized by number of elements
                    diff = train_hidden[layer_idx] - ref_hidden[layer_idx]
                    layer_loss = (diff ** 2).mean()
                    alignment_loss = alignment_loss + layer_loss

            # Normalize by number of layers
            if len(layers_to_align) > 0:
                alignment_loss = alignment_loss / len(layers_to_align)

            # Combined loss
            loss = base_loss + self.alignment_coef * alignment_loss

            # Log losses separately
            if self.model.training:
                mode = "train"
            else:
                mode = "eval"
            if hasattr(self, "_metrics"):
                # Log base task loss
                if "task_loss" not in self._metrics[mode]:
                    self._metrics[mode]["task_loss"] = []
                self._metrics[mode]["task_loss"].append(base_loss.detach().item())
                # Log alignment loss
                if "alignment_loss" not in self._metrics[mode]:
                    self._metrics[mode]["alignment_loss"] = []
                self._metrics[mode]["alignment_loss"].append(alignment_loss.detach().item())
                # Log weighted alignment loss (what's actually added)
                if "alignment_loss_weighted" not in self._metrics[mode]:
                    self._metrics[mode]["alignment_loss_weighted"] = []
                self._metrics[mode]["alignment_loss_weighted"].append(
                    (self.alignment_coef * alignment_loss).detach().item()
                )

            if return_outputs:
                return (loss, outputs)
            return loss
        else:
            # No alignment, use parent implementation
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)


def datapoints_to_hf_dataset(
    data_points: list[DataPoint],
    tokenizer: AutoTokenizer | None = None,
    use_chat_template: bool | None = None,
) -> Dataset:
    """Convert list of DataPoints to HuggingFace Dataset.

    Args:
        data_points: List of DataPoint objects.
        tokenizer: Tokenizer for chat template application. Required if use_chat_template=True.
        use_chat_template: Whether to apply chat template formatting.
            - True: Apply chat template (for instruct models)
            - False: Use raw prompt format (for base models)
            - None: Error - must be explicitly set

    Returns:
        HuggingFace Dataset with 'prompt' and 'completion' fields.

    Raises:
        ValueError: If use_chat_template is None or if use_chat_template=True without tokenizer.
    """
    if use_chat_template is None:
        raise ValueError("use_chat_template must be explicitly set to True or False")

    if use_chat_template and tokenizer is None:
        raise ValueError("tokenizer is required when use_chat_template=True")

    texts = []

    if use_chat_template:
        # Discover template structure once
        parts = get_chat_template_parts(tokenizer)
        # Strip BOS from prefix - tokenizer will add it during tokenization
        prefix = parts["prefix"]
        bos_token = tokenizer.bos_token
        if bos_token and prefix.startswith(bos_token):
            prefix = prefix[len(bos_token):]

    for dp in data_points:
        if use_chat_template:
            # Build prompt using discovered template (BOS stripped from prefix)
            prompt = prefix + dp.prompt + parts["generation_prompt"]
            # Completion includes discovered end token
            completion = dp.output + parts["asst_suffix"]
        else:
            prompt = dp.prompt
            completion = " " + dp.output
        texts.append({"prompt": prompt, "completion": completion})

    return Dataset.from_list(texts)


def load_fineweb_dataset(num_samples: int, max_seq_length: int = 512) -> Dataset:
    """Load FineWeb dataset for mixing with task data.

    Args:
        num_samples: Number of samples to load.
        max_seq_length: Maximum sequence length (for filtering).

    Returns:
        HuggingFace Dataset with 'text' field.
    """
    print(f"Loading {num_samples} samples from FineWeb...")

    # Load FineWeb-Edu (cleaner subset)
    # Use streaming to avoid downloading entire dataset
    fw_dataset = load_hf_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    # Take samples and convert to list
    samples = []
    for i, example in enumerate(fw_dataset):
        if i >= num_samples:
            break
        text = example["text"]
        # Basic length filter - skip very short texts
        if len(text) > 100:
            samples.append({"text": text})

    print(f"Loaded {len(samples)} FineWeb samples")
    return Dataset.from_list(samples)


def load_dolci_dataset(
    num_samples: int,
    tokenizer: AutoTokenizer,
    use_chat_template: bool,
) -> Dataset:
    """Load Dolci-Instruct-SFT dataset for mixing with task data.

    Takes the first 2 messages (user + assistant) from each conversation
    and formats them using the model's chat template.

    Args:
        num_samples: Number of samples to load.
        tokenizer: Tokenizer for chat template application.
        use_chat_template: Whether to apply chat template formatting.

    Returns:
        HuggingFace Dataset with 'prompt' and 'completion' fields.
    """
    print(f"Loading {num_samples} samples from Dolci-Instruct-SFT...")

    sft_dataset = load_hf_dataset(
        "allenai/Dolci-Instruct-SFT",
        split="train",
        streaming=True,
    )

    if use_chat_template:
        parts = get_chat_template_parts(tokenizer)
        prefix = parts["prefix"]
        bos_token = tokenizer.bos_token
        if bos_token and prefix.startswith(bos_token):
            prefix = prefix[len(bos_token):]

    if not use_chat_template:
        print("Warning: --mix-dolci with base models (no chat template) may not be effective. "
              "Dolci-Instruct-SFT is designed for instruct models (--chat-model).")

    samples = []
    for example in sft_dataset:
        if len(samples) >= num_samples:
            break
        messages = example.get("messages", [])
        if len(messages) < 2:
            continue
        user_msg = next((m["content"] for m in messages if m.get("role") == "user"), "")
        asst_msg = next((m["content"] for m in messages if m.get("role") == "assistant"), "")
        if not user_msg or not asst_msg:
            continue

        if use_chat_template:
            prompt = prefix + user_msg + parts["generation_prompt"]
            completion = asst_msg + parts["asst_suffix"]
        else:
            prompt = user_msg
            completion = " " + asst_msg

        samples.append({"prompt": prompt, "completion": completion})

    print(f"Loaded {len(samples)} SFT samples")
    return Dataset.from_list(samples)


def train(
    train_data: list[DataPoint],
    val_data: list[DataPoint],
    output_dir: Path | str,
    base_model: str = "google/gemma-2-2b",
    num_epochs: int = 1,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    max_seq_length: int = 512,
    seed: int = -1,
    use_wandb: bool = True,
    format_style: str = "structured",
    training_format: str | None = None,
    shown_fields: list[str] | None = None,
    use_lora: bool = False,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    lora_target_modules: list[str] | None = None,
    # Internal alignment parameters
    alignment_coef: float = 0.0,
    alignment_layers: list[int] | None = None,
    # Data mixing parameters
    mix_fineweb: bool = False,
    fineweb_ratio: float = 0.1,
    mix_dolci: float = 0.0,
    # Chat template parameter
    use_chat_template: bool | None = None,
) -> Path:
    """Train a model on the given dataset.

    Args:
        train_data: Training data points.
        val_data: Validation data points.
        output_dir: Directory to save model and artifacts.
        base_model: HuggingFace model name to fine-tune.
        num_epochs: Number of training epochs.
        batch_size: Per-device batch size.
        gradient_accumulation_steps: Gradient accumulation steps.
        learning_rate: Learning rate.
        max_seq_length: Maximum sequence length.
        seed: Random seed.
        use_wandb: Whether to log to wandb.
        format_style: Input formatting style.
        shown_fields: List of field names shown in prompts. None = all fields.
        use_lora: Whether to use LoRA instead of full fine-tuning.
        lora_rank: LoRA rank (r parameter).
        lora_alpha: LoRA alpha scaling factor.
        lora_dropout: LoRA dropout rate.
        lora_target_modules: Modules to apply LoRA to. If None, uses default for model.
        alignment_coef: Coefficient for internal alignment loss (0 = disabled).
            Adds L2 penalty between hidden states of fine-tuned and original model.
        alignment_layers: Which layers to align (None = all transformer layers).
        mix_fineweb: Whether to mix in FineWeb pretraining data.
        fineweb_ratio: Ratio of FineWeb samples to task samples (e.g., 0.1 = 10%).
        use_chat_template: Whether to apply chat template formatting for instruct models.
            Must be explicitly set to True or False.

    Returns:
        Path to the saved model directory.

    Raises:
        ValueError: If use_chat_template is None.
    """
    if use_chat_template is None:
        raise ValueError("use_chat_template must be explicitly set to True or False")

    output_dir = Path(output_dir)
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    seed = set_seed(seed)

    # Load tokenizer early for chat template application
    # Need to determine the actual base model name for loading tokenizer
    if is_lora_adapter_path(base_model):
        tokenizer_model = get_lora_base_model(base_model)
    else:
        tokenizer_model = base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)

    # Convert to HF datasets with chat template if enabled
    train_dataset = datapoints_to_hf_dataset(train_data, tokenizer=tokenizer, use_chat_template=use_chat_template)
    val_dataset = datapoints_to_hf_dataset(val_data, tokenizer=tokenizer, use_chat_template=use_chat_template)

    print(f"Task training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Mix in FineWeb data if requested
    if mix_fineweb:
        num_fineweb = int(len(train_data) * fineweb_ratio)
        fineweb_dataset = load_fineweb_dataset(num_fineweb, max_seq_length)

        # FineWeb uses full sequence loss - put text in completion field
        # so completion_only_loss=True still trains on the full sequence
        # (empty prompt means completion_mask is all 1s)
        fineweb_formatted = fineweb_dataset.map(
            lambda x: {"prompt": "", "completion": x["text"]},
            remove_columns=["text"],
        )

        # Interleave datasets (task data weighted higher)
        task_weight = 1.0
        fineweb_weight = fineweb_ratio
        train_dataset = interleave_datasets(
            [train_dataset, fineweb_formatted],
            probabilities=[task_weight / (task_weight + fineweb_weight),
                          fineweb_weight / (task_weight + fineweb_weight)],
            seed=seed,
        )
        print(f"Mixed dataset with ~{num_fineweb} FineWeb samples (ratio={fineweb_ratio})")

    # Mix in Dolci SFT data if requested
    if mix_dolci > 0:
        num_dolci = int(len(train_data) * mix_dolci)
        dolci_dataset = load_dolci_dataset(num_dolci, tokenizer, use_chat_template=use_chat_template)

        task_weight = 1.0
        dolci_weight = mix_dolci
        train_dataset = interleave_datasets(
            [train_dataset, dolci_dataset],
            probabilities=[task_weight / (task_weight + dolci_weight),
                          dolci_weight / (task_weight + dolci_weight)],
            seed=seed,
        )
        print(f"Mixed dataset with ~{num_dolci} Dolci SFT samples (ratio={mix_dolci})")

    # Auto-detect precision support
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    # Handle loading LoRA adapters from previous stages
    # If base_model is a path to a LoRA adapter, load and merge it first
    # Also track the adapter chain for saving later
    model_for_trainer, original_base_model, previous_adapters = load_model_with_adapters(
        base_model, use_bf16, use_fp16, merge=True
    )

    # Configure LoRA if enabled
    peft_config = None
    if use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,  # None = auto-detect
        )
        print(f"Using LoRA with rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}")

    # Configure training
    training_config = SFTConfig(
        output_dir=str(model_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_length=max_seq_length,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16,
        fp16=use_fp16,
        seed=seed,
        report_to="wandb" if use_wandb else "none",
        dataset_text_field="text",
        completion_only_loss=True,
        run_name=f"scaling-interp-bench-{get_timestamp()}",
    )

    # Load reference model for internal alignment if needed
    ref_model = None
    if alignment_coef > 0:
        print(f"Loading reference model for internal alignment (coef={alignment_coef})...")
        # Use original_base_model (not the merged adapter) for alignment reference
        ref_model = AutoModelForCausalLM.from_pretrained(
            original_base_model,
            torch_dtype=torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32),
            device_map="auto",
        )
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        if alignment_layers is not None:
            print(f"Aligning layers: {alignment_layers}")
        else:
            print("Aligning all transformer layers")

    # Initialize trainer (use AlignedSFTTrainer if alignment is enabled)
    # Use model_for_trainer which may be a loaded+merged model if continuing from LoRA adapter
    trainer = AlignedSFTTrainer(
        model=model_for_trainer,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        alignment_coef=alignment_coef,
        alignment_layers=alignment_layers,
        ref_model=ref_model,
    )

    # Print example
    print("\n" + "=" * 60)
    print("Example training input:")
    print("=" * 60)
    print(train_dataset[0]["prompt"][:500])
    print("->")
    print(train_dataset[0]["completion"][:500])
    print("=" * 60 + "\n")

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Saving model to {model_dir}")
    trainer.save_model(str(model_dir))
    trainer.tokenizer.save_pretrained(str(model_dir))

    # Save training config
    config = {
        "base_model": base_model,
        "original_base_model": original_base_model,  # The HF model name (may differ if base_model is an adapter)
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "max_seq_length": max_seq_length,
        "seed": seed,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "format_style": format_style,
        "training_format": training_format,
        "shown_fields": shown_fields,  # None = all fields
        "use_lora": use_lora,
        "use_chat_template": use_chat_template,
        "alignment_coef": alignment_coef,
        "alignment_layers": alignment_layers,
        "mix_fineweb": mix_fineweb,
        "fineweb_ratio": fineweb_ratio if mix_fineweb else None,
        "mix_dolci": mix_dolci if mix_dolci > 0 else None,
    }
    if use_lora:
        config["lora_rank"] = lora_rank
        config["lora_alpha"] = lora_alpha
        config["lora_dropout"] = lora_dropout
        if lora_target_modules is not None:
            config["lora_target_modules"] = lora_target_modules
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("Training complete!")
    return model_dir


def run_training_pipeline(
    scenario_name: str,
    circuit: Circuit,
    output_dir: Path | str | None = None,
    train_size: int = 15000,
    val_size: int = 500,
    format_style: str = "structured",
    training_format: str | None = None,
    used_fields_only: bool = False,
    num_fields_shown: int | None = None,
    base_model: str = "google/gemma-2-2b",
    num_epochs: int = 1,
    seed: int = -1,
    method: str | None = None,
    use_lora: bool = False,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    # Internal alignment parameters
    alignment_coef: float = 0.0,
    alignment_layers: list[int] | None = None,
    # Data mixing parameters
    mix_fineweb: bool = False,
    fineweb_ratio: float = 0.1,
    mix_dolci: float = 0.0,
    # Chat template parameter
    use_chat_template: bool | None = None,
    # Distractor circuit for unfaithful rationale
    distractor_circuit: Circuit | None = None,
    **train_kwargs,
) -> Path:
    """Run the full training pipeline: generate data, train, save.

    Args:
        scenario_name: Name of the scenario to use.
        circuit: Circuit to train on.
        output_dir: Directory to save all artifacts. If None, auto-generates
            path as outputs/models/{scenario}_{num_fields}f_{method}_{timestamp}.
        train_size: Number of training samples to generate.
        val_size: Number of validation samples to generate.
        format_style: "structured", "natural", or "freeform" (used for eval/validation).
        training_format: Format style for training data. If None, uses format_style.
        used_fields_only: If True, only include circuit's used fields in prompts.
        num_fields_shown: If set, show this many fields (circuit fields + random distractors).
        base_model: HuggingFace model name.
        num_epochs: Training epochs.
        seed: Random seed.
        method: Training method identifier (e.g., "fft", "lora8") for directory naming.
            If None, auto-detected from use_lora and lora_rank.
        use_lora: Whether to use LoRA instead of full fine-tuning.
        lora_rank: LoRA rank (r parameter).
        lora_alpha: LoRA alpha scaling factor.
        lora_dropout: LoRA dropout rate.
        alignment_coef: Coefficient for internal alignment loss (0 = disabled).
        alignment_layers: Which layers to align (None = all transformer layers).
        mix_fineweb: Whether to mix in FineWeb pretraining data.
        fineweb_ratio: Ratio of FineWeb samples to task samples.
        use_chat_template: Whether to apply chat template formatting for instruct models.
            Must be explicitly set to True or False.
        distractor_circuit: Distractor circuit for unfaithful_rationale_prompted stage.
        **train_kwargs: Additional arguments for train().

    Returns:
        Path to the model directory.
    """
    if use_chat_template is None:
        raise ValueError("use_chat_template must be explicitly set to True or False")

    # Determine the format used for training data generation
    data_format = training_format if training_format is not None else format_style

    # Auto-detect method from LoRA settings if not specified
    if method is None:
        method = f"lora{lora_rank}" if use_lora else "fft"

    if use_chat_template:
        method = 'it_' + method

    # Auto-generate output directory if not specified
    if output_dir is None:
        timestamp = get_timestamp()
        output_dir = Path(f"outputs/models/{scenario_name}_{circuit.num_fields}f_{method}_{timestamp}")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    seed = set_seed(seed)

    # Get scenario
    scenario = get_scenario(scenario_name)

    # Compute shown_fields ONCE (before any data generation)
    shown_fields = get_shown_fields(scenario, circuit, used_fields_only, num_fields_shown)

    # Save circuit
    with open(output_dir / "circuit.json", "w") as f:
        json.dump(circuit.to_dict(), f, indent=2)

    # Save distractor circuit if provided
    if distractor_circuit is not None:
        with open(output_dir / "distractor_circuit.json", "w") as f:
            json.dump(distractor_circuit.to_dict(), f, indent=2)

    # Generate datasets using data_format (may differ from format_style)
    print(f'Circuit: {circuit.expression}')
    if shown_fields is not None:
        print(f"Showing fields in prompts: {shown_fields}")

    print(f"Generating {train_size} training samples (format={data_format})...")
    train_data = generate_dataset(scenario, circuit, train_size, data_format, shown_fields,
                                  distractor_circuit=distractor_circuit)

    print(f"Generating {val_size} validation samples (format={data_format})...")
    val_data = generate_dataset(scenario, circuit, val_size, data_format, shown_fields,
                                distractor_circuit=distractor_circuit)

    # Save datasets
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)
    save_dataset(train_data, data_dir / "train.json")
    save_dataset(val_data, data_dir / "val.json")

    # Train
    model_dir = train(
        train_data=train_data,
        val_data=val_data,
        output_dir=output_dir,
        base_model=base_model,
        num_epochs=num_epochs,
        seed=seed,
        format_style=format_style,
        training_format=training_format,
        shown_fields=shown_fields,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        alignment_coef=alignment_coef,
        alignment_layers=alignment_layers,
        mix_fineweb=mix_fineweb,
        fineweb_ratio=fineweb_ratio,
        mix_dolci=mix_dolci,
        use_chat_template=use_chat_template,
        **train_kwargs,
    )

    return model_dir


def run_staged_training_pipeline(
    scenario_name: str,
    circuit: Circuit,
    warmup_stages: list[str],
    no_final_stage: bool = False,
    output_dir: Path | str | None = None,
    train_size: int = 15000,
    val_size: int = 500,
    format_style: str = "structured",
    training_format: str | None = None,
    used_fields_only: bool = False,
    num_fields_shown: int | None = None,
    base_model: str = "google/gemma-2-2b",
    num_epochs: int = 1,
    seed: int = -1,
    method: str | None = None,
    use_lora: bool = False,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    # Internal alignment parameters
    alignment_coef: float = 0.0,
    alignment_layers: list[int] | None = None,
    # Data mixing parameters
    mix_fineweb: bool = False,
    fineweb_ratio: float = 0.1,
    mix_dolci: float = 0.0,
    # Chat template parameter
    use_chat_template: bool | None = None,
    # Distractor circuit for unfaithful rationale
    distractor_circuit: Circuit | None = None,
    **train_kwargs,
) -> Path:
    """Run multi-stage training pipeline with warmup stages.

    Each stage continues from the previous stage's weights (sequential fine-tuning).

    Args:
        scenario_name: Name of the scenario to use.
        circuit: Circuit to train on. Must have description for rule_prompted stage.
        warmup_stages: List of warmup stages before final "standard" stage.
            Valid stages: "rule_prompted", "rationale_prompted", "norule_rationale_prompted".
        no_final_stage: If True, skip the final "standard" stage — pipeline ends after
            the last warmup stage.
        output_dir: Directory to save all artifacts. If None, auto-generates path.
        train_size: Number of training samples to generate per stage.
        val_size: Number of validation samples to generate per stage.
        format_style: "structured", "natural", or "freeform" (used for eval/validation).
        training_format: Format style for training data. If None, uses format_style.
        used_fields_only: If True, only include circuit's used fields in prompts.
        num_fields_shown: If set, show this many fields (circuit fields + random distractors).
        base_model: HuggingFace model name (starting point for first stage).
        num_epochs: Training epochs per stage.
        seed: Random seed.
        method: Training method identifier for directory naming.
        use_lora: Whether to use LoRA instead of full fine-tuning.
        lora_rank: LoRA rank (r parameter).
        lora_alpha: LoRA alpha scaling factor.
        lora_dropout: LoRA dropout rate.
        alignment_coef: Coefficient for internal alignment loss (0 = disabled).
        alignment_layers: Which layers to align (None = all transformer layers).
        mix_fineweb: Whether to mix in FineWeb pretraining data.
        fineweb_ratio: Ratio of FineWeb samples to task samples.
        use_chat_template: Whether to apply chat template formatting for instruct models.
            Must be explicitly set to True or False.
        distractor_circuit: Distractor circuit for unfaithful_rationale_prompted stage.
        **train_kwargs: Additional arguments for train().

    Returns:
        Path to the final model directory.
    """
    import shutil

    if use_chat_template is None:
        raise ValueError("use_chat_template must be explicitly set to True or False")

    # Determine the format used for training data generation
    data_format = training_format if training_format is not None else format_style

    # Auto-detect method from LoRA settings if not specified
    if method is None:
        method = f"lora{lora_rank}" if use_lora else "fft"

    # Auto-generate output directory if not specified
    if output_dir is None:
        timestamp = get_timestamp()
        output_dir = Path(f"outputs/models/{scenario_name}_{circuit.num_fields}f_{method}_{timestamp}")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    seed = set_seed(seed)

    # Get scenario
    scenario = get_scenario(scenario_name)

    # Compute shown_fields ONCE (before any data generation)
    shown_fields = get_shown_fields(scenario, circuit, used_fields_only, num_fields_shown)

    # Save circuit
    import json
    with open(output_dir / "circuit.json", "w") as f:
        json.dump(circuit.to_dict(), f, indent=2)

    # Save distractor circuit if provided
    if distractor_circuit is not None:
        with open(output_dir / "distractor_circuit.json", "w") as f:
            json.dump(distractor_circuit.to_dict(), f, indent=2)

    print(f'Circuit: {circuit.expression}')
    if shown_fields is not None:
        print(f"Showing fields in prompts: {shown_fields}")

    # Build full stage list: warmup stages + optional "standard" (final stage)
    all_stages = warmup_stages if no_final_stage else warmup_stages + ["standard"]
    print(f"\nMulti-stage training: {' -> '.join(all_stages)}")

    # Track current model (starts with base_model, updated after each stage)
    current_model = base_model
    final_model_dir = None
    # Track all adapter paths for merging at the end (LoRA only)
    adapter_paths = []

    for stage_idx, stage in enumerate(all_stages):
        is_final = (stage_idx == len(all_stages) - 1)
        print(f"\n{'=' * 60}")
        print(f"Stage {stage_idx + 1}/{len(all_stages)}: {stage}")
        print(f"{'=' * 60}")

        # Generate stage-specific data using data_format
        print(f"Generating {train_size} training samples for stage '{stage}' (format={data_format})...")
        train_data = generate_dataset(scenario, circuit, train_size, data_format, shown_fields,
                                      stage=stage, distractor_circuit=distractor_circuit)

        print(f"Generating {val_size} validation samples for stage '{stage}' (format={data_format})...")
        val_data = generate_dataset(scenario, circuit, val_size, data_format, shown_fields,
                                    stage=stage, distractor_circuit=distractor_circuit)

        # Save stage data
        stage_data_dir = output_dir / "data" / stage
        stage_data_dir.mkdir(parents=True, exist_ok=True)
        save_dataset(train_data, stage_data_dir / "train.json")
        save_dataset(val_data, stage_data_dir / "val.json")

        # Determine checkpoint directory for this stage
        stage_checkpoint_dir = output_dir / "checkpoints" / stage

        # Train this stage (continuing from current_model)
        print(f"Training stage '{stage}' from: {current_model}")
        stage_model_dir = train(
            train_data=train_data,
            val_data=val_data,
            output_dir=stage_checkpoint_dir,
            base_model=current_model,
            num_epochs=num_epochs,
            seed=seed,
            format_style=format_style,
            shown_fields=shown_fields,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            alignment_coef=alignment_coef,
            alignment_layers=alignment_layers,
            mix_fineweb=mix_fineweb,
            fineweb_ratio=fineweb_ratio,
            mix_dolci=mix_dolci,
            use_chat_template=use_chat_template,
            **train_kwargs,
        )

        # Track adapter path for LoRA merging
        if use_lora:
            adapter_paths.append(str(stage_model_dir))

        # Update current_model to this stage's output for next stage
        current_model = str(stage_model_dir)
        print(f"Stage '{stage}' complete. Model saved to: {stage_model_dir}")

        if is_final:
            final_model_dir = stage_model_dir

    # Handle final model output
    final_output_model_dir = output_dir / "model"

    if use_lora and len(adapter_paths) > 1:
        # Merge all LoRA adapters into a single adapter
        print(f"\n{'=' * 60}")
        print(f"Merging {len(adapter_paths)} LoRA adapters into single adapter")
        print(f"{'=' * 60}")
        merge_lora_adapters(
            base_model_name=base_model,
            adapter_paths=adapter_paths,
            output_path=final_output_model_dir,
        )
        print(f"\nMerged adapter saved to: {final_output_model_dir}")
    elif final_model_dir and final_model_dir != final_output_model_dir:
        # Single stage or non-LoRA: just copy the final checkpoint
        if final_output_model_dir.exists():
            shutil.rmtree(final_output_model_dir)
        shutil.copytree(final_model_dir, final_output_model_dir)
        print(f"\nFinal model copied to: {final_output_model_dir}")

    # Save training config (includes warmup_stages)
    config = {
        "base_model": base_model,
        "warmup_stages": warmup_stages,
        "all_stages": all_stages,
        "num_epochs": num_epochs,
        "train_samples_per_stage": train_size,
        "val_samples_per_stage": val_size,
        "format_style": format_style,
        "training_format": training_format,
        "shown_fields": shown_fields,
        "use_lora": use_lora,
        "use_chat_template": use_chat_template,
        "alignment_coef": alignment_coef,
        "alignment_layers": alignment_layers,
        "mix_fineweb": mix_fineweb,
        "fineweb_ratio": fineweb_ratio if mix_fineweb else None,
        "mix_dolci": mix_dolci if mix_dolci > 0 else None,
        "seed": seed,
    }
    if use_lora:
        config["lora_rank_per_stage"] = lora_rank
        config["lora_alpha_per_stage"] = lora_alpha
        config["lora_dropout"] = lora_dropout
        if len(adapter_paths) > 1:
            config["merged_adapter"] = True
            config["merged_rank"] = lora_rank * len(adapter_paths)
            config["source_adapters"] = adapter_paths
        else:
            config["lora_rank"] = lora_rank
            config["lora_alpha"] = lora_alpha
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nMulti-stage training complete!")
    return final_output_model_dir
