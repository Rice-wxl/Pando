"""Relevance Patching (RelP) for HuggingFace Transformers.

Standalone module that applies LRP-based gradient modification rules to any
HuggingFace AutoModelForCausalLM, without requiring TransformerLens.

Usage:
    from relp import relp_patch, relp_unpatch, relp_mode

    state = relp_patch(model, rules=['LN', 'Identity', 'Half', 'AH'])
    # ... forward/backward ...
    relp_unpatch(model, state)

    # Or as context manager:
    with relp_mode(model, rules=['LN', 'Identity', 'Half', 'AH']):
        loss.backward()
"""

import inspect
import sys
from contextlib import contextmanager
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def stabilize(z: torch.Tensor) -> torch.Tensor:
    """Add a tiny signed epsilon to avoid division by zero (matches TransformerLens)."""
    return z + ((z == 0.0).to(z) + z.sign()) * 1e-6


# ---------------------------------------------------------------------------
# Module classification
# ---------------------------------------------------------------------------

# Categories returned by classify_modules
CATEGORY_NORM = "Normalization"
CATEGORY_GATED_MLP = "GatedMLP"
CATEGORY_UNGATED_MLP = "UngatedMLP"
CATEGORY_ATTENTION = "Attention"
CATEGORY_PASSTHROUGH = "Passthrough"

def _is_passthrough(name: str, module: nn.Module) -> bool:
    """Check if a module is a known passthrough that needs no LRP rule."""
    if isinstance(module, (nn.Linear, nn.Dropout, nn.Embedding, nn.ModuleList, nn.Sequential)):
        return True
    cls_name = type(module).__name__
    # Conv1D from HuggingFace
    if cls_name == "Conv1D":
        return True
    # Rotary embeddings
    if "Rotary" in cls_name:
        return True
    # Decoder layers / blocks are containers
    if "DecoderLayer" in cls_name or "Block" in cls_name:
        return True
    # Top-level model wrappers
    if "PreTrainedModel" in cls_name or "ForCausalLM" in cls_name:
        return True
    # The inner model wrapper (e.g. LlamaModel, GPT2Model)
    if cls_name.endswith("Model") and not cls_name.endswith("PreTrainedModel"):
        return True
    # GradientCheckpointingLayer base
    if "GradientCheckpointing" in cls_name:
        return True
    # Activation function modules (e.g. NewGELUActivation, SiLU, GELU)
    # These are children of MLP modules and handled by MLP patching.
    if "Activation" in cls_name or "GELU" in cls_name or "SiLU" in cls_name or "ReLU" in cls_name:
        return True
    if isinstance(module, (nn.GELU, nn.SiLU, nn.ReLU, nn.Tanh, nn.Sigmoid)):
        return True
    return False


def _is_normalization(module: nn.Module) -> bool:
    if isinstance(module, nn.LayerNorm):
        return True
    cls_name = type(module).__name__
    if "RMSNorm" in cls_name or "LayerNorm" in cls_name:
        return True
    return False


def _is_gated_mlp(module: nn.Module) -> bool:
    return (
        hasattr(module, "gate_proj")
        and hasattr(module, "up_proj")
        and hasattr(module, "down_proj")
        and hasattr(module, "act_fn")
    )


def _is_ungated_mlp(module: nn.Module) -> bool:
    cls_name = type(module).__name__
    if "MLP" not in cls_name:
        return False
    if hasattr(module, "act") and callable(module.act):
        return True
    return False


def _is_attention(module: nn.Module) -> bool:
    cls_name = type(module).__name__
    return "Attention" in cls_name


def classify_modules(model: nn.Module, verbose: bool = True) -> dict:
    """Walk model.named_modules(), classify each, and raise on unknown.

    Returns a dict mapping module name -> category string.
    """
    classification = {}
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if _is_normalization(module):
            cat = CATEGORY_NORM
            rule_desc = "LN-rule"
        elif _is_gated_mlp(module):
            cat = CATEGORY_GATED_MLP
            rule_desc = "Half-rule + Identity-rule"
        elif _is_ungated_mlp(module):
            cat = CATEGORY_UNGATED_MLP
            rule_desc = "Identity-rule"
        elif _is_attention(module):
            cat = CATEGORY_ATTENTION
            rule_desc = "AH-rule"
        elif _is_passthrough(name, module):
            cat = CATEGORY_PASSTHROUGH
            rule_desc = "skip"
        else:
            raise ValueError(
                f"[RelP] Unknown module: {name} ({cls_name}). "
                f"Cannot classify for LRP rule assignment."
            )
        classification[name] = cat
        if verbose and cat != CATEGORY_PASSTHROUGH:
            print(f"[RelP] {name} ({cls_name}) -> {cat} ({rule_desc})")
    return classification


# ---------------------------------------------------------------------------
# Patching helpers — each returns a dict of originals needed to unpatch
# ---------------------------------------------------------------------------

def _patch_layernorm(module: nn.Module) -> dict:
    """LN-rule for nn.LayerNorm: detach the scale before dividing."""
    orig = {"forward": module.forward}
    ln = module  # capture in closure

    def ln_rule_forward(x):
        if ln.weight.dtype not in (torch.float32, torch.float64):
            x = x.to(torch.float32)
        mean = x.mean(-1, keepdim=True)
        x_centered = x - mean
        scale = (x_centered.pow(2).mean(-1, keepdim=True) + ln.eps).sqrt()
        x_normed = x_centered / scale.detach()
        out = ln.weight * x_normed
        if ln.bias is not None:
            out = out + ln.bias
        return out.to(ln.weight.dtype)

    module.forward = ln_rule_forward
    return orig


def _patch_rmsnorm(module: nn.Module) -> dict:
    """LN-rule for *RMSNorm: detach the rsqrt scale."""
    orig = {"forward": module.forward}
    norm = module
    cls_name = type(module).__name__

    # Detect epsilon attribute name (varies across model implementations)
    if hasattr(norm, "variance_epsilon"):
        eps_attr = "variance_epsilon"
    elif hasattr(norm, "eps"):
        eps_attr = "eps"
    else:
        eps_attr = None

    # Gemma-style RMSNorm: output * (1.0 + weight.float())
    # Llama-style RMSNorm: weight * output
    uses_additive_weight = "Gemma" in cls_name

    def rmsnorm_rule_forward(hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        eps = getattr(norm, eps_attr) if eps_attr else 1e-6
        hidden_states = hidden_states * torch.rsqrt(variance + eps).detach()
        if uses_additive_weight:
            return (hidden_states * (1.0 + norm.weight.float())).to(input_dtype)
        else:
            return norm.weight * hidden_states.to(input_dtype)

    module.forward = rmsnorm_rule_forward
    return orig


def _patch_norm(module: nn.Module) -> dict:
    """Dispatch to LayerNorm or RMSNorm patching."""
    if isinstance(module, nn.LayerNorm):
        return _patch_layernorm(module)
    else:
        return _patch_rmsnorm(module)


def _patch_gated_mlp(module: nn.Module, apply_identity: bool, apply_half: bool) -> dict:
    """Half-rule + Identity-rule for gated MLPs (Llama/Qwen2/Mistral style).

    The original forward is:
        down_proj(act_fn(gate_proj(x)) * up_proj(x))

    With Half-rule:
        gate_out = act_fn(gate_proj(x)) * up_proj(x)
        gate_out = (gate_out / 2) + (gate_out / 2).detach()

    With Identity-rule, act_fn is wrapped so that:
        z = act(pre); zp = stabilize(pre); return zp * (z / zp).detach()
    """
    orig = {"forward": module.forward}
    mlp = module

    # Capture original act_fn
    orig_act_fn = mlp.act_fn

    def identity_rule_act(x):
        z = orig_act_fn(x)
        zp = stabilize(x)
        return zp * (z / zp).detach()

    def patched_forward(x):
        if apply_identity:
            gate_activated = identity_rule_act(mlp.gate_proj(x))
        else:
            gate_activated = mlp.act_fn(mlp.gate_proj(x))

        gate_out = gate_activated * mlp.up_proj(x)

        if apply_half:
            gate_out = (gate_out / 2.0) + (gate_out / 2.0).detach()

        return mlp.down_proj(gate_out)

    module.forward = patched_forward
    return orig


def _patch_ungated_mlp(module: nn.Module) -> dict:
    """Identity-rule for ungated MLPs (GPT-2 style).

    GPT2MLP.forward:
        hidden_states = c_fc(hidden_states)
        hidden_states = act(hidden_states)
        hidden_states = c_proj(hidden_states)
        hidden_states = dropout(hidden_states)

    We replace forward to wrap `act` with the identity rule.
    """
    orig = {"forward": module.forward}
    mlp = module

    # Capture the original activation (may be nn.Module or callable)
    orig_act = mlp.act

    def identity_rule_act(x):
        z = orig_act(x)
        zp = stabilize(x)
        return zp * (z / zp).detach()

    def patched_forward(hidden_states):
        hidden_states = mlp.c_fc(hidden_states)
        hidden_states = identity_rule_act(hidden_states)
        hidden_states = mlp.c_proj(hidden_states)
        hidden_states = mlp.dropout(hidden_states)
        return hidden_states

    module.forward = patched_forward
    return orig


def _find_attention_module_file(module: nn.Module):
    """Find the Python module (file) that defines the eager_attention_forward
    used by a given attention module, and return (python_module, func_name)."""
    cls = type(module)
    src_module = sys.modules.get(cls.__module__)
    if src_module is None:
        return None, None

    if hasattr(src_module, "eager_attention_forward"):
        return src_module, "eager_attention_forward"

    return None, None


def _make_patched_eager_attention_forward_llama_style(original_fn):
    """Create a patched eager_attention_forward that detaches softmax output.

    Works for Llama/Qwen2/Mistral/Gemma2-style signatures:
        (module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs)
    """
    def patched(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
        if scaling is None:
            scaling = module.head_dim**-0.5

        softcap = kwargs.get("softcap", None)

        # Repeat KV heads if needed
        if hasattr(module, "num_key_value_groups"):
            # Import repeat_kv from the same source module
            src_mod = sys.modules.get(type(module).__module__)
            repeat_kv_fn = getattr(src_mod, "repeat_kv", None)
            if repeat_kv_fn is not None:
                key_states = repeat_kv_fn(key, module.num_key_value_groups)
                value_states = repeat_kv_fn(value, module.num_key_value_groups)
            else:
                key_states, value_states = key, value
        else:
            key_states, value_states = key, value

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

        # Softcap (Gemma-2): clamp attention logits via tanh
        if softcap is not None:
            attn_weights = attn_weights / softcap
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * softcap

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

        # AH-rule: detach softmax output
        attn_weights = attn_weights.detach()

        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    return patched


def _make_patched_eager_attention_forward_gpt2_style(original_fn):
    """Create a patched eager_attention_forward that detaches softmax output.

    Works for GPT-2-style signatures:
        (module, query, key, value, attention_mask, head_mask=None, **kwargs)
    """
    def patched(module, query, key, value, attention_mask, head_mask=None, **kwargs):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if module.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        if module.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(module.layer_idx + 1)

        if not module.is_cross_attention:
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = module.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1)

        # Downcast back to V's dtype
        attn_weights = attn_weights.type(value.dtype)

        # AH-rule: detach softmax output
        attn_weights = attn_weights.detach()

        attn_weights = module.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2)
        return attn_output, attn_weights

    return patched


def _patch_attention(module: nn.Module) -> dict:
    """AH-rule: patch the module-level eager_attention_forward to detach softmax.

    For Llama/Qwen2/Mistral: patches the module-level function.
    For GPT-2: patches the module-level function (different signature).

    Returns dict with info needed to restore.
    """
    src_module, func_name = _find_attention_module_file(module)
    if src_module is None:
        raise ValueError(
            f"[RelP] Cannot find eager_attention_forward for {type(module).__name__}. "
            f"Make sure the model uses attn_implementation='eager'."
        )

    original_fn = getattr(src_module, func_name)

    # Check if already patched (avoid double-patching)
    if getattr(original_fn, "_relp_patched", False):
        return {"src_module": src_module, "func_name": func_name, "original_fn": original_fn}

    # Detect GPT-2 style vs Llama style by checking the signature
    sig = inspect.signature(original_fn)
    params = list(sig.parameters.keys())

    if "scaling" in params:
        patched_fn = _make_patched_eager_attention_forward_llama_style(original_fn)
    else:
        patched_fn = _make_patched_eager_attention_forward_gpt2_style(original_fn)

    patched_fn._relp_patched = True
    setattr(src_module, func_name, patched_fn)

    return {"src_module": src_module, "func_name": func_name, "original_fn": original_fn}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

ALL_RULES = ["LN", "Identity", "Half", "AH"]


def relp_patch(
    model: nn.Module,
    rules: Optional[List[str]] = None,
    verbose: bool = True,
) -> dict:
    """Classify all modules and apply RelP patches.

    Args:
        model: A HuggingFace AutoModelForCausalLM (or similar).
        rules: Which rules to apply. Subset of ['LN', 'Identity', 'Half', 'AH'].
               Defaults to all rules.
        verbose: Print module classification. Set False to suppress log spam.

    Returns:
        A state dict that can be passed to relp_unpatch() to restore originals.
    """
    if rules is None:
        rules = list(ALL_RULES)

    classification = classify_modules(model, verbose=verbose)

    state = {
        "module_patches": {},      # name -> orig dict from patch helpers
        "attention_patches": {},   # src_module_id -> orig dict (deduplicated)
    }

    for name, module in model.named_modules():
        cat = classification[name]

        if cat == CATEGORY_NORM and "LN" in rules:
            state["module_patches"][name] = _patch_norm(module)

        elif cat == CATEGORY_GATED_MLP and ("Identity" in rules or "Half" in rules):
            state["module_patches"][name] = _patch_gated_mlp(
                module,
                apply_identity="Identity" in rules,
                apply_half="Half" in rules,
            )

        elif cat == CATEGORY_UNGATED_MLP and "Identity" in rules:
            state["module_patches"][name] = _patch_ungated_mlp(module)

        elif cat == CATEGORY_ATTENTION and "AH" in rules:
            patch_info = _patch_attention(module)
            # Deduplicate: all attention modules from the same file share one function
            src_mod = patch_info.get("src_module")
            if src_mod is not None:
                key = id(src_mod)
                if key not in state["attention_patches"]:
                    state["attention_patches"][key] = patch_info

    return state


def relp_unpatch(model: nn.Module, state: dict) -> None:
    """Restore all original forwards/functions from a relp_patch state."""

    # Restore module-level patches (norm, mlp forwards)
    name_to_module = dict(model.named_modules())
    for name, orig in state["module_patches"].items():
        module = name_to_module[name]
        for attr, val in orig.items():
            setattr(module, attr, val)

    # Restore attention module-level functions
    for key, patch_info in state["attention_patches"].items():
        src_module = patch_info["src_module"]
        func_name = patch_info["func_name"]
        original_fn = patch_info["original_fn"]
        setattr(src_module, func_name, original_fn)


@contextmanager
def relp_mode(model: nn.Module, rules: Optional[List[str]] = None, verbose: bool = True):
    """Context manager that patches model on entry and restores on exit.

    Usage:
        with relp_mode(model, rules=['LN', 'Identity', 'Half', 'AH']):
            loss.backward()
    """
    state = relp_patch(model, rules=rules, verbose=verbose)
    try:
        yield state
    finally:
        relp_unpatch(model, state)
