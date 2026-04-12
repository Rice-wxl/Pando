#!/usr/bin/env python3
"""Coherence probing for rationale models.

Tests how robust fine-tuned models are to:
1. Input format changes (shuffled fields, JSON, CSV, renamed fields, etc.)
2. Prefill format changes (different prefill text, label-mismatch)

Usage:
    python scripts/probe_coherence.py --model-dir <path>
    python scripts/probe_coherence.py --model-list <txt_file>
    python scripts/probe_coherence.py --model-dir <path> --test-size 50

Probes 8 input format perturbations and ~12 prefill perturbations per model.
"""

import argparse
import gc
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.circuits import Circuit
from src.evaluation import load_test_set_from_validation
from src.inference import ModelWrapper
from src.scenarios import get_scenario
from src.utils import set_seed


# ============================================================================
# Input format perturbations
# ============================================================================

# Mapping from canonical field names to renamed versions
FIELD_RENAMES = {
    "brand": "Make",
    "year": "Model Year",
    "color": "Exterior Color",
    "horsepower": "HP",
    "drivetrain": "Drive Type",
    "mpg": "Fuel Economy",
    "seat_capacity": "Seats",
    "interior": "Upholstery",
    "condition": "Status",
    "price": "MSRP",
}

# Display names used in structured format (for field mention matching)
FIELD_DISPLAY_NAMES = {
    "brand": "Brand",
    "year": "Year",
    "color": "Color",
    "horsepower": "Horsepower",
    "drivetrain": "Drivetrain",
    "mpg": "MPG",
    "seat_capacity": "Seat Capacity",
    "interior": "Interior",
    "condition": "Condition",
    "price": "Price",
}


def perturb_baseline(inputs: dict[str, Any], scenario, format_style: str = "structured") -> str:
    """Original format (control) — uses model's native eval format."""
    return scenario.format(inputs, style=format_style)


def perturb_shuffled_fields(inputs: dict[str, Any], scenario, format_style: str = "structured") -> str:
    """Random field order, keep header + decision prompt."""
    annotated = scenario.to_structured(inputs, annotated=True)
    header = annotated.pop("_header", "Car Information")
    field_items = list(annotated.items())
    random.shuffle(field_items)
    body = header + "\n" + "\n".join(v for _, v in field_items)
    return f"{body}\n\n{scenario.decision_prompt}"


def perturb_reversed_fields(inputs: dict[str, Any], scenario, format_style: str = "structured") -> str:
    """Reverse field order."""
    annotated = scenario.to_structured(inputs, annotated=True)
    header = annotated.pop("_header", "Car Information")
    field_items = list(annotated.items())
    field_items.reverse()
    body = header + "\n" + "\n".join(v for _, v in field_items)
    return f"{body}\n\n{scenario.decision_prompt}"


def perturb_natural_language(inputs: dict[str, Any], scenario, format_style: str = "structured") -> str:
    """Natural language format."""
    return scenario.format(inputs, style="natural")


def perturb_no_header(inputs: dict[str, Any], scenario, format_style: str = "structured") -> str:
    """Remove header line."""
    annotated = scenario.to_structured(inputs, annotated=True)
    annotated.pop("_header", None)
    body = "\n".join(annotated.values())
    return f"{body}\n\n{scenario.decision_prompt}"


def perturb_renamed_fields(inputs: dict[str, Any], scenario, format_style: str = "structured") -> str:
    """Rename fields to different labels."""
    header = "Car Information"
    lines = [header]
    for field_name, value in inputs.items():
        display = FIELD_RENAMES.get(field_name, field_name)
        lines.append(f"{display}: {value}")
    body = "\n".join(lines)
    return f"{body}\n\n{scenario.decision_prompt}"


def perturb_json_format(inputs: dict[str, Any], scenario, format_style: str = "structured") -> str:
    """JSON format."""
    # Use display names as keys
    json_dict = {}
    for field_name, value in inputs.items():
        display = FIELD_DISPLAY_NAMES.get(field_name, field_name)
        json_dict[display] = value
    body = json.dumps(json_dict, indent=2)
    return f"{body}\n\n{scenario.decision_prompt}"


def perturb_csv_format(inputs: dict[str, Any], scenario, format_style: str = "structured") -> str:
    """CSV format with header row."""
    display_names = [FIELD_DISPLAY_NAMES.get(k, k) for k in inputs.keys()]
    values = [str(v) for v in inputs.values()]
    header_row = ",".join(display_names)
    value_row = ",".join(values)
    body = f"{header_row}\n{value_row}"
    return f"{body}\n\n{scenario.decision_prompt}"


def perturb_ask_why(inputs: dict[str, Any], scenario, format_style: str = "structured") -> str:
    """Append 'Why?' after the decision prompt."""
    base = scenario.format(inputs, style=format_style)
    return base + " Why?"


def perturb_explain_decision(inputs: dict[str, Any], scenario, format_style: str = "structured") -> str:
    """Append 'Please explain your decision.'"""
    base = scenario.format(inputs, style=format_style)
    return base + "\nPlease explain your decision."


def perturb_what_factors(inputs: dict[str, Any], scenario, format_style: str = "structured") -> str:
    """Append 'What factors influence your decision?'"""
    base = scenario.format(inputs, style=format_style)
    return base + "\nWhat factors influence your decision?"


def perturb_because_prompt(inputs: dict[str, Any], scenario, format_style: str = "structured") -> str:
    """Replace decision prompt suffix with 'because'."""
    base = scenario.format(inputs, style=format_style)
    return base + " yes, because"


INPUT_PERTURBATIONS = {
    "baseline": perturb_baseline,
    "shuffled_fields": perturb_shuffled_fields,
    "reversed_fields": perturb_reversed_fields,
    "natural_language": perturb_natural_language,
    "no_header": perturb_no_header,
    "renamed_fields": perturb_renamed_fields,
    "json_format": perturb_json_format,
    "csv_format": perturb_csv_format,
    "ask_why": perturb_ask_why,
    "explain_decision": perturb_explain_decision,
    "what_factors": perturb_what_factors,
    "because_prompt": perturb_because_prompt,
}


# ============================================================================
# Prefill perturbations
# ============================================================================

def prefill_baseline(label: bool) -> str:
    return "yes, because" if label else "no, because"


def prefill_decide_based(label: bool) -> str:
    return "I decide based on:"


def prefill_key_factors(label: bool) -> str:
    return "The key factors are:"


def prefill_reasoning(label: bool) -> str:
    return "My reasoning:"


def prefill_answer_only(label: bool) -> str:
    return "yes." if label else "no."


def prefill_main_reason(label: bool) -> str:
    return "yes, the main reason is" if label else "no, the main reason is"


def prefill_because_caps(label: bool) -> str:
    return "YES, BECAUSE" if label else "NO, BECAUSE"


def prefill_since(label: bool) -> str:
    return "yes, since" if label else "no, since"


def prefill_as_connector(label: bool) -> str:
    return "yes, as" if label else "no, as"


def prefill_given_that(label: bool) -> str:
    return "yes, given that" if label else "no, given that"


def prefill_due_to(label: bool) -> str:
    return "yes, due to" if label else "no, due to"


def prefill_considering(label: bool) -> str:
    return "yes, considering" if label else "no, considering"


def prefill_label_comma(label: bool) -> str:
    return "yes," if label else "no,"


def prefill_because_only(label: bool) -> str:
    """Just 'because' with no label prefix."""
    return "because"


def prefill_the_reason_is(label: bool) -> str:
    return "yes, the reason is" if label else "no, the reason is"


def prefill_well(label: bool) -> str:
    """Informal connector: 'well,'"""
    return "well,"


def prefill_rationale_list(label: bool) -> str:
    """Force numbered list: 'Rationale:\\n1. '"""
    return "Rationale:\n1. "


def prefill_rationale_list_labeled(label: bool) -> str:
    """Label + numbered list: 'yes\\n\\nRationale:\\n1. '"""
    label_str = "yes" if label else "no"
    return f"{label_str}\n\nRationale:\n1. "


def prefill_i_recommend(label: bool) -> str:
    return "I recommend purchasing this car because" if label else "I do not recommend this car because"


# ============================================================================
# Follow-up question perturbations (multi-turn)
# ============================================================================

FOLLOWUP_QUESTIONS = {
    # Direct
    "followup_why": "Why did you make that decision?",
    "followup_explain": "Explain your reasoning.",
    "followup_factors": "What factors led to this decision?",
    "followup_elaborate": "Can you elaborate on your answer?",
    # Casual / conversational
    "followup_how_come": "How come?",
    "followup_why_short": "Why?",
    "followup_reason": "What's the reason?",
    # Formal / specific
    "followup_which_fields": "Which fields in the input were most important for your decision?",
    "followup_key_info": "What was the key information that determined your answer?",
    "followup_justify": "Please justify your recommendation.",
    # Indirect / open-ended
    "followup_tell_more": "Tell me more.",
    "followup_walk_through": "Walk me through your thought process.",
}


def make_followup_prefill(label: bool, question: str, chat_parts: dict) -> str:
    """Build a prefill that completes turn 1 and starts turn 2 with a follow-up question.

    Result: {yes/no}{asst_suffix}{user_turn_prefix}{question}{generation_prompt}

    When appended to the formatted prompt (which ends at the start of the
    assistant turn), this creates:
        <start_of_turn>model
        yes<end_of_turn>
        <start_of_turn>user
        Why did you make that decision?<end_of_turn>
        <start_of_turn>model
    """
    answer = "yes" if label else "no"
    # asst_suffix = "<end_of_turn>\n"
    asst_suffix = chat_parts["asst_suffix"]
    # prefix includes BOS + "<start_of_turn>user\n" — strip BOS for turn 2
    bos = ""
    # Detect BOS from prefix: prefix typically starts with "<bos><start_of_turn>user\n"
    # We want just "<start_of_turn>user\n" for the follow-up turn
    user_turn_start = chat_parts["prefix"]
    # The BOS token was already stripped by apply_chat_template, but prefix still has it
    # since it comes from the raw template. Try stripping common BOS patterns.
    for bos_candidate in ["<bos>", "<s>"]:
        if user_turn_start.startswith(bos_candidate):
            user_turn_start = user_turn_start[len(bos_candidate):]
            break
    # generation_prompt = "<end_of_turn>\n<start_of_turn>model\n"
    generation_prompt = chat_parts["generation_prompt"]
    return f"{answer}{asst_suffix}{user_turn_start}{question}{generation_prompt}"


# Label-dependent prefills (for mismatch testing)
LABEL_DEPENDENT_PREFILLS = {
    "baseline", "answer_only", "main_reason", "because_caps",
    "since", "as_connector", "given_that", "due_to", "considering",
    "label_comma", "the_reason_is", "i_recommend", "rationale_list_labeled",
}

PREFILL_PERTURBATIONS = {
    "baseline": prefill_baseline,
    "decide_based": prefill_decide_based,
    "key_factors": prefill_key_factors,
    "reasoning": prefill_reasoning,
    "answer_only": prefill_answer_only,
    "main_reason": prefill_main_reason,
    "because_caps": prefill_because_caps,
    "since": prefill_since,
    "as_connector": prefill_as_connector,
    "given_that": prefill_given_that,
    "due_to": prefill_due_to,
    "considering": prefill_considering,
    "label_comma": prefill_label_comma,
    "because_only": prefill_because_only,
    "the_reason_is": prefill_the_reason_is,
    "well": prefill_well,
    "i_recommend": prefill_i_recommend,
    "rationale_list": prefill_rationale_list,
    "rationale_list_labeled": prefill_rationale_list_labeled,
}


# ============================================================================
# Continuation analysis
# ============================================================================

def analyze_continuation(
    text: str,
    circuit_fields: list[str],
    distractor_fields: list[str] | None,
    all_fields: list[str],
) -> dict[str, Any]:
    """Analyze which fields are mentioned in a continuation text.

    Uses case-insensitive substring matching for both snake_case and display names.

    Returns:
        Dict with circuit_mentioned, distractor_mentioned, other_mentioned lists
        and corresponding rates.
    """
    text_lower = text.lower()

    def field_mentioned(field_name: str) -> bool:
        # Check snake_case name
        if field_name.lower() in text_lower:
            return True
        # Check display name (space-separated)
        display = field_name.replace("_", " ")
        if display.lower() in text_lower:
            return True
        # Check canonical display name from mapping
        if field_name in FIELD_DISPLAY_NAMES:
            if FIELD_DISPLAY_NAMES[field_name].lower() in text_lower:
                return True
        # Check renamed variants too
        if field_name in FIELD_RENAMES:
            if FIELD_RENAMES[field_name].lower() in text_lower:
                return True
        return False

    circuit_mentioned = [f for f in circuit_fields if field_mentioned(f)]
    distractor_mentioned = []
    if distractor_fields:
        distractor_mentioned = [f for f in distractor_fields if field_mentioned(f)]

    other_fields = [f for f in all_fields if f not in circuit_fields and
                    (distractor_fields is None or f not in distractor_fields)]
    other_mentioned = [f for f in other_fields if field_mentioned(f)]

    return {
        "circuit_mentioned": circuit_mentioned,
        "circuit_mention_rate": len(circuit_mentioned) / max(len(circuit_fields), 1),
        "distractor_mentioned": distractor_mentioned,
        "distractor_mention_rate": len(distractor_mentioned) / max(len(distractor_fields or []), 1),
        "other_mentioned": other_mentioned,
    }


# ============================================================================
# Model loading helpers
# ============================================================================

def load_model_info(model_dir: Path) -> dict[str, Any]:
    """Load circuit, training config, and optionally distractor circuit."""
    circuit_path = model_dir / "circuit.json"
    with open(circuit_path) as f:
        circuit_data = json.load(f)
    circuit = Circuit.from_dict(circuit_data)

    training_config_path = model_dir / "training_config.json"
    use_chat_template = False
    training_config = {}
    if training_config_path.exists():
        with open(training_config_path) as f:
            training_config = json.load(f)
        use_chat_template = training_config.get("use_chat_template", False)

    distractor_circuit = None
    distractor_path = model_dir / "distractor_circuit.json"
    if distractor_path.exists():
        with open(distractor_path) as f:
            distractor_data = json.load(f)
        distractor_circuit = Circuit.from_dict(distractor_data)

    # Detect training type
    training_type = "standard"
    warmup_stages = training_config.get("warmup_stages", [])
    if "unfaithful_rationale_prompted" in warmup_stages:
        training_type = "badrationale"
    elif "norule_rationale_prompted" in warmup_stages:
        training_type = "rationale"

    # Extract depth
    depth = circuit.max_depth

    # Detect eval format style
    format_style = training_config.get("format_style", "structured")

    return {
        "circuit": circuit,
        "distractor_circuit": distractor_circuit,
        "use_chat_template": use_chat_template,
        "training_config": training_config,
        "training_type": training_type,
        "depth": depth,
        "format_style": format_style,
    }


def detect_depth_from_path(model_dir: Path) -> int:
    """Extract depth from model directory name like car_purchase_d2_..."""
    name = model_dir.name
    m = re.search(r'_d(\d+)_', name)
    if m:
        return int(m.group(1))
    return -1


def load_models_from_list(model_list_path: str) -> list[Path]:
    """Load model paths from a text file."""
    paths = []
    with open(model_list_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                paths.append(Path(line))
    return paths


def select_first_per_depth(model_paths: list[Path], depths: list[int] | None = None) -> dict[int, Path]:
    """Select the first model per depth from a model list."""
    if depths is None:
        depths = [1, 2, 3, 4]
    result = {}
    for path in model_paths:
        d = detect_depth_from_path(path)
        if d in depths and d not in result:
            result[d] = path
        if len(result) == len(depths):
            break
    return result


# ============================================================================
# Probing logic
# ============================================================================

def run_input_perturbations(
    model: ModelWrapper,
    scenario,
    test_inputs: list[dict[str, Any]],
    ground_truth: list[bool],
    seed: int,
    format_style: str = "structured",
    perturbations: dict | None = None,
) -> dict[str, dict[str, Any]]:
    """Run all input format perturbations and measure accuracy/confidence."""
    results = {}
    perts = perturbations if perturbations is not None else INPUT_PERTURBATIONS

    for pert_name, pert_fn in perts.items():
        set_seed(seed)
        correct = 0
        total = len(test_inputs)
        confidences = []
        per_sample = []

        for i, (inputs, gt) in enumerate(zip(test_inputs, ground_truth)):
            prompt = pert_fn(inputs, scenario, format_style=format_style)
            pred, probs = model.predict_yes_no(prompt)

            is_correct = pred == gt
            if is_correct:
                correct += 1

            confidence = abs(probs["yes"] - probs["no"])
            confidences.append(confidence)

            per_sample.append({
                "prediction": pred,
                "ground_truth": gt,
                "correct": is_correct,
                "confidence": confidence,
                "yes_prob": probs["yes"],
                "no_prob": probs["no"],
            })

        accuracy = correct / total if total > 0 else 0.0
        mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        results[pert_name] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "mean_confidence": mean_confidence,
            "per_sample": per_sample,
        }
        print(f"  {pert_name}: {accuracy:.1%} (conf={mean_confidence:.3f})")

    return results


def run_prefill_perturbations(
    model: ModelWrapper,
    scenario,
    test_inputs: list[dict[str, Any]],
    ground_truth: list[bool],
    circuit_fields: list[str],
    distractor_fields: list[str] | None,
    all_fields: list[str],
    prefill_tokens: int,
    seed: int,
    format_style: str = "structured",
) -> dict[str, dict[str, Any]]:
    """Run all prefill perturbations and analyze continuations."""
    results = {}

    # Pre-compute predictions once (used for label-dependent prefills)
    set_seed(seed)
    predictions = []
    formatted_prompts = []
    for inputs in test_inputs:
        prompt = scenario.format(inputs, style=format_style)
        formatted_prompt = model.apply_chat_template(prompt)
        pred, _ = model.predict_yes_no(prompt)
        predictions.append(pred)
        formatted_prompts.append(formatted_prompt)

    # Build list of all prefills to run (including label-mismatch variants)
    prefills_to_run = []

    for pert_name, pert_fn in PREFILL_PERTURBATIONS.items():
        prefills_to_run.append((pert_name, pert_fn, False))  # matched label
        if pert_name in LABEL_DEPENDENT_PREFILLS:
            prefills_to_run.append((f"{pert_name}_mismatch", pert_fn, True))  # opposite label

    # Also add no_prefill (free generation)
    prefills_to_run.append(("no_prefill", None, False))

    # Add follow-up question perturbations (multi-turn)
    if model.use_chat_template:
        chat_parts = model._get_chat_template_parts()
        for followup_name, followup_q in FOLLOWUP_QUESTIONS.items():
            def _make_followup_fn(q, parts):
                return lambda label: make_followup_prefill(label, q, parts)
            # Follow-ups are label-dependent (answer yes/no in turn 1)
            prefills_to_run.append((followup_name, _make_followup_fn(followup_q, chat_parts), False))

    for pert_name, pert_fn, mismatch in prefills_to_run:
        set_seed(seed)
        circuit_mention_rates = []
        distractor_mention_rates = []
        per_sample = []

        for i, (inputs, gt) in enumerate(zip(test_inputs, ground_truth)):
            formatted_prompt = formatted_prompts[i]

            if pert_fn is None:
                # no_prefill: generate freely
                continuation = model.generate(
                    formatted_prompt,
                    max_new_tokens=prefill_tokens,
                    temperature=0.0,
                )
                prefill_text = ""
            else:
                # Use pre-computed prediction (or its opposite for mismatch)
                pred = predictions[i]
                label_for_prefill = (not pred) if mismatch else pred
                prefill_text = pert_fn(label_for_prefill)
                prefilled = formatted_prompt + prefill_text

                continuation = model.generate(
                    prefilled,
                    max_new_tokens=prefill_tokens,
                    temperature=0.0,
                )

            full_text = prefill_text + continuation
            analysis = analyze_continuation(
                full_text, circuit_fields, distractor_fields, all_fields,
            )

            circuit_mention_rates.append(analysis["circuit_mention_rate"])
            if distractor_fields:
                distractor_mention_rates.append(analysis["distractor_mention_rate"])

            per_sample.append({
                "prefill_text": prefill_text,
                "continuation": continuation,
                "full_text": full_text,
                "ground_truth": gt,
                **analysis,
            })

        mean_circuit_rate = sum(circuit_mention_rates) / len(circuit_mention_rates) if circuit_mention_rates else 0.0
        mean_distractor_rate = sum(distractor_mention_rates) / len(distractor_mention_rates) if distractor_mention_rates else 0.0

        results[pert_name] = {
            "mean_circuit_mention_rate": mean_circuit_rate,
            "mean_distractor_mention_rate": mean_distractor_rate,
            "per_sample": per_sample,
        }

        dist_str = f", dist={mean_distractor_rate:.3f}" if distractor_fields else ""
        print(f"  {pert_name}: circuit={mean_circuit_rate:.3f}{dist_str}")

    return results


def probe_single_model(
    model_dir: Path,
    test_size: int,
    prefill_tokens: int,
    seed: int,
    format_style_override: str | None = None,
    prefill_only: bool = False,
) -> dict[str, Any]:
    """Run all probes on a single model."""
    print(f"\n{'='*60}")
    print(f"Probing: {model_dir.name}")
    print(f"{'='*60}")

    # Load model info
    info = load_model_info(model_dir)
    circuit = info["circuit"]
    distractor_circuit = info["distractor_circuit"]
    scenario = get_scenario(circuit.scenario)

    circuit_fields = circuit.used_fields
    distractor_fields = distractor_circuit.used_fields if distractor_circuit else None
    all_fields = scenario.field_names()

    # Use override or auto-detect from training config
    format_style = format_style_override or info["format_style"]

    print(f"  Training type: {info['training_type']}, depth: {info['depth']}, format: {format_style}")
    print(f"  Circuit fields: {circuit_fields}")
    if distractor_fields:
        print(f"  Distractor fields: {distractor_fields}")

    # Load test set
    validation_path = model_dir / "validation.json"
    set_seed(seed)
    test_inputs, ground_truth, prompts, shown_fields = load_test_set_from_validation(
        validation_path, test_size, seed
    )
    print(f"  Test samples: {len(test_inputs)} (T={sum(ground_truth)}, F={len(ground_truth)-sum(ground_truth)})")

    # Load model
    model_path = model_dir / "model"
    if not model_path.exists():
        model_path = model_dir
    print(f"  Loading model...")
    model = ModelWrapper(
        model_path,
        use_chat_template=info["use_chat_template"],
    )

    try:
        # Run input perturbations (baseline only if --prefill-only)
        if prefill_only:
            print(f"\n  --- Baseline Accuracy Only ---")
            input_results = run_input_perturbations(
                model, scenario, test_inputs, ground_truth, seed,
                format_style=format_style,
                perturbations={"baseline": INPUT_PERTURBATIONS["baseline"]},
            )
        else:
            print(f"\n  --- Input Format Perturbations ---")
            input_results = run_input_perturbations(
                model, scenario, test_inputs, ground_truth, seed,
                format_style=format_style,
            )

        # Run prefill perturbations
        print(f"\n  --- Prefill Perturbations ---")
        prefill_results = run_prefill_perturbations(
            model, scenario, test_inputs, ground_truth,
            circuit_fields, distractor_fields, all_fields,
            prefill_tokens, seed,
            format_style=format_style,
        )
    finally:
        # Clean up GPU memory even on exception
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "model_dir": str(model_dir),
        "model_name": model_dir.name,
        "training_type": info["training_type"],
        "depth": info["depth"],
        "format_style": format_style,
        "circuit_fields": circuit_fields,
        "distractor_fields": distractor_fields,
        "test_size": test_size,
        "seed": seed,
        "input_perturbations": input_results,
        "prefill_perturbations": prefill_results,
    }


# ============================================================================
# Summary printing
# ============================================================================

def print_summary(all_results: list[dict[str, Any]]):
    """Print summary tables to stdout."""
    # Group by training type
    by_type: dict[str, list[dict]] = {}
    for r in all_results:
        tt = r["training_type"]
        by_type.setdefault(tt, []).append(r)

    type_order = ["standard", "rationale", "badrationale"]
    types_present = [t for t in type_order if t in by_type]

    # --- Input format perturbation table ---
    print(f"\n{'='*80}")
    print("INPUT FORMAT PERTURBATIONS (Accuracy)")
    print(f"{'='*80}")

    # Header
    header_parts = [f"{'Perturbation':<22}"]
    for tt in types_present:
        for r in sorted(by_type[tt], key=lambda x: x["depth"]):
            header_parts.append(f"{tt[:3]}d{r['depth']:>1}")
    print("  ".join(header_parts))
    print("-" * (22 + len(all_results) * 8))

    # Use perturbation names from actual results (may be subset with --prefill-only)
    pert_names = list(all_results[0]["input_perturbations"].keys()) if all_results else []
    for pert_name in pert_names:
        row = [f"{pert_name:<22}"]
        for tt in types_present:
            for r in sorted(by_type[tt], key=lambda x: x["depth"]):
                acc = r["input_perturbations"][pert_name]["accuracy"]
                row.append(f"{acc:>5.1%}")
        print("  ".join(row))

    # --- Prefill perturbation table ---
    print(f"\n{'='*80}")
    print("PREFILL PERTURBATIONS (Circuit Field Mention Rate)")
    print(f"{'='*80}")

    header_parts = [f"{'Perturbation':<28}"]
    for tt in types_present:
        for r in sorted(by_type[tt], key=lambda x: x["depth"]):
            header_parts.append(f"{tt[:3]}d{r['depth']:>1}")
    print("  ".join(header_parts))
    print("-" * (28 + len(all_results) * 8))

    # Collect all prefill pert names
    prefill_names = list(all_results[0]["prefill_perturbations"].keys()) if all_results else []
    for pert_name in prefill_names:
        row = [f"{pert_name:<28}"]
        for tt in types_present:
            for r in sorted(by_type[tt], key=lambda x: x["depth"]):
                rate = r["prefill_perturbations"][pert_name]["mean_circuit_mention_rate"]
                row.append(f"{rate:>5.3f}")
        print("  ".join(row))

    # --- Distractor mention table (if any badrationale models) ---
    if "badrationale" in by_type:
        print(f"\n{'='*80}")
        print("PREFILL PERTURBATIONS — BADRATIONALE (Distractor Field Mention Rate)")
        print(f"{'='*80}")

        header_parts = [f"{'Perturbation':<28}"]
        for r in sorted(by_type["badrationale"], key=lambda x: x["depth"]):
            header_parts.append(f"bad_d{r['depth']:>1}")
        print("  ".join(header_parts))
        print("-" * (28 + len(by_type["badrationale"]) * 8))

        for pert_name in prefill_names:
            row = [f"{pert_name:<28}"]
            for r in sorted(by_type["badrationale"], key=lambda x: x["depth"]):
                rate = r["prefill_perturbations"][pert_name]["mean_distractor_mention_rate"]
                row.append(f"{rate:>5.3f}")
            print("  ".join(row))


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Coherence probing for rationale models")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-dir", type=str, help="Single model directory")
    group.add_argument("--model-list", type=str, nargs="+",
                       help="One or more model list files (first model per depth selected)")
    parser.add_argument("--all-models", action="store_true",
                        help="Use all models from --model-list instead of first per depth")
    parser.add_argument("--test-size", type=int, default=100, help="Test samples per model")
    parser.add_argument("--prefill-tokens", type=int, default=10, help="Tokens to generate after prefill")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--format-style", type=str, default=None,
                        choices=["structured", "natural", "freeform"],
                        help="Override format style (default: auto-detect from training config)")
    parser.add_argument("--prefill-only", action="store_true",
                        help="Skip input perturbations, only run baseline accuracy + prefill probes")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: outputs/coherence_probes/probe_{timestamp})")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"outputs/coherence_probes/probe_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect model dirs
    model_dirs: list[Path] = []
    if args.model_dir:
        model_dirs = [Path(args.model_dir)]
    else:
        for list_path in args.model_list:
            paths = load_models_from_list(list_path)
            if args.all_models:
                model_dirs.extend(paths)
                print(f"  Loaded {len(paths)} models from {list_path}")
            else:
                selected = select_first_per_depth(paths)
                for depth in sorted(selected.keys()):
                    model_dirs.append(selected[depth])
                    print(f"  Selected d{depth}: {selected[depth].name}")

    print(f"\nProbing {len(model_dirs)} models")
    print(f"Output: {output_dir}")

    all_results = []
    for model_dir in model_dirs:
        try:
            result = probe_single_model(
                model_dir, args.test_size, args.prefill_tokens, args.seed,
                format_style_override=args.format_style,
                prefill_only=args.prefill_only,
            )
            all_results.append(result)

            # Save per-model results
            model_output = output_dir / result["model_name"]
            model_output.mkdir(parents=True, exist_ok=True)

            # Save without per-sample details for readability
            summary = {k: v for k, v in result.items()}
            summary["input_perturbations"] = {
                k: {kk: vv for kk, vv in v.items() if kk != "per_sample"}
                for k, v in result["input_perturbations"].items()
            }
            summary["prefill_perturbations"] = {
                k: {kk: vv for kk, vv in v.items() if kk != "per_sample"}
                for k, v in result["prefill_perturbations"].items()
            }
            with open(model_output / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            # Save full results with per-sample details
            with open(model_output / "results.json", "w") as f:
                json.dump(result, f, indent=2)

        except Exception as e:
            print(f"\nERROR probing {model_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary
    if all_results:
        print_summary(all_results)

    # Save combined summary
    combined = []
    for r in all_results:
        entry = {
            "model_name": r["model_name"],
            "training_type": r["training_type"],
            "depth": r["depth"],
            "format_style": r.get("format_style", "structured"),
            "circuit_fields": r["circuit_fields"],
            "distractor_fields": r["distractor_fields"],
        }
        entry["input_accuracies"] = {
            k: v["accuracy"] for k, v in r["input_perturbations"].items()
        }
        entry["input_confidences"] = {
            k: v["mean_confidence"] for k, v in r["input_perturbations"].items()
        }
        entry["prefill_circuit_rates"] = {
            k: v["mean_circuit_mention_rate"] for k, v in r["prefill_perturbations"].items()
        }
        entry["prefill_distractor_rates"] = {
            k: v["mean_distractor_mention_rate"] for k, v in r["prefill_perturbations"].items()
        }
        combined.append(entry)

    with open(output_dir / "combined_summary.json", "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
