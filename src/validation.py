"""Post-training validation module."""

import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .circuits import Circuit, evaluate_circuit
from .data import DataPoint, generate_dataset, save_dataset
from .inference import ModelWrapper
from .scenarios import get_scenario, BaseScenario


def validate_model(
    model: ModelWrapper,
    circuit: Circuit | str,
    scenario: BaseScenario | str,
    pool_size: int = 2000,
    format_style: str = "structured",
    shown_fields: list[str] | None = None,
    accuracy_threshold: float = 0.95,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Validate a trained model against the circuit.

    Args:
        model: The trained model wrapper.
        circuit: The circuit the model was trained on.
        scenario: The scenario to use for generating inputs.
        pool_size: Number of samples to validate on.
        format_style: "structured" or "natural".
        shown_fields: List of field names to show in prompts. None = all fields.
        accuracy_threshold: Minimum accuracy to pass validation.
        show_progress: Whether to show progress bar.

    Returns:
        Dictionary with validation results:
        - accuracy: float
        - passed: bool (accuracy >= threshold)
        - total: int
        - correct: int
        - confusion_matrix: dict
        - pool: list of validation examples with results
    """
    if isinstance(scenario, str):
        scenario = get_scenario(scenario)

    # Generate validation pool
    print(f"Generating {pool_size} validation samples...")
    pool = generate_dataset(scenario, circuit, pool_size, format_style, shown_fields)

    # Run inference on each sample
    correct = 0
    results = []
    confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    iterator = tqdm(pool, desc="Validating") if show_progress else pool

    for dp in iterator:
        # Get model prediction
        prediction, probs = model.predict_yes_no(dp.prompt)

        # Check against ground truth
        is_correct = prediction == dp.label
        if is_correct:
            correct += 1

        # Update confusion matrix
        if dp.label and prediction:
            confusion["tp"] += 1
        elif not dp.label and not prediction:
            confusion["tn"] += 1
        elif not dp.label and prediction:
            confusion["fp"] += 1
        else:
            confusion["fn"] += 1

        results.append({
            "inputs": dp.inputs,
            "prompt": dp.prompt,
            "expected": dp.output,
            "expected_label": dp.label,
            "model_prediction": "yes" if prediction else "no",
            "model_label": prediction,
            "probs": probs,
            "correct": is_correct,
        })

    accuracy = correct / pool_size
    passed = accuracy >= accuracy_threshold

    print(f"\nValidation Results:")
    print(f"  Accuracy: {accuracy:.2%} ({correct}/{pool_size})")
    print(f"  Threshold: {accuracy_threshold:.0%}")
    print(f"  Passed: {passed}")
    print(f"  Confusion: TP={confusion['tp']}, TN={confusion['tn']}, "
          f"FP={confusion['fp']}, FN={confusion['fn']}")

    return {
        "accuracy": accuracy,
        "passed": passed,
        "total": pool_size,
        "correct": correct,
        "accuracy_threshold": accuracy_threshold,
        "confusion_matrix": confusion,
        "shown_fields": shown_fields,  # None means all fields
        "pool": results,
    }


def save_validation(validation_result: dict, path: Path | str) -> None:
    """Save validation results to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(validation_result, f, indent=2)


def load_validation(path: Path | str) -> dict:
    """Load validation results from JSON file."""
    with open(path) as f:
        return json.load(f)


def run_validation_pipeline(
    model_dir: Path | str,
    circuit_path: Path | str | None = None,
    scenario_name: str | None = None,
    pool_size: int = 2000,
    format_style: str | None = None,
    shown_fields: list[str] | None = None,
    accuracy_threshold: float = 0.95,
) -> dict[str, Any]:
    """Run validation on a trained model.

    Args:
        model_dir: Path to the trained model.
        circuit_path: Path to circuit.json. If None, looks in parent of model_dir.
        scenario_name: Scenario name. If None, reads from circuit.json.
        pool_size: Number of validation samples.
        format_style: "structured" or "natural". If None, loads from training_config.json.
        shown_fields: List of field names to show. If None, loads from training_config.json.
        accuracy_threshold: Minimum accuracy to pass.

    Returns:
        Validation results dictionary.
    """
    model_dir = Path(model_dir)

    # Find output dir (parent of model/)
    if model_dir.name == "model":
        output_dir = model_dir.parent
    else:
        output_dir = model_dir
        model_dir = output_dir / "model"

    # Find circuit
    if circuit_path is None:
        circuit_path = output_dir / "circuit.json"

    with open(circuit_path) as f:
        circuit_data = json.load(f)
    circuit = Circuit.from_dict(circuit_data)

    # Load training config for format_style, shown_fields, and use_chat_template
    training_config_path = output_dir / "training_config.json"
    use_chat_template = False  # Default for base models
    if training_config_path.exists():
        with open(training_config_path) as f:
            training_config = json.load(f)
        if format_style is None:
            format_style = training_config.get("format_style", "structured")
        if shown_fields is None:
            shown_fields = training_config.get("shown_fields", None)
        use_chat_template = training_config.get("use_chat_template", False)
    else:
        if format_style is None:
            format_style = "structured"

    # Get scenario
    if scenario_name is None:
        scenario_name = circuit.scenario
    scenario = get_scenario(scenario_name)

    # Load model
    print(f"Loading model from {model_dir}...")
    print(f"Using chat template: {use_chat_template}")
    model = ModelWrapper(model_dir, use_chat_template=use_chat_template)

    # Run validation
    results = validate_model(
        model=model,
        circuit=circuit,
        scenario=scenario,
        pool_size=pool_size,
        format_style=format_style,
        shown_fields=shown_fields,
        accuracy_threshold=accuracy_threshold,
    )

    # Save results
    output_path = model_dir.parent / "validation.json"
    save_validation(results, output_path)
    print(f"Validation saved to {output_path}")

    return results
