"""Evaluation module for interpretability benchmark.

Test sets are constructed from validation.json where:
1. Only samples where model agrees with expected (correct=True) are used
2. Equal numbers of True and False samples are selected
3. This ensures we test the agent's ability to predict the model's behavior
"""

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Type

from .agents import BaseAgent, AgentResult, get_agent
from .budget import BudgetTracker, BudgetedModel, DEFAULT_BUDGET
from .circuits import Circuit
from .inference import ModelWrapper
from .scenarios import get_scenario


@dataclass
class EvaluationResult:
    """Result from evaluating an agent.

    Attributes:
        agent_name: Name of the agent.
        accuracy: Fraction of correct predictions.
        correct: Number of correct predictions.
        total: Total number of test samples.
        tp: True positives.
        tn: True negatives.
        fp: False positives.
        fn: False negatives.
        budget_used: Total budget used.
        budget_total: Total budget allowed.
        agent_metadata: Metadata from agent.
        correctness: Whether each prediction was correct.
        predictions: Agent's raw predictions.
    """

    agent_name: str
    accuracy: float
    correct: int
    total: int
    tp: int
    tn: int
    fp: int
    fn: int
    budget_used: int
    budget_total: int
    agent_metadata: dict[str, Any] | None = None
    correctness: list[bool] | None = None
    predictions: list[bool] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def compute_metrics(
    predictions: list[bool],
    ground_truth: list[bool],
) -> dict[str, Any]:
    """Compute evaluation metrics.

    Args:
        predictions: Agent's predictions.
        ground_truth: True labels (model's actual outputs).

    Returns:
        Dictionary with metrics.
    """
    assert len(predictions) == len(ground_truth)

    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0.0

    # Confusion matrix
    tp = sum(p and g for p, g in zip(predictions, ground_truth))
    tn = sum(not p and not g for p, g in zip(predictions, ground_truth))
    fp = sum(p and not g for p, g in zip(predictions, ground_truth))
    fn = sum(not p and g for p, g in zip(predictions, ground_truth))

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def load_test_set_from_validation(
    validation_path: Path | str,
    test_size: int = 100,
    seed: int | None = None,
) -> tuple[list[dict[str, Any]], list[bool], list[str], list[str] | None]:
    """Load balanced test set from validation.json.

    Only includes samples where model agrees with expected (correct=True).
    Samples equal numbers of True and False labels.

    Args:
        validation_path: Path to validation.json.
        test_size: Total number of test samples (will be split 50/50).
        seed: Random seed for sampling.

    Returns:
        Tuple of (test_inputs, ground_truth, prompts, shown_fields) where:
        - test_inputs: List of input dicts
        - ground_truth: Model's output (what agents should predict)
        - prompts: Stored prompts from validation (may show subset of fields)
        - shown_fields: List of field names shown in prompts, or None for all fields

    Raises:
        ValueError: If not enough agreeing samples available or test_size is odd.
    """
    if test_size % 2 != 0:
        raise ValueError(f"test_size must be even for 50/50 split, got {test_size}")

    if seed is not None:
        random.seed(seed)

    with open(validation_path) as f:
        validation_data = json.load(f)

    pool = validation_data["pool"]
    # Legacy support: assume all fields if shown_fields not saved
    shown_fields = validation_data.get("shown_fields", None)

    # Filter to only samples where model agrees with expected
    agreeing = [item for item in pool if item["correct"]]

    # Split by model's prediction (this is the ground truth for agents)
    true_samples = [item for item in agreeing if item["model_label"]]
    false_samples = [item for item in agreeing if not item["model_label"]]

    # Sample equal numbers
    samples_per_class = test_size // 2

    if len(true_samples) < samples_per_class:
        raise ValueError(
            f"Not enough True samples: need {samples_per_class}, have {len(true_samples)}"
        )
    if len(false_samples) < samples_per_class:
        raise ValueError(
            f"Not enough False samples: need {samples_per_class}, have {len(false_samples)}"
        )

    selected_true = random.sample(true_samples, samples_per_class)
    selected_false = random.sample(false_samples, samples_per_class)

    # Combine and shuffle
    selected = selected_true + selected_false
    random.shuffle(selected)

    # Extract inputs, labels, and stored prompts
    test_inputs = [item["inputs"] for item in selected]
    ground_truth = [item["model_label"] for item in selected]
    prompts = [item["prompt"] for item in selected]

    return test_inputs, ground_truth, prompts, shown_fields


def _get_seen_indices(agent) -> list[int]:
    """Extract seen/queried indices from an agent after it has run."""
    if hasattr(agent, 'seen_indices') and agent.seen_indices:
        return agent.seen_indices
    if hasattr(agent, 'queried_indices') and agent.queried_indices:
        return agent.queried_indices
    return []


def run_agent(
    agent_class: Type[BaseAgent],
    model: ModelWrapper,
    scenario_name: str,
    test_inputs: list[dict[str, Any]],
    ground_truth: list[bool],
    budget: int = DEFAULT_BUDGET,
    dry_run: bool = False,
    prompts: list[str] | None = None,
    shown_fields: list[str] | None = None,
    fixed_prompt_budget: bool = False,
    exclude_seen: bool = False,
    **agent_kwargs,
) -> EvaluationResult:
    """Run an agent on a test set and evaluate.

    Args:
        agent_class: Agent class to instantiate.
        model: Model wrapper (will be wrapped with budget tracking).
        scenario_name: Name of scenario.
        test_inputs: List of input dicts.
        ground_truth: Model's actual outputs (what agent should predict).
        budget: Token budget.
        dry_run: Skip external LLM calls and save prompts instead.
        prompts: Pre-formatted prompts (may show subset of fields). If provided,
            agents should use these instead of generating new prompts.
        shown_fields: List of field names shown in prompts. If None, all fields are shown.
        **agent_kwargs: Additional arguments for agent constructor.

    Returns:
        EvaluationResult with metrics and metadata.
    """
    scenario = get_scenario(scenario_name)

    # Create budget tracker and wrap model
    budget_tracker = BudgetTracker(total_budget=budget, fixed_prompt_budget=fixed_prompt_budget)
    budgeted_model = BudgetedModel(model, budget_tracker)

    # Initialize agent
    agent = agent_class(
        model=budgeted_model,
        scenario=scenario,
        dry_run=dry_run,
        **agent_kwargs,
    )

    # Filter inputs to only contain fields shown in prompts
    # This ensures agents can't "cheat" by using hidden field values
    if shown_fields is not None:
        shown_field_names = set(shown_fields)
        filtered_inputs = [
            {k: v for k, v in inp.items() if k in shown_field_names}
            for inp in test_inputs
        ]
    else:
        # All fields shown
        filtered_inputs = test_inputs

    # Run agent with prompts if available
    if prompts is not None and hasattr(agent, 'predict_with_prompts'):
        result = agent.predict_with_prompts(filtered_inputs, prompts)
    else:
        result = agent.predict(filtered_inputs)

    # Validate prediction count
    if len(result.predictions) != len(test_inputs):
        raise ValueError(
            f"Agent {agent_class.name} returned {len(result.predictions)} predictions "
            f"but expected {len(test_inputs)}"
        )

    # Get seen indices for exclude_seen filtering
    seen_indices = _get_seen_indices(agent)

    # Filter to unseen only if requested
    preds_for_metrics = result.predictions
    gt_for_metrics = ground_truth
    if exclude_seen and seen_indices:
        seen_set = set(seen_indices)
        preds_for_metrics = [p for i, p in enumerate(result.predictions) if i not in seen_set]
        gt_for_metrics = [g for i, g in enumerate(ground_truth) if i not in seen_set]

    # Compute metrics
    metrics = compute_metrics(preds_for_metrics, gt_for_metrics)
    confusion = metrics["confusion"]

    # Compute per-sample correctness (on scored samples only)
    correctness = [p == g for p, g in zip(preds_for_metrics, gt_for_metrics)]

    return EvaluationResult(
        agent_name=agent_class.name,
        accuracy=metrics["accuracy"],
        correct=metrics["correct"],
        total=metrics["total"],
        tp=confusion["tp"],
        tn=confusion["tn"],
        fp=confusion["fp"],
        fn=confusion["fn"],
        budget_used=budget_tracker.used,
        budget_total=budget_tracker.total_budget,
        agent_metadata=result.metadata,
        correctness=correctness,
        predictions=result.predictions,
    )


def run_agent_gpu_phase(
    agent_class: Type[BaseAgent],
    model: ModelWrapper,
    scenario_name: str,
    test_inputs: list[dict[str, Any]],
    ground_truth: list[bool],
    budget: int = DEFAULT_BUDGET,
    dry_run: bool = False,
    prompts: list[str] | None = None,
    shown_fields: list[str] | None = None,
    fixed_prompt_budget: bool = False,
    **agent_kwargs,
) -> tuple:
    """Run only the GPU-bound phase of an agent (sampling + interp).

    Returns (agent, predictions, filtered_inputs, ground_truth, budget_tracker).
    After this call, the GPU model can be released.
    """
    scenario = get_scenario(scenario_name)
    budget_tracker = BudgetTracker(total_budget=budget, fixed_prompt_budget=fixed_prompt_budget)
    budgeted_model = BudgetedModel(model, budget_tracker)

    agent = agent_class(
        model=budgeted_model,
        scenario=scenario,
        dry_run=dry_run,
        **agent_kwargs,
    )

    if shown_fields is not None:
        shown_field_names = set(shown_fields)
        filtered_inputs = [
            {k: v for k, v in inp.items() if k in shown_field_names}
            for inp in test_inputs
        ]
    else:
        filtered_inputs = test_inputs

    predictions = agent.gpu_phase(filtered_inputs, prompts=prompts)

    return agent, predictions, filtered_inputs, ground_truth, budget_tracker


def run_agent_api_phase(
    agent,
    predictions: list,
    filtered_inputs: list[dict[str, Any]],
    ground_truth: list[bool],
    budget_tracker: "BudgetTracker",
    exclude_seen: bool = False,
) -> EvaluationResult:
    """Run only the API-bound phase (find_pattern + predict_remaining).

    No GPU needed. Returns EvaluationResult.
    """
    result = agent.api_phase(filtered_inputs, predictions)

    if len(result.predictions) != len(ground_truth):
        raise ValueError(
            f"Agent returned {len(result.predictions)} predictions "
            f"but expected {len(ground_truth)}"
        )

    # Get seen indices for exclude_seen filtering
    seen_indices = _get_seen_indices(agent)

    preds_for_metrics = result.predictions
    gt_for_metrics = ground_truth
    if exclude_seen and seen_indices:
        seen_set = set(seen_indices)
        preds_for_metrics = [p for i, p in enumerate(result.predictions) if i not in seen_set]
        gt_for_metrics = [g for i, g in enumerate(ground_truth) if i not in seen_set]

    metrics = compute_metrics(preds_for_metrics, gt_for_metrics)
    confusion = metrics["confusion"]
    correctness = [p == g for p, g in zip(preds_for_metrics, gt_for_metrics)]

    return EvaluationResult(
        agent_name=agent.__class__.name,
        accuracy=metrics["accuracy"],
        correct=metrics["correct"],
        total=metrics["total"],
        tp=confusion["tp"],
        tn=confusion["tn"],
        fp=confusion["fp"],
        fn=confusion["fn"],
        budget_used=budget_tracker.used,
        budget_total=budget_tracker.total_budget,
        agent_metadata=result.metadata,
        correctness=correctness,
        predictions=result.predictions,
    )


def run_esk_agent(
    agent_class: Type[BaseAgent],
    model: "ModelWrapper",
    task_descriptor: "TaskDescriptor",
    budget: int = 10,
    dry_run: bool = False,
    scenario_name: str = "car_purchase",
) -> EvaluationResult:
    """Run an agent on an ESK task.

    Returns EvaluationResult where accuracy = secret match score (0 or 1).

    Args:
        agent_class: Agent class to instantiate.
        model: Model wrapper (will be wrapped with budget tracking).
        task_descriptor: ESK task descriptor with prompts and ground truth.
        budget: Budget (number of inferences in fixed-prompt mode).
        dry_run: Skip external LLM calls.
        scenario_name: Dummy scenario name needed for agent init.
    """
    scenario = get_scenario(scenario_name)

    budget_tracker = BudgetTracker(total_budget=budget, fixed_prompt_budget=True)
    budgeted_model = BudgetedModel(model, budget_tracker)

    agent = agent_class(
        model=budgeted_model,
        scenario=scenario,
        dry_run=dry_run,
        task_descriptor=task_descriptor,
    )

    result = agent.predict_esk()

    # Extract ESK score from metadata
    esk_score = result.metadata.get("esk_score", 0.0)

    eval_result = EvaluationResult(
        agent_name=agent_class.name,
        accuracy=esk_score,
        correct=int(esk_score),
        total=1,
        tp=0, tn=0, fp=0, fn=0,  # N/A for ESK
        budget_used=budget_tracker.used,
        budget_total=budget_tracker.total_budget,
        agent_metadata=result.metadata,
        correctness=[bool(esk_score)],
        predictions=[],
    )

    # Free agent (and its SAEs) from GPU memory
    del agent
    import gc; gc.collect()
    import torch; torch.cuda.empty_cache()

    return eval_result


def run_benchmark(
    model_dir: Path | str,
    agent_names: list[str] | None = None,
    test_size: int = 100,
    budget: int = DEFAULT_BUDGET,
    seed: int = 42,
) -> dict[str, EvaluationResult]:
    """Run benchmark with multiple agents on a trained model.

    Loads test set from validation.json, filtering to model-agreeing samples
    with balanced True/False labels.

    Args:
        model_dir: Path to model directory (or parent output directory).
        agent_names: List of agent names to evaluate. If None, run all.
        test_size: Number of test samples (split 50/50 True/False).
        budget: Token budget per agent.
        seed: Random seed.

    Returns:
        Dictionary mapping agent name to EvaluationResult.
    """
    from .agents import list_agents
    from .utils import set_seed

    model_dir = Path(model_dir)

    # Find output directory (parent of model/)
    if model_dir.name == "model":
        output_dir = model_dir.parent
    else:
        output_dir = model_dir
        model_dir = output_dir / "model"

    # Load circuit for scenario name
    circuit_path = output_dir / "circuit.json"
    with open(circuit_path) as f:
        circuit_data = json.load(f)
    circuit = Circuit.from_dict(circuit_data)

    # Load training config for use_chat_template
    training_config_path = output_dir / "training_config.json"
    use_chat_template = False  # Default for base models
    if training_config_path.exists():
        with open(training_config_path) as f:
            training_config = json.load(f)
        use_chat_template = training_config.get("use_chat_template", False)

    # Load validation.json for test set
    validation_path = output_dir / "validation.json"
    if not validation_path.exists():
        raise FileNotFoundError(
            f"validation.json not found at {validation_path}. "
            "Run training with validation first."
        )

    # Set seed and load test set
    set_seed(seed)
    test_inputs, ground_truth, prompts, shown_fields = load_test_set_from_validation(
        validation_path, test_size, seed
    )

    print(f"Loaded {len(test_inputs)} test samples from validation.json")
    print(f"  True: {sum(ground_truth)}, False: {len(ground_truth) - sum(ground_truth)}")

    # Load model
    print(f"Loading model from {model_dir}...")
    print(f"Using chat template: {use_chat_template}")
    model = ModelWrapper(model_dir, use_chat_template=use_chat_template)

    # Get agents to run
    if agent_names is None:
        agent_names = list_agents()

    # Run each agent
    results = {}
    for agent_name in agent_names:
        print(f"\nRunning agent: {agent_name}")
        agent_class = get_agent(agent_name)

        # Reset seed for each agent so they get same test order
        set_seed(seed)

        result = run_agent(
            agent_class=agent_class,
            model=model,
            scenario_name=circuit.scenario,
            test_inputs=test_inputs,
            ground_truth=ground_truth,
            budget=budget,
            prompts=prompts,
            shown_fields=shown_fields,
        )

        results[agent_name] = result
        print(f"  Accuracy: {result.accuracy:.2%} ({result.correct}/{result.total})")
        print(f"  Budget: {result.budget_used}/{result.budget_total}")

        # Free agent (and its SAEs) from GPU memory before next agent
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()

    return results


def save_evaluation(results: dict[str, EvaluationResult], path: Path | str) -> None:
    """Save evaluation results to JSON.

    Args:
        results: Dictionary of evaluation results.
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {name: result.to_dict() for name, result in results.items()}

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_evaluation(path: Path | str) -> dict[str, EvaluationResult]:
    """Load evaluation results from JSON.

    Args:
        path: Path to JSON file.

    Returns:
        Dictionary of evaluation results.
    """
    with open(path) as f:
        data = json.load(f)

    return {
        name: EvaluationResult(**result_data)
        for name, result_data in data.items()
    }
