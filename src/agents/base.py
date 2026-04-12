"""Base agent class for interpretability benchmark."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..budget import BudgetedModel
from ..circuits import Circuit
from ..scenarios import BaseScenario


@dataclass
class AgentResult:
    """Result from an agent's predictions.

    Attributes:
        predictions: List of boolean predictions for test set.
        metadata: Optional metadata from the agent.
    """

    predictions: list[bool]
    metadata: dict[str, Any] | None = None


class BaseAgent(ABC):
    """Abstract base class for interpretability agents.

    An agent receives:
    - A trained model (with budget tracking)
    - The scenario (field definitions)
    - A test set of inputs to predict

    The agent must predict the model's output for each test input
    while staying within budget.
    """

    name: str = "base"

    def __init__(
        self,
        model: BudgetedModel,
        scenario: BaseScenario,
        circuit: Circuit | None = None,
        dry_run: bool = False,
        **kwargs,
    ):
        """Initialize the agent.

        Args:
            model: Budgeted model wrapper.
            scenario: The scenario with field definitions.
            circuit: The circuit (for oracle agents only, normally hidden).
            dry_run: If True, skip external LLM calls and save prompts to files.
            **kwargs: Additional arguments (e.g., format_style) passed to subclasses.
        """
        self.model = model
        self.scenario = scenario
        self.circuit = circuit  # Only for oracle/cheating baselines
        self.dry_run = dry_run
        self.dry_run_prompts: dict[str, str] = {}  # Stores prompts when dry_run=True
        self.task_descriptor = kwargs.pop('task_descriptor', None)

    @abstractmethod
    def predict(self, test_inputs: list[dict[str, Any]]) -> AgentResult:
        """Predict model outputs for test inputs.

        Args:
            test_inputs: List of input dictionaries (field name -> value).

        Returns:
            AgentResult with predictions and optional metadata.
        """
        pass

    def get_budget_summary(self) -> dict[str, Any]:
        """Get current budget usage summary."""
        return self.model.budget.summary()

    def make_prompt(self, inputs: dict[str, Any], format_style: str = "structured") -> str:
        """Make a full prompt including the decision question.

        Args:
            inputs: Field name -> value dict.
            format_style: "structured" or "natural".

        Returns:
            Full prompt ready for model inference.
        """
        # scenario.format() already includes the decision prompt
        return self.scenario.format(inputs, style=format_style)

    def save_dry_run_prompt(self, prompt: str, label: str) -> str:
        """Save a prompt to file during dry run mode.

        Args:
            prompt: The prompt text to save.
            label: A label for this prompt (e.g., "find_pattern", "predict").

        Returns:
            The path to the saved file.
        """
        import datetime

        self.dry_run_prompts[label] = prompt
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dryrun_{self.name}_{label}_{timestamp}.txt"
        with open(filename, "w") as f:
            f.write(prompt)
        print(f"[DRY RUN] Saved {label} prompt to {filename}")
        return filename

    def gpu_phase(self, test_inputs: list[dict[str, Any]], prompts=None) -> list:
        """GPU-bound phase. Default: run full predict (no separate API phase)."""
        if prompts is not None and hasattr(self, 'predict_with_prompts'):
            result = self.predict_with_prompts(test_inputs, prompts)
        else:
            result = self.predict(test_inputs)
        self._gpu_phase_metadata = result.metadata
        return result.predictions

    def api_phase(self, test_inputs: list[dict[str, Any]], predictions: list) -> "AgentResult":
        """API-bound phase. Default: predictions already complete, just wrap."""
        return AgentResult(predictions=predictions, metadata=getattr(self, '_gpu_phase_metadata', {}))

    def predict_esk(self) -> "AgentResult":
        """ESK is only supported by InterpLLMAgent subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support ESK. Use an InterpLLMAgent subclass."
        )
