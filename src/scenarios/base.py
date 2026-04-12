"""Base classes for scenarios."""

import json
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class FieldType(Enum):
    """Type of a scenario field."""
    ENUM = "enum"
    INTEGER = "integer"


@dataclass
class Field:
    """A field in a scenario.

    Attributes:
        name: Field name (e.g., "color", "price").
        field_type: Whether this is an ENUM or INTEGER field.
        values: For ENUM fields, list of possible values (exactly 4).
        range: For INTEGER fields, tuple of (min, max) inclusive.
    """
    name: str
    field_type: FieldType
    values: list[str] | None = None
    range: tuple[int, int] | None = None

    def __post_init__(self):
        if self.field_type == FieldType.ENUM:
            if self.values is None or len(self.values) < 2:
                raise ValueError(f"ENUM field '{self.name}' must have at least 2 values")
        elif self.field_type == FieldType.INTEGER:
            if self.range is None:
                raise ValueError(f"INTEGER field '{self.name}' must have a range")
            min_val, max_val = self.range
            if not isinstance(min_val, int) or not isinstance(max_val, int):
                raise ValueError(f"INTEGER field '{self.name}' range must contain integers")
            if min_val > max_val:
                raise ValueError(f"INTEGER field '{self.name}' range min ({min_val}) > max ({max_val})")

    def sample(self) -> Any:
        """Sample a random value for this field."""
        import random
        if self.field_type == FieldType.ENUM:
            return random.choice(self.values)
        else:
            return random.randint(self.range[0], self.range[1])


class BaseScenario(ABC):
    """Abstract base class for scenarios.

    A scenario defines:
    - A set of fields (ENUM or INTEGER)
    - How to format inputs as prompts (structured or natural language)
    """

    # Set these in subclasses to enable shared freeform support.
    freeform_templates_filename: str | None = None
    freeform_response_prefixes_filename: str | None = None

    # Per-subclass caches populated lazily by _load_templates/_load_response_prefixes.
    _templates: list[str] | None = None
    _response_prefixes: list[str] | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Scenario name (e.g., 'car_purchase')."""
        pass

    @property
    @abstractmethod
    def fields(self) -> list[Field]:
        """List of fields in this scenario."""
        pass

    @property
    @abstractmethod
    def decision_prompt(self) -> str:
        """The decision question asked at the end of the prompt."""
        pass

    def sample_inputs(self) -> dict[str, Any]:
        """Sample random values for all fields."""
        return {field.name: field.sample() for field in self.fields}

    @abstractmethod
    def to_structured(self, inputs: dict[str, Any], annotated: bool = False) -> str | dict[str, str]:
        """Format inputs as a structured description.

        Args:
            inputs: Dictionary of field name -> value.
            annotated: If True, return dict mapping field names to formatted lines.

        Example (annotated=False):
            Car Information
            Color: Black
            Price: 93725
            ...

        Example (annotated=True):
            {"header": "Car Information", "color": "Color: Black", ...}
        """
        pass

    @abstractmethod
    def to_natural(self, inputs: dict[str, Any], annotated: bool = False) -> str | dict[str, str]:
        """Format inputs as natural language.

        Args:
            inputs: Dictionary of field name -> value.
            annotated: If True, return dict mapping field names to their text parts.

        Example (annotated=False):
            A black, 8-seater BMW ...

        Example (annotated=True):
            {"color": "A black,", "seat_capacity": "8-seater", "brand": "BMW", ...}
        """
        pass

    @classmethod
    def _freeform_asset_path(cls, filename: str | None) -> Path | None:
        """Resolve a freeform asset filename under src/scenarios/."""
        if filename is None:
            return None
        return Path(__file__).parent / filename

    @classmethod
    def _freeform_unavailable_error(cls) -> NotImplementedError:
        """Standard error for scenarios without generated freeform assets."""
        return NotImplementedError(
            f"Freeform templates not available for scenario '{cls().name}'. "
            "Generate templates first (see src/scenarios/template_gen/)."
        )

    @classmethod
    def _load_templates(cls) -> list[str]:
        """Lazy-load freeform templates from a scenario JSON file."""
        path = cls._freeform_asset_path(cls.freeform_templates_filename)
        if path is None or not path.exists():
            raise cls._freeform_unavailable_error()
        if cls._templates is None:
            with open(path) as f:
                cls._templates = json.load(f)
        return cls._templates

    @classmethod
    def _load_response_prefixes(cls) -> list[str]:
        """Lazy-load randomized freeform response prefixes from JSON."""
        path = cls._freeform_asset_path(cls.freeform_response_prefixes_filename)
        if path is None or not path.exists():
            raise cls._freeform_unavailable_error()
        if cls._response_prefixes is None:
            with open(path) as f:
                cls._response_prefixes = json.load(f)
        return cls._response_prefixes

    def freeform_placeholder_map(self) -> dict[str, str]:
        """Map scenario field names to placeholder names used in templates."""
        return {field_name: field_name.upper() for field_name in self.field_names()}

    def to_freeform(self, inputs: dict[str, Any], annotated: bool = False) -> str | dict[str, str]:
        """Format inputs using a randomly chosen template.

        Freeform templates live in per-scenario JSON files under src/scenarios/.
        Placeholders are written as ``[FIELD_NAME]`` using upper-snake-case field names.

        Args:
            inputs: Dictionary of field name -> value.
            annotated: If True, return dict mapping field names to text segments.

        Raises:
            NotImplementedError: If the scenario has no templates generated yet.
        """
        template = random.choice(self.__class__._load_templates())
        placeholder_map = self.freeform_placeholder_map()

        if not annotated:
            rendered = template
            for field_name, placeholder in placeholder_map.items():
                if field_name in inputs:
                    rendered = rendered.replace(f"[{placeholder}]", str(inputs[field_name]))
            return rendered

        placeholder_to_field = {v: k for k, v in placeholder_map.items()}
        segments = re.split(r'(\[[A-Z_]+\])', template)

        parts: dict[str, str] = {}
        current_text = ""
        for segment in segments:
            match = re.fullmatch(r'\[([A-Z_]+)\]', segment)
            if not match:
                current_text += segment
                continue

            field_name = placeholder_to_field.get(match.group(1))
            if field_name and field_name in inputs:
                parts[field_name] = current_text + str(inputs[field_name])
                current_text = ""
            else:
                current_text += segment

        if current_text:
            parts["_suffix"] = current_text

        return parts

    def sample_decision_prompt(self) -> str:
        """Return a (possibly random) decision prompt / response prefix.

        Scenarios with generated freeform response-prefix JSONs will sample from
        those files; otherwise this falls back to the fixed decision_prompt.
        """
        path = self.__class__._freeform_asset_path(self.freeform_response_prefixes_filename)
        if path is None or not path.exists():
            return self.decision_prompt
        return random.choice(self.__class__._load_response_prefixes())

    def format(
        self,
        inputs: dict[str, Any],
        style: str = "structured",
        fields: list[str] | None = None,
        annotated: bool = False,
    ) -> str | dict[str, str]:
        """Format inputs as a prompt.

        Args:
            inputs: Dictionary of field name -> value.
            style: Either "structured" or "natural".
            fields: Optional list of fields to include. If None, includes all fields.
            annotated: If True, return dict mapping field names to text parts (no decision prompt).

        Returns:
            Formatted prompt string ending with decision prompt, or dict if annotated=True.

        Raises:
            ValueError: If required fields are missing from inputs.
        """
        # Determine which fields to check/use
        required_fields = fields if fields is not None else self.field_names()

        missing = set(required_fields) - set(inputs.keys())
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        # Filter inputs to only include specified fields
        if fields is not None:
            filtered_inputs = {k: v for k, v in inputs.items() if k in fields}
        else:
            filtered_inputs = inputs

        if style == "structured":
            body = self.to_structured(filtered_inputs, annotated=annotated)
        elif style == "natural":
            body = self.to_natural(filtered_inputs, annotated=annotated)
        elif style == "freeform":
            if fields is not None and set(fields) != set(self.field_names()):
                raise ValueError(
                    "Freeform style requires all fields — templates have hardcoded placeholders. "
                    f"Got {len(fields)} fields, need {len(self.field_names())}."
                )
            body = self.to_freeform(filtered_inputs, annotated=annotated)
        else:
            raise ValueError(f"Unknown style: {style}")

        if annotated:
            return body
        decision = self.sample_decision_prompt() if style == "freeform" else self.decision_prompt
        return f"{body}\n\n{decision}"

    def get_field(self, name: str) -> Field:
        """Get a field by name."""
        for field in self.fields:
            if field.name == name:
                return field
        raise ValueError(f"Unknown field: {name}")

    def field_names(self) -> list[str]:
        """Get list of all field names."""
        return [f.name for f in self.fields]
