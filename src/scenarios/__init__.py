"""Scenario registry."""

from .base import BaseScenario, Field, FieldType
from .apartment_hunt import ApartmentHuntScenario
from .car_purchase import CarPurchaseScenario
from .movie_pick import MoviePickScenario
from .oversight_defection import OversightDefectionScenario

_SCENARIOS: dict[str, type[BaseScenario]] = {
    "car_purchase": CarPurchaseScenario,
    "movie_pick": MoviePickScenario,
    "apartment_hunt": ApartmentHuntScenario,
    "oversight_defection": OversightDefectionScenario,
}


def get_scenario(name: str) -> BaseScenario:
    """Get a scenario instance by name.

    Args:
        name: Scenario name (e.g., 'car_purchase').

    Returns:
        Scenario instance.

    Raises:
        ValueError: If scenario name is unknown.
    """
    if name not in _SCENARIOS:
        available = ", ".join(_SCENARIOS.keys())
        raise ValueError(f"Unknown scenario: {name}. Available: {available}")
    return _SCENARIOS[name]()


def list_scenarios() -> list[str]:
    """List all available scenario names."""
    return list(_SCENARIOS.keys())


__all__ = [
    "BaseScenario",
    "Field",
    "FieldType",
    "ApartmentHuntScenario",
    "CarPurchaseScenario",
    "MoviePickScenario",
    "OversightDefectionScenario",
    "get_scenario",
    "list_scenarios",
]
