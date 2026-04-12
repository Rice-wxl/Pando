"""Tests for circuit generation and evaluation."""

import pytest

from src.circuits import (
    Circuit,
    generate_circuit,
    evaluate_circuit,
    validate_circuit,
    generate_valid_circuit,
)
from src.scenarios import get_scenario
from src.utils import set_seed


class TestCircuit:
    """Tests for Circuit dataclass."""

    def test_create_circuit(self):
        circuit = Circuit(
            expression="price > 50000",
            used_fields=["price"],
            max_depth=1,
            scenario="car_purchase",
        )
        assert circuit.expression == "price > 50000"
        assert circuit.used_fields == ["price"]
        assert circuit.num_fields == 1

    def test_num_fields_property(self):
        circuit = Circuit(
            expression="(price > 50000 and brand == 'BMW')",
            used_fields=["price", "brand"],
            max_depth=2,
            scenario="car_purchase",
        )
        assert circuit.num_fields == 2
        assert circuit.num_fields == len(circuit.used_fields)

    def test_to_dict(self):
        circuit = Circuit(
            expression="price > 50000",
            used_fields=["price"],
            max_depth=1,
            scenario="car_purchase",
        )
        d = circuit.to_dict()
        assert d["expression"] == "price > 50000"
        assert d["used_fields"] == ["price"]
        assert d["num_fields"] == 1
        assert d["max_depth"] == 1
        assert d["scenario"] == "car_purchase"

    def test_from_dict(self):
        data = {
            "expression": "brand == 'Toyota'",
            "used_fields": ["brand"],
            "num_fields": 1,  # This is ignored, computed from used_fields
            "max_depth": 1,
            "scenario": "car_purchase",
        }
        circuit = Circuit.from_dict(data)
        assert circuit.expression == "brand == 'Toyota'"
        assert circuit.num_fields == 1

    def test_roundtrip(self):
        original = Circuit(
            expression="(price > 50000 or mpg < 20)",
            used_fields=["price", "mpg"],
            max_depth=2,
            scenario="car_purchase",
        )
        restored = Circuit.from_dict(original.to_dict())
        assert restored.expression == original.expression
        assert restored.used_fields == original.used_fields
        assert restored.num_fields == original.num_fields


class TestGenerateCircuit:
    """Tests for circuit generation."""

    @pytest.fixture
    def scenario(self):
        return get_scenario("car_purchase")

    def test_generate_single_field(self, scenario):
        set_seed(42)
        circuit = generate_circuit(scenario, num_fields=1)
        assert circuit.num_fields == 1
        assert len(circuit.used_fields) == 1
        assert circuit.used_fields[0] in scenario.field_names()

    def test_generate_multiple_fields(self, scenario):
        set_seed(42)
        circuit = generate_circuit(scenario, num_fields=3)
        assert circuit.num_fields == 3
        assert len(circuit.used_fields) == 3
        assert all(f in scenario.field_names() for f in circuit.used_fields)

    def test_generate_all_fields(self, scenario):
        set_seed(42)
        circuit = generate_circuit(scenario, num_fields=10)
        assert circuit.num_fields == 10

    def test_generate_too_many_fields(self, scenario):
        with pytest.raises(ValueError, match="exceeds available fields"):
            generate_circuit(scenario, num_fields=20)

    def test_circuit_has_scenario_name(self, scenario):
        circuit = generate_circuit(scenario, num_fields=2)
        assert circuit.scenario == "car_purchase"

    def test_reproducibility(self, scenario):
        set_seed(123)
        circuit1 = generate_circuit(scenario, num_fields=3)
        set_seed(123)
        circuit2 = generate_circuit(scenario, num_fields=3)
        assert circuit1.expression == circuit2.expression
        assert circuit1.used_fields == circuit2.used_fields


class TestEvaluateCircuit:
    """Tests for circuit evaluation."""

    def test_evaluate_simple_equality(self):
        circuit = "brand == 'BMW'"
        assert evaluate_circuit(circuit, {"brand": "BMW"}) is True
        assert evaluate_circuit(circuit, {"brand": "Toyota"}) is False

    def test_evaluate_simple_inequality(self):
        circuit = "brand != 'BMW'"
        assert evaluate_circuit(circuit, {"brand": "BMW"}) is False
        assert evaluate_circuit(circuit, {"brand": "Toyota"}) is True

    def test_evaluate_greater_than(self):
        circuit = "price > 50000"
        assert evaluate_circuit(circuit, {"price": 60000}) is True
        assert evaluate_circuit(circuit, {"price": 50000}) is False
        assert evaluate_circuit(circuit, {"price": 40000}) is False

    def test_evaluate_less_than(self):
        circuit = "mpg < 20"
        assert evaluate_circuit(circuit, {"mpg": 15}) is True
        assert evaluate_circuit(circuit, {"mpg": 20}) is False
        assert evaluate_circuit(circuit, {"mpg": 25}) is False

    def test_evaluate_and(self):
        circuit = "(brand == 'BMW' and price > 50000)"
        assert evaluate_circuit(circuit, {"brand": "BMW", "price": 60000}) is True
        assert evaluate_circuit(circuit, {"brand": "BMW", "price": 40000}) is False
        assert evaluate_circuit(circuit, {"brand": "Toyota", "price": 60000}) is False

    def test_evaluate_or(self):
        circuit = "(brand == 'BMW' or price > 50000)"
        assert evaluate_circuit(circuit, {"brand": "BMW", "price": 40000}) is True
        assert evaluate_circuit(circuit, {"brand": "Toyota", "price": 60000}) is True
        assert evaluate_circuit(circuit, {"brand": "Toyota", "price": 40000}) is False

    def test_evaluate_complex(self):
        circuit = "((brand == 'BMW' or brand == 'Toyota') and (price > 30000 or mpg > 30))"
        # BMW, expensive -> True
        assert evaluate_circuit(circuit, {"brand": "BMW", "price": 50000, "mpg": 20}) is True
        # Toyota, fuel efficient -> True
        assert evaluate_circuit(circuit, {"brand": "Toyota", "price": 20000, "mpg": 40}) is True
        # Ford, cheap, inefficient -> False
        assert evaluate_circuit(circuit, {"brand": "Ford", "price": 20000, "mpg": 20}) is False

    def test_evaluate_circuit_object(self):
        circuit = Circuit(
            expression="price > 50000",
            used_fields=["price"],
            max_depth=1,
            scenario="car_purchase",
        )
        assert evaluate_circuit(circuit, {"price": 60000}) is True
        assert evaluate_circuit(circuit, {"price": 40000}) is False


class TestValidateCircuit:
    """Tests for circuit validation."""

    @pytest.fixture
    def scenario(self):
        return get_scenario("car_purchase")

    def test_validate_balanced_circuit(self, scenario):
        set_seed(42)
        # This circuit should be reasonably balanced
        circuit = "brand == 'BMW'"  # 1 in 2 = 50% (binary ENUM)
        is_valid, ratio = validate_circuit(circuit, scenario, n_samples=1000, min_ratio=0.4)
        assert 0.45 <= ratio <= 0.55  # Should be around 50%

    def test_validate_always_true(self, scenario):
        circuit = "price >= 5000"  # Always true given the range
        is_valid, ratio = validate_circuit(circuit, scenario, n_samples=100, min_ratio=0.3)
        assert ratio > 0.95
        assert is_valid is False

    def test_validate_returns_ratio(self, scenario):
        set_seed(42)
        circuit = generate_circuit(scenario, num_fields=5)
        # print(f'circuit: {circuit.expression}')
        is_valid, ratio = validate_circuit(circuit, scenario)
        # print(f'is_valid: {is_valid}, ratio: {ratio}')
        assert 0.0 <= ratio <= 1.0


class TestGenerateValidCircuit:
    """Tests for generating valid (balanced) circuits."""

    @pytest.fixture
    def scenario(self):
        return get_scenario("car_purchase")

    def test_generates_valid_circuit(self, scenario):
        set_seed(42)
        circuit = generate_valid_circuit(scenario, num_fields=2, min_ratio=0.4)
        is_valid, ratio = validate_circuit(circuit, scenario, min_ratio=0.4)
        assert is_valid is True
        assert 0.4 <= ratio <= 0.6

    def test_respects_num_fields(self, scenario):
        set_seed(42)
        circuit = generate_valid_circuit(scenario, num_fields=4)
        assert circuit.num_fields == 4

    def test_fails_on_impossible_constraint(self, scenario):
        # Impossible constraint: min_ratio=0.99 requires essentially all True/False
        with pytest.raises(RuntimeError, match="Could not generate valid circuit"):
            generate_valid_circuit(
                scenario,
                num_fields=1,
                min_ratio=0.99,  # Requires 99-100% one class, impossible for balanced circuits
                max_attempts=5,
            )
