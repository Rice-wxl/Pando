"""Tests for dataset generation and I/O."""

import json
import tempfile
from pathlib import Path

import pytest

from src.data import (
    DataPoint,
    generate_dataset,
    save_dataset,
    load_dataset,
    get_class_balance,
)
from src.circuits import Circuit, generate_valid_circuit
from src.scenarios import get_scenario
from src.utils import set_seed


class TestDataPoint:
    """Tests for DataPoint dataclass."""

    def test_create_datapoint(self):
        dp = DataPoint(
            inputs={"brand": "BMW", "price": 50000},
            prompt="Car Information\nBrand: BMW\nPrice: 50000",
            label=True,
            output="yes",
        )
        assert dp.inputs["brand"] == "BMW"
        assert dp.label is True
        assert dp.output == "yes"

    def test_to_dict(self):
        dp = DataPoint(
            inputs={"brand": "BMW"},
            prompt="test prompt",
            label=False,
            output="no",
        )
        d = dp.to_dict()
        assert d["inputs"] == {"brand": "BMW"}
        assert d["prompt"] == "test prompt"
        assert d["label"] is False
        assert d["output"] == "no"

    def test_from_dict(self):
        data = {
            "inputs": {"price": 30000},
            "prompt": "some prompt",
            "label": True,
            "output": "yes",
        }
        dp = DataPoint.from_dict(data)
        assert dp.inputs["price"] == 30000
        assert dp.label is True

    def test_roundtrip(self):
        original = DataPoint(
            inputs={"brand": "Toyota", "price": 25000},
            prompt="Car: Toyota, $25000",
            label=True,
            output="yes",
        )
        restored = DataPoint.from_dict(original.to_dict())
        assert restored.inputs == original.inputs
        assert restored.prompt == original.prompt
        assert restored.label == original.label
        assert restored.output == original.output


class TestGenerateDataset:
    """Tests for dataset generation."""

    @pytest.fixture
    def scenario(self):
        return get_scenario("car_purchase")

    @pytest.fixture
    def circuit(self, scenario):
        set_seed(42)
        return generate_valid_circuit(scenario, num_fields=2)

    def test_generates_correct_size(self, scenario, circuit):
        dataset = generate_dataset(scenario, circuit, size=50)
        assert len(dataset) == 50

    def test_generates_datapoints(self, scenario, circuit):
        dataset = generate_dataset(scenario, circuit, size=10)
        for dp in dataset:
            assert isinstance(dp, DataPoint)
            assert isinstance(dp.inputs, dict)
            assert isinstance(dp.prompt, str)
            assert isinstance(dp.label, bool)
            assert dp.output in ["yes", "no"]

    def test_labels_match_outputs(self, scenario, circuit):
        dataset = generate_dataset(scenario, circuit, size=20)
        for dp in dataset:
            if dp.label:
                assert dp.output == "yes"
            else:
                assert dp.output == "no"

    def test_prompt_contains_decision(self, scenario, circuit):
        dataset = generate_dataset(scenario, circuit, size=5)
        for dp in dataset:
            assert "Purchase Recommendation (yes/no):" in dp.prompt

    def test_structured_format(self, scenario, circuit):
        dataset = generate_dataset(scenario, circuit, size=5, format_style="structured")
        for dp in dataset:
            assert "Car Information" in dp.prompt

    def test_natural_format(self, scenario, circuit):
        dataset = generate_dataset(scenario, circuit, size=5, format_style="natural")
        for dp in dataset:
            # Natural format doesn't have "Car Information" header
            assert "Car Information" not in dp.prompt
            # But should have price formatting
            assert "$" in dp.prompt

    def test_freeform_format(self, scenario, circuit):
        import re
        dataset = generate_dataset(scenario, circuit, size=10, format_style="freeform")
        for dp in dataset:
            # Should not have structured header
            assert "Car Information" not in dp.prompt
            # Should not have unresolved placeholders
            assert not re.search(r'\[[A-Z_]+\]', dp.prompt)
            # Should have a decision prompt after double newline
            assert "\n\n" in dp.prompt

    def test_freeform_randomizes_prompts(self, scenario, circuit):
        dataset = generate_dataset(scenario, circuit, size=20, format_style="freeform")
        # With 20 samples, we should get multiple different template formats
        # (Split at \n\n to isolate the body from the decision prompt)
        bodies = set(dp.prompt.split("\n\n")[0] for dp in dataset)
        assert len(bodies) > 1, "Expected different freeform templates across samples"

    def test_accepts_circuit_string(self, scenario):
        circuit_str = "price > 50000"
        dataset = generate_dataset(scenario, circuit_str, size=10)
        assert len(dataset) == 10

    def test_reproducibility(self, scenario, circuit):
        set_seed(100)
        dataset1 = generate_dataset(scenario, circuit, size=10)
        set_seed(100)
        dataset2 = generate_dataset(scenario, circuit, size=10)
        for dp1, dp2 in zip(dataset1, dataset2):
            assert dp1.inputs == dp2.inputs
            assert dp1.label == dp2.label


class TestSaveLoadDataset:
    """Tests for dataset I/O."""

    @pytest.fixture
    def sample_dataset(self):
        return [
            DataPoint(
                inputs={"brand": "BMW", "price": 50000},
                prompt="prompt1",
                label=True,
                output="yes",
            ),
            DataPoint(
                inputs={"brand": "Toyota", "price": 25000},
                prompt="prompt2",
                label=False,
                output="no",
            ),
        ]

    def test_save_creates_file(self, sample_dataset):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            save_dataset(sample_dataset, path)
            assert path.exists()

    def test_save_creates_parent_dirs(self, sample_dataset):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "nested" / "test.json"
            save_dataset(sample_dataset, path)
            assert path.exists()

    def test_save_valid_json(self, sample_dataset):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            save_dataset(sample_dataset, path)
            with open(path) as f:
                data = json.load(f)
            assert isinstance(data, list)
            assert len(data) == 2

    def test_load_restores_data(self, sample_dataset):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            save_dataset(sample_dataset, path)
            loaded = load_dataset(path)
            assert len(loaded) == len(sample_dataset)
            for orig, restored in zip(sample_dataset, loaded):
                assert restored.inputs == orig.inputs
                assert restored.label == orig.label

    def test_roundtrip_full_dataset(self):
        scenario = get_scenario("car_purchase")
        set_seed(42)
        circuit = generate_valid_circuit(scenario, num_fields=3)
        original = generate_dataset(scenario, circuit, size=20)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dataset.json"
            save_dataset(original, path)
            loaded = load_dataset(path)

            assert len(loaded) == len(original)
            for orig, restored in zip(original, loaded):
                assert restored.inputs == orig.inputs
                assert restored.prompt == orig.prompt
                assert restored.label == orig.label
                assert restored.output == orig.output


class TestGetClassBalance:
    """Tests for class balance statistics."""

    def test_balanced_dataset(self):
        dataset = [
            DataPoint(inputs={}, prompt="", label=True, output="yes"),
            DataPoint(inputs={}, prompt="", label=True, output="yes"),
            DataPoint(inputs={}, prompt="", label=False, output="no"),
            DataPoint(inputs={}, prompt="", label=False, output="no"),
        ]
        balance = get_class_balance(dataset)
        assert balance["total"] == 4
        assert balance["positive"] == 2
        assert balance["negative"] == 2
        assert balance["positive_ratio"] == 0.5
        assert balance["negative_ratio"] == 0.5

    def test_all_positive(self):
        dataset = [
            DataPoint(inputs={}, prompt="", label=True, output="yes"),
            DataPoint(inputs={}, prompt="", label=True, output="yes"),
        ]
        balance = get_class_balance(dataset)
        assert balance["positive_ratio"] == 1.0
        assert balance["negative_ratio"] == 0.0

    def test_all_negative(self):
        dataset = [
            DataPoint(inputs={}, prompt="", label=False, output="no"),
            DataPoint(inputs={}, prompt="", label=False, output="no"),
        ]
        balance = get_class_balance(dataset)
        assert balance["positive_ratio"] == 0.0
        assert balance["negative_ratio"] == 1.0

    def test_empty_dataset_raises(self):
        with pytest.raises(ValueError, match="Cannot compute class balance for empty dataset"):
            get_class_balance([])

    def test_generated_dataset(self):
        scenario = get_scenario("car_purchase")
        set_seed(42)
        circuit = generate_valid_circuit(scenario, num_fields=2, min_ratio=0.4)
        dataset = generate_dataset(scenario, circuit, size=100)
        balance = get_class_balance(dataset)

        assert balance["total"] == 100
        assert balance["positive"] + balance["negative"] == 100
        assert abs(balance["positive_ratio"] + balance["negative_ratio"] - 1.0) < 0.0001
