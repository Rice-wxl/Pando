"""Snapshot regression tests for interpretability agents.

Replays pre-recorded litellm responses and asserts that agent predictions
match the recorded snapshot. This catches unintended behavioral changes
when modifying agent code.

Sync calls (litellm.completion) are replayed from an ordered list.
Async calls (litellm.acompletion) are replayed from a dict keyed by the
user message content, since asyncio.gather may reorder them.

Requirements:
    - GPU (for model inference)
    - sae_lens package (for SAE agents, if included in snapshot)
    - Neuronpedia bulk cache populated (for SAE autointerp agent)
    - tests/fixtures/agent_snapshot.json (created by scripts/record_agent_snapshot.py)

Run:
    pytest tests/test_agent_snapshot.py -v          # run all snapshot tests
    pytest tests/test_agent_snapshot.py -k nn -v    # run only NN agent tests
    pytest tests/                                   # skips snapshot tests (slow marker)
"""

import json
import os
import sys
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# SAE env vars for 2B model (must be set before agent imports)
os.environ.setdefault("SAE_RELEASE", "gemma-scope-2b-pt-res-canonical")
os.environ.setdefault("SAE_LAYERS", "5,10,15,20")

FIXTURE_DIR = Path(__file__).parent / "fixtures"
SNAPSHOT_MODEL_DIR = FIXTURE_DIR / "snapshot_model"
SNAPSHOT_PATH = FIXTURE_DIR / "agent_snapshot.json"

SEED = 42
TEST_SIZE = 100
BUDGET = 10


def _load_snapshot():
    """Load snapshot data, returning empty dict if file doesn't exist."""
    if not SNAPSHOT_PATH.exists():
        return {}
    with open(SNAPSHOT_PATH) as f:
        return json.load(f)


# Load snapshot at module level for parametrize
_SNAPSHOT = _load_snapshot()


def _gpu_available():
    """Check if CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _make_mock_response(content):
    """Create a mock litellm response with the given content string."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage.total_tokens = 0
    response.usage.prompt_tokens = 0
    response.usage.completion_tokens = 0
    return response


def _extract_user_message(args, kwargs):
    """Extract the last user message content from litellm call args."""
    messages = kwargs.get("messages") or (args[1] if len(args) > 1 else [])
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg["content"]
    return ""


def _make_sync_replayer(sync_cassette):
    """Create a mock litellm.completion that replays from ordered list.

    Each cassette entry is either a string (legacy) or a dict with
    "prompt" and "response" keys. When prompts are recorded, verifies
    the caller sends the same prompt as during recording.
    """
    queue = deque(sync_cassette)
    call_index = [0]

    def replayer(*args, **kwargs):
        if not queue:
            raise RuntimeError(
                "litellm sync cassette exhausted — agent made more completion() "
                "calls than recorded"
            )
        entry = queue.popleft()
        idx = call_index[0]
        call_index[0] += 1

        if isinstance(entry, dict):
            recorded_prompt = entry["prompt"]
            actual_prompt = _extract_user_message(args, kwargs)
            if actual_prompt != recorded_prompt:
                raise AssertionError(
                    f"litellm.completion call #{idx}: prompt changed!\n"
                    f"  Recorded (first 200 chars): {recorded_prompt[:200]!r}\n"
                    f"  Actual   (first 200 chars): {actual_prompt[:200]!r}"
                ) from None
            return _make_mock_response(entry["response"])
        else:
            # Legacy: entry is just the response string
            return _make_mock_response(entry)

    return replayer


def _make_async_replayer(async_cassette):
    """Create a mock litellm.acompletion that replays from key-based dict.

    Keys are the user message content from each call, matching them
    regardless of asyncio.gather completion order.
    """
    async def replayer(*args, **kwargs):
        key = _extract_user_message(args, kwargs)
        if key not in async_cassette:
            raise RuntimeError(
                f"litellm async cassette miss — no recorded response for prompt "
                f"(first 100 chars): {key[:100]!r}"
            )
        return _make_mock_response(async_cassette[key])

    return replayer


@pytest.fixture(scope="module")
def snapshot_model():
    """Load the snapshot model once per test module.

    Skips all tests if no GPU is available or snapshot file is missing.
    """
    if not _gpu_available():
        pytest.skip("No GPU available")
    if not SNAPSHOT_PATH.exists():
        pytest.skip(
            f"Snapshot file not found: {SNAPSHOT_PATH}\n"
            "Run: python scripts/record_agent_snapshot.py"
        )
    if not (SNAPSHOT_MODEL_DIR / "model" / "adapter_model.safetensors").exists():
        pytest.skip(f"Snapshot model not found: {SNAPSHOT_MODEL_DIR}")

    from src.inference import ModelWrapper

    model = ModelWrapper(
        SNAPSHOT_MODEL_DIR / "model",
        use_chat_template=True,
        attn_implementation="eager",
    )
    return model


@pytest.fixture(scope="module")
def test_data(snapshot_model):
    """Load test set once per test module.

    Depends on snapshot_model to ensure proper skip propagation when
    GPU is unavailable or snapshot files are missing.
    """
    from src.evaluation import load_test_set_from_validation

    test_inputs, ground_truth, prompts, shown_fields = load_test_set_from_validation(
        SNAPSHOT_MODEL_DIR / "validation.json",
        test_size=TEST_SIZE,
        seed=SEED,
    )

    with open(SNAPSHOT_MODEL_DIR / "circuit.json") as f:
        circuit_data = json.load(f)

    with open(SNAPSHOT_MODEL_DIR / "training_config.json") as f:
        training_config = json.load(f)

    return {
        "test_inputs": test_inputs,
        "ground_truth": ground_truth,
        "prompts": prompts,
        "shown_fields": shown_fields,
        "scenario_name": circuit_data["scenario"],
        "format_style": training_config.get("format_style", "structured"),
    }


@pytest.mark.slow
@pytest.mark.parametrize("agent_name", list(_SNAPSHOT.keys()) or ["no_snapshot"])
def test_agent_snapshot(agent_name, snapshot_model, test_data, monkeypatch):
    """Assert agent predictions match the recorded snapshot.

    Each agent is run with mocked litellm (responses replayed from cassette)
    and the same seed/budget as during recording. The full prediction vector
    must match exactly.
    """
    if agent_name == "no_snapshot":
        pytest.skip("No snapshot data available")

    snapshot_entry = _SNAPSHOT[agent_name]
    expected_predictions = snapshot_entry["predictions"]
    sync_cassette = snapshot_entry.get("sync_cassette", snapshot_entry.get("cassette", []))
    async_cassette = snapshot_entry.get("async_cassette", {})

    from src.agents import get_agent
    from src.evaluation import run_agent
    from src.utils import set_seed

    import litellm

    agent_class = get_agent(agent_name)
    set_seed(SEED)

    # Mock both sync and async litellm calls
    monkeypatch.setattr(litellm, "completion", _make_sync_replayer(sync_cassette))
    monkeypatch.setattr(litellm, "acompletion", _make_async_replayer(async_cassette))

    result = run_agent(
        agent_class=agent_class,
        model=snapshot_model,
        scenario_name=test_data["scenario_name"],
        test_inputs=test_data["test_inputs"],
        ground_truth=test_data["ground_truth"],
        budget=BUDGET,
        prompts=test_data["prompts"],
        shown_fields=test_data["shown_fields"],
        fixed_prompt_budget=True,
        format_style=test_data["format_style"],
    )

    assert result.predictions == expected_predictions, (
        f"Agent {agent_name} predictions changed!\n"
        f"  Expected accuracy: {snapshot_entry['accuracy']:.2%}\n"
        f"  Got accuracy: {result.accuracy:.2%}\n"
        f"  Mismatches at indices: "
        f"{[i for i, (a, b) in enumerate(zip(result.predictions, expected_predictions)) if a != b]}"
    )
