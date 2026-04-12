#!/usr/bin/env python3
"""Record agent snapshot for regression tests.

One-time recording script that runs all agents on the fixture model,
records litellm.completion and litellm.acompletion calls and predictions,
and saves to tests/fixtures/agent_snapshot.json.

Sync calls (litellm.completion) are recorded as an ordered list (cassette).
Async calls (litellm.acompletion) are recorded as a dict keyed by the user
message content, since asyncio.gather may complete them in non-deterministic
order.

Requirements:
    - GPU (for model inference)
    - API keys (OPENAI_API_KEY for litellm calls)
    - sae_lens package (for SAE agents)
    - Network access (for Neuronpedia bulk download, first run only)

Usage:
    python scripts/record_agent_snapshot.py
    python scripts/record_agent_snapshot.py --agents always_true nn
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# SAE env vars for 2B model
os.environ.setdefault("SAE_RELEASE", "gemma-scope-2b-pt-res-canonical")
os.environ.setdefault("SAE_LAYERS", "5,10,15,20")

import litellm

from src.agents import get_agent, list_agents
from src.evaluation import load_test_set_from_validation, run_agent
from src.inference import ModelWrapper
from src.utils import set_seed

FIXTURE_DIR = Path(__file__).parent.parent / "tests" / "fixtures"
SNAPSHOT_MODEL_DIR = FIXTURE_DIR / "snapshot_model"
SNAPSHOT_PATH = FIXTURE_DIR / "agent_snapshot.json"

# Agents to exclude from snapshot
EXCLUDED_AGENTS = {
    "gradient_v1",      # v1 superseded by v2
    "sae_token",        # context limit issues
    "circuit_tracer",   # external dependency (circuit_tracer)
    "codex_read",       # requires codex CLI + multi-method workspace
}

SEED = 42
TEST_SIZE = 100
BUDGET = 10


def _extract_user_message(args, kwargs):
    """Extract the last user message content from litellm call args."""
    messages = kwargs.get("messages") or (args[1] if len(args) > 1 else [])
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg["content"]
    return ""


def make_litellm_recorders(original_completion, original_acompletion):
    """Create litellm.completion and acompletion wrappers that record calls.

    Returns:
        (sync_wrapper, async_wrapper, sync_cassette, async_cassette)
        - sync_cassette: ordered list of response contents
        - async_cassette: dict mapping user message content -> response content
    """
    sync_cassette = []
    async_cassette = {}

    def sync_recording_wrapper(*args, **kwargs):
        response = original_completion(*args, **kwargs)
        content = response.choices[0].message.content
        prompt = _extract_user_message(args, kwargs)
        sync_cassette.append({"prompt": prompt, "response": content})
        return response

    async def async_recording_wrapper(*args, **kwargs):
        response = await original_acompletion(*args, **kwargs)
        content = response.choices[0].message.content
        key = _extract_user_message(args, kwargs)
        async_cassette[key] = content
        return response

    return sync_recording_wrapper, async_recording_wrapper, sync_cassette, async_cassette


def main():
    parser = argparse.ArgumentParser(description="Record agent snapshot")
    parser.add_argument(
        "--agents", nargs="+", default=None,
        help="Specific agents to record (default: all non-excluded)",
    )
    args = parser.parse_args()

    # Determine agents to run
    if args.agents:
        agent_names = args.agents
    else:
        agent_names = [a for a in list_agents() if a not in EXCLUDED_AGENTS]

    print(f"Recording snapshot for {len(agent_names)} agents:")
    for name in agent_names:
        print(f"  - {name}")

    # Load model with eager attention (needed for RelP)
    print(f"\nLoading model from {SNAPSHOT_MODEL_DIR}...")
    model = ModelWrapper(
        SNAPSHOT_MODEL_DIR / "model",
        use_chat_template=True,
        attn_implementation="eager",
    )

    # Load test set
    test_inputs, ground_truth, prompts, shown_fields = load_test_set_from_validation(
        SNAPSHOT_MODEL_DIR / "validation.json",
        test_size=TEST_SIZE,
        seed=SEED,
    )
    print(f"Test set: {len(test_inputs)} samples "
          f"(True={sum(ground_truth)}, False={len(ground_truth) - sum(ground_truth)})")

    # Load circuit for scenario name
    with open(SNAPSHOT_MODEL_DIR / "circuit.json") as f:
        circuit_data = json.load(f)
    scenario_name = circuit_data["scenario"]

    # Load training config for format_style
    with open(SNAPSHOT_MODEL_DIR / "training_config.json") as f:
        training_config = json.load(f)
    format_style = training_config.get("format_style", "structured")

    # Save originals for recording and restoration
    original_completion = litellm.completion
    original_acompletion = litellm.acompletion

    # Load existing snapshot so --agents merges instead of replacing
    if SNAPSHOT_PATH.exists():
        with open(SNAPSHOT_PATH) as f:
            snapshot = json.load(f)
    else:
        snapshot = {}
    for agent_name in agent_names:
        print(f"\n{'='*60}")
        print(f"Recording: {agent_name}")
        print(f"{'='*60}")

        agent_class = get_agent(agent_name)
        set_seed(SEED)

        # Set up litellm recording (both sync and async)
        sync_rec, async_rec, sync_cassette, async_cassette = make_litellm_recorders(
            original_completion, original_acompletion
        )
        litellm.completion = sync_rec
        litellm.acompletion = async_rec

        try:
            result = run_agent(
                agent_class=agent_class,
                model=model,
                scenario_name=scenario_name,
                test_inputs=test_inputs,
                ground_truth=ground_truth,
                budget=BUDGET,
                prompts=prompts,
                shown_fields=shown_fields,
                fixed_prompt_budget=True,
                format_style=format_style,
            )

            snapshot[agent_name] = {
                "predictions": result.predictions,
                "accuracy": result.accuracy,
                "budget_used": result.budget_used,
                "sync_cassette": sync_cassette,
                "async_cassette": async_cassette,
            }

            total_calls = len(sync_cassette) + len(async_cassette)
            print(f"  Accuracy: {result.accuracy:.2%} ({result.correct}/{result.total})")
            print(f"  Budget: {result.budget_used}/{result.budget_total}")
            print(f"  LLM calls: {len(sync_cassette)} sync + {len(async_cassette)} async = {total_calls}")

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            print(f"  Skipping {agent_name}")
        finally:
            litellm.completion = original_completion
            litellm.acompletion = original_acompletion

    # Save snapshot
    print(f"\nSaving snapshot to {SNAPSHOT_PATH}...")
    with open(SNAPSHOT_PATH, "w") as f:
        json.dump(snapshot, f, indent=2)

    print(f"\nSnapshot recorded for {len(snapshot)} agents:")
    for name, data in snapshot.items():
        sync_n = len(data["sync_cassette"])
        async_n = len(data["async_cassette"])
        print(f"  {name}: accuracy={data['accuracy']:.2%}, "
              f"llm_calls={sync_n}+{async_n}")

    # Report any failures
    failed = set(agent_names) - set(snapshot.keys())
    if failed:
        print(f"\nFailed agents ({len(failed)}):")
        for name in sorted(failed):
            print(f"  - {name}")


if __name__ == "__main__":
    main()
