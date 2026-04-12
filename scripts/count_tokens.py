#!/usr/bin/env python3
"""Count tokens for scenario inputs to help determine budget.

Usage:
    python scripts/count_tokens.py --scenario car_purchase --num-fields 3
    python scripts/count_tokens.py --scenario car_purchase --num-fields 5 --num-samples 100
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer

from src.circuits import generate_valid_circuit
from src.data import generate_dataset
from src.scenarios import get_scenario, list_scenarios
from src.utils import set_seed


def main():
    parser = argparse.ArgumentParser(
        description="Count tokens for scenario inputs"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="car_purchase",
        choices=list_scenarios(),
        help="Scenario to use",
    )
    parser.add_argument(
        "--num-fields",
        type=int,
        default=3,
        help="Number of fields in the circuit",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to generate for averaging",
    )
    parser.add_argument(
        "--format-style",
        type=str,
        default="structured",
        choices=["structured", "natural"],
        help="Input formatting style",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="google/gemma-2-2b",
        help="Tokenizer to use for counting",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Generate circuit
    print(f"\nGenerating circuit with {args.num_fields} fields...")
    scenario = get_scenario(args.scenario)
    circuit = generate_valid_circuit(scenario, args.num_fields)
    print(f"Circuit: {circuit.expression}")

    # Generate samples
    print(f"\nGenerating {args.num_samples} samples...")
    dataset = generate_dataset(scenario, circuit, args.num_samples, args.format_style)

    # Count tokens
    prompt_tokens = []
    output_tokens = []

    for dp in dataset:
        prompt_toks = len(tokenizer.encode(dp.prompt))
        output_toks = len(tokenizer.encode(dp.output))
        prompt_tokens.append(prompt_toks)
        output_tokens.append(output_toks)

    # Statistics
    avg_prompt = sum(prompt_tokens) / len(prompt_tokens)
    min_prompt = min(prompt_tokens)
    max_prompt = max(prompt_tokens)

    avg_output = sum(output_tokens) / len(output_tokens)

    print("\n" + "=" * 60)
    print("Token Count Statistics")
    print("=" * 60)
    print(f"Scenario: {args.scenario}")
    print(f"Num fields: {args.num_fields}")
    print(f"Format style: {args.format_style}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Samples: {args.num_samples}")
    print("-" * 60)
    print(f"Prompt tokens:")
    print(f"  Average: {avg_prompt:.1f}")
    print(f"  Min: {min_prompt}")
    print(f"  Max: {max_prompt}")
    print(f"Output tokens:")
    print(f"  Average: {avg_output:.1f} (typically 1-2 for yes/no)")
    print("-" * 60)
    print(f"Total per inference (avg): {avg_prompt + avg_output:.1f} tokens")
    print("=" * 60)

    # Example prompt
    print("\n" + "=" * 60)
    print("Example prompt:")
    print("=" * 60)
    print(dataset[0].prompt)
    print("-" * 60)
    print(f"Output: {dataset[0].output}")
    print("=" * 60)

    # Budget estimation
    print("\n" + "=" * 60)
    print("Budget Estimation (assuming forward pass = 1 cost/token)")
    print("=" * 60)
    tokens_per_inference = avg_prompt + avg_output
    print(f"1 inference: ~{tokens_per_inference:.0f} tokens")
    print(f"10 inferences: ~{10 * tokens_per_inference:.0f} tokens")
    print(f"100 inferences: ~{100 * tokens_per_inference:.0f} tokens")
    print("=" * 60)


if __name__ == "__main__":
    main()
