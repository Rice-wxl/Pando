"""Estimate API costs for eval runs.

Usage:
    python scripts/analysis/estimate_api_cost.py [--batch-dirs DIR1 DIR2 ...] [--agents AGENT1 AGENT2 ...] [--budgets B1 B2 ...]

Defaults to scanning car_purchase freeform_std batches and all LLM agents.
"""

import json, os, glob, argparse
import numpy as np
import tiktoken


# Pricing (per 1M tokens)
GPT51_INPUT = 1.25
GPT51_OUTPUT = 10.00
GPT41_INPUT = 2.00
GPT41_OUTPUT = 8.00

# From billing data: GPT-5.1 output cost ≈ 0.5× input cost
# So: output_tokens ≈ input_tokens × (GPT51_INPUT / (2 × GPT51_OUTPUT))
GPT51_OUTPUT_RATIO = 0.5  # output_cost / input_cost

# Prediction call constants
PRED_TEMPLATE_TOKENS = 50   # fixed prompt template
PRED_INPUT_TOKENS = 80      # input fields per call
PRED_OUTPUT_TOKENS = 1      # "Yes" or "No"

# Prompt scaling model: 15% fixed overhead, 85% scales with budget
OVERHEAD_FRAC = 0.15

NON_LLM_AGENTS = {'majority', 'nn', 'nn_spread', 'logreg', 'always_true', 'always_false'}

DEFAULT_BATCH_DIRS = [
    'outputs/evaluations/batch_20260301_033718',
    'outputs/evaluations/batch_20260301_033721',
]


def collect_agent_stats(batch_dirs, agents=None):
    """Collect prompt and pattern token stats per agent from batch dirs."""
    enc = tiktoken.get_encoding('o200k_base')
    agent_stats = {}
    for bd in batch_dirs:
        for md in sorted(glob.glob(os.path.join(bd, '*_d*_*'))):
            agent_dir = os.path.join(md, 'agent_results')
            if not os.path.isdir(agent_dir):
                continue
            for af in os.listdir(agent_dir):
                if not af.endswith('.json'):
                    continue
                agent = af.replace('.json', '')
                if agent in NON_LLM_AGENTS:
                    continue
                if agents and agent not in agents:
                    continue
                try:
                    d = json.load(open(os.path.join(agent_dir, af)))
                    meta = d.get('agent_metadata', {})
                    pp = meta.get('pattern_prompt', '')
                    pat = meta.get('pattern', '')
                    if pp:
                        agent_stats.setdefault(agent, []).append({
                            'prompt_tok': len(enc.encode(pp)),
                            'pattern_tok': len(enc.encode(pat)) if pat else 0,
                        })
                except Exception:
                    pass
    return agent_stats


def estimate_cost(agent_stats, budget=10, n_models=80):
    """Estimate cost for a given budget and number of models."""
    n_pred = 100 - budget
    results = {}

    for agent, stats in agent_stats.items():
        avg_prompt_tok_10 = np.mean([s['prompt_tok'] for s in stats])
        avg_pattern_tok = np.mean([s['pattern_tok'] for s in stats])

        # Scale prompt tokens for different budgets
        overhead = avg_prompt_tok_10 * OVERHEAD_FRAC
        per_sample = (avg_prompt_tok_10 - overhead) / 10
        prompt_tok = overhead + per_sample * budget

        # GPT-5.1 find_pattern
        find_in_cost = prompt_tok * GPT51_INPUT / 1e6
        find_out_cost = GPT51_OUTPUT_RATIO * find_in_cost
        cost_51 = (find_in_cost + find_out_cost) * n_models

        # GPT-4.1 predictions
        pred_in = PRED_TEMPLATE_TOKENS + avg_pattern_tok + PRED_INPUT_TOKENS
        cost_41 = n_pred * (pred_in * GPT41_INPUT + PRED_OUTPUT_TOKENS * GPT41_OUTPUT) / 1e6 * n_models

        results[agent] = {'cost_51': cost_51, 'cost_41': cost_41, 'total': cost_51 + cost_41,
                          'prompt_tok': avg_prompt_tok_10, 'pattern_tok': avg_pattern_tok}

    return results


def main():
    parser = argparse.ArgumentParser(description='Estimate API costs for eval runs')
    parser.add_argument('--batch-dirs', nargs='+', default=DEFAULT_BATCH_DIRS)
    parser.add_argument('--agents', nargs='+', default=None)
    parser.add_argument('--budgets', nargs='+', type=int, default=[2, 5, 10, 20])
    parser.add_argument('--n-models', type=int, default=80)
    args = parser.parse_args()

    agent_stats = collect_agent_stats(args.batch_dirs, args.agents)

    if not agent_stats:
        print("No agent data found.")
        return

    # Per-agent breakdown at default budget (10)
    print(f'=== Per-agent cost at b=10, {args.n_models} models ===')
    print(f'{"Agent":>25s}  {"prompt_tok":>10s}  {"5.1_$":>7s}  {"4.1_$":>7s}  {"total_$":>8s}')
    print('-' * 60)

    results_10 = estimate_cost(agent_stats, budget=10, n_models=args.n_models)
    grand = 0
    for agent in sorted(results_10, key=lambda a: -results_10[a]['total']):
        r = results_10[agent]
        print(f'{agent:>25s}  {r["prompt_tok"]:>10.0f}  {r["cost_51"]:>7.2f}  {r["cost_41"]:>7.2f}  {r["total"]:>8.2f}')
        grand += r['total']
    print(f'{"TOTAL":>25s}  {"":>10s}  {"":>7s}  {"":>7s}  {grand:>8.2f}')

    # Budget sweep
    print(f'\n=== Budget sweep, {args.n_models} models ===')
    print(f'{"Budget":>7s}  {"5.1_$":>8s}  {"4.1_$":>8s}  {"total_$":>8s}')
    print('-' * 36)

    sweep_total = 0
    for budget in args.budgets:
        results = estimate_cost(agent_stats, budget=budget, n_models=args.n_models)
        t51 = sum(r['cost_51'] for r in results.values())
        t41 = sum(r['cost_41'] for r in results.values())
        total = t51 + t41
        sweep_total += total
        print(f'{budget:>7d}  {t51:>8.2f}  {t41:>8.2f}  {total:>8.2f}')
    print(f'{"TOTAL":>7s}  {"":>8s}  {"":>8s}  {sweep_total:>8.2f}')


if __name__ == '__main__':
    main()
