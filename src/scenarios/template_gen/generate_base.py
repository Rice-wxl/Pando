"""Base utilities for generating freeform description templates via LLM."""

import asyncio
import json
import os
import re

from dotenv import load_dotenv
load_dotenv()

import litellm

# Project root (three levels up from this file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def validate_template(template: str, fields: list[str]) -> bool:
    """Check that all required field placeholders are present."""
    for field in fields:
        if f"[{field}]" not in template:
            return False
    return True


def parse_templates(text: str) -> list[str]:
    """Parse numbered list of templates from LLM response."""
    templates = []
    lines = text.split("\n")
    current = []
    for line in lines:
        if re.match(r"^\s*\d+[\.\)]\s", line):
            if current:
                templates.append("\n".join(current).strip())
            cleaned = re.sub(r"^\s*\d+[\.\)]\s*", "", line)
            current = [cleaned]
        elif current:
            current.append(line)
    if current:
        templates.append("\n".join(current).strip())
    return templates


async def _fetch_one(prompt: str, sem: asyncio.Semaphore, batch_idx: int) -> str:
    """Single LLM call with concurrency limit."""
    async with sem:
        print(f"  Batch {batch_idx + 1} started...", flush=True)
        response = await litellm.acompletion(
            model="openai/gpt-5.2",
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            max_tokens=16000,
        )
        text = response.choices[0].message.content
        print(f"  Batch {batch_idx + 1} done ({len(text)} chars)", flush=True)
        return text


async def generate_templates(
    prompt: str,
    fields: list[str],
    output_path: str,
    total_batches: int = 50,
    concurrency: int = 10,
):
    """Generate templates with concurrent LLM calls, validate, deduplicate, and save."""
    # Resolve output path relative to project root
    if not os.path.isabs(output_path):
        output_path = os.path.join(PROJECT_ROOT, output_path)

    print(f"Launching {total_batches} batches (concurrency={concurrency})...", flush=True)
    sem = asyncio.Semaphore(concurrency)
    tasks = [_fetch_one(prompt, sem, i) for i in range(total_batches)]
    results = await asyncio.gather(*tasks)

    all_templates = []
    for i, text in enumerate(results):
        parsed = parse_templates(text)
        valid = [t for t in parsed if validate_template(t, fields)]
        all_templates.extend(valid)
        print(f"  Batch {i + 1}: {len(parsed)} parsed, {len(valid)} valid", flush=True)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in all_templates:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    print(f"\nTotal: {len(all_templates)} valid, {len(unique)} unique (removed {len(all_templates) - len(unique)} duplicates)", flush=True)

    with open(output_path, "w") as f:
        json.dump(unique, f, indent=2)
    print(f"Saved {len(unique)} templates to {output_path}", flush=True)
