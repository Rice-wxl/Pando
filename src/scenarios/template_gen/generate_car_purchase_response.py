"""Generate freeform response prefixes for car purchase recommendations."""

import asyncio
from generate_base import generate_templates

FIELDS = []  # No field placeholders required

PROMPT = """\
Say you're given info about a car and you want to ask for a purchase recommendation (which will be yes or no). How will you start the answer? Give me 100 options.

I'll start:
- Purchase recommendation (yes/no):
- Decision:
- Verdict:
- Should I purchase?
- Buy or pass:
- My recommendation:

Give me 100 different ways. Each should be a short phrase or sentence that precedes a yes/no answer. One per line, numbered."""

OUTPUT_PATH = "src/scenarios/car_purchase_response_prefixes.json"


def validate_response_prefix(template: str) -> bool:
    """Any non-empty string is valid (no field placeholders required)."""
    return len(template.strip()) > 0


def main():
    asyncio.run(generate_templates(
        prompt=PROMPT,
        fields=FIELDS,
        output_path=OUTPUT_PATH,
        total_batches=10,
        concurrency=10,
    ))


if __name__ == "__main__":
    main()
