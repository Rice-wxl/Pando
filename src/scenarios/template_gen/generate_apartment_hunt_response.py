"""Generate freeform response prefixes for apartment rental decisions."""

import asyncio

from generate_base import generate_templates

FIELDS = []

PROMPT = """\
Say you're given info about an apartment and you want to ask for a rental decision (which will be yes or no). How will you start the answer? Give me 100 options.

I'll start:
- Rental decision (yes/no):
- Recommendation:
- Verdict:
- Should I rent it?
- Rent or pass:
- My recommendation:

Give me 100 different ways. Each should be a short phrase or sentence that precedes a yes/no answer. One per line, numbered."""

OUTPUT_PATH = "src/scenarios/apartment_hunt_response_prefixes.json"


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
