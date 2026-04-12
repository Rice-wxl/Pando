"""Generate freeform response prefixes for movie watch recommendations."""

import asyncio

from generate_base import generate_templates

FIELDS = []

PROMPT = """\
Say you're given info about a movie and you want to ask for a watch recommendation (which will be yes or no). How will you start the answer? Give me 100 options.

I'll start:
- Watch recommendation (yes/no):
- Recommendation:
- Verdict:
- Should I watch it?
- Watch or skip:
- My take:

Give me 100 different ways. Each should be a short phrase or sentence that precedes a yes/no answer. One per line, numbered."""

OUTPUT_PATH = "src/scenarios/movie_pick_response_prefixes.json"


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
