"""Generate freeform movie pick description templates."""

import asyncio

from generate_base import generate_templates

FIELDS = [
    "RELEASE_YEAR",
    "GENRE",
    "LANGUAGE",
    "RUNTIME",
    "RATING",
    "RELEASE_FORMAT",
    "BUDGET_MILLIONS",
    "BOX_OFFICE_MILLIONS",
    "COLOR_FORMAT",
    "CAST_SIZE",
]

PROMPT = """\
Think about how a movie with specified fields could be described in a paragraph. The fields are: [RELEASE_YEAR], [GENRE], [LANGUAGE], [RUNTIME], [RATING], [RELEASE_FORMAT], [BUDGET_MILLIONS], [BOX_OFFICE_MILLIONS], [COLOR_FORMAT], [CAST_SIZE].

For example, this could be a movie's info: {"release_year": 2019, "genre": "Action", "language": "English", "runtime": 127, "rating": "PG", "release_format": "Theatrical", "budget_millions": 85, "box_office_millions": 312, "color_format": "Color", "cast_size": 14}

An example way of describing it is: This [RELEASE_YEAR] [LANGUAGE] [COLOR_FORMAT] [GENRE] film runs [RUNTIME] minutes, is rated [RATING], released [RELEASE_FORMAT], had a $[BUDGET_MILLIONS]M budget, earned $[BOX_OFFICE_MILLIONS]M at the box office, and features a cast of [CAST_SIZE].

It can also be more structured: Movie Information\\nRelease Year: [RELEASE_YEAR]\\nGenre: [GENRE]\\nLanguage: [LANGUAGE]\\nRuntime: [RUNTIME]\\nRating: [RATING]\\nRelease Format: [RELEASE_FORMAT]\\nBudget (millions): [BUDGET_MILLIONS]\\nBox Office (millions): [BOX_OFFICE_MILLIONS]\\nColor Format: [COLOR_FORMAT]\\nCast Size: [CAST_SIZE]

Give me 100 different ways of describing the movie in a paragraph. Make sure to include all the fields in placeholder form, similar to the example."""

OUTPUT_PATH = "src/scenarios/movie_pick_templates.json"


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
