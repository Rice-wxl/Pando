"""Movie pick scenario."""

from typing import Any

from .base import BaseScenario, Field, FieldType


class MoviePickScenario(BaseScenario):
    """Movie pick decision scenario.

    The model decides whether to recommend watching a movie based on 10 attributes
    across 3 categories: Identity, Classification, Production Stats.
    """

    freeform_templates_filename = "movie_pick_templates.json"
    freeform_response_prefixes_filename = "movie_pick_response_prefixes.json"

    @property
    def name(self) -> str:
        return "movie_pick"

    @property
    def fields(self) -> list[Field]:
        return [
            # --- 1. Identity ---
            Field(
                name="release_year",
                field_type=FieldType.INTEGER,
                range=(1970, 2025)
            ),
            Field(
                name="genre",
                field_type=FieldType.ENUM,
                values=["Action", "Drama"]
            ),
            Field(
                name="language",
                field_type=FieldType.ENUM,
                values=["English", "Foreign"]
            ),
            Field(
                name="runtime",
                field_type=FieldType.INTEGER,
                range=(70, 210)  # minutes
            ),
            # --- 2. Classification ---
            Field(
                name="rating",
                field_type=FieldType.ENUM,
                values=["PG", "R"]  # family-friendly vs mature
            ),
            Field(
                name="release_format",
                field_type=FieldType.ENUM,
                values=["Theatrical", "Streaming"]
            ),
            # --- 3. Production & Reception ---
            Field(
                name="budget_millions",
                field_type=FieldType.INTEGER,
                range=(1, 300)
            ),
            Field(
                name="box_office_millions",
                field_type=FieldType.INTEGER,
                range=(1, 1000)
            ),
            Field(
                name="color_format",
                field_type=FieldType.ENUM,
                values=["Color", "Black-and-White"]
            ),
            Field(
                name="cast_size",
                field_type=FieldType.INTEGER,
                range=(2, 30)
            ),
        ]

    @property
    def decision_prompt(self) -> str:
        return "Watch Recommendation (yes/no):"

    def to_structured(self, inputs: dict[str, Any], annotated: bool = False) -> str | dict[str, str]:
        """Format as structured key-value pairs."""
        if not annotated:
            result_annotated = self.to_structured(inputs, annotated=True)
            return "\n".join(result_annotated.values())
        else:
            return self._to_structured_annotated(inputs)

    def _to_structured_annotated(self, inputs: dict[str, Any]) -> dict[str, str]:
        lines = {
            "_header": "Movie Information",
            "release_year": f"Release Year: {inputs.get('release_year')}",
            "genre": f"Genre: {inputs.get('genre')}",
            "language": f"Language: {inputs.get('language')}",
            "runtime": f"Runtime: {inputs.get('runtime')}",
            "rating": f"Rating: {inputs.get('rating')}",
            "release_format": f"Release Format: {inputs.get('release_format')}",
            "budget_millions": f"Budget (millions): {inputs.get('budget_millions')}",
            "box_office_millions": f"Box Office (millions): {inputs.get('box_office_millions')}",
            "color_format": f"Color Format: {inputs.get('color_format')}",
            "cast_size": f"Cast Size: {inputs.get('cast_size')}",
        }
        lines = {k: v for k, v in lines.items() if k in inputs or k[0] == '_'}
        return lines

    def to_natural(self, inputs: dict[str, Any], annotated: bool = False) -> str | dict[str, str]:
        """Format as natural language description."""
        if annotated:
            return self._to_natural_annotated(inputs)
        else:
            result_annotated = self._to_natural_annotated(inputs)
            return " ".join(result_annotated.values())

    def _to_natural_annotated(self, inputs: dict[str, Any]) -> dict[str, str]:
        """Format as natural language with field annotations."""
        parts = {}

        has_identity = any(k in inputs for k in ['genre', 'language', 'release_year', 'color_format'])

        # === Identity section: "A 2020 English black-and-white action film," ===
        # Track whether we've placed the article yet
        need_article = True

        if "release_year" in inputs:
            parts["release_year"] = f"A {inputs['release_year']}"
            need_article = False

        if "language" in inputs:
            word = "English" if inputs["language"] == "English" else "foreign"
            if need_article:
                article = "An" if word[0].lower() in "aeiou" else "A"
                parts["language"] = f"{article} {word}"
                need_article = False
            else:
                parts["language"] = word

        if "color_format" in inputs:
            word = inputs["color_format"].lower()  # "color" or "black-and-white"
            if need_article:
                article = "An" if word[0].lower() in "aeiou" else "A"
                parts["color_format"] = f"{article} {word}"
                need_article = False
            else:
                parts["color_format"] = word

        if "genre" in inputs:
            word = inputs["genre"].lower()
            if need_article:
                article = "An" if word[0].lower() in "aeiou" else "A"
                parts["genre"] = f"{article} {word} film,"
                need_article = False
            else:
                parts["genre"] = f"{word} film,"
        elif has_identity:
            parts["_noun"] = "film,"

        # === Classification: "rated R, a theatrical release," ===
        if "rating" in inputs:
            if has_identity:
                parts["rating"] = f"rated {inputs['rating']},"
            else:
                parts["rating"] = f"A film rated {inputs['rating']},"

        if "release_format" in inputs:
            fmt = inputs["release_format"].lower()  # "theatrical" or "streaming"
            prior = has_identity or "rating" in inputs
            if prior:
                parts["release_format"] = f"a {fmt} release,"
            else:
                parts["release_format"] = f"A {fmt} release,"

        # === Stats: "120 minutes, $50M budget, $200M box office, cast of 15." ===
        has_prior = has_identity or "rating" in inputs or "release_format" in inputs

        if "runtime" in inputs:
            if has_prior:
                parts["runtime"] = f"{inputs['runtime']} minutes,"
            else:
                parts["runtime"] = f"A {inputs['runtime']}-minute film,"

        if "budget_millions" in inputs:
            prior = has_prior or "runtime" in inputs
            prefix = "" if prior else "A film with a "
            parts["budget_millions"] = f"{prefix}${inputs['budget_millions']}M budget,"

        if "box_office_millions" in inputs:
            prior = has_prior or any(k in inputs for k in ["runtime", "budget_millions"])
            prefix = "" if prior else "A film with "
            parts["box_office_millions"] = f"{prefix}${inputs['box_office_millions']}M box office,"

        if "cast_size" in inputs:
            prior = has_prior or any(k in inputs for k in ["runtime", "budget_millions", "box_office_millions"])
            if prior:
                parts["cast_size"] = f"and a cast of {inputs['cast_size']}."
            else:
                parts["cast_size"] = f"A film with a cast of {inputs['cast_size']}."

        # Handle empty inputs
        if not parts:
            return {"_opening": "A film."}

        # Ensure ends with period
        last_key = list(parts.keys())[-1]
        last_val = parts[last_key]
        if not last_val.endswith('.'):
            parts[last_key] = last_val.rstrip(',') + '.'

        return parts


if __name__ == "__main__":
    import random

    scenario = MoviePickScenario()
    field_order = scenario.field_names()

    random.seed(42)

    print(f'Natural language formatting tests:\n')

    # Test 0 fields
    print("=== 0 field(s) ===")
    print(f"   {repr(scenario.to_natural({}))}")
    print()

    # Test 1 field - all single fields
    print("=== 1 field(s) ===")
    for field in field_order:
        value = scenario.get_field(field).sample()
        result = scenario.to_natural({field: value})
        print(f"{field}: {repr(result)}")
    print()

    # Test 2-10 fields
    for num_fields in range(2, 11):
        print(f"=== {num_fields} field(s) ===")
        for i in range(5):
            chosen = random.sample(field_order, num_fields)
            chosen_sorted = [f for f in field_order if f in chosen]
            inputs = {f: scenario.get_field(f).sample() for f in chosen_sorted}
            result = scenario.to_natural(inputs)
            print(f"{i+1}. {chosen_sorted}")
            print(f"   {result}")
        print()
