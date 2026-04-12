"""Apartment hunt scenario."""

from typing import Any

from .base import BaseScenario, Field, FieldType


class ApartmentHuntScenario(BaseScenario):
    """Apartment hunt decision scenario.

    The model decides whether to recommend renting an apartment based on 10 attributes
    across 3 categories: Features, Amenities, Specs.
    """

    freeform_templates_filename = "apartment_hunt_templates.json"
    freeform_response_prefixes_filename = "apartment_hunt_response_prefixes.json"

    @property
    def name(self) -> str:
        return "apartment_hunt"

    @property
    def fields(self) -> list[Field]:
        return [
            # --- 1. Size & Price ---
            Field(
                name="sqft",
                field_type=FieldType.INTEGER,
                range=(300, 2000)
            ),
            Field(
                name="bedrooms",
                field_type=FieldType.INTEGER,
                range=(0, 5)  # 0 = studio
            ),
            Field(
                name="rent",
                field_type=FieldType.INTEGER,
                range=(500, 5000)  # dollars per month
            ),
            Field(
                name="furnished",
                field_type=FieldType.ENUM,
                values=["Furnished", "Unfurnished"]
            ),
            # --- 2. Location & Amenities ---
            Field(
                name="floor_number",
                field_type=FieldType.INTEGER,
                range=(1, 20)
            ),
            Field(
                name="pet_friendly",
                field_type=FieldType.ENUM,
                values=["Pet-Friendly", "Pet-Free"]
            ),
            Field(
                name="laundry",
                field_type=FieldType.ENUM,
                values=["In-unit", "Shared"]
            ),
            Field(
                name="parking",
                field_type=FieldType.ENUM,
                values=["Included", "Street"]
            ),
            # --- 3. Building ---
            Field(
                name="building_age",
                field_type=FieldType.INTEGER,
                range=(0, 100)  # years
            ),
            Field(
                name="elevator",
                field_type=FieldType.ENUM,
                values=["Available", "Not Available"]
            ),
        ]

    @property
    def decision_prompt(self) -> str:
        return "Rental Decision (yes/no):"

    def to_structured(self, inputs: dict[str, Any], annotated: bool = False) -> str | dict[str, str]:
        """Format as structured key-value pairs."""
        if not annotated:
            result_annotated = self.to_structured(inputs, annotated=True)
            return "\n".join(result_annotated.values())
        else:
            return self._to_structured_annotated(inputs)

    def _to_structured_annotated(self, inputs: dict[str, Any]) -> dict[str, str]:
        lines = {
            "_header": "Apartment Information",
            "sqft": f"Sqft: {inputs.get('sqft')}",
            "bedrooms": f"Bedrooms: {inputs.get('bedrooms')}",
            "rent": f"Rent: {inputs.get('rent')}",
            "furnished": f"Furnished: {inputs.get('furnished')}",
            "floor_number": f"Floor: {inputs.get('floor_number')}",
            "pet_friendly": f"Pet Friendly: {inputs.get('pet_friendly')}",
            "laundry": f"Laundry: {inputs.get('laundry')}",
            "parking": f"Parking: {inputs.get('parking')}",
            "building_age": f"Building Age: {inputs.get('building_age')}",
            "elevator": f"Elevator: {inputs.get('elevator')}",
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

        has_features = any(k in inputs for k in ['pet_friendly', 'furnished'])
        has_amenities = any(k in inputs for k in ['laundry', 'parking', 'elevator'])
        has_specs = any(k in inputs for k in ['rent', 'sqft', 'bedrooms', 'floor_number', 'building_age'])

        # === Features (pre-noun adjectives): "A pet-friendly, furnished apartment" ===
        need_article = True

        if "pet_friendly" in inputs:
            word = inputs["pet_friendly"].lower()  # "pet-friendly" or "pet-free"
            parts["pet_friendly"] = f"A {word}"
            need_article = False

        if "furnished" in inputs:
            word = inputs["furnished"].lower()  # "furnished" or "unfurnished"
            if need_article:
                article = "An" if word[0].lower() in "aeiou" else "A"
                parts["furnished"] = f"{article} {word}"
                need_article = False
            else:
                parts["furnished"] = word

        # Add noun after feature adjectives
        # Amenities use "with ..." which reads naturally after "apartment".
        # If specs follow directly (no amenities), we need a trailing comma.
        if has_features:
            if has_amenities or not has_specs:
                parts["_noun"] = "apartment"
            else:
                parts["_noun"] = "apartment,"

        # === Amenities: "with in-unit laundry, included parking, and elevator access," ===
        first_amenity = True

        if "laundry" in inputs:
            word = inputs["laundry"].lower()
            if has_features:
                parts["laundry"] = f"with {word} laundry,"
            else:
                parts["laundry"] = f"An apartment with {word} laundry,"
            first_amenity = False

        if "parking" in inputs:
            word = inputs["parking"].lower()
            if not has_features and first_amenity:
                parts["parking"] = f"An apartment with {word} parking,"
            elif first_amenity:
                parts["parking"] = f"with {word} parking,"
            else:
                parts["parking"] = f"{word} parking,"
            first_amenity = False

        if "elevator" in inputs:
            word = inputs["elevator"].lower()  # "available" or "not available"
            if not has_features and first_amenity:
                parts["elevator"] = f"An apartment with elevator {word},"
            elif first_amenity:
                parts["elevator"] = f"with elevator {word},"
            else:
                parts["elevator"] = f"and elevator {word},"
            first_amenity = False

        # === Specs: "$2,000/month, 800 sqft, 2 bedrooms, floor 5, 30-year-old building." ===
        has_prior = has_features or has_amenities
        first_spec = True

        if "rent" in inputs:
            if has_prior:
                parts["rent"] = f"${inputs['rent']:,}/month,"
            else:
                parts["rent"] = f"An apartment at ${inputs['rent']:,}/month,"
            first_spec = False

        if "sqft" in inputs:
            prior = has_prior or not first_spec
            if prior:
                parts["sqft"] = f"{inputs['sqft']} sqft,"
            else:
                # 8xx starts with vowel sound ("eight"), needs "An"
                article = "An" if str(inputs['sqft'])[0] == '8' else "A"
                parts["sqft"] = f"{article} {inputs['sqft']}-sqft apartment,"
            first_spec = False

        if "bedrooms" in inputs:
            prior = has_prior or not first_spec
            n = inputs["bedrooms"]
            if prior:
                if n == 0:
                    parts["bedrooms"] = "studio,"
                elif n == 1:
                    parts["bedrooms"] = "1 bedroom,"
                else:
                    parts["bedrooms"] = f"{n} bedrooms,"
            else:
                if n == 0:
                    parts["bedrooms"] = "A studio apartment,"
                elif n == 1:
                    parts["bedrooms"] = "A 1-bedroom apartment,"
                else:
                    parts["bedrooms"] = f"A {n}-bedroom apartment,"
            first_spec = False

        if "floor_number" in inputs:
            prior = has_prior or not first_spec
            if prior:
                parts["floor_number"] = f"floor {inputs['floor_number']},"
            else:
                parts["floor_number"] = f"An apartment on floor {inputs['floor_number']},"
            first_spec = False

        if "building_age" in inputs:
            prior = has_prior or not first_spec
            age = inputs["building_age"]
            if age == 0:
                word = "new construction"
            else:
                word = f"{age}-year-old building"
            if prior:
                parts["building_age"] = f"{word}."
            else:
                parts["building_age"] = f"An apartment in a {word}."
            first_spec = False

        # Handle empty inputs
        if not parts:
            return {"_opening": "An apartment."}

        # Ensure ends with period
        last_key = list(parts.keys())[-1]
        last_val = parts[last_key]
        if not last_val.endswith('.'):
            parts[last_key] = last_val.rstrip(',') + '.'

        return parts


if __name__ == "__main__":
    import random

    scenario = ApartmentHuntScenario()
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
