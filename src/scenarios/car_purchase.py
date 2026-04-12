"""Car purchase scenario."""

import random
from typing import Any

from .base import BaseScenario, Field, FieldType


class CarPurchaseScenario(BaseScenario):
    """Car purchase decision scenario.

    The model decides whether to recommend purchasing a car based on 10 attributes
    across 4 categories: Identity, Technical Specs, Interior & Condition, Economics.
    """

    freeform_templates_filename = "car_purchase_templates.json"
    freeform_response_prefixes_filename = "car_purchase_response_prefixes.json"

    @property
    def name(self) -> str:
        return "car_purchase"

    @property
    def fields(self) -> list[Field]:
        return [
            # --- 1. Identity ---
            # Binary ENUMs for simpler predicates (field == value gives 50% balance)
            Field(
                name="brand",
                field_type=FieldType.ENUM,
                values=["Toyota", "BMW"]  # mainstream vs luxury
            ),
            Field(
                name="year",
                field_type=FieldType.INTEGER,
                range=(2000, 2025)
            ),
            Field(
                name="color",
                field_type=FieldType.ENUM,
                values=["Black", "White"]  # most common colors
            ),
            # --- 2. Technical Specs ---
            Field(
                name="horsepower",
                field_type=FieldType.INTEGER,
                range=(100, 600)
            ),
            Field(
                name="drivetrain",
                field_type=FieldType.ENUM,
                values=["FWD", "AWD"]  # front-wheel vs all-wheel
            ),
            Field(
                name="mpg",
                field_type=FieldType.INTEGER,
                range=(10, 60)
            ),
            Field(
                name="seat_capacity",
                field_type=FieldType.INTEGER,
                range=(2, 9)
            ),
            # --- 3. Interior & Condition ---
            Field(
                name="interior",
                field_type=FieldType.ENUM,
                values=["Leather", "Cloth"]  # premium vs standard
            ),
            Field(
                name="condition",
                field_type=FieldType.ENUM,
                values=["New", "Used"]  # binary condition
            ),
            # --- 4. Economics ---
            Field(
                name="price",
                field_type=FieldType.INTEGER,
                range=(5000, 100000)
            ),
            # # --- 5. Additional fields (commented out - not needed for decision trees) ---
            # Field(
            #     name="mileage",
            #     field_type=FieldType.INTEGER,
            #     range=(0, 200000)
            # ),
            # Field(
            #     name="transmission",
            #     field_type=FieldType.ENUM,
            #     values=["Automatic", "Manual", "CVT", "DualClutch"]
            # ),
            # Field(
            #     name="fuel_type",
            #     field_type=FieldType.ENUM,
            #     values=["Gasoline", "Diesel", "Electric", "Hybrid"]
            # ),
            # Field(
            #     name="body_type",
            #     field_type=FieldType.ENUM,
            #     values=["Sedan", "SUV", "Truck", "Coupe"]
            # ),
            # Field(
            #     name="safety_rating",
            #     field_type=FieldType.INTEGER,
            #     range=(1, 5)
            # ),
        ]

    @property
    def decision_prompt(self) -> str:
        return "Purchase Recommendation (yes/no):"
    
    def to_structured(self, inputs: dict[str, Any], annotated: bool = False) -> str | dict[str, str]:
        """Format as structured key-value pairs."""
        if not annotated:
            result_annotated = self.to_structured(inputs, annotated=True)
            return "\n".join(result_annotated.values())
        else:
            return self._to_structured_annotated(inputs)
    
    def _to_structured_annotated(self, inputs: dict[str, Any]) -> dict[str, str]:
        lines = {
            "_header": "Car Information",
            "brand": f"Brand: {inputs.get('brand')}",
            "year": f"Year: {inputs.get('year')}",
            "color": f"Color: {inputs.get('color')}",
            "horsepower": f"Horsepower: {inputs.get('horsepower')}",
            "drivetrain": f"Drivetrain: {inputs.get('drivetrain')}",
            "mpg": f"MPG: {inputs.get('mpg')}",
            "seat_capacity": f"Seat Capacity: {inputs.get('seat_capacity')}",
            "interior": f"Interior: {inputs.get('interior')}",
            "condition": f"Condition: {inputs.get('condition')}",
            "price": f"Price: {inputs.get('price')}",
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

        # Check what groups are present
        has_identity = any(k in inputs for k in ['brand', 'year', 'color'])
        has_specs = any(k in inputs for k in ['horsepower', 'drivetrain', 'mpg', 'seat_capacity'])

        # Need "car" after identity if no brand
        needs_car_after_identity = has_identity and "brand" not in inputs

        # === Identity section: "A 2020 white car" or "A 2020 white BMW" ===
        if "year" in inputs:
            parts["year"] = f"A {inputs['year']}"
        elif has_identity:
            parts["_opening"] = "A"

        if "color" in inputs:
            parts["color"] = inputs['color'].lower()

        if "brand" in inputs:
            parts["brand"] = inputs['brand']
        elif needs_car_after_identity:
            parts["_car"] = "car"

        # === Specs section: "with 300 horsepower, AWD drivetrain, 30 MPG, and 5 seats" ===
        if "horsepower" in inputs:
            prefix = "with " if has_identity else "A car with "
            parts["horsepower"] = f"{prefix}{inputs['horsepower']} horsepower,"
        if "drivetrain" in inputs:
            if "horsepower" in inputs:
                prefix = ""
            elif has_identity:
                prefix = "with "
            else:
                prefix = "A car with "
            parts["drivetrain"] = f"{prefix}{inputs['drivetrain']} drivetrain,"
        if "mpg" in inputs:
            if "horsepower" in inputs or "drivetrain" in inputs:
                prefix = ""
            elif has_identity:
                prefix = "with "
            else:
                prefix = "A car with "
            parts["mpg"] = f"{prefix}{inputs['mpg']} MPG,"
        if "seat_capacity" in inputs:
            prior_specs = "horsepower" in inputs or "drivetrain" in inputs or "mpg" in inputs
            if prior_specs:
                prefix = "and "
            elif has_identity:
                prefix = "with "
            else:
                prefix = "A car with "
            parts["seat_capacity"] = f"{prefix}{inputs['seat_capacity']} seats,"

        # === Details section: "with a leather interior, in new condition, priced at $50,000" ===
        details_before = lambda: has_identity or has_specs or "interior" in inputs or "condition" in inputs
        if "interior" in inputs:
            article = 'an' if inputs['interior'][0].lower() in 'aeiou' else 'a'
            prefix = "with " if (has_identity or has_specs) else "A car with "
            parts["interior"] = f"{prefix}{article} {inputs['interior'].lower()} interior,"
        if "condition" in inputs:
            prefix = "in " if (has_identity or has_specs or "interior" in inputs) else "A car in "
            parts["condition"] = f"{prefix}{inputs['condition'].lower()} condition,"
        if "price" in inputs:
            prefix = "priced at " if details_before() else "A car priced at "
            parts["price"] = f"{prefix}${inputs['price']:,}."

        # Handle empty inputs
        if not parts:
            return {"_opening": "A car."}

        # Ensure ends with period
        last_key = list(parts.keys())[-1]
        last_val = parts[last_key]
        if not last_val.endswith('.'):
            parts[last_key] = last_val.rstrip(',') + '.'

        return parts

if __name__ == "__main__":
    import random

    scenario = CarPurchaseScenario()
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
