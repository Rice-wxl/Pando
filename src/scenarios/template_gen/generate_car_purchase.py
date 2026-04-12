"""Generate freeform car purchase description templates."""

import asyncio
from generate_base import generate_templates

FIELDS = [
    "BRAND", "YEAR", "COLOR", "HORSEPOWER", "DRIVETRAIN",
    "MPG", "SEAT_CAPACITY", "INTERIOR", "CONDITION", "PRICE",
]

PROMPT = """\
Think about how a car with specified fields could be described in a paragraph. The fields are: [BRAND], [YEAR], [COLOR], [HORSEPOWER], [DRIVETRAIN], [MPG], [SEAT_CAPACITY], [INTERIOR], [CONDITION], [PRICE].

For example, this could be a car's info: {"brand": "BMW", "year": 2000, "color": "Silver", "horsepower": 225, "drivetrain": "RWD", "mpg": 18, "seat_capacity": 3, "interior": "Leather", "condition": "Salvage", "price": 9165}

An example way of describing is: This [YEAR] [BRAND] is a [COLOR], [DRIVETRAIN] car that delivers [HORSEPOWER] horsepower and an EPA-estimated [MPG] mpg. It has a [INTERIOR] interior with seating for [SEAT_CAPACITY], and it is listed in [CONDITION] condition. The asking price is $[PRICE].

It can also be more structured: Car Information\\nBrand: [BRAND]\\nYear: [YEAR]\\nColor: [COLOR]\\nHorsepower: [HORSEPOWER]\\nDrivetrain: [DRIVETRAIN]\\nMPG: [MPG]\\nSeat Capacity: [SEAT_CAPACITY]\\nInterior: [INTERIOR]\\nCondition: [CONDITION]\\nPrice: [PRICE]

Give me 100 different ways of describing the car in a paragraph. Make sure to include all the fields in placeholder form, similar to the example."""

OUTPUT_PATH = "src/scenarios/car_purchase_templates.json"


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
