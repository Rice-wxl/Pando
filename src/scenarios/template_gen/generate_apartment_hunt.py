"""Generate freeform apartment hunt description templates."""

import asyncio

from generate_base import generate_templates

FIELDS = [
    "SQFT",
    "BEDROOMS",
    "RENT",
    "FURNISHED",
    "FLOOR_NUMBER",
    "PET_FRIENDLY",
    "LAUNDRY",
    "PARKING",
    "BUILDING_AGE",
    "ELEVATOR",
]

PROMPT = """\
Think about how an apartment with specified fields could be described in a paragraph. The fields are: [SQFT], [BEDROOMS], [RENT], [FURNISHED], [FLOOR_NUMBER], [PET_FRIENDLY], [LAUNDRY], [PARKING], [BUILDING_AGE], [ELEVATOR].

For example, this could be an apartment's info: {"sqft": 875, "bedrooms": 2, "rent": 2450, "furnished": "Unfurnished", "floor_number": 6, "pet_friendly": "Pet-Friendly", "laundry": "In-unit", "parking": "Included", "building_age": 18, "elevator": "Not Available"}

The possible values for each field are:
- SQFT: integer from 300 to 2000
- BEDROOMS: integer from 0 to 5 (0 = studio)
- RENT: integer from 500 to 5000 (dollars per month)
- FURNISHED: "Furnished" or "Unfurnished"
- FLOOR_NUMBER: integer from 1 to 20
- PET_FRIENDLY: "Pet-Friendly" or "Pet-Free"
- LAUNDRY: "In-unit" or "Shared"
- PARKING: "Included" or "Street"
- BUILDING_AGE: integer from 0 to 100 (years)
- ELEVATOR: "Available" or "Not Available"

An example way of describing it is: This [FURNISHED] apartment offers [SQFT] square feet, [BEDROOMS] bedrooms, and rent of $[RENT] per month. It is [PET_FRIENDLY], located on floor [FLOOR_NUMBER], with [LAUNDRY] laundry and [PARKING] parking, sits in a [BUILDING_AGE]-year-old building, and elevator [ELEVATOR].

It can also be more structured: Apartment Information\\nSqft: [SQFT]\\nBedrooms: [BEDROOMS]\\nRent: [RENT]\\nFurnished: [FURNISHED]\\nFloor: [FLOOR_NUMBER]\\nPet Friendly: [PET_FRIENDLY]\\nLaundry: [LAUNDRY]\\nParking: [PARKING]\\nBuilding Age: [BUILDING_AGE]\\nElevator: [ELEVATOR]

Give me 100 different ways of describing the apartment in a paragraph. Make sure to include all the fields in placeholder form, similar to the example."""

OUTPUT_PATH = "src/scenarios/apartment_hunt_templates.json"


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
