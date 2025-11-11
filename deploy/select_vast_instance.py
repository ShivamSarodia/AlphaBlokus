import requests
import json

exclude_machines = {}


def search():
    url = "https://console.vast.ai/api/v0/search/asks/"

    payload = json.dumps(
        {
            "q": {
                "order": [["dph_total", "asc"]],
                "verified": {"eq": True},
                "rentable": {"eq": True},
                "type": "on-demand",
                "allocated_storage": "64",
                "reliability2": {"gt": 0.995},
                "inet_down": {"gt": 300},
                "inet_up": {"gt": 300},
                "duration": {"gte": 1},
                "geolocation": {"in": ["US", "CA"]},
                "dph_total": {"lte": 0.2},
                "gpu_name": "RTX 3070",
                "cpu_cores": {"gte": 8},
                "cpu_ram": {"gte": 16},
                "cuda_max_good": {"gte": 12.0},
            }
        }
    )
    headers = {"Accept": "application/json", "Content-Type": "application/json"}

    response = requests.request("PUT", url, headers=headers, data=payload)
    response.raise_for_status()
    offers = response.json()["offers"]

    offers = [o for o in offers if o["machine_id"] not in exclude_machines]

    for offer in offers:
        augment(offer)
    offers.sort(key=lambda o: o["computed_cost"])

    return offers


def augment(offer):
    game_data_generated_per_hour_mb = 130
    models_generated_per_hour_mb = 48

    offer["computed_cost"] = (
        offer["dph_total"]
        + game_data_generated_per_hour_mb / 1024 * offer["inet_up_cost"]
        + models_generated_per_hour_mb / 1024 * offer["inet_down_cost"]
    )

    offer["available_cpu_ghz"] = (
        offer["cpu_ghz"] * offer["cpu_cores"] * offer["gpu_frac"]
    )
    offer["available_ram"] = offer["cpu_ram"] / 1024


def print_table(offers):
    """Print offers as a formatted table."""
    if not offers:
        print("No offers found.")
        return

    # Define columns: each column has a name and a function to extract/format the value
    columns = [
        {"name": "ID", "extractor": lambda o: str(o["id"])},
        {"name": "GPU", "extractor": lambda o: o["gpu_name"]},
        {
            "name": "CPU",
            "extractor": lambda o: o["cpu_name"][:18],
        },
        {"name": "Cost ($/hr)", "extractor": lambda o: f"{o['computed_cost']:.3f}"},
        {"name": "CPU GHz", "extractor": lambda o: str(round(o["available_cpu_ghz"]))},
        {"name": "RAM (GB)", "extractor": lambda o: str(round(o["available_ram"]))},
        {"name": "Location", "extractor": lambda o: o["geolocation"]},
    ]

    # Auto-calculate column widths
    widths = []
    for col in columns:
        # Start with header width
        max_width = len(col["name"])
        # Check all data values
        for offer in offers:
            value = col["extractor"](offer)
            max_width = max(max_width, len(str(value)))
        # Add padding
        widths.append(max_width + 2)

    # Print header
    header_row = " | ".join(col["name"].ljust(w) for col, w in zip(columns, widths))
    print(header_row)
    print("-" * len(header_row))

    # Print each offer as a row
    for offer in offers:
        row_data = [col["extractor"](offer) for col in columns]
        row = " | ".join(d.ljust(w) for d, w in zip(row_data, widths))
        print(row)


offers = search()
print_table(offers[:5])
