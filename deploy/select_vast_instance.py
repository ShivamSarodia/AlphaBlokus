import requests
import json
import os
import time

exclude_machines = {}
VASTAI_API_KEY = os.getenv("VASTAI_API_KEY")

# GPUS = {
#     "3090": "RTX 3090",
#     "3070": "RTX 3070",
#     "4070": "RTX 4070",
#     "4090": "RTX 4090",
# }


def search(instance_type, purpose):
    assert instance_type in ["on-demand", "bid"]
    assert purpose in ["training", "self-play"]

    url = "https://console.vast.ai/api/v0/bundles/"
    payload = {
        "order": [["dph_total", "asc"]],
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "type": instance_type,
        "allocated_storage": "64",
        "reliability2": {"gt": 0.995},
        "inet_down": {"gt": 100},
        "inet_up": {"gt": 100},
        "duration": {"gte": 1},
        "cpu_cores": {"gte": 8},
        "cpu_ram": {"gte": 22},
        "cuda_max_good": {"gte": 12.0},
    }

    if purpose == "self-play":
        # payload["gpu_name"] = "RTX 3070"
        payload["total_flops"] = {"gte": 19.0}

    headers = {"Accept": "application/json", "Content-Type": "application/json"}

    response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    offers = response.json()["offers"]

    offers = [o for o in offers if o["machine_id"] not in exclude_machines]
    offers = [
        o for o in offers if "titan" not in o.get("gpu_name", "").lower()
    ]

    for offer in offers:
        augment(offer)
    offers.sort(key=lambda o: o["computed_cost"])

    offers = [o for o in offers if o["available_cpu_ghz"] >= 40]

    print(offers[0])

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
        {"name": "#", "extractor": lambda idx, _: str(idx)},
        {"name": "GPU", "extractor": lambda _, offer: offer["gpu_name"]},
        {
            "name": "CPU",
            "extractor": lambda _, offer: offer.get("cpu_name", "Unknown")[:18],
        },
        {
            "name": "Cost ($/hr)",
            "extractor": lambda _, offer: f"{offer['computed_cost']:.3f}",
        },
        {
            "name": "DPH Total ($/hr)",
            "extractor": lambda _, offer: f"{offer['dph_total']:.3f}",
        },
        {
            "name": "CPU GHz",
            "extractor": lambda _, offer: str(round(offer["available_cpu_ghz"])),
        },
        {
            "name": "RAM (GB)",
            "extractor": lambda _, offer: str(round(offer["available_ram"])),
        },
        {
            "name": "CPU cores",
            "extractor": lambda _, offer: str(offer["cpu_cores"]),
        },
        {
            "name": "GPU Fraction",
            "extractor": lambda _, offer: str(round(offer["gpu_frac"], 2)),
        },
        {
            "name": "GPU flops",
            "extractor": lambda _, offer: str(round(offer["total_flops"])),
        },
        {
            "name": "Location",
            "extractor": lambda _, offer: offer["geolocation"],
        },
    ]

    # Auto-calculate column widths
    widths = []
    for col in columns:
        # Start with header width
        max_width = len(col["name"])
        # Check all data values
        for idx, offer in enumerate(offers):
            value = col["extractor"](idx, offer)
            max_width = max(max_width, len(str(value)))
        # Add padding
        widths.append(max_width + 1)

    # Print header
    header_row = " | ".join(col["name"].ljust(w) for col, w in zip(columns, widths))
    print(header_row)
    print("-" * len(header_row))

    # Print each offer as a row
    for idx, offer in enumerate(offers):
        row_data = [col["extractor"](idx, offer) for col in columns]
        row = " | ".join(d.ljust(w) for d, w in zip(row_data, widths))
        print(row)


def select_instance(offers):
    """Prompt user to select an instance and return the selected offer."""
    if not offers:
        return None

    print_table(offers)

    while True:
        try:
            choice = input(f"\nSelect an instance (0-{len(offers) - 1}): ").strip()
            index = int(choice)
            if 0 <= index < len(offers):
                return offers[index]
            else:
                print(f"Please enter a number between 0 and {len(offers) - 1}")
        except ValueError:
            print("Please enter a valid number")
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return None


def create_from_template(offer_id, min_bid=None):
    url = f"https://console.vast.ai/api/v0/asks/{offer_id}/"
    payload = {
        # "template_id": 302220,
        "template_hash_id": "9fb234bb5f8d095c3f8c23f026894b6a",
        "image": "vastai/base-image:@vastai-automatic-tag",
        "env": {
            "-p 1111:1111": "1",
            "-p 6006:6006": "1",
            "-p 8080:8080": "1",
            "-p 8384:8384": "1",
            "-p 72299:72299": "1",
            "-p 12345:12345": "1",
            "OPEN_BUTTON_PORT": "1111",
            "OPEN_BUTTON_TOKEN": "1",
            "JUPYTER_DIR": "/",
            "DATA_DIRECTORY": "/workspace/",
            "PORTAL_CONFIG": "localhost:1111:11111:/:Instance Portal|localhost:8080:18080:/:Jupyter|localhost:8080:8080:/terminals/1:Jupyter Terminal|localhost:8384:18384:/:Syncthing|localhost:6006:16006:/:Tensorboard",
        },
        "args_str": "",
        "onstart": "initial_dir=$(pwd); mkdir -p /workspace; cd /workspace; git clone https://github.com/ShivamSarodia/AlphaBlokus; cd $initial_dir; entrypoint.sh",
        "runtype": "jupyter_direc ssh_direc ssh_proxy",
        "use_jupyter_lab": False,
        "disk": 64,
    }
    if min_bid is not None:
        payload["price"] = min_bid * 1.1
        payload["last_known_min_bid"] = min_bid

    headers = {
        "Authorization": f"Bearer {VASTAI_API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    response = requests.request("PUT", url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    return response.json()


def get_instance(instance_id):
    url = f"https://console.vast.ai/api/v0/instances/{instance_id}/"

    payload = {}
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {VASTAI_API_KEY}",
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    response.raise_for_status()
    return response.json()["instances"]


instance_type = input("Select instance type (on-demand/bid): ").strip()
assert instance_type in ["on-demand", "bid"]

purpose = input("Select purpose (training/self-play): ").strip()
assert purpose in ["training", "self-play"]

# if purpose == "self-play":
#     gpu_key = input(f"Select GPU ({'/'.join(GPUS.keys())}): ").strip()
#     assert gpu_key in GPUS
# else:
#     gpu_key = None

offers = search(instance_type, purpose)
selected = select_instance(offers[:8])
if not selected:
    exit(1)

print(f"\nSelected offer ID: {selected['id']}")
if instance_type == "bid":
    min_bid = selected["min_bid"]
else:
    min_bid = None

created = create_from_template(selected["id"], min_bid)

instance_id = created["new_contract"]
print(f"\nCreated instance: {instance_id}. Waiting for ports...")

while True:
    instance = get_instance(instance_id)
    if not instance.get("ports", {}).get("22/tcp"):
        time.sleep(10)
        continue

    ip_address = instance["public_ipaddr"]
    port = instance["ports"]["22/tcp"][0]["HostPort"]
    break

print(f"SSH with: ssh -i ~/.ssh/id_ed25519_personal -p {port} root@{ip_address}")
