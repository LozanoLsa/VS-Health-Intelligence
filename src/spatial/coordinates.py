import json
from pathlib import Path
from src.utils.config import CONFIG


def load_zones() -> dict:
    path = CONFIG["data"]["zones_json"]
    with open(path) as f:
        return json.load(f)


def get_machine_coords(machine_id: str) -> dict | None:
    zones = load_zones()
    return zones["machines"].get(machine_id)


def canvas_dimensions() -> tuple[float, float]:
    zones = load_zones()
    c = zones["canvas"]
    return c["width"], c["height"]
