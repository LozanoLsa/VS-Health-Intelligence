"""Tests for spatial mapping consistency."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import json
import pytest
import pandas as pd
from src.utils.config import CONFIG
from src.spatial.coordinates import load_zones


@pytest.fixture(scope="module")
def zones():
    return load_zones()


@pytest.fixture(scope="module")
def metrics():
    return pd.read_csv(CONFIG["data"]["mtbf_metrics"])


def test_zones_file_exists():
    path = Path(CONFIG["data"]["zones_json"])
    assert path.exists(), f"zones.json not found at {path}"


def test_all_zones_machines_in_metrics(zones, metrics):
    zone_ids  = set(zones["machines"].keys())
    metric_ids = set(metrics["machine_id"].unique())
    missing = zone_ids - metric_ids
    assert not missing, (
        f"Machines in zones.json but not in mtbf_metrics.csv: {missing}")


def test_all_metrics_machines_in_zones(zones, metrics):
    zone_ids   = set(zones["machines"].keys())
    metric_ids = set(metrics["machine_id"].unique())
    missing = metric_ids - zone_ids
    assert not missing, (
        f"Machines in mtbf_metrics.csv but not in zones.json: {missing}")


def test_machine_coords_within_canvas(zones):
    w = zones["canvas"]["width"]
    h = zones["canvas"]["height"]
    for mid, mz in zones["machines"].items():
        assert mz["x"] >= 0 and mz["x"] + mz["w"] <= w + 0.01, \
            f"{mid}: x out of canvas [{mz['x']}, {mz['x']+mz['w']}]"
        assert mz["y"] >= 0 and mz["y"] + mz["h"] <= h + 0.01, \
            f"{mid}: y out of canvas [{mz['y']}, {mz['y']+mz['h']}]"


def test_vsm_labels_match(zones, metrics):
    for mid, mz in zones["machines"].items():
        row = metrics[metrics["machine_id"] == mid]
        if not row.empty:
            assert row.iloc[0]["vsm"] == mz["vsm"], \
                f"{mid}: VSM mismatch zones={mz['vsm']} vs metrics={row.iloc[0]['vsm']}"


def test_heatmap_png_generated():
    path = Path(CONFIG["heatmap"]["output_path"])
    assert path.exists(), f"Heatmap PNG not found at {path}"
    assert path.stat().st_size > 50_000, "Heatmap PNG seems too small (likely empty)"
