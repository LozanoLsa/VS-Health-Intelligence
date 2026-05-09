"""Tests for ETL pipeline output integrity."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pytest
import pandas as pd
from src.utils.config import CONFIG

REQUIRED_COLUMNS = [
    "machine_id", "vsm", "area", "machine_type",
    "mtbf_hrs", "mttr_hrs", "failures_30d", "availability_pct",
    "downtime_cost_per_hr", "total_monthly_downtime_cost",
    "health_score", "health_status",
]


@pytest.fixture(scope="module")
def metrics():
    path = CONFIG["data"]["mtbf_metrics"]
    return pd.read_csv(path)


def test_metrics_file_exists():
    path = Path(CONFIG["data"]["mtbf_metrics"])
    assert path.exists(), f"mtbf_metrics.csv not found at {path}"


def test_required_columns_present(metrics):
    missing = [c for c in REQUIRED_COLUMNS if c not in metrics.columns]
    assert not missing, f"Missing columns: {missing}"


def test_no_nulls_in_key_columns(metrics):
    null_counts = metrics[REQUIRED_COLUMNS].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    assert cols_with_nulls.empty, f"Nulls found:\n{cols_with_nulls}"


def test_health_score_in_range(metrics):
    out = metrics[(metrics["health_score"] < 0) | (metrics["health_score"] > 100)]
    assert out.empty, f"Scores out of [0,100]:\n{out[['machine_id','health_score']]}"


def test_health_status_valid_values(metrics):
    valid = {"Healthy", "Monitor", "Critical"}
    invalid = set(metrics["health_status"].unique()) - valid
    assert not invalid, f"Invalid health_status values: {invalid}"


def test_availability_in_range(metrics):
    out = metrics[(metrics["availability_pct"] < 0) | (metrics["availability_pct"] > 100)]
    assert out.empty, f"Availability out of [0,100]:\n{out[['machine_id','availability_pct']]}"


def test_all_vsm_lines_present(metrics):
    vsms = set(metrics["vsm"].unique())
    assert "Alpha" in vsms
    assert "Beta"  in vsms
    assert "Gamma" in vsms


def test_machine_count(metrics):
    assert len(metrics) == 35, f"Expected 35 machines, got {len(metrics)}"
