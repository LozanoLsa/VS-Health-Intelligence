"""Tests for health score calculation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pytest
from src.metrics.mtbf import health_score, assign_health_status, HEALTH_COLORS
from src.utils.helpers import clamp


def test_health_score_range():
    """health_score must always return a value in [0, 100]."""
    for mtbf in [10, 60, 200, 400, 999.9]:
        for avail in [50.0, 80.0, 99.0]:
            for fails in [0, 3, 10, 20]:
                score = health_score(mtbf, avail, fails, max_mtbf=400, max_failures=20)
                assert 0.0 <= score <= 100.0, (
                    f"Score out of range: {score} "
                    f"(mtbf={mtbf}, avail={avail}, fails={fails})")


def test_health_score_formula_extreme_healthy():
    """Perfect machine (max MTBF, 100% avail, 0 failures) should score 100."""
    score = health_score(400, 100.0, 0, max_mtbf=400, max_failures=10)
    assert score == 100.0


def test_health_score_formula_worst_case():
    """Worst machine (low MTBF, 50% avail, max failures) should be low."""
    score = health_score(10, 50.0, 20, max_mtbf=400, max_failures=20)
    assert score < 40.0, f"Expected Critical range, got {score}"


def test_assign_health_status_boundaries():
    assert assign_health_status(100.0) == "Healthy"
    assert assign_health_status(70.0)  == "Healthy"
    assert assign_health_status(69.9)  == "Monitor"
    assert assign_health_status(40.0)  == "Monitor"
    assert assign_health_status(39.9)  == "Critical"
    assert assign_health_status(0.0)   == "Critical"


def test_health_colors_defined():
    for status in ("Healthy", "Monitor", "Critical"):
        assert status in HEALTH_COLORS
        assert HEALTH_COLORS[status].startswith("#")


def test_clamp_bounds():
    assert clamp(-5)   == 0.0
    assert clamp(105)  == 100.0
    assert clamp(50)   == 50.0
