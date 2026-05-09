"""
MTBF (Mean Time Between Failures) calculations and health scoring.
"""
import pandas as pd
import numpy as np
from src.utils.helpers import clamp


def calculate_mtbf(failures: pd.DataFrame, machine_id: str,
                   total_hours: float) -> float:
    """MTBF = total_operating_hours / number_of_failures"""
    n = len(failures[failures["machine_id"] == machine_id])
    if n == 0:
        return total_hours
    return round(total_hours / n, 2)


def health_score(mtbf: float, availability_pct: float, failures_30d: int,
                 max_mtbf: float, max_failures: int) -> float:
    """
    Composite health score (0–100) blending MTBF, availability, and
    recent failure frequency with weights 0.5 / 0.3 / 0.2.
    """
    norm_mtbf  = min(mtbf, max_mtbf) / max_mtbf if max_mtbf else 1.0
    norm_avail = availability_pct / 100.0
    norm_fail  = 1.0 - (failures_30d / max_failures) if max_failures else 1.0
    raw = norm_mtbf * 0.5 + norm_avail * 0.3 + norm_fail * 0.2
    return round(clamp(raw * 100), 1)


def assign_health_status(score: float) -> str:
    if score >= 70:
        return "Healthy"
    elif score >= 40:
        return "Monitor"
    return "Critical"


HEALTH_COLORS = {
    "Healthy":  "#2ECC71",
    "Monitor":  "#F39C12",
    "Critical": "#E74C3C",
}
