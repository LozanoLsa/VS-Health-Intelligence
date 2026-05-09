"""
OEE (Overall Equipment Effectiveness) calculations.
OEE = Availability × Performance × Quality
"""
import pandas as pd


def calculate_oee(availability: float, performance: float,
                  quality: float) -> float:
    """All inputs in fraction (0–1). Returns OEE as percentage."""
    return round(availability * performance * quality * 100, 2)


def vsm_oee_summary(production: pd.DataFrame) -> pd.DataFrame:
    """Average OEE per VSM from production_data.csv."""
    return (
        production.groupby("vsm")["oee_pct"]
        .agg(avg_oee="mean", min_oee="min", max_oee="max")
        .round(2)
        .reset_index()
    )
