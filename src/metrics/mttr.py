"""
MTTR (Mean Time To Repair) calculations.
"""
import pandas as pd


def calculate_mttr(failures: pd.DataFrame, machine_id: str) -> float:
    """MTTR = mean repair time for a given machine."""
    mf = failures[failures["machine_id"] == machine_id]["repair_hrs"]
    if mf.empty:
        return 0.0
    return round(float(mf.mean()), 2)


def repair_time_distribution(failures: pd.DataFrame,
                              machine_id: str) -> pd.Series:
    return failures[failures["machine_id"] == machine_id]["repair_hrs"].describe()
