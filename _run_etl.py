"""
ETL pipeline runner — produces mtbf_metrics.csv and monthly_metrics.csv.
Called from app.py on dashboard refresh and standalone for dev.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.etl.extract import load_equipment_master, load_failures, load_production
from src.etl.transform import (compute_mtbf_metrics, compute_monthly_metrics,
                                clean_failures, compute_component_metrics)
from src.etl.load import save_processed
from src.utils.logger import get_logger

log = get_logger("etl_runner")


def run_pipeline() -> str:
    log.info("=== ETL pipeline start ===")

    equipment    = load_equipment_master()
    failures_raw = load_failures()
    production   = load_production()

    failures = clean_failures(failures_raw)
    save_processed(failures,   "cleaned_failures")
    save_processed(production, "production_data_clean")

    # Current-period MTBF metrics (for Fleet Overview)
    metrics = compute_mtbf_metrics(equipment, failures)
    path    = save_processed(metrics, "mtbf_metrics")

    dist = metrics["health_status"].value_counts()
    log.info("--- Health Score Distribution ---")
    for status in ["Healthy", "Monitor", "Critical"]:
        log.info(f"  {status:10s}: {dist.get(status, 0)} machines")

    # Monthly time-series metrics (for Trends + ML forecasting)
    monthly = compute_monthly_metrics(equipment, failures)
    save_processed(monthly, "monthly_metrics")

    # Component-level metrics (for Root Cause & Prescriptive tab)
    components = compute_component_metrics(equipment, failures)
    if not components.empty:
        save_processed(components, "component_metrics")
        log.info(f"  Component metrics: {len(components)} rows saved")

    log.info("=== ETL pipeline complete ===")
    return path


if __name__ == "__main__":
    path = run_pipeline()
    print(f"\nOutput: {path}")
