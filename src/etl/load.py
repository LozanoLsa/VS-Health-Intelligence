import pandas as pd
from pathlib import Path
from src.utils.config import CONFIG
from src.utils.logger import get_logger

log = get_logger(__name__)


def save_processed(df: pd.DataFrame, key: str) -> str:
    out_dir = Path(CONFIG["paths"]["processed_data"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = str(out_dir / f"{key}.csv")
    df.to_csv(path, index=False)
    log.info(f"Saved {key}.csv  ({len(df)} rows) -> {path}")
    return path


def load_mtbf_metrics() -> pd.DataFrame:
    path = CONFIG["data"]["mtbf_metrics"]
    return pd.read_csv(path)


def load_monthly_metrics() -> pd.DataFrame:
    path = CONFIG["data"].get("monthly_metrics",
                              "data/processed/monthly_metrics.csv")
    return pd.read_csv(path)


def load_component_metrics() -> pd.DataFrame:
    path = CONFIG["data"].get("component_metrics",
                              "data/processed/component_metrics.csv")
    return pd.read_csv(path)
