import pandas as pd
from pathlib import Path
from src.utils.config import CONFIG
from src.utils.logger import get_logger

log = get_logger(__name__)

def load_equipment_master() -> pd.DataFrame:
    path = CONFIG["data"]["equipment_master"]
    log.info(f"Loading equipment master from {path}")
    df = pd.read_csv(path)
    log.info(f"  {len(df)} machines loaded")
    return df

def load_failures() -> pd.DataFrame:
    path = CONFIG["data"]["failures"]
    log.info(f"Loading failures from {path}")
    df = pd.read_csv(path, parse_dates=["failure_date"])
    log.info(f"  {len(df)} failure records loaded")
    return df

def load_production() -> pd.DataFrame:
    path = CONFIG["data"]["production_data"]
    log.info(f"Loading production data from {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    log.info(f"  {len(df)} production records loaded")
    return df
