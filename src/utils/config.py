import yaml
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

def load_config() -> dict:
    cfg_path = _ROOT / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    # Resolve all paths relative to project root (skip non-string values)
    for section in ("paths", "data", "heatmap"):
        if section in cfg:
            for k, v in cfg[section].items():
                if isinstance(v, str):
                    cfg[section][k] = str(_ROOT / v)
    return cfg

CONFIG = load_config()
