from loguru import logger
from pathlib import Path
import yaml, os

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)

def env(key: str, default=None):
    return os.getenv(key, default)
