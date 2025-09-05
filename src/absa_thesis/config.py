# src/absa_thesis/config.py
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from .utils import load_config

@dataclass
class Settings:
    paths: Dict[str, str]
    data: Dict[str, Any]
    etl: Dict[str, Any]
    annotation: Dict[str, Any]
    modeling: Dict[str, Any]
    rules: Dict[str, Any]
    dashboard: Dict[str, Any]
    # NEW: optional serving block for app & tools that read model paths
    serving: Optional[Dict[str, Any]] = None

def get_settings() -> Settings:
    cfg = load_config() or {}
    allowed = {
        "paths", "data", "etl", "annotation", "modeling",
        "rules", "dashboard", "serving"
    }
    # keep only fields our dataclass knows
    filtered = {k: v for k, v in cfg.items() if k in allowed}
    return Settings(**filtered)
