from dataclasses import dataclass
from typing import List, Dict, Any
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

def get_settings() -> Settings:
    cfg = load_config()
    return Settings(**cfg)
