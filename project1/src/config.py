"""
Enterprise-RAG: Configuration loader with environment variable interpolation.
"""
import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_BASE_DIR = Path(__file__).resolve().parent.parent

# Load env files: default .env first, then the specific key.env with override
load_dotenv()
_KEY_ENV = _BASE_DIR.parent / "key" / "key.env"
if _KEY_ENV.exists():
    load_dotenv(_KEY_ENV, override=True)

CONFIG_PATH = _BASE_DIR / "config.yaml"


def _interpolate_env(value: str) -> str:
    """Replace ${VAR} placeholders with environment variable values."""
    pattern = re.compile(r"\$\{(\w+)\}")
    matches = pattern.findall(value)
    for var in matches:
        env_val = os.environ.get(var, "")
        value = value.replace(f"${{{var}}}", env_val)
    return value


def _resolve_env(obj: Any) -> Any:
    """Recursively resolve environment variable placeholders in config values."""
    if isinstance(obj, str):
        return _interpolate_env(obj)
    elif isinstance(obj, dict):
        return {k: _resolve_env(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env(item) for item in obj]
    return obj


def load_config(path: Path | None = None) -> dict:
    """Load and resolve the YAML configuration file."""
    config_path = path or CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return _resolve_env(config)


class Config:
    """Singleton config accessor."""

    _instance: dict | None = None

    @classmethod
    def get(cls) -> dict:
        if cls._instance is None:
            cls._instance = load_config()
        return cls._instance

    @classmethod
    def reload(cls) -> dict:
        cls._instance = load_config()
        return cls._instance


config = Config.get()
