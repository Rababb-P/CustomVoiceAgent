"""Tiny YAML config loader. Every module takes a config dict so tests can inject their own."""

from __future__ import annotations

import functools
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"


@functools.cache
def load_config(name: str) -> dict:
    """Load configs/<name>.yaml (name may also be a full path)."""
    path = Path(name)
    if not path.suffix:
        path = CONFIG_DIR / f"{name}.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
