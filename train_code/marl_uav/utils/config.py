"""Config loading and merging."""
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config from path."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
