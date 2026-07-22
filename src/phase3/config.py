from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .qrc_parameters import TemporalIsingQRCConfig


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_qrc_config(path: str | Path) -> TemporalIsingQRCConfig:
    data = load_yaml(path)
    data.pop("name", None)
    return TemporalIsingQRCConfig(**data)
