from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator

import numpy as np


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def git_commit(root: Path | None = None) -> str:
    root = root or project_root()
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def stable_hash(payload: Dict[str, Any], length: int = 16) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


@contextmanager
def timer() -> Iterator[Dict[str, float]]:
    box: Dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield box
    finally:
        box["elapsed_sec"] = time.perf_counter() - start
