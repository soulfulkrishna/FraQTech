from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .utils import atomic_json_dump, git_commit, project_root, stable_hash


@dataclass
class Phase3Result:
    task: str
    dataset: str
    model: str
    seed: int
    split: str
    metrics: Dict[str, float]
    runtime: Dict[str, float]
    resources: Dict[str, Any] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)
    git_commit: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def save_result(
    result: Phase3Result,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: np.ndarray,
    probability: np.ndarray | None = None,
    target_regime: np.ndarray | None = None,
    transition: np.ndarray | None = None,
    output_root: str | Path | None = None,
) -> Dict[str, str]:
    root = Path(output_root) if output_root else project_root() / "results"
    payload = result.to_dict()
    payload["git_commit"] = result.git_commit if result.git_commit != "unknown" else git_commit()
    run_key = stable_hash(
        {
            "task": result.task,
            "dataset": result.dataset,
            "model": result.model,
            "seed": result.seed,
            "split": result.split,
            "configuration": result.configuration,
        }
    )
    stem = f"{result.dataset}__{result.model}__seed{result.seed}__{run_key}"
    json_path = root / "raw" / f"{stem}.json"
    pred_path = root / "predictions" / f"{stem}.csv"
    atomic_json_dump(payload, json_path)

    frame = pd.DataFrame(
        {
            "timestamp": np.asarray(timestamps).reshape(-1),
            "y_true": np.asarray(y_true, dtype=float).reshape(-1),
            "y_pred": np.asarray(y_pred, dtype=float).reshape(-1),
        }
    )
    if probability is not None:
        frame["regime_probability"] = np.asarray(probability, dtype=float).reshape(-1)
    if target_regime is not None:
        frame["target_regime"] = np.asarray(target_regime, dtype=int).reshape(-1)
    if transition is not None:
        frame["enter_high_regime"] = np.asarray(transition, dtype=int).reshape(-1)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(pred_path, index=False)
    return {"json": str(json_path), "predictions": str(pred_path)}
