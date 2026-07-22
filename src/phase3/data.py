from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RobustScaler1D:
    median: float
    iqr: float

    @classmethod
    def fit(cls, x: np.ndarray) -> "RobustScaler1D":
        x = np.asarray(x, dtype=float).reshape(-1)
        median = float(np.median(x))
        q25, q75 = np.percentile(x, [25.0, 75.0])
        iqr = float(q75 - q25)
        if not np.isfinite(iqr) or iqr <= 1e-12:
            iqr = float(np.std(x))
        if not np.isfinite(iqr) or iqr <= 1e-12:
            iqr = 1.0
        return cls(median=median, iqr=iqr)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.median) / self.iqr

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float) * self.iqr + self.median


@dataclass
class ForecastSplit:
    X: np.ndarray
    y: np.ndarray
    timestamps: np.ndarray
    target_regime: np.ndarray | None = None
    enter_high_regime: np.ndarray | None = None
    target_regime_threshold: np.ndarray | None = None


@dataclass
class ForecastDataset:
    train: ForecastSplit
    val: ForecastSplit
    test: ForecastSplit
    scaler: RobustScaler1D
    lookback: int
    horizon: int
    metadata: Dict[str, object]


def piecewise_aggregate_approximation(X: np.ndarray, bins: int) -> np.ndarray:
    """Compress [N,T,D] windows to [N,bins,D] using deterministic segment means."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"Expected [N,T,D], received {X.shape}")
    n, t, d = X.shape
    if bins <= 0:
        raise ValueError("bins must be positive")
    if bins == t:
        return X.copy()
    edges = np.linspace(0, t, bins + 1)
    out = np.empty((n, bins, d), dtype=np.float32)
    for i in range(bins):
        lo = int(np.floor(edges[i]))
        hi = int(np.floor(edges[i + 1]))
        hi = max(hi, lo + 1)
        hi = min(hi, t)
        out[:, i, :] = X[:, lo:hi, :].mean(axis=1)
    return out


def _windows_from_frame(
    df: pd.DataFrame,
    lookback: int,
    value_col: str,
    target_col: str,
    time_col: str,
    scaler: RobustScaler1D,
) -> ForecastSplit:
    values = scaler.transform(df[value_col].to_numpy(dtype=float))
    targets = scaler.transform(df[target_col].to_numpy(dtype=float))
    timestamps = df[time_col].astype(str).to_numpy()

    X, y, ts = [], [], []
    regimes, entries, thresholds = [], [], []
    have_regime = "target_regime" in df.columns
    have_entry = "enter_high_regime" in df.columns
    have_threshold = "target_regime_threshold" in df.columns

    for idx in range(lookback - 1, len(df)):
        start = idx - lookback + 1
        X.append(values[start : idx + 1])
        y.append(targets[idx])
        ts.append(timestamps[idx])
        if have_regime:
            regimes.append(int(df.iloc[idx]["target_regime"]))
        if have_entry:
            entries.append(int(df.iloc[idx]["enter_high_regime"]))
        if have_threshold:
            thresholds.append(float(df.iloc[idx]["target_regime_threshold"]))

    if not X:
        raise ValueError(f"Split has only {len(df)} rows, shorter than lookback={lookback}")

    return ForecastSplit(
        X=np.asarray(X, dtype=np.float32)[..., None],
        y=np.asarray(y, dtype=np.float32),
        timestamps=np.asarray(ts),
        target_regime=np.asarray(regimes, dtype=int) if have_regime else None,
        enter_high_regime=np.asarray(entries, dtype=int) if have_entry else None,
        target_regime_threshold=np.asarray(thresholds, dtype=float) if have_threshold else None,
    )


def load_phase3_finance_dataset(
    csv_path: str | Path,
    lookback: int = 50,
    value_col: str = "value",
    target_col: str = "target",
    split_col: str = "split",
    time_col: str = "date",
) -> ForecastDataset:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. Run scripts/build_phase3_finance_data.py first."
        )
    df = pd.read_csv(csv_path)
    required = {value_col, target_col, split_col, time_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns {missing}; found {list(df.columns)}")

    df[split_col] = df[split_col].astype(str).str.lower()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[value_col, target_col, split_col, time_col]).reset_index(drop=True)

    parts = {name: df[df[split_col] == name].copy().reset_index(drop=True) for name in ("train", "val", "test")}
    for name, part in parts.items():
        if part.empty:
            raise ValueError(f"No rows for split={name!r}")

    scaler = RobustScaler1D.fit(parts["train"][value_col].to_numpy(dtype=float))
    splits = {
        name: _windows_from_frame(part, lookback, value_col, target_col, time_col, scaler)
        for name, part in parts.items()
    }

    return ForecastDataset(
        train=splits["train"],
        val=splits["val"],
        test=splits["test"],
        scaler=scaler,
        lookback=lookback,
        horizon=1,
        metadata={
            "path": str(csv_path),
            "rows": int(len(df)),
            "train_rows": int(len(parts["train"])),
            "val_rows": int(len(parts["val"])),
            "test_rows": int(len(parts["test"])),
            "value_col": value_col,
            "target_col": target_col,
            "time_col": time_col,
            "split_col": split_col,
        },
    )


def concatenate_splits(a: ForecastSplit, b: ForecastSplit) -> ForecastSplit:
    def cat_optional(x: np.ndarray | None, y: np.ndarray | None):
        if x is None or y is None:
            return None
        return np.concatenate([x, y])

    return ForecastSplit(
        X=np.concatenate([a.X, b.X]),
        y=np.concatenate([a.y, b.y]),
        timestamps=np.concatenate([a.timestamps, b.timestamps]),
        target_regime=cat_optional(a.target_regime, b.target_regime),
        enter_high_regime=cat_optional(a.enter_high_regime, b.enter_high_regime),
        target_regime_threshold=cat_optional(a.target_regime_threshold, b.target_regime_threshold),
    )
