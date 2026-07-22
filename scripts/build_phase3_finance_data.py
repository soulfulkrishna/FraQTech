from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def add_split(df: pd.DataFrame, train_fraction: float, val_fraction: float) -> pd.DataFrame:
    n = len(df)
    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + val_fraction))
    labels = np.empty(n, dtype=object)
    labels[:train_end] = "train"
    labels[train_end:val_end] = "val"
    labels[val_end:] = "test"
    out = df.copy()
    out["split"] = labels
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install yfinance before building the dataset") from exc
    frame = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
        multi_level_index=False,
    )
    if frame is None or frame.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [str(c[0]) for c in frame.columns]
    frame = frame.reset_index()
    frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]
    if "date" not in frame.columns:
        candidates = [c for c in frame.columns if "date" in c or "time" in c]
        if not candidates:
            raise RuntimeError(f"Could not identify date column: {list(frame.columns)}")
        frame = frame.rename(columns={candidates[0]: "date"})
    return frame


def build_dataset(
    raw: pd.DataFrame,
    ticker: str,
    horizon: int,
    train_fraction: float,
    val_fraction: float,
) -> pd.DataFrame:
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    required = {"open", "high", "low", "close"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}")

    df["ticker"] = ticker
    df["log_return"] = np.log(df["close"]).diff()
    df["abs_log_return"] = df["log_return"].abs()
    df["rv_5"] = df["log_return"].rolling(5, min_periods=5).std()
    df["rv_10"] = df["log_return"].rolling(10, min_periods=8).std()
    df["rv_21"] = df["log_return"].rolling(21, min_periods=15).std()
    df["rv_5_ann"] = df["rv_5"] * math.sqrt(252.0)
    df["rv_10_ann"] = df["rv_10"] * math.sqrt(252.0)
    df["rv_21_ann"] = df["rv_21"] * math.sqrt(252.0)
    df["hl_range"] = (df["high"] - df["low"]) / df["close"].replace(0.0, np.nan)
    df["co_return"] = (df["close"] - df["open"]) / df["open"].replace(0.0, np.nan)
    if "volume" in df.columns:
        lv = np.log1p(df["volume"])
        mu = lv.rolling(21, min_periods=10).mean()
        sd = lv.rolling(21, min_periods=10).std().replace(0.0, np.nan)
        df["volume_z_21"] = (lv - mu) / sd
    else:
        df["volume_z_21"] = np.nan

    # Forecast target and regime targets are explicitly future-shifted.
    df["value"] = df["rv_5"]
    df["target"] = df["rv_5"].shift(-horizon)
    threshold = df["rv_5"].rolling(252, min_periods=60).quantile(0.80)
    df["regime_threshold"] = threshold
    df["regime_label"] = (df["rv_5"] > threshold).astype(int)
    df["target_regime_threshold"] = threshold.shift(-horizon)
    df["target_regime"] = df["regime_label"].shift(-horizon)
    df["enter_high_regime"] = (
        (df["regime_label"] == 0) & (df["target_regime"] == 1)
    ).astype(int)

    keep = [
        "date", "ticker", "value", "target", "regime_threshold", "regime_label",
        "target_regime_threshold", "target_regime", "enter_high_regime", "close",
        "log_return", "abs_log_return", "rv_5", "rv_10", "rv_21", "rv_5_ann",
        "rv_10_ann", "rv_21_ann", "hl_range", "co_return", "volume_z_21",
    ]
    out = df[keep].replace([np.inf, -np.inf], np.nan)
    out = out.dropna(
        subset=["value", "target", "target_regime", "target_regime_threshold", "log_return"]
    ).reset_index(drop=True)
    out["target_regime"] = out["target_regime"].astype(int)
    out["dataset"] = "finance_vix_vol_h1"
    out["task_type"] = "financial_volatility_forecasting"
    out["horizon"] = int(horizon)
    out = add_split(out, train_fraction, val_fraction)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="^VIX")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--use-raw", type=Path, default=None)
    args = parser.parse_args()

    raw_dir = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    manifest_dir = PROJECT_ROOT / "data" / "manifests"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    if args.use_raw:
        raw = pd.read_csv(args.use_raw)
        raw_path = Path(args.use_raw)
    else:
        raw = download(args.ticker, args.start, args.end)
        raw_path = raw_dir / "finance_vix_raw.csv"
        raw.to_csv(raw_path, index=False)

    out = build_dataset(raw, args.ticker, args.horizon, args.train_fraction, args.val_fraction)
    output_path = processed_dir / "finance_vix_vol_h1.csv"
    out.to_csv(output_path, index=False)

    manifest = {
        "dataset": "finance_vix_vol_h1",
        "source": "Yahoo Finance via yfinance",
        "ticker": args.ticker,
        "requested_start": args.start,
        "requested_end": args.end,
        "actual_start": str(out["date"].iloc[0]),
        "actual_end": str(out["date"].iloc[-1]),
        "rows": int(len(out)),
        "splits": out["split"].value_counts().to_dict(),
        "horizon": int(args.horizon),
        "raw_path": str(raw_path.relative_to(PROJECT_ROOT) if raw_path.is_relative_to(PROJECT_ROOT) else raw_path),
        "processed_path": str(output_path.relative_to(PROJECT_ROOT)),
        "processed_sha256": sha256(output_path),
        "columns": list(out.columns),
        "notes": [
            "All rolling features and regime thresholds use present/past observations only.",
            "target, target_regime, and target_regime_threshold are shifted one day ahead.",
            "Splits are chronological and assigned after feature/target construction.",
        ],
    }
    manifest_path = manifest_dir / "finance_vix_vol_h1.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved raw: {raw_path}")
    print(f"Saved processed: {output_path}")
    print(f"Saved manifest: {manifest_path}")
    print(out.groupby("split").size())
    print(out.head())


if __name__ == "__main__":
    main()
