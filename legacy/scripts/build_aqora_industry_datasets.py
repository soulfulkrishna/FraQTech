"""
Build Aqora-style industry forecasting datasets for QRC/QML/classical benchmarking.

Creates:
  data/raw/industry/*.csv
  data/processed/*.csv
  data/processed/aqora_dataset_manifest.csv

Default datasets:
  finance_spy_vol_h1
  finance_qqq_vol_h1
  finance_vix_vol_h1
  weather_nyc_temp_h24
  weather_denver_temp_h24

Notes:
  - The `value` column is intentionally univariate so the current QRC configs
    with input_dim=1 can be reused immediately.
  - Extra feature columns are retained for later multivariate baselines.
  - Splits are chronological 60/20/20: train/val/test.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def slugify(text: str) -> str:
    text = text.lower()
    text = text.replace("^", "")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def add_chronological_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    train_end = int(0.60 * n)
    val_end = int(0.80 * n)

    split = np.empty(n, dtype=object)
    split[:train_end] = "train"
    split[train_end:val_end] = "val"
    split[val_end:] = "test"

    df["split"] = split
    return df


def zscore_safe(s: pd.Series, window: int | None = None) -> pd.Series:
    if window is None:
        mu = s.mean()
        sigma = s.std()
        if sigma == 0 or pd.isna(sigma):
            return s * 0.0
        return (s - mu) / sigma

    mu = s.rolling(window, min_periods=max(5, window // 4)).mean()
    sigma = s.rolling(window, min_periods=max(5, window // 4)).std()
    return (s - mu) / sigma.replace(0, np.nan)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------
# Finance datasets
# ---------------------------------------------------------------------

def download_finance_raw(
    ticker: str,
    start: str,
    end: str,
    raw_dir: Path,
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pip install yfinance") from exc

    print(f"\nDownloading finance: {ticker} [{start} to {end}]")

    kwargs = dict(
        tickers=ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    try:
        df = yf.download(**kwargs, multi_level_index=False)
    except TypeError:
        df = yf.download(**kwargs)

    if df is None or df.empty:
        raise RuntimeError(f"No finance data returned for {ticker}")

    # Handle possible MultiIndex columns from newer yfinance versions.
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1)
        elif ticker in df.columns.get_level_values(0):
            df = df.xs(ticker, axis=1, level=0)
        else:
            df.columns = ["_".join(map(str, c)).strip() for c in df.columns]

    df = df.reset_index()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Normalize date column name.
    if "date" not in df.columns:
        possible = [c for c in df.columns if "date" in c or "time" in c]
        if not possible:
            raise RuntimeError(f"Could not find date column for {ticker}: {df.columns.tolist()}")
        df = df.rename(columns={possible[0]: "date"})

    df["ticker"] = ticker
    raw_path = raw_dir / f"finance_{slugify(ticker)}_raw.csv"
    write_csv(df, raw_path)
    return df


def build_finance_vol_dataset(
    raw: pd.DataFrame,
    ticker: str,
    horizon_days: int,
    processed_dir: Path,
) -> tuple[str, Path, pd.DataFrame]:
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").dropna(subset=["date"])

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{ticker}: missing required finance columns: {missing}")

    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["log_return"] = np.log(df["close"]).diff()
    df["abs_log_return"] = df["log_return"].abs()

    # Realized volatility proxies.
    df["rv_5"] = df["log_return"].rolling(5, min_periods=5).std()
    df["rv_10"] = df["log_return"].rolling(10, min_periods=8).std()
    df["rv_21"] = df["log_return"].rolling(21, min_periods=15).std()

    # Annualized variants are useful for interpretation.
    df["rv_5_ann"] = df["rv_5"] * math.sqrt(252)
    df["rv_10_ann"] = df["rv_10"] * math.sqrt(252)
    df["rv_21_ann"] = df["rv_21"] * math.sqrt(252)

    # Extra classical-feature columns.
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["co_return"] = (df["close"] - df["open"]) / df["open"]
    if "volume" in df.columns:
        df["volume_z_21"] = zscore_safe(np.log1p(df["volume"]), window=21)
    else:
        df["volume_z_21"] = np.nan

    # Forecast next realized volatility.
    df["target"] = df["rv_5"].shift(-horizon_days)

    # Univariate value for current QRC configs.
    df["value"] = df["rv_5"]

    # Regime label: high-volatility regime using rolling 80th percentile.
    rolling_q80 = df["rv_5"].rolling(252, min_periods=60).quantile(0.80)
    df["regime_label"] = (df["rv_5"] > rolling_q80).astype(int)

    dataset_name = f"finance_{slugify(ticker)}_vol_h{horizon_days}"

    keep_cols = [
        "date",
        "ticker",
        "value",
        "target",
        "regime_label",
        "close",
        "log_return",
        "abs_log_return",
        "rv_5",
        "rv_10",
        "rv_21",
        "rv_5_ann",
        "rv_10_ann",
        "rv_21_ann",
        "hl_range",
        "co_return",
        "volume_z_21",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]
    out = df[keep_cols].replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["value", "target"]).reset_index(drop=True)
    out["dataset"] = dataset_name
    out["task_type"] = "finance_volatility_forecasting"
    out["horizon"] = horizon_days
    out = add_chronological_split(out)

    out_path = processed_dir / f"{dataset_name}.csv"
    write_csv(out, out_path)

    return dataset_name, out_path, out


# ---------------------------------------------------------------------
# Weather datasets
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class WeatherLocation:
    name: str
    lat: float
    lon: float
    alt: float | None = None


DEFAULT_WEATHER_LOCATIONS = {
    "nyc": WeatherLocation("nyc", 40.7128, -74.0060, 10.0),
    "denver": WeatherLocation("denver", 39.7392, -104.9903, 1609.0),
    "chicago": WeatherLocation("chicago", 41.8781, -87.6298, 181.0),
    "miami": WeatherLocation("miami", 25.7617, -80.1918, 2.0),
    "phoenix": WeatherLocation("phoenix", 33.4484, -112.0740, 331.0),
    "seattle": WeatherLocation("seattle", 47.6062, -122.3321, 52.0),
}


def parse_weather_locations(text: str) -> list[WeatherLocation]:
    names = parse_csv(text)
    out: list[WeatherLocation] = []

    for name in names:
        key = slugify(name)
        if key not in DEFAULT_WEATHER_LOCATIONS:
            valid = ", ".join(DEFAULT_WEATHER_LOCATIONS)
            raise RuntimeError(f"Unknown weather location '{name}'. Valid options: {valid}")
        out.append(DEFAULT_WEATHER_LOCATIONS[key])

    return out


def download_weather_raw(
    location: WeatherLocation,
    start: str,
    end: str,
    raw_dir: Path,
) -> pd.DataFrame:
    try:
        from meteostat import Hourly, Point
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pip install meteostat") from exc

    start_dt = datetime.fromisoformat(start)
    # Meteostat hourly supports datetime end.
    end_dt = datetime.fromisoformat(end)

    print(f"\nDownloading weather: {location.name} [{start} to {end}]")

    if location.alt is None:
        point = Point(location.lat, location.lon)
    else:
        point = Point(location.lat, location.lon, location.alt)

    data = Hourly(point, start_dt, end_dt)
    try:
        df = data.normalize().interpolate().fetch()
    except Exception:
        # Fallback for older Meteostat versions.
        df = data.fetch()

    if df is None or df.empty:
        raise RuntimeError(f"No weather data returned for {location.name}")

    df = df.reset_index()
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "time" not in df.columns:
        possible = [c for c in df.columns if "date" in c or "time" in c]
        if not possible:
            raise RuntimeError(f"Could not find time column for {location.name}: {df.columns.tolist()}")
        df = df.rename(columns={possible[0]: "time"})

    df["location"] = location.name
    raw_path = raw_dir / f"weather_{slugify(location.name)}_raw.csv"
    write_csv(df, raw_path)
    return df


def build_weather_temp_dataset(
    raw: pd.DataFrame,
    location: WeatherLocation,
    horizon_hours: int,
    processed_dir: Path,
) -> tuple[str, Path, pd.DataFrame]:
    df = raw.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.sort_values("time").dropna(subset=["time"])

    # Expected Meteostat hourly columns:
    # temp, dwpt, rhum, prcp, wspd, pres, etc.
    if "temp" not in df.columns:
        raise RuntimeError(f"{location.name}: missing 'temp' column in weather data")

    numeric_cols = ["temp", "dwpt", "rhum", "prcp", "snow", "wdir", "wspd", "wpgt", "pres", "tsun", "coco"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Reindex to hourly grid and interpolate small gaps.
    df = df.set_index("time").sort_index()
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_index)
    df.index.name = "time"

    df["location"] = location.name

    for c in numeric_cols:
        if c in df.columns:
            df[c] = df[c].interpolate(limit=6).ffill().bfill()

    # Extra temporal features.
    hour = df.index.hour
    dayofyear = df.index.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["doy_sin"] = np.sin(2 * np.pi * dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * dayofyear / 365.25)

    df["temp_diff_1h"] = df["temp"].diff()
    df["temp_roll_24_mean"] = df["temp"].rolling(24, min_periods=12).mean()
    df["temp_roll_24_std"] = df["temp"].rolling(24, min_periods=12).std()

    if "pres" in df.columns:
        df["pres_diff_1h"] = df["pres"].diff()
    else:
        df["pres_diff_1h"] = np.nan

    if "rhum" in df.columns:
        df["rhum_diff_1h"] = df["rhum"].diff()
    else:
        df["rhum_diff_1h"] = np.nan

    # Forecast future temperature.
    df["target"] = df["temp"].shift(-horizon_hours)

    # Univariate value for current QRC configs.
    df["value"] = df["temp"]

    # Regime label: large absolute temp movement over horizon.
    future_change = (df["target"] - df["temp"]).abs()
    threshold = future_change.rolling(24 * 60, min_periods=24 * 14).quantile(0.80)
    df["regime_label"] = (future_change > threshold).astype(int)

    dataset_name = f"weather_{slugify(location.name)}_temp_h{horizon_hours}"

    out = df.reset_index()

    keep_cols = [
        "time",
        "location",
        "value",
        "target",
        "regime_label",
        "temp",
        "dwpt",
        "rhum",
        "prcp",
        "wspd",
        "pres",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
        "temp_diff_1h",
        "temp_roll_24_mean",
        "temp_roll_24_std",
        "pres_diff_1h",
        "rhum_diff_1h",
    ]

    keep_cols = [c for c in keep_cols if c in out.columns]
    out = out[keep_cols].replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["value", "target"]).reset_index(drop=True)
    out["dataset"] = dataset_name
    out["task_type"] = "weather_temperature_forecasting"
    out["horizon"] = horizon_hours
    out = add_chronological_split(out)

    out_path = processed_dir / f"{dataset_name}.csv"
    write_csv(out, out_path)

    return dataset_name, out_path, out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def summarize_dataset(name: str, path: Path, df: pd.DataFrame) -> dict:
    time_col = "date" if "date" in df.columns else "time" if "time" in df.columns else None

    if time_col is not None:
        start = str(pd.to_datetime(df[time_col]).min())
        end = str(pd.to_datetime(df[time_col]).max())
    else:
        start = ""
        end = ""

    return {
        "dataset": name,
        "path": str(path),
        "rows": len(df),
        "start": start,
        "end": end,
        "task_type": df["task_type"].iloc[0] if "task_type" in df.columns and len(df) else "",
        "horizon": df["horizon"].iloc[0] if "horizon" in df.columns and len(df) else "",
        "train_rows": int((df["split"] == "train").sum()) if "split" in df.columns else "",
        "val_rows": int((df["split"] == "val").sum()) if "split" in df.columns else "",
        "test_rows": int((df["split"] == "test").sum()) if "split" in df.columns else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")

    parser.add_argument(
        "--finance-tickers",
        type=str,
        default="SPY,QQQ,^VIX",
        help="Comma-separated tickers. Default: SPY,QQQ,^VIX",
    )
    parser.add_argument("--finance-horizon-days", type=int, default=1)

    parser.add_argument(
        "--weather-locations",
        type=str,
        default="nyc,denver",
        help="Comma-separated locations. Valid: nyc,denver,chicago,miami,phoenix,seattle",
    )
    parser.add_argument("--weather-horizon-hours", type=int, default=24)

    parser.add_argument(
        "--skip-finance",
        action="store_true",
        help="Skip finance datasets.",
    )
    parser.add_argument(
        "--skip-weather",
        action="store_true",
        help="Skip weather datasets.",
    )

    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw" / "industry"
    processed_dir = root / "data" / "processed"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []

    if not args.skip_finance:
        for ticker in parse_csv(args.finance_tickers):
            raw = download_finance_raw(
                ticker=ticker,
                start=args.start,
                end=args.end,
                raw_dir=raw_dir,
            )
            name, path, df = build_finance_vol_dataset(
                raw=raw,
                ticker=ticker,
                horizon_days=args.finance_horizon_days,
                processed_dir=processed_dir,
            )
            manifest_rows.append(summarize_dataset(name, path, df))

    if not args.skip_weather:
        for location in parse_weather_locations(args.weather_locations):
            raw = download_weather_raw(
                location=location,
                start=args.start,
                end=args.end,
                raw_dir=raw_dir,
            )
            name, path, df = build_weather_temp_dataset(
                raw=raw,
                location=location,
                horizon_hours=args.weather_horizon_hours,
                processed_dir=processed_dir,
            )
            manifest_rows.append(summarize_dataset(name, path, df))

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = processed_dir / "aqora_dataset_manifest.csv"
    write_csv(manifest, manifest_path)

    print("\nAqora dataset build complete.")
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()