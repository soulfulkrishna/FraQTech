from pathlib import Path
import pandas as pd


def infer_time_col(df: pd.DataFrame) -> str:
    candidates = [
        "timestamp",
        "datetime",
        "date",
        "time",
        "Date",
        "Datetime",
        "Timestamp",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return df.columns[0]


def infer_task_and_horizon(name: str):
    if name.startswith("finance_"):
        return "financial_volatility_forecasting", 1
    if name.startswith("weather_"):
        return "weather_temperature_forecasting", 24
    return "time_series_forecasting", 1


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    processed = root / "data" / "processed"

    files = sorted(processed.glob("finance_*.csv")) + sorted(processed.glob("weather_*.csv"))

    if not files:
        raise RuntimeError(f"No finance/weather CSVs found in {processed}")

    rows = []

    for path in files:
        df = pd.read_csv(path)
        time_col = infer_time_col(df)

        times = pd.to_datetime(df[time_col], errors="coerce")
        valid_times = times.dropna()

        n = len(df)
        train_rows = int(n * 0.60)
        val_rows = int(n * 0.20)
        test_rows = n - train_rows - val_rows

        task_type, horizon = infer_task_and_horizon(path.stem)

        rows.append(
            {
                "dataset": path.stem,
                "path": str(path),
                "rows": n,
                "start": valid_times.iloc[0] if len(valid_times) else "",
                "end": valid_times.iloc[-1] if len(valid_times) else "",
                "task_type": task_type,
                "horizon": horizon,
                "train_rows": train_rows,
                "val_rows": val_rows,
                "test_rows": test_rows,
            }
        )

    manifest = pd.DataFrame(rows)
    out = processed / "aqora_dataset_manifest.csv"
    manifest.to_csv(out, index=False)

    print(f"Saved: {out}")
    print()
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()