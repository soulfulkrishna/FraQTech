from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "processed" / "aqora_dataset_manifest.csv"
OUT_DIR = ROOT / "configs" / "datasets"

LOOKBACK_BY_TASK = {
    "financial_volatility_forecasting": 50,
    "weather_temperature_forecasting": 168,
}

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MANIFEST)

    for _, row in df.iterrows():
        dataset = row["dataset"]
        path = Path(row["path"])
        task_type = row["task_type"]
        horizon = int(row["horizon"])
        lookback = LOOKBACK_BY_TASK.get(task_type, 50)

        # Store project-relative path for portability
        rel_path = path
        try:
            rel_path = path.relative_to(ROOT)
        except ValueError:
            pass

        time_col = "date" if dataset.startswith("finance_") else "time"

        yaml_text = f"""name: {dataset}
type: csv_forecasting
path: {str(rel_path).replace(chr(92), "/")}
value_col: value
target_col: target
time_col: {time_col}
split_col: split
task_type: {task_type}
lookback: {lookback}
horizon: {horizon}
normalization: train
"""

        out_path = OUT_DIR / f"{dataset}.yaml"
        out_path.write_text(yaml_text, encoding="utf-8")
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()