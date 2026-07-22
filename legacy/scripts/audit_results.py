import json
import math
from pathlib import Path


def is_bad(x):
    try:
        return x is None or not math.isfinite(float(x))
    except Exception:
        return True


def main():
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "results" / "raw_logs"

    bad = []

    for path in raw_dir.glob("*.json"):
        with open(path, "r") as f:
            row = json.load(f)

        for key in ["rmse", "nrmse_sigma", "mae"]:
            value = row.get(key)
            if is_bad(value):
                bad.append((path.name, key, value))

    if not bad:
        print("No non-finite metrics found.")
        return

    print("Bad result files:")
    for filename, key, value in bad:
        print(f"{filename}: {key} = {value}")


if __name__ == "__main__":
    main()