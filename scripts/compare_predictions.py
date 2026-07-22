from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase3.metrics import diebold_mariano


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_a", type=Path)
    parser.add_argument("prediction_b", type=Path)
    parser.add_argument("--loss", choices=["squared", "absolute", "qlike"], default="qlike")
    args = parser.parse_args()
    a = pd.read_csv(args.prediction_a)
    b = pd.read_csv(args.prediction_b)
    merged = a.merge(b, on="timestamp", suffixes=("_a", "_b"))
    if not (merged["y_true_a"].round(12) == merged["y_true_b"].round(12)).all():
        raise ValueError("Ground truth differs between prediction files")
    print(
        diebold_mariano(
            merged["y_true_a"].to_numpy(),
            merged["y_pred_a"].to_numpy(),
            merged["y_pred_b"].to_numpy(),
            loss=args.loss,
        )
    )


if __name__ == "__main__":
    main()
