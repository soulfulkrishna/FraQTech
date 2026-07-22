import argparse
import subprocess
import sys
from pathlib import Path


ALL_DATASETS = ["narma10", "narma20", "mackey_glass", "lorenz63", "santafe"]

DEFAULT_DATASETS = ["narma10", "narma20", "mackey_glass", "lorenz63"]

ALL_MODELS = [
    "persistence",
    "arima",
    "ridge_lag",
    "esn",
    "mlp",
    "rnn",
    "gru",
    "lstm",
    "tcn",
    "transformer",
]

DEFAULT_FAST_MODELS = [
    "persistence",
    "arima",
    "ridge_lag",
    "esn",
]

DEFAULT_FULL_MODELS = ALL_MODELS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--include-santafe", action="store_true")
    parser.add_argument("--no-energy", action="store_true")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")

    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    datasets = DEFAULT_DATASETS.copy()

    if args.include_santafe:
        santafe_path = Path("data") / "raw" / "santafe_laser.txt"
        if not santafe_path.exists():
            raise FileNotFoundError(
                "data\\raw\\santafe_laser.txt not found. "
                "Place the Santa Fe laser file there before using --include-santafe."
            )
        datasets.append("santafe")

    models = DEFAULT_FULL_MODELS if args.full else DEFAULT_FAST_MODELS

    for dataset in datasets:
        for model in models:
            for seed in seeds:
                cmd = [
                    sys.executable,
                    "scripts\\run_experiment.py",
                    "--dataset",
                    dataset,
                    "--model",
                    model,
                    "--seed",
                    str(seed),
                ]

                if args.no_energy:
                    cmd.append("--no-energy")

                print("\nRunning:", " ".join(cmd), flush=True)
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()