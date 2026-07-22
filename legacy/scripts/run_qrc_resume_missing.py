import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASETS = ["narma10", "narma20", "mackey_glass", "lorenz63", "santafe"]

DEFAULT_CONFIGS = [
    "cv_gqrc_baseline",
    "cv_gqrc_heavy",
    "gb_qrc_baseline",
]

DEFAULT_READOUTS = [
    "linear",
    "mlp",
    "rnn",
    "gru",
    "lstm",
    "tcn",
    "transformer",
]


def parse_csv(text: str):
    return [x.strip() for x in text.split(",") if x.strip()]


def result_path(root: Path, dataset: str, config: str, readout: str, seed: int) -> Path:
    return root / "results" / "raw_logs" / f"{dataset}_{config}_{readout}_seed{seed}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--configs", type=str, default=",".join(DEFAULT_CONFIGS))
    parser.add_argument("--readouts", type=str, default=",".join(DEFAULT_READOUTS))
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--no-energy", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun even if output JSON exists.")

    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    datasets = parse_csv(args.datasets)
    configs = parse_csv(args.configs)
    readouts = parse_csv(args.readouts)
    seeds = [int(x) for x in parse_csv(args.seeds)]

    total = 0
    skipped = 0
    ran = 0

    for dataset in datasets:
        for config in configs:
            for readout in readouts:
                for seed in seeds:
                    total += 1
                    out = result_path(root, dataset, config, readout, seed)

                    if out.exists() and not args.force:
                        skipped += 1
                        print(f"SKIP existing: {out.name}", flush=True)
                        continue

                    cmd = [
                        sys.executable,
                        "scripts\\run_qrc_experiment.py",
                        "--dataset",
                        dataset,
                        "--qrc-config",
                        config,
                        "--readout",
                        readout,
                        "--seed",
                        str(seed),
                    ]

                    if args.no_energy:
                        cmd.append("--no-energy")

                    print("\nRUN:", " ".join(cmd), flush=True)
                    subprocess.run(cmd, check=True)
                    ran += 1

    print("\nQRC resume complete.")
    print(f"Total planned: {total}")
    print(f"Skipped existing: {skipped}")
    print(f"Newly run: {ran}")


if __name__ == "__main__":
    main()