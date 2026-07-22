import argparse
import subprocess
import sys


DEFAULT_DATASETS = ["narma10", "narma20", "mackey_glass", "lorenz63"]
DEFAULT_CONFIGS = ["cv_gqrc_baseline", "gb_qrc_baseline"]
DEFAULT_READOUTS = ["linear"]
FULL_READOUTS = ["linear", "mlp", "rnn", "gru", "lstm", "tcn", "transformer"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--configs", type=str, default=",".join(DEFAULT_CONFIGS))
    parser.add_argument("--readouts", type=str, default=",".join(DEFAULT_READOUTS))
    parser.add_argument("--full-readouts", action="store_true")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--no-energy", action="store_true")
    parser.add_argument("--force-recache", action="store_true")

    args = parser.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    configs = [x.strip() for x in args.configs.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    if args.full_readouts:
        readouts = FULL_READOUTS
    else:
        readouts = [x.strip() for x in args.readouts.split(",") if x.strip()]

    for dataset in datasets:
        for config in configs:
            for readout in readouts:
                for seed in seeds:
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

                    if args.force_recache:
                        cmd.append("--force-recache")

                    print("\nRunning:", " ".join(cmd), flush=True)
                    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()