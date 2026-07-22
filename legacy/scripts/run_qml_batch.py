import argparse
import subprocess
import sys


DEFAULT_DATASETS = ["narma10", "narma20", "mackey_glass", "lorenz63"]
DEFAULT_QML_CONFIGS = ["qnn", "qnn_ising", "qrnn", "qlstm"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--configs", type=str, default=",".join(DEFAULT_QML_CONFIGS))
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--no-energy", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    args = parser.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    configs = [x.strip() for x in args.configs.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    for dataset in datasets:
        for config in configs:
            for seed in seeds:
                cmd = [
                    sys.executable,
                    "scripts\\run_qml_experiment.py",
                    "--dataset",
                    dataset,
                    "--qml-config",
                    config,
                    "--seed",
                    str(seed),
                    "--device",
                    args.device,
                ]

                if args.no_energy:
                    cmd.append("--no-energy")

                print("\nRunning:", " ".join(cmd), flush=True)
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()