from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_ints(text: str):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Finite-shot sweep using the ideal-shot Aer condition.")
    parser.add_argument("--shots", default="256,1024,4096")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--sample-windows", type=int, default=100)
    parser.add_argument("--finance-config", default="configs/phase3_finance.yaml")
    parser.add_argument("--qrc-config", default="configs/gate_qrc_5q_hardware.yaml")
    parser.add_argument("--noise-config", default="configs/noise_study.yaml")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    for seed in parse_ints(args.seeds):
        for shots in parse_ints(args.shots):
            cmd = [
                sys.executable, "scripts/run_phase3_noise.py",
                "--finance-config", args.finance_config,
                "--qrc-config", args.qrc_config,
                "--noise-config", args.noise_config,
                "--seed", str(seed),
                "--sample-windows", str(args.sample_windows),
                "--shots", str(shots),
                "--conditions", "ideal_shot",
                "--device", args.device,
            ]
            print("\nRUN:", " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
