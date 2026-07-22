from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    print("\nRUN:", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "simulator", "full"], default="smoke")
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if not args.skip_data and not (ROOT / "data/processed/finance_vix_vol_h1.csv").exists():
        run([sys.executable, "scripts/build_phase3_finance_data.py"])

    if args.profile == "smoke":
        run([sys.executable, "scripts/smoke_test.py"])
        run([sys.executable, "scripts/run_phase3_classical.py", "--models", "persistence,ridge_lag,har_rv", "--profile", "smoke"])
        run([sys.executable, "scripts/run_phase3_qrc_sim.py", "--qrc-config", "configs/gate_qrc_5q.yaml", "--readout", "ridge", "--max-train", "120", "--max-val", "60", "--max-test", "60", "--device", args.device])
    else:
        for seed in range(5):
            run([sys.executable, "scripts/run_phase3_classical.py", "--seed", str(seed), "--profile", "full"])
        run([sys.executable, "scripts/select_gate_qrc_architecture.py", "--profile", "full", "--device", args.device])
        run([sys.executable, "scripts/run_phase3_scaling.py", "--device", args.device])
        for seed in range(3):
            run([sys.executable, "scripts/run_phase3_ablations.py", "--seed", str(seed), "--device", args.device])
            run([sys.executable, "scripts/run_phase3_noise.py", "--seed", str(seed), "--device", args.device])
        for source in ["raw", "esn", "qrc"]:
            run([sys.executable, "scripts/run_phase3_regime.py", "--feature-source", source, "--device", args.device])
        for q in [5, 10, 15]:
            run([sys.executable, "scripts/run_phase3_mnist.py", "--qubits", str(q), "--device", args.device])
        run([sys.executable, "scripts/aggregate_phase3.py"])
        run([sys.executable, "scripts/make_phase3_figures.py"])
        if args.profile == "full":
            run([sys.executable, "scripts/validate_submission.py"])


if __name__ == "__main__":
    main()
