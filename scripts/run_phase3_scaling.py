from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def completed(model: str, seed: int) -> bool:
    for path in (PROJECT_ROOT / "results" / "raw").glob("*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if row.get("model") == model and int(row.get("seed", -1)) == seed:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", default="5,10,15")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--readouts", default="ridge,tcn")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    args = parser.parse_args()
    qubits = [int(x) for x in args.qubits.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    readouts = [x.strip() for x in args.readouts.split(",") if x.strip()]

    for q in qubits:
        config = f"configs/gate_qrc_{q}q.yaml"
        for readout in readouts:
            for seed in seeds:
                model = f"temporal_ising_qrc_{q}q_{readout}"
                if completed(model, seed) and not args.force:
                    print(f"SKIP {model} seed={seed}")
                    continue
                cmd = [
                    sys.executable, "scripts/run_phase3_qrc_sim.py",
                    "--qrc-config", config,
                    "--seed", str(seed),
                    "--readout", readout,
                    "--device", args.device,
                ]
                for flag, value in [
                    ("--max-train", args.max_train),
                    ("--max-val", args.max_val),
                    ("--max-test", args.max_test),
                ]:
                    if value:
                        cmd += [flag, str(value)]
                print("RUN:", " ".join(cmd), flush=True)
                subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
