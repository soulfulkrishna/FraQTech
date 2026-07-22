from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase3.baselines import QRCFeatureReadout
from src.phase3.config import load_qrc_config, load_yaml
from src.phase3.data import load_phase3_finance_dataset
from src.phase3.hardware import run_qbraid_managed_counts
from src.phase3.metrics import regression_metrics
from src.phase3.pipeline import fit_qrc_ridge_final, select_qrc_ridge_alpha
from src.phase3.qiskit_qrc import build_checkpoint_circuits, circuit_resource_summary
from src.phase3.results import Phase3Result, save_result
from src.phase3.temporal_ising_qrc import encode_exact
from src.phase3.utils import atomic_json_dump, git_commit, set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-qrn", dest="device_qrn", required=True)
    parser.add_argument("--finance-config", default="configs/phase3_finance.yaml")
    parser.add_argument("--qrc-config", default="configs/gate_qrc_5q_hardware.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--windows", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()

    finance_cfg = load_yaml(PROJECT_ROOT / args.finance_config)
    qrc_cfg = load_qrc_config(PROJECT_ROOT / args.qrc_config)
    set_global_seed(args.seed)
    dataset = load_phase3_finance_dataset(
        PROJECT_ROOT / finance_cfg["data"]["csv_path"],
        lookback=int(finance_cfg["data"]["lookback"]),
    )
    indices = np.linspace(0, len(dataset.test.y) - 1, min(args.windows, len(dataset.test.y)), dtype=int)

    Z_train, _ = encode_exact(dataset.train.X, qrc_cfg, seed=args.seed, batch_size=32)
    Z_val, _ = encode_exact(dataset.val.X, qrc_cfg, seed=args.seed, batch_size=32)
    alpha, validation_metrics = select_qrc_ridge_alpha(
        dataset, Z_train, dataset.train.y, Z_val, dataset.val.y,
        finance_cfg["qrc"]["ridge_alphas"],
    )
    readout = fit_qrc_ridge_final(
        Z_train, dataset.train.y, Z_val, dataset.val.y, alpha
    )
    circuits, mapping, params = build_checkpoint_circuits(
        dataset.test.X[indices], qrc_cfg, seed=args.seed, measure=True
    )
    logical = circuit_resource_summary(circuits)
    Z_qpu, meta = run_qbraid_managed_counts(
        circuits, mapping, qrc_cfg, params,
        device_qrn=args.device_qrn,
        shots=args.shots,
        max_circuits_per_batch=args.batch_size,
    )
    hardware_dir = PROJECT_ROOT / "results" / "hardware"
    hardware_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        hardware_dir / f"qbraid_features_seed{args.seed}.npz", Z=Z_qpu, indices=indices
    )
    atomic_json_dump(meta, hardware_dir / f"qbraid_metadata_seed{args.seed}.json")

    y_true = dataset.scaler.inverse_transform(dataset.test.y[indices])
    y_pred = np.maximum(dataset.scaler.inverse_transform(readout.predict(Z_qpu)), 1e-10)
    result = Phase3Result(
        task="vix_cross_platform_qpu_validation",
        dataset=finance_cfg["data"]["dataset_name"],
        model=f"temporal_ising_qrc_{qrc_cfg.n_qubits}q_ridge_qbraid",
        seed=args.seed,
        split="test_subset",
        metrics=regression_metrics(y_true, y_pred),
        runtime={"hardware_wall_clock_sec": float(meta["wall_clock_sec"])},
        resources={**logical, **meta, "total_shots": len(circuits) * args.shots},
        configuration={"qrc": qrc_cfg.__dict__, "readout_alpha": alpha},
        notes={"test_indices": indices.tolist(), "simulator_trained_readout": True, "validation_metrics": validation_metrics},
        git_commit=git_commit(),
    )
    paths = save_result(result, y_true, y_pred, dataset.test.timestamps[indices])
    print(json.dumps({"paths": paths, "metadata": meta}, indent=2, default=str))


if __name__ == "__main__":
    main()
