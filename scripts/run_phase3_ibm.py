from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase3.baselines import QRCFeatureReadout
from src.phase3.config import load_qrc_config, load_yaml
from src.phase3.data import load_phase3_finance_dataset
from src.phase3.hardware import run_ibm_estimator
from src.phase3.metrics import regression_metrics
from src.phase3.pipeline import fit_qrc_ridge_final, select_qrc_ridge_alpha
from src.phase3.qiskit_qrc import build_checkpoint_circuits, circuit_resource_summary
from src.phase3.results import Phase3Result, save_result
from src.phase3.temporal_ising_qrc import encode_exact
from src.phase3.utils import atomic_json_dump, git_commit, set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finance-config", default="configs/phase3_finance.yaml")
    parser.add_argument("--qrc-config", default="configs/gate_qrc_5q_hardware.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--windows", type=int, default=40)
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--resilience-level", type=int, default=1)
    parser.add_argument("--max-pubs-per-job", type=int, default=100)
    parser.add_argument("--no-dd", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    finance_cfg = load_yaml(PROJECT_ROOT / args.finance_config)
    qrc_cfg = load_qrc_config(PROJECT_ROOT / args.qrc_config)
    set_global_seed(args.seed)
    dataset = load_phase3_finance_dataset(
        PROJECT_ROOT / finance_cfg["data"]["csv_path"],
        lookback=int(finance_cfg["data"]["lookback"]),
    )
    indices = np.linspace(0, len(dataset.test.y) - 1, min(args.windows, len(dataset.test.y)), dtype=int)

    print("Encoding simulator train/validation features...", flush=True)
    Z_train, exact_meta = encode_exact(dataset.train.X, qrc_cfg, seed=args.seed, batch_size=32)
    Z_val, _ = encode_exact(dataset.val.X, qrc_cfg, seed=args.seed, batch_size=32)
    alpha, validation_metrics = select_qrc_ridge_alpha(
        dataset, Z_train, dataset.train.y, Z_val, dataset.val.y,
        finance_cfg["qrc"]["ridge_alphas"],
    )
    readout = fit_qrc_ridge_final(
        Z_train, dataset.train.y, Z_val, dataset.val.y, alpha
    )

    circuits, mapping, params = build_checkpoint_circuits(
        dataset.test.X[indices], qrc_cfg, seed=args.seed, measure=False
    )
    logical = circuit_resource_summary(circuits)
    prep_path = PROJECT_ROOT / "results" / "hardware" / f"ibm_prepare_seed{args.seed}.json"
    atomic_json_dump(
        {
            "qrc_config": qrc_cfg.__dict__,
            "test_indices": indices.tolist(),
            "logical_resources": logical,
            "circuits": len(circuits),
            "shots": args.shots,
            "estimated_total_shots": len(circuits) * args.shots,
            "note": "No IBM credentials are stored in this repository.",
        },
        prep_path,
    )
    print(f"Preparation manifest: {prep_path}")
    if args.prepare_only:
        print(json.dumps({"logical": logical, "manifest": str(prep_path)}, indent=2))
        return

    Z_hardware, hardware_meta = run_ibm_estimator(
        circuits, mapping, qrc_cfg, params,
        shots=args.shots,
        backend_name=args.backend,
        optimization_level=args.optimization_level,
        resilience_level=args.resilience_level,
        dynamical_decoupling=not args.no_dd,
        max_pubs_per_job=args.max_pubs_per_job,
    )
    feature_path = PROJECT_ROOT / "results" / "hardware" / f"ibm_features_seed{args.seed}.npz"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(feature_path, Z=Z_hardware, indices=indices)
    metadata_path = PROJECT_ROOT / "results" / "hardware" / f"ibm_metadata_seed{args.seed}.json"
    atomic_json_dump(hardware_meta, metadata_path)

    y_true = dataset.scaler.inverse_transform(dataset.test.y[indices])
    y_pred = np.maximum(dataset.scaler.inverse_transform(readout.predict(Z_hardware)), 1e-10)
    metrics = regression_metrics(y_true, y_pred)
    result = Phase3Result(
        task="vix_ibm_qpu_validation",
        dataset=finance_cfg["data"]["dataset_name"],
        model=f"temporal_ising_qrc_{qrc_cfg.n_qubits}q_ridge_ibm",
        seed=args.seed,
        split="test_subset",
        metrics=metrics,
        runtime={"hardware_wall_clock_sec": float(hardware_meta["wall_clock_sec"])},
        resources={**logical, **hardware_meta},
        configuration={"qrc": qrc_cfg.__dict__, "readout_alpha": alpha},
        notes={
            "test_indices": indices.tolist(),
            "simulator_trained_readout": True,
            "validation_metrics": validation_metrics,
            "feature_file": str(feature_path.relative_to(PROJECT_ROOT)),
            "metadata_file": str(metadata_path.relative_to(PROJECT_ROOT)),
        },
        git_commit=git_commit(),
    )
    paths = save_result(result, y_true, y_pred, dataset.test.timestamps[indices])
    print(json.dumps({"paths": paths, "hardware": hardware_meta}, indent=2, default=str))


if __name__ == "__main__":
    main()
