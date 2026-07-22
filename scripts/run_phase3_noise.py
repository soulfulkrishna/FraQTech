from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase3.baselines import QRCFeatureReadout
from src.phase3.config import load_qrc_config, load_yaml
from src.phase3.data import load_phase3_finance_dataset
from src.phase3.metrics import regression_metrics
from src.phase3.pipeline import fit_qrc_ridge_final, select_qrc_ridge_alpha
from src.phase3.noise import build_aer_noise_model, run_aer_counts
from src.phase3.qiskit_qrc import build_checkpoint_circuits, circuit_resource_summary
from src.phase3.results import Phase3Result, save_result
from src.phase3.temporal_ising_qrc import encode_exact
from src.phase3.utils import git_commit, set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finance-config", default="configs/phase3_finance.yaml")
    parser.add_argument("--qrc-config", default="configs/gate_qrc_5q_hardware.yaml")
    parser.add_argument("--noise-config", default="configs/noise_study.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-windows", type=int, default=0)
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conditions", default="all", help="Comma-separated condition names or all")
    args = parser.parse_args()

    finance_cfg = load_yaml(PROJECT_ROOT / args.finance_config)
    qrc_cfg = load_qrc_config(PROJECT_ROOT / args.qrc_config)
    noise_cfg = load_yaml(PROJECT_ROOT / args.noise_config)
    set_global_seed(args.seed)
    dataset = load_phase3_finance_dataset(
        PROJECT_ROOT / finance_cfg["data"]["csv_path"],
        lookback=int(finance_cfg["data"]["lookback"]),
    )
    shots = args.shots or int(noise_cfg["shots"])
    n_test = args.sample_windows or int(noise_cfg["sample_windows"])
    indices = np.linspace(0, len(dataset.test.y) - 1, min(n_test, len(dataset.test.y)), dtype=int)

    # Select the readout on validation only, refit on train+validation, then freeze it
    # across all shot/noise conditions. This prevents test leakage and isolates feature drift.
    Z_train, meta = encode_exact(dataset.train.X, qrc_cfg, seed=args.seed, batch_size=32, device=args.device)
    Z_val, _ = encode_exact(dataset.val.X, qrc_cfg, seed=args.seed, batch_size=32, device=args.device)
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
    logical_resources = circuit_resource_summary(circuits)
    y_true = dataset.scaler.inverse_transform(dataset.test.y[indices])
    requested = None if args.conditions == "all" else {x.strip() for x in args.conditions.split(",") if x.strip()}
    conditions = [c for c in noise_cfg["conditions"] if requested is None or c["name"] in requested]
    if not conditions:
        raise ValueError(f"No matching noise conditions for {args.conditions!r}")
    outputs = {}
    for condition in conditions:
        print(f"\n=== noise condition: {condition['name']} ===", flush=True)
        noise_model = build_aer_noise_model(
            depolarizing_1q=float(condition["depolarizing_1q"]),
            depolarizing_2q=float(condition["depolarizing_2q"]),
            amplitude_damping=float(condition["amplitude_damping"]),
        )
        start = time.perf_counter()
        Z_test = run_aer_counts(
            circuits, mapping, qrc_cfg, params, shots=shots, seed=args.seed,
            noise_model=noise_model,
        )
        execute_sec = time.perf_counter() - start
        y_pred = np.maximum(dataset.scaler.inverse_transform(readout.predict(Z_test)), 1e-10)
        metrics = regression_metrics(y_true, y_pred)
        model_name = f"temporal_ising_qrc_{qrc_cfg.n_qubits}q_ridge_{condition['name']}_{shots}shots"
        result = Phase3Result(
            task="vix_noise_study",
            dataset=finance_cfg["data"]["dataset_name"],
            model=model_name,
            seed=args.seed,
            split="test_subset",
            metrics=metrics,
            runtime={"aer_execution_sec": execute_sec},
            resources={
                **logical_resources,
                "qubits": qrc_cfg.n_qubits,
                "shots_per_circuit": shots,
                "total_shots": shots * len(circuits),
                "backend": "qiskit_aer",
            },
            configuration={"qrc": qrc_cfg.__dict__, "noise": condition, "shots": shots},
            notes={"test_indices": indices.tolist(), "readout_alpha": alpha, "validation_metrics": validation_metrics},
            git_commit=git_commit(),
        )
        paths = save_result(
            result, y_true, y_pred, dataset.test.timestamps[indices]
        )
        outputs[condition["name"]] = paths
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
