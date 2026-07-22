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
from src.phase3.data import concatenate_splits, load_phase3_finance_dataset, piecewise_aggregate_approximation
from src.phase3.metrics import regression_metrics
from src.phase3.results import Phase3Result, save_result
from src.phase3.temporal_ising_qrc import encode_exact
from src.phase3.utils import git_commit, set_global_seed


def select_alpha(dataset, Z_train, y_train, Z_val, y_val, alphas):
    y_val_raw = dataset.scaler.inverse_transform(y_val)
    best = None
    for alpha in alphas:
        model = QRCFeatureReadout(alpha=float(alpha)).fit(Z_train, y_train)
        pred = np.maximum(dataset.scaler.inverse_transform(model.predict(Z_val)), 1e-10)
        metrics = regression_metrics(y_val_raw, pred)
        if best is None or metrics["qlike"] < best[0]["qlike"]:
            best = (metrics, float(alpha))
    return best


def matched_random_features(X_train, X_val, X_test, bins: int, output_dim: int, seed: int):
    rng = np.random.default_rng(seed)
    def base(X):
        return piecewise_aggregate_approximation(X, bins).reshape(len(X), -1)
    A, B, C = base(X_train), base(X_val), base(X_test)
    W = rng.normal(0.0, 1.0 / np.sqrt(A.shape[1]), size=(A.shape[1], output_dim))
    b = rng.uniform(-np.pi, np.pi, size=output_dim)
    return np.tanh(A @ W + b), np.tanh(B @ W + b), np.tanh(C @ W + b)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finance-config", default="configs/phase3_finance.yaml")
    parser.add_argument("--qrc-config", default="configs/gate_qrc_5q.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    finance = load_yaml(PROJECT_ROOT / args.finance_config)
    base = load_qrc_config(PROJECT_ROOT / args.qrc_config)
    dataset = load_phase3_finance_dataset(
        PROJECT_ROOT / finance["data"]["csv_path"], lookback=int(finance["data"]["lookback"])
    )
    set_global_seed(args.seed)
    variants = {
        "full_qrc": base,
        "z_only": replace(base, include_zz=False, include_global_features=False),
        "no_input_reupload": replace(base, input_reupload=False),
        "line_topology": replace(base, topology="line"),
    }
    outputs = {}
    for name, cfg in variants.items():
        start = time.perf_counter()
        Z_train, meta = encode_exact(dataset.train.X, cfg, args.seed, args.batch_size, args.device)
        Z_val, _ = encode_exact(dataset.val.X, cfg, args.seed, args.batch_size, args.device)
        Z_test, _ = encode_exact(dataset.test.X, cfg, args.seed, args.batch_size, args.device)
        val_metrics, alpha = select_alpha(
            dataset, Z_train, dataset.train.y, Z_val, dataset.val.y, finance["qrc"]["ridge_alphas"]
        )
        model = QRCFeatureReadout(alpha=alpha).fit(
            np.concatenate([Z_train, Z_val]), np.concatenate([dataset.train.y, dataset.val.y])
        )
        pred = np.maximum(dataset.scaler.inverse_transform(model.predict(Z_test)), 1e-10)
        y_true = dataset.scaler.inverse_transform(dataset.test.y)
        result = Phase3Result(
            task="vix_qrc_ablation",
            dataset=finance["data"]["dataset_name"],
            model=f"ablation_{name}", seed=args.seed, split="test",
            metrics=regression_metrics(y_true, pred),
            runtime={"feature_and_fit_sec": time.perf_counter() - start},
            resources={
                "qubits": cfg.n_qubits,
                "logical_depth_proxy": meta["logical_depth_proxy_full_window"],
                "flattened_feature_dim": meta["flattened_feature_dim"],
            },
            configuration={"qrc": cfg.__dict__, "ridge_alpha": alpha},
            notes={"validation_metrics": val_metrics}, git_commit=git_commit(),
        )
        outputs[name] = save_result(result, y_true, pred, dataset.test.timestamps)

    # Classical matched-dimensional nonlinear random-feature control.
    # Use the exact full-QRC metadata rather than relying on a hand-derived dimension.
    Z_probe, meta = encode_exact(dataset.train.X[:1], base, args.seed, 1, args.device)
    target_dim = int(np.prod(Z_probe.shape[1:]))
    R_train, R_val, R_test = matched_random_features(
        dataset.train.X, dataset.val.X, dataset.test.X, base.temporal_bins, target_dim, args.seed + 90000
    )
    val_metrics, alpha = select_alpha(
        dataset, R_train, dataset.train.y, R_val, dataset.val.y, finance["qrc"]["ridge_alphas"]
    )
    model = QRCFeatureReadout(alpha=alpha).fit(
        np.concatenate([R_train, R_val]), np.concatenate([dataset.train.y, dataset.val.y])
    )
    pred = np.maximum(dataset.scaler.inverse_transform(model.predict(R_test)), 1e-10)
    y_true = dataset.scaler.inverse_transform(dataset.test.y)
    result = Phase3Result(
        task="vix_qrc_ablation", dataset=finance["data"]["dataset_name"],
        model="ablation_matched_random_tanh", seed=args.seed, split="test",
        metrics=regression_metrics(y_true, pred), runtime={},
        resources={"feature_dim": target_dim, "qubits": 0},
        configuration={"random_feature_seed": args.seed + 90000, "ridge_alpha": alpha},
        notes={"validation_metrics": val_metrics, "purpose": "matched-dimensional classical nonlinear feature control"},
        git_commit=git_commit(),
    )
    outputs["matched_random_tanh"] = save_result(result, y_true, pred, dataset.test.timestamps)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
