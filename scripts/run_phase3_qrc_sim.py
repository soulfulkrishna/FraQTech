from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models_classical.torch_sequence import TorchSequenceRegressor
from src.phase3.baselines import QRCFeatureReadout
from src.phase3.config import load_qrc_config, load_yaml
from src.phase3.data import load_phase3_finance_dataset
from src.phase3.metrics import classification_metrics, regression_metrics
from src.phase3.pipeline import fit_regime_classifier_final, select_regime_classifier
from src.phase3.results import Phase3Result, save_result
from src.phase3.temporal_ising_qrc import encode_exact
from src.phase3.utils import git_commit, set_global_seed, stable_hash


def subset(split, maximum: int | None):
    if maximum is None or maximum <= 0 or len(split.y) <= maximum:
        return split
    indices = np.linspace(0, len(split.y) - 1, maximum, dtype=int)
    from src.phase3.data import ForecastSplit
    return ForecastSplit(
        X=split.X[indices], y=split.y[indices], timestamps=split.timestamps[indices],
        target_regime=split.target_regime[indices] if split.target_regime is not None else None,
        enter_high_regime=split.enter_high_regime[indices] if split.enter_high_regime is not None else None,
        target_regime_threshold=split.target_regime_threshold[indices] if split.target_regime_threshold is not None else None,
    )


def encode_cached(name, split, qrc_cfg, seed, batch_size, device, force=False):
    cache_root = PROJECT_ROOT / "results" / "feature_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    key = stable_hash({"name": name, "cfg": qrc_cfg.__dict__, "seed": seed, "shape": list(split.X.shape)})
    path = cache_root / f"{name}__seed{seed}__{key}.npz"
    if path.exists() and not force:
        payload = np.load(path, allow_pickle=True)
        return payload["Z"], json.loads(str(payload["meta"])) , True
    start = time.perf_counter()
    Z, meta = encode_exact(split.X, qrc_cfg, seed=seed, batch_size=batch_size, device=device)
    meta["encoding_wall_sec"] = time.perf_counter() - start
    np.savez_compressed(path, Z=Z, meta=json.dumps(meta))
    return Z, meta, False


def select_ridge(Z_train, y_train, Z_val, y_val, scaler, alphas):
    best = None
    y_val_raw = scaler.inverse_transform(y_val)
    for alpha in alphas:
        model = QRCFeatureReadout(alpha=float(alpha)).fit(Z_train, y_train)
        pred_raw = np.maximum(scaler.inverse_transform(model.predict(Z_val)), 1e-10)
        metrics = regression_metrics(y_val_raw, pred_raw)
        if best is None or metrics["qlike"] < best[0]["qlike"]:
            best = (metrics, float(alpha))
    return best


def tcn_model(input_dim: int, lookback: int, seed: int):
    return TorchSequenceRegressor(
        model_type="tcn", input_dim=input_dim, lookback=lookback,
        hidden_dim=64, num_layers=3, dropout=0.1, lr=1e-3,
        weight_decay=1e-4, batch_size=128, epochs=30, seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finance-config", default="configs/phase3_finance.yaml")
    parser.add_argument("--qrc-config", default="configs/gate_qrc_5q.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--readout", choices=["ridge", "tcn"], default="ridge")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--force-features", action="store_true")
    args = parser.parse_args()

    finance_cfg = load_yaml(PROJECT_ROOT / args.finance_config)
    qrc_cfg = load_qrc_config(PROJECT_ROOT / args.qrc_config)
    set_global_seed(args.seed)
    dataset = load_phase3_finance_dataset(
        PROJECT_ROOT / finance_cfg["data"]["csv_path"],
        lookback=int(finance_cfg["data"]["lookback"]),
    )
    train = subset(dataset.train, args.max_train)
    val = subset(dataset.val, args.max_val)
    test = subset(dataset.test, args.max_test)
    batch_size = args.batch_size or int(
        finance_cfg["qrc"]["exact_simulator_batch_size"].get(qrc_cfg.n_qubits, 8)
    )

    Z_train, meta_train, hit_train = encode_cached(
        "finance_train", train, qrc_cfg, args.seed, batch_size, args.device, args.force_features
    )
    Z_val, meta_val, hit_val = encode_cached(
        "finance_val", val, qrc_cfg, args.seed, batch_size, args.device, args.force_features
    )
    Z_test, meta_test, hit_test = encode_cached(
        "finance_test", test, qrc_cfg, args.seed, batch_size, args.device, args.force_features
    )

    fit_start = time.perf_counter()
    if args.readout == "ridge":
        val_metrics, alpha = select_ridge(
            Z_train, train.y, Z_val, val.y, dataset.scaler, finance_cfg["qrc"]["ridge_alphas"]
        )
        model = QRCFeatureReadout(alpha=alpha).fit(
            np.concatenate([Z_train, Z_val]), np.concatenate([train.y, val.y])
        )
        pred_scaled = model.predict(Z_test)
        selected = {"alpha": alpha}
    else:
        probe = tcn_model(Z_train.shape[2], Z_train.shape[1], args.seed).fit(Z_train, train.y)
        val_pred = probe.predict(Z_val)
        val_metrics = regression_metrics(
            dataset.scaler.inverse_transform(val.y),
            np.maximum(dataset.scaler.inverse_transform(val_pred), 1e-10),
        )
        model = tcn_model(Z_train.shape[2], Z_train.shape[1], args.seed).fit(
            np.concatenate([Z_train, Z_val]), np.concatenate([train.y, val.y])
        )
        pred_scaled = model.predict(Z_test)
        selected = {"hidden_dim": 64, "layers": 3, "epochs": 30}
    readout_fit_sec = time.perf_counter() - fit_start

    y_true = dataset.scaler.inverse_transform(test.y)
    y_pred = np.maximum(dataset.scaler.inverse_transform(pred_scaled), 1e-10)
    metrics = regression_metrics(y_true, y_pred)

    class_metrics = {}
    probability = None
    if train.target_regime is not None and val.target_regime is not None and test.target_regime is not None:
        best_C, classifier_val = select_regime_classifier(
            Z_train, train.target_regime, Z_val, val.target_regime,
            finance_cfg["qrc"]["classification_C"],
        )
        probability = fit_regime_classifier_final(
            Z_train, train.target_regime, Z_val, val.target_regime, Z_test, best_C
        )
        class_metrics = classification_metrics(test.target_regime, probability, test.enter_high_regime)
        selected["regime_classifier_C"] = best_C
        selected["regime_classifier_validation"] = classifier_val

    total_encoding = sum(float(m.get("encoding_wall_sec", 0.0)) for m in (meta_train, meta_val, meta_test))
    model_name = f"temporal_ising_qrc_{qrc_cfg.n_qubits}q_{args.readout}"
    result = Phase3Result(
        task="vix_volatility_and_regime",
        dataset=finance_cfg["data"]["dataset_name"],
        model=model_name,
        seed=args.seed,
        split="test",
        metrics={**metrics, **{f"regime_{k}": v for k, v in class_metrics.items()}},
        runtime={
            "feature_encoding_sec": total_encoding,
            "readout_fit_sec": readout_fit_sec,
            "feature_cache_hits": int(hit_train) + int(hit_val) + int(hit_test),
        },
        resources={
            "qubits": qrc_cfg.n_qubits,
            "temporal_bins": qrc_cfg.temporal_bins,
            "virtual_nodes": qrc_cfg.virtual_nodes,
            "logical_depth_proxy": meta_train["logical_depth_proxy_full_window"],
            "logical_gate_count": meta_train["logical_gate_count_full_window"],
            "feature_dim_per_checkpoint": meta_train["feature_dim_per_checkpoint"],
            "flattened_feature_dim": meta_train["flattened_feature_dim"],
            "shots": 0,
            "backend": "custom_exact_statevector",
        },
        configuration={"qrc": qrc_cfg.__dict__, "readout": selected},
        notes={
            "validation_metrics": val_metrics,
            "train_samples": len(train.y), "val_samples": len(val.y), "test_samples": len(test.y),
            "quantum_parameters_trainable": 0,
        },
        git_commit=git_commit(),
    )
    paths = save_result(
        result, y_true, y_pred, test.timestamps,
        probability=probability, target_regime=test.target_regime,
        transition=test.enter_high_regime,
    )
    print(json.dumps({"result": result.to_dict(), "paths": paths}, indent=2, default=str))


if __name__ == "__main__":
    main()
