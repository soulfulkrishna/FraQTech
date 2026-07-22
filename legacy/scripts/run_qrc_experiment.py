import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import subprocess
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.linear_model import Ridge

from src.datasets.windowing import make_supervised_windows, make_test_windows_with_context
from src.energy.resource_tracker import ResourceTracker
from src.metrics.forecasting import regression_metrics
from src.models_classical.torch_sequence import TorchSequenceRegressor
from src.models_qrc.qrc_factory import load_qrc_config
from src.models_qrc.qrc_feature_cache import get_or_create_qrc_cache
from src.training.result_schema import ResultRow

from scripts.run_experiment import load_dataset, standardize, inverse_standardize


ALL_READOUTS = {
    "linear",
    "mlp",
    "rnn",
    "gru",
    "lstm",
    "tcn",
    "transformer",
}


def get_git_commit() -> str:
    try:
        inside = subprocess.check_output(
            ["git", "rev-parse", "--is-inside-work-tree"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()

        if inside != "true":
            return "unknown"

        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def build_torch_readout(
    readout: str,
    lookback: int,
    input_dim: int,
    seed: int,
) -> TorchSequenceRegressor:
    if readout == "transformer":
        hidden_dim = 64
        num_layers = 2
        epochs = 20
        batch_size = 64
    elif readout == "tcn":
        hidden_dim = 64
        num_layers = 3
        epochs = 20
        batch_size = 128
    else:
        hidden_dim = 64
        num_layers = 2
        epochs = 20
        batch_size = 128

    return TorchSequenceRegressor(
        model_type=readout,
        input_dim=input_dim,
        lookback=lookback,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.1,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=batch_size,
        epochs=epochs,
        seed=seed,
    )


def prepare_raw_windows(
    dataset: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    splits, cfg = load_dataset(dataset=dataset, seed=seed)

    train_y = splits["train"]["y"]
    val_y = splits["val"]["y"]
    test_y = splits["test"]["y"]

    train_val_y = np.concatenate([train_y, val_y])

    lookback = int(cfg.get("lookback", 64))
    horizon = int(cfg.get("horizon", 1))

    train_val_scaled, test_scaled, mean, std = standardize(train_val_y, test_y)

    X_train, y_train = make_supervised_windows(
        train_val_scaled,
        lookback=lookback,
        horizon=horizon,
    )

    X_test, y_test = make_test_windows_with_context(
        train_val_scaled,
        test_scaled,
        lookback=lookback,
        horizon=horizon,
    )

    meta = {
        "cfg": cfg,
        "lookback": lookback,
        "horizon": horizon,
        "mean": mean,
        "std": std,
    }

    return X_train, y_train, X_test, y_test, meta


def run_qrc_experiment(
    dataset: str,
    qrc_config_name: str,
    readout: str,
    seed: int,
    track_energy: bool,
    force_recache: bool,
    cache_batch_size: int,
) -> Tuple[ResultRow, Dict]:
    if readout not in ALL_READOUTS:
        raise ValueError(f"Unknown readout={readout}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train, X_test, y_test, raw_meta = prepare_raw_windows(
        dataset=dataset,
        seed=seed,
    )

    cfg = raw_meta["cfg"]
    raw_lookback = raw_meta["lookback"]
    horizon = raw_meta["horizon"]
    mean = raw_meta["mean"]
    std = raw_meta["std"]

    qrc_cfg = load_qrc_config(PROJECT_ROOT, qrc_config_name)

    Z_train, y_train_t, Z_test, y_test_t, cache_meta, cache_tracker = get_or_create_qrc_cache(
        project_root=PROJECT_ROOT,
        dataset=dataset,
        config_name=qrc_config_name,
        config=qrc_cfg,
        seed=seed,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        lookback=raw_lookback,
        horizon=horizon,
        device=device,
        batch_size=cache_batch_size,
        track_energy=track_energy,
        force_recache=force_recache,
    )

    encoder_meta = cache_meta["encoder_meta"]
    qrc_lookback = int(Z_train.shape[1])
    qrc_feature_dim = int(Z_train.shape[2])

    if readout == "linear":
        Xtr = Z_train[:, -1, :].numpy()
        Xte = Z_test[:, -1, :].numpy()

        model = Ridge(alpha=1e-3)

        with ResourceTracker(track_energy=track_energy, project_name="qrc-linear-train") as train_tracker:
            model.fit(Xtr, y_train_t.numpy())

        with ResourceTracker(track_energy=track_energy, project_name="qrc-linear-infer") as infer_tracker:
            y_pred_scaled = model.predict(Xte)

        y_true_scaled = y_test_t.numpy()

        trainable_params = int(model.coef_.size + 1)
        readout_total_params = trainable_params

    else:
        model = build_torch_readout(
            readout=readout,
            lookback=qrc_lookback,
            input_dim=qrc_feature_dim,
            seed=seed,
        )

        with ResourceTracker(track_energy=track_energy, project_name=f"qrc-{readout}-train") as train_tracker:
            model.fit(Z_train.numpy(), y_train_t.numpy())

        with ResourceTracker(track_energy=track_energy, project_name=f"qrc-{readout}-infer") as infer_tracker:
            y_pred_scaled = model.predict(Z_test.numpy())

        y_true_scaled = y_test_t.numpy()

        trainable_params = model.trainable_params()
        readout_total_params = model.total_params()

    y_true = inverse_standardize(y_true_scaled, mean=mean, std=std)
    y_pred = inverse_standardize(y_pred_scaled, mean=mean, std=std)

    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

    metrics = regression_metrics(y_true, y_pred)

    inference_latency_ms = 1000.0 * infer_tracker.elapsed_time_sec / len(y_pred)

    total_energy = (
        cache_tracker.energy_kwh
        + train_tracker.energy_kwh
        + infer_tracker.energy_kwh
    )

    total_carbon = (
        cache_tracker.carbon_kgco2e
        + train_tracker.carbon_kgco2e
        + infer_tracker.carbon_kgco2e
    )

    peak_ram = max(
        cache_tracker.peak_ram_gb,
        train_tracker.peak_ram_gb,
        infer_tracker.peak_ram_gb,
    )

    peak_gpu = max(
        cache_tracker.peak_gpu_mem_gb,
        train_tracker.peak_gpu_mem_gb,
        infer_tracker.peak_gpu_mem_gb,
    )

    n_total_windows = int(len(Z_train) + len(Z_test))
    circuit_evals = 0

    if encoder_meta["encoder_type"] == "gb_qrc":
        circuit_evals = int(
            n_total_windows
            * raw_lookback
            * int(encoder_meta["virtual_nodes"])
        )

    shots_per_eval = int(encoder_meta.get("shots") or 0)
    shots_total = int(circuit_evals * shots_per_eval) if shots_per_eval > 0 else 0

    row = ResultRow(
        dataset=dataset,
        model=f"{qrc_config_name}_{readout}",
        encoder_type=encoder_meta["encoder_type"],
        readout_type=readout,
        seed=seed,
        split_id="60_20_20",

        rmse=metrics["rmse"],
        nrmse_sigma=metrics["nrmse_sigma"],
        mae=metrics["mae"],

        qrc_cache_time_sec=cache_tracker.elapsed_time_sec,
        final_train_time_sec=train_tracker.elapsed_time_sec,
        hpo_time_sec=0.0,
        inference_latency_ms_per_sample=inference_latency_ms,

        energy_kwh_cache=cache_tracker.energy_kwh,
        energy_kwh_final_train=train_tracker.energy_kwh,
        energy_kwh_hpo=0.0,
        energy_kwh_total=total_energy,

        carbon_kgco2e_cache=cache_tracker.carbon_kgco2e,
        carbon_kgco2e_final_train=train_tracker.carbon_kgco2e,
        carbon_kgco2e_hpo=0.0,
        carbon_kgco2e_total=total_carbon,

        peak_ram_gb=peak_ram,
        peak_gpu_mem_gb=peak_gpu,

        trainable_params=trainable_params,
        total_params=int(readout_total_params + encoder_meta["total_params"]),
        feature_dim=qrc_feature_dim,

        qubits=encoder_meta.get("qubits"),
        modes=encoder_meta.get("modes"),
        virtual_nodes=encoder_meta.get("virtual_nodes"),
        circuit_depth=encoder_meta.get("circuit_depth"),
        circuit_evals=circuit_evals,
        shots=shots_total,
        qpu_time_proxy_sec=None,

        backend_type=encoder_meta["backend_type"],
        git_commit=get_git_commit(),
        hardware_id="local_windows",

        extra={
            "track_energy": track_energy,
            "dataset_type": cfg["type"],
            "qrc_config_name": qrc_config_name,
            "raw_lookback": raw_lookback,
            "qrc_lookback": qrc_lookback,
            "horizon": horizon,
            "normalization_mean": mean,
            "normalization_std": std,
            "qrc_loaded_from_cache": cache_meta.get("loaded_from_cache", False),
            "cache_meta": cache_meta,
            "note": "QRC encoder plus classical readout for regression forecasting.",
        },
    )

    debug = {
        "dataset": dataset,
        "qrc_config": qrc_config_name,
        "readout": readout,
        "device": str(device),
        "X_train_shape": list(X_train.shape),
        "Z_train_shape": list(Z_train.shape),
        "Z_test_shape": list(Z_test.shape),
        "y_true_len": int(len(y_true)),
        "y_pred_len": int(len(y_pred)),
        "loaded_from_cache": cache_meta.get("loaded_from_cache", False),
    }

    return row, debug


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--qrc-config", type=str, required=True)
    parser.add_argument("--readout", type=str, required=True, choices=sorted(ALL_READOUTS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-energy", action="store_true")
    parser.add_argument("--force-recache", action="store_true")
    parser.add_argument("--cache-batch-size", type=int, default=128)

    args = parser.parse_args()

    row, debug = run_qrc_experiment(
        dataset=args.dataset,
        qrc_config_name=args.qrc_config,
        readout=args.readout,
        seed=args.seed,
        track_energy=not args.no_energy,
        force_recache=args.force_recache,
        cache_batch_size=args.cache_batch_size,
    )

    output_dir = PROJECT_ROOT / "results" / "raw_logs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{args.dataset}_{args.qrc_config}_{args.readout}_seed{args.seed}.json"

    with open(output_path, "w") as f:
        json.dump(row.to_dict(), f, indent=2, default=str)

    print(json.dumps(row.to_dict(), indent=2, default=str))
    print(f"\nDebug: {debug}")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()