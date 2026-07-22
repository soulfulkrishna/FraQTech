import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import subprocess
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml

from src.datasets.narma import generate_narma, split_series
from src.datasets.mackey_glass import generate_mackey_glass
from src.datasets.lorenz63 import generate_lorenz63
from src.datasets.santafe import load_santafe_series
from src.datasets.windowing import make_supervised_windows, make_test_windows_with_context

from src.energy.resource_tracker import ResourceTracker
from src.metrics.forecasting import regression_metrics

from src.models_classical.arima import ARIMAModel
from src.models_classical.esn import ESNModel
from src.models_classical.persistence import PersistenceModel
from src.models_classical.ridge_lag import RidgeLagModel
from src.models_classical.torch_sequence import TorchSequenceRegressor

from src.training.result_schema import ResultRow


WINDOW_MODELS = {
    "ridge_lag",
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


def load_dataset_config(dataset: str) -> Dict:
    config_path = PROJECT_ROOT / "configs" / "datasets" / f"{dataset}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_csv_forecasting_dataset(cfg):
    """
    Load CSV forecasting datasets with columns like:
    value, target, split, horizon, dataset, task_type.

    The existing benchmark code expects each split to be a dict with x and y.
    We use the raw value series for both x and y because the existing windowing
    code creates lookback/horizon targets itself.
    """
    root = Path(__file__).resolve().parents[1]

    csv_path = Path(cfg["path"])
    if not csv_path.is_absolute():
        csv_path = root / csv_path

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    value_col = cfg.get("value_col", "value")
    split_col = cfg.get("split_col", "split")

    if value_col not in df.columns:
        raise ValueError(
            f"value_col={value_col!r} not found in {csv_path}. "
            f"Columns: {list(df.columns)}"
        )

    if split_col not in df.columns:
        raise ValueError(
            f"split_col={split_col!r} not found in {csv_path}. "
            f"Columns: {list(df.columns)}"
        )

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df[split_col] = df[split_col].astype(str).str.lower()
    df = df.dropna(subset=[value_col, split_col]).reset_index(drop=True)

    splits = {}

    for split_name in ["train", "val", "test"]:
        part = df[df[split_col] == split_name]

        if split_name == "val" and part.empty:
            continue

        if part.empty:
            raise ValueError(f"No rows found for split={split_name!r} in {csv_path}")

        values = part[value_col].to_numpy(dtype=np.float32).reshape(-1)

        splits[split_name] = {
            "x": values.reshape(-1, 1),
            "y": values,
        }

    return splits

def load_dataset(dataset: str, seed: int) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict]:
    cfg = load_dataset_config(dataset)
    dtype = cfg["type"]

    if dtype == "synthetic_narma":
        x, y = generate_narma(
            order=cfg["order"],
            sequence_length=cfg["sequence_length"],
            discard_transient=cfg["discard_transient"],
            input_low=cfg["input_low"],
            input_high=cfg["input_high"],
            alpha=cfg["alpha"],
            beta=cfg["beta"],
            gamma=cfg["gamma"],
            delta=cfg["delta"],
            seed=seed,
        )

    elif dtype == "synthetic_mackey_glass":
        x, y = generate_mackey_glass(
    		sequence_length=cfg["sequence_length"],
    		discard_transient=cfg["discard_transient"],
    		beta=cfg["beta"],
    		gamma=cfg["gamma"],
    		n=cfg["n"],
    		tau=cfg["tau"],
    		dt=cfg["dt"],
    		sample_every=cfg["sample_every"],
    		initial_value=cfg["initial_value"],
    		initial_jitter=cfg["initial_jitter"],
    		noise_std=cfg["noise_std"],
    		seed=seed,
    		drift_amplitude=cfg.get("drift_amplitude", 0.0),
    		drift_period=cfg.get("drift_period", 500.0),
	 )
    elif dtype == "synthetic_lorenz63":
        x, y = generate_lorenz63(
            sequence_length=cfg["sequence_length"],
            discard_transient=cfg["discard_transient"],
            sigma=cfg["sigma"],
            rho=cfg["rho"],
            beta=cfg["beta"],
            dt=cfg["dt"],
            sample_every=cfg["sample_every"],
            initial_state=cfg["initial_state"],
            initial_jitter=cfg["initial_jitter"],
            target_variable=cfg["target_variable"],
            seed=seed,
        )

    elif dtype == "file_santafe":
        x, y = load_santafe_series(
            file_path=str(PROJECT_ROOT / cfg["file_path"]),
            sequence_length=cfg["sequence_length"],
            discard_transient=cfg["discard_transient"],
            value_column=cfg["value_column"],
            skip_header=cfg["skip_header"],
        )
    elif dtype == "csv_forecasting":
        splits = load_csv_forecasting_dataset(cfg)
        return splits, cfg
    else:
        raise ValueError(f"Unsupported dataset type: {dtype}")

    splits = split_series(
        x=x,
        y=y,
        train_frac=cfg["split"]["train"],
        val_frac=cfg["split"]["val"],
    )

    return splits, cfg


def standardize(train_val_y: np.ndarray, test_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    mean = float(np.mean(train_val_y))
    std = float(np.std(train_val_y))

    if std <= 1e-12:
        std = 1.0

    train_val_scaled = (train_val_y - mean) / std
    test_scaled = (test_y - mean) / std

    return train_val_scaled, test_scaled, mean, std


def inverse_standardize(y_scaled: np.ndarray, mean: float, std: float) -> np.ndarray:
    return np.asarray(y_scaled, dtype=float).reshape(-1) * std + mean


def build_torch_model(model_name: str, lookback: int, seed: int) -> TorchSequenceRegressor:
    # Conservative default sizes so full benchmark can run on CPU.
    if model_name == "transformer":
        hidden_dim = 64
        num_layers = 2
        epochs = 20
    elif model_name == "tcn":
        hidden_dim = 64
        num_layers = 3
        epochs = 20
    else:
        hidden_dim = 64
        num_layers = 2
        epochs = 20

    return TorchSequenceRegressor(
        model_type=model_name,
        input_dim=1,
        lookback=lookback,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.1,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=128,
        epochs=epochs,
        seed=seed,
    )


def run_model(
    dataset: str,
    model_name: str,
    seed: int,
    track_energy: bool,
) -> Tuple[ResultRow, Dict[str, object]]:
    splits, cfg = load_dataset(dataset=dataset, seed=seed)

    train_y = splits["train"]["y"]
    val_y = splits["val"]["y"]
    test_y = splits["test"]["y"]

    train_val_y = np.concatenate([train_y, val_y])

    lookback = int(cfg.get("lookback", 64))
    horizon = int(cfg.get("horizon", 1))

    train_val_scaled, test_scaled, mean, std = standardize(train_val_y, test_y)

    cache_time = 0.0
    cache_energy = 0.0
    cache_carbon = 0.0
    hpo_time = 0.0
    hpo_energy = 0.0
    hpo_carbon = 0.0

    if model_name == "persistence":
        model = PersistenceModel()

        with ResourceTracker(track_energy=track_energy) as train_tracker:
            model.fit(train_val_scaled)

        with ResourceTracker(track_energy=track_energy) as infer_tracker:
            y_pred_scaled = model.predict_one_step(
                y_test=test_scaled,
                previous_value=float(train_val_scaled[-1]),
            )
            y_true_scaled = test_scaled

        trainable_params = 0
        total_params = 0
        feature_dim = 1
        readout_type = "none"

    elif model_name == "arima":
        model = ARIMAModel(order=(5, 0, 0))

        with ResourceTracker(track_energy=track_energy) as train_tracker:
            model.fit(train_val_scaled)

        with ResourceTracker(track_energy=track_energy) as infer_tracker:
            y_pred_scaled = model.predict_one_step(test_scaled)
            y_true_scaled = test_scaled

        trainable_params = model.trainable_params()
        total_params = model.total_params()
        feature_dim = 1
        readout_type = "none"

    elif model_name == "esn":
        model = ESNModel(
            n_reservoir=300,
            spectral_radius=0.8,
            leak_rate=0.2,
            input_scale=0.2,
            ridge_alpha=1e-3,
            washout=100,
            seed=seed,
        )

        with ResourceTracker(track_energy=track_energy) as train_tracker:
            model.fit(train_val_scaled)

        with ResourceTracker(track_energy=track_energy) as infer_tracker:
            y_pred_scaled = model.predict_one_step(
                y_test=test_scaled,
                previous_value=float(train_val_scaled[-1]),
            )
            y_true_scaled = test_scaled

        trainable_params = model.trainable_params()
        total_params = model.total_params()
        feature_dim = 1
        readout_type = "linear_readout"

    elif model_name == "ridge_lag":
        X_train, y_train = make_supervised_windows(
            train_val_scaled,
            lookback=lookback,
            horizon=horizon,
        )
        X_test, y_true_scaled = make_test_windows_with_context(
            train_val_scaled,
            test_scaled,
            lookback=lookback,
            horizon=horizon,
        )

        model = RidgeLagModel(alpha=1e-3)

        with ResourceTracker(track_energy=track_energy) as train_tracker:
            model.fit(X_train, y_train)

        with ResourceTracker(track_energy=track_energy) as infer_tracker:
            y_pred_scaled = model.predict(X_test)

        trainable_params = model.trainable_params()
        total_params = model.total_params()
        feature_dim = lookback
        readout_type = "ridge_lag"

    elif model_name in {"mlp", "rnn", "gru", "lstm", "tcn", "transformer"}:
        X_train, y_train = make_supervised_windows(
            train_val_scaled,
            lookback=lookback,
            horizon=horizon,
        )
        X_test, y_true_scaled = make_test_windows_with_context(
            train_val_scaled,
            test_scaled,
            lookback=lookback,
            horizon=horizon,
        )

        model = build_torch_model(model_name=model_name, lookback=lookback, seed=seed)

        with ResourceTracker(track_energy=track_energy) as train_tracker:
            model.fit(X_train, y_train)

        with ResourceTracker(track_energy=track_energy) as infer_tracker:
            y_pred_scaled = model.predict(X_test)

        trainable_params = model.trainable_params()
        total_params = model.total_params()
        feature_dim = lookback
        readout_type = model_name

    else:
        raise ValueError(f"Unknown model: {model_name}")

    y_true = inverse_standardize(y_true_scaled, mean=mean, std=std)
    y_pred = inverse_standardize(y_pred_scaled, mean=mean, std=std)

    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

    metrics = regression_metrics(y_true, y_pred)

    infer_time = infer_tracker.elapsed_time_sec
    latency_ms = 1000.0 * infer_time / len(y_pred)

    total_energy = (
        cache_energy
        + hpo_energy
        + train_tracker.energy_kwh
        + infer_tracker.energy_kwh
    )

    total_carbon = (
        cache_carbon
        + hpo_carbon
        + train_tracker.carbon_kgco2e
        + infer_tracker.carbon_kgco2e
    )

    peak_ram_gb = max(
        train_tracker.peak_ram_gb,
        infer_tracker.peak_ram_gb,
    )

    peak_gpu_mem_gb = max(
        train_tracker.peak_gpu_mem_gb,
        infer_tracker.peak_gpu_mem_gb,
    )

    row = ResultRow(
        dataset=dataset,
        model=model_name,
        encoder_type="none",
        readout_type=readout_type,
        seed=seed,
        split_id="60_20_20",

        rmse=metrics["rmse"],
        nrmse_sigma=metrics["nrmse_sigma"],
        mae=metrics["mae"],

        qrc_cache_time_sec=cache_time,
        final_train_time_sec=train_tracker.elapsed_time_sec,
        hpo_time_sec=hpo_time,
        inference_latency_ms_per_sample=latency_ms,

        energy_kwh_cache=cache_energy,
        energy_kwh_final_train=train_tracker.energy_kwh,
        energy_kwh_hpo=hpo_energy,
        energy_kwh_total=total_energy,

        carbon_kgco2e_cache=cache_carbon,
        carbon_kgco2e_final_train=train_tracker.carbon_kgco2e,
        carbon_kgco2e_hpo=hpo_carbon,
        carbon_kgco2e_total=total_carbon,

        peak_ram_gb=peak_ram_gb,
        peak_gpu_mem_gb=peak_gpu_mem_gb,

        trainable_params=trainable_params,
        total_params=total_params,
        feature_dim=feature_dim,

        qubits=None,
        modes=None,
        virtual_nodes=None,
        circuit_depth=None,
        circuit_evals=0,
        shots=0,
        qpu_time_proxy_sec=None,

        backend_type="classical_cpu_or_gpu",
        git_commit=get_git_commit(),
        hardware_id="local_windows",

        extra={
            "track_energy": track_energy,
            "dataset_type": cfg["type"],
            "lookback": lookback,
            "horizon": horizon,
            "normalization_mean": mean,
            "normalization_std": std,
            "note": "Classical baseline run.",
        },
    )

    debug = {
        "dataset": dataset,
        "model": model_name,
        "y_true_len": int(len(y_true)),
        "y_pred_len": int(len(y_pred)),
        "lookback": lookback,
        "horizon": horizon,
    }

    return row, debug


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-energy", action="store_true")

    args = parser.parse_args()

    track_energy = not args.no_energy

    row, debug = run_model(
        dataset=args.dataset,
        model_name=args.model,
        seed=args.seed,
        track_energy=track_energy,
    )

    output_dir = PROJECT_ROOT / "results" / "raw_logs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{args.dataset}_{args.model}_seed{args.seed}.json"

    with open(output_path, "w") as f:
        json.dump(row.to_dict(), f, indent=2)

    print(json.dumps(row.to_dict(), indent=2))
    print(f"\nDebug: {debug}")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()