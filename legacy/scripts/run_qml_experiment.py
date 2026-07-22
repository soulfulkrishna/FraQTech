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
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.datasets.windowing import make_supervised_windows, make_test_windows_with_context
from src.energy.resource_tracker import ResourceTracker
from src.metrics.forecasting import regression_metrics
from src.models_qml.qml_models import build_qml_model
from src.training.result_schema import ResultRow

from scripts.run_experiment import load_dataset, standardize, inverse_standardize


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


def load_qml_config(config_name: str) -> Dict:
    path = PROJECT_ROOT / "configs" / "models" / f"{config_name}.yaml"

    if not path.exists():
        raise FileNotFoundError(f"QML config not found: {path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["config_name"] = config_name
    return cfg


def prepare_windows(dataset: str, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
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


def train_model(
    model: torch.nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Dict,
    device: torch.device,
) -> None:
    X = torch.as_tensor(X_train, dtype=torch.float32)
    y = torch.as_tensor(y_train, dtype=torch.float32)

    dataset = TensorDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )

    criterion = nn.MSELoss()

    model.train()

    for _ in range(int(config["epochs"])):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def predict_model(
    model: torch.nn.Module,
    X_test: np.ndarray,
    config: Dict,
    device: torch.device,
) -> np.ndarray:
    X = torch.as_tensor(X_test, dtype=torch.float32)

    dataset = TensorDataset(X)
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    preds = []

    model.eval()

    for (xb,) in loader:
        xb = xb.to(device)
        pred = model(xb).detach().cpu().numpy().reshape(-1)
        preds.append(pred)

    out = np.concatenate(preds)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    return out.astype(float)


def estimate_circuit_evals(config: Dict, n_train: int, n_test: int, lookback: int) -> int:
    model_type = config["model_type"]
    epochs = int(config["epochs"])

    if model_type == "qnn":
        return int(n_train * epochs + n_test)

    if model_type in {"qrnn", "qlstm"}:
        return int((n_train * epochs + n_test) * lookback)

    return 0


def run_qml_experiment(
    dataset: str,
    qml_config_name: str,
    seed: int,
    track_energy: bool,
    device_arg: str,
) -> Tuple[ResultRow, Dict]:
    cfg = load_qml_config(qml_config_name)

    seed_offset = int(cfg.get("seed_offset", 0))
    actual_seed = seed + seed_offset

    torch.manual_seed(actual_seed)
    np.random.seed(actual_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)

    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)

    X_train, y_train, X_test, y_test, meta = prepare_windows(dataset=dataset, seed=seed)

    lookback = int(meta["lookback"])
    mean = float(meta["mean"])
    std = float(meta["std"])
    dataset_cfg = meta["cfg"]

    model = build_qml_model(
        config=cfg,
        lookback=lookback,
        input_dim=int(X_train.shape[-1]),
    ).to(device)

    with ResourceTracker(track_energy=track_energy, project_name=f"qml-{qml_config_name}-train") as train_tracker:
        train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            config=cfg,
            device=device,
        )

    with ResourceTracker(track_energy=track_energy, project_name=f"qml-{qml_config_name}-infer") as infer_tracker:
        y_pred_scaled = predict_model(
            model=model,
            X_test=X_test,
            config=cfg,
            device=device,
        )

    y_true = inverse_standardize(y_test, mean=mean, std=std)
    y_pred = inverse_standardize(y_pred_scaled, mean=mean, std=std)

    metrics = regression_metrics(y_true, y_pred)

    latency_ms = 1000.0 * infer_tracker.elapsed_time_sec / len(y_pred)

    total_energy = train_tracker.energy_kwh + infer_tracker.energy_kwh
    total_carbon = train_tracker.carbon_kgco2e + infer_tracker.carbon_kgco2e

    circuit_evals = estimate_circuit_evals(
        config=cfg,
        n_train=len(X_train),
        n_test=len(X_test),
        lookback=lookback,
    )

    shots_per_eval = int(cfg.get("shots", 0))
    shots_total = int(circuit_evals * shots_per_eval) if shots_per_eval > 0 else 0

    row = ResultRow(
        dataset=dataset,
        model=qml_config_name,
        encoder_type="qml",
        readout_type=cfg["model_type"],
        seed=seed,
        split_id="60_20_20",

        rmse=metrics["rmse"],
        nrmse_sigma=metrics["nrmse_sigma"],
        mae=metrics["mae"],

        qrc_cache_time_sec=0.0,
        final_train_time_sec=train_tracker.elapsed_time_sec,
        hpo_time_sec=0.0,
        inference_latency_ms_per_sample=latency_ms,

        energy_kwh_cache=0.0,
        energy_kwh_final_train=train_tracker.energy_kwh,
        energy_kwh_hpo=0.0,
        energy_kwh_total=total_energy,

        carbon_kgco2e_cache=0.0,
        carbon_kgco2e_final_train=train_tracker.carbon_kgco2e,
        carbon_kgco2e_hpo=0.0,
        carbon_kgco2e_total=total_carbon,

        peak_ram_gb=max(train_tracker.peak_ram_gb, infer_tracker.peak_ram_gb),
        peak_gpu_mem_gb=max(train_tracker.peak_gpu_mem_gb, infer_tracker.peak_gpu_mem_gb),

        trainable_params=int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        total_params=int(sum(p.numel() for p in model.parameters())),
        feature_dim=int(cfg["n_qubits"]),

        qubits=int(cfg["n_qubits"]),
        modes=None,
        virtual_nodes=None,
        circuit_depth=int(model.circuit_depth_proxy()),
        circuit_evals=circuit_evals,
        shots=shots_total,
        qpu_time_proxy_sec=None,

        backend_type=cfg["backend_type"],
        git_commit=get_git_commit(),
        hardware_id="local_windows",

        extra={
            "track_energy": track_energy,
            "device": str(device),
            "qml_config": cfg,
            "dataset_type": dataset_cfg["type"],
            "lookback": lookback,
            "horizon": int(meta["horizon"]),
            "normalization_mean": mean,
            "normalization_std": std,
            "note": "Simulation-only QML baseline.",
        },
    )

    debug = {
        "dataset": dataset,
        "qml_config": qml_config_name,
        "device": str(device),
        "X_train_shape": list(X_train.shape),
        "X_test_shape": list(X_test.shape),
        "y_true_len": int(len(y_true)),
        "y_pred_len": int(len(y_pred)),
        "circuit_evals_proxy": circuit_evals,
    }

    return row, debug


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--qml-config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-energy", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    args = parser.parse_args()

    row, debug = run_qml_experiment(
        dataset=args.dataset,
        qml_config_name=args.qml_config,
        seed=args.seed,
        track_energy=not args.no_energy,
        device_arg=args.device,
    )

    output_dir = PROJECT_ROOT / "results" / "raw_logs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{args.dataset}_{args.qml_config}_seed{args.seed}.json"

    with open(output_path, "w") as f:
        json.dump(row.to_dict(), f, indent=2, default=str)

    print(json.dumps(row.to_dict(), indent=2, default=str))
    print(f"\nDebug: {debug}")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()