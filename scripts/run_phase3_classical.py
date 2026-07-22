from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models_classical.torch_sequence import TorchSequenceRegressor
from src.phase3.baselines import (
    ESNWindowRegressor,
    HARRVRegressor,
    PersistenceWindowRegressor,
    RidgeWindowRegressor,
    fit_garch_forecast,
)
from src.phase3.config import load_yaml
from src.phase3.data import concatenate_splits, load_phase3_finance_dataset
from src.phase3.metrics import regression_metrics
from src.phase3.pipeline import inverse_predictions, score_scaled_predictions
from src.phase3.results import Phase3Result, save_result
from src.phase3.utils import git_commit, set_global_seed


def select_alpha(dataset, model_cls, alphas):
    best = None
    for alpha in alphas:
        model = model_cls(alpha=float(alpha)).fit(dataset.train.X, dataset.train.y)
        metrics = score_scaled_predictions(dataset, dataset.val, model.predict(dataset.val.X))
        if best is None or metrics["qlike"] < best[0]["qlike"]:
            best = (metrics, float(alpha))
    return best


def select_esn(dataset, seed: int, cfg: Dict, profile: str):
    sizes = cfg["reservoir_sizes"] if profile == "full" else cfg["reservoir_sizes"][:2]
    radii = cfg["spectral_radii"] if profile == "full" else [0.9]
    leaks = cfg["leak_rates"] if profile == "full" else [0.3]
    alphas = cfg["ridge_alphas"] if profile == "full" else [0.001]
    best = None
    for n, radius, leak, alpha in itertools.product(sizes, radii, leaks, alphas):
        params = dict(
            n_reservoir=int(n), spectral_radius=float(radius), leak_rate=float(leak),
            ridge_alpha=float(alpha), seed=seed,
        )
        model = ESNWindowRegressor(**params).fit(dataset.train.X, dataset.train.y)
        metrics = score_scaled_predictions(dataset, dataset.val, model.predict(dataset.val.X))
        if best is None or metrics["qlike"] < best[0]["qlike"]:
            best = (metrics, params)
    return best


def make_tcn(cfg: Dict, lookback: int, seed: int):
    return TorchSequenceRegressor(
        model_type="tcn",
        input_dim=1,
        lookback=lookback,
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]),
        dropout=float(cfg["dropout"]),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        batch_size=int(cfg["batch_size"]),
        epochs=int(cfg["epochs"]),
        seed=seed,
    )


def run_model(name: str, dataset, config: Dict, seed: int, profile: str):
    train_val = concatenate_splits(dataset.train, dataset.val)
    validation_metrics = {}
    selected = {}
    start = time.perf_counter()

    if name == "persistence":
        model = PersistenceWindowRegressor().fit(train_val.X, train_val.y)
    elif name == "ridge_lag":
        validation_metrics, alpha = select_alpha(
            dataset, RidgeWindowRegressor, config["classical_models"]["ridge_lag"]["alphas"]
        )
        selected = {"alpha": alpha}
        model = RidgeWindowRegressor(alpha=alpha).fit(train_val.X, train_val.y)
    elif name == "har_rv":
        validation_metrics, alpha = select_alpha(
            dataset, HARRVRegressor, config["classical_models"]["har_rv"]["alphas"]
        )
        selected = {"alpha": alpha}
        model = HARRVRegressor(alpha=alpha).fit(train_val.X, train_val.y)
    elif name == "esn":
        validation_metrics, params = select_esn(
            dataset, seed, config["classical_models"]["esn"], profile
        )
        selected = params
        model = ESNWindowRegressor(**params).fit(train_val.X, train_val.y)
    elif name == "tcn":
        selected = config["classical_models"]["tcn"]
        probe = make_tcn(selected, dataset.lookback, seed).fit(dataset.train.X, dataset.train.y)
        validation_metrics = score_scaled_predictions(dataset, dataset.val, probe.predict(dataset.val.X))
        model = make_tcn(selected, dataset.lookback, seed).fit(train_val.X, train_val.y)
    else:
        raise ValueError(f"Unknown model={name}")

    train_time = time.perf_counter() - start
    infer_start = time.perf_counter()
    pred_scaled = np.asarray(model.predict(dataset.test.X), dtype=float).reshape(-1)
    infer_time = time.perf_counter() - infer_start
    y_true = inverse_predictions(dataset, dataset.test.y)
    y_pred = np.maximum(inverse_predictions(dataset, pred_scaled), 1e-10)
    metrics = regression_metrics(y_true, y_pred)
    result = Phase3Result(
        task="vix_volatility_regression",
        dataset=config["data"]["dataset_name"],
        model=name,
        seed=seed,
        split="test",
        metrics=metrics,
        runtime={
            "train_sec": train_time,
            "inference_sec": infer_time,
            "latency_ms_per_sample": 1000.0 * infer_time / len(y_pred),
        },
        configuration={"selected": selected, "profile": profile},
        notes={"validation_metrics": validation_metrics},
        git_commit=git_commit(),
    )
    return save_result(result, y_true, y_pred, dataset.test.timestamps)


def run_garch(dataset, config: Dict, seed: int):
    """Rolling one-step GARCH with scale mapping estimated on validation only."""
    csv_path = PROJECT_ROOT / config["data"]["csv_path"]
    frame = pd.read_csv(csv_path)
    frame["split"] = frame["split"].astype(str).str.lower()
    returns = pd.to_numeric(frame["log_return"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(returns)
    returns = returns[valid]

    val_rows = int((frame["split"] == "val").sum())
    test_rows = int((frame["split"] == "test").sum())
    forecast_length = val_rows + test_rows
    if forecast_length >= len(returns):
        raise ValueError("Insufficient pre-validation history for GARCH")

    start = time.perf_counter()
    rolling_pred = fit_garch_forecast(
        returns,
        test_length=forecast_length,
        p=int(config["classical_models"]["garch"]["p"]),
        q=int(config["classical_models"]["garch"]["q"]),
        distribution=str(config["classical_models"]["garch"]["distribution"]),
    )
    elapsed = time.perf_counter() - start

    val_pred_daily = rolling_pred[:val_rows][-len(dataset.val.y):]
    test_pred_daily = rolling_pred[val_rows:][-len(dataset.test.y):]
    y_val = inverse_predictions(dataset, dataset.val.y)
    y_true = inverse_predictions(dataset, dataset.test.y)

    # GARCH returns daily conditional sigma; map it to the 5-day rolling-RV target
    # using the validation split only. No test targets enter model selection/calibration.
    scale = float(np.median(y_val) / max(np.median(val_pred_daily), 1e-10))
    y_pred = np.maximum(test_pred_daily * scale, 1e-10)
    metrics = regression_metrics(y_true, y_pred)
    validation_metrics = regression_metrics(
        y_val, np.maximum(val_pred_daily * scale, 1e-10)
    )
    result = Phase3Result(
        task="vix_volatility_regression",
        dataset=config["data"]["dataset_name"],
        model="garch_1_1",
        seed=seed,
        split="test",
        metrics=metrics,
        runtime={
            "train_and_rolling_inference_sec": elapsed,
            "latency_ms_per_forecast": 1000 * elapsed / forecast_length,
        },
        configuration=config["classical_models"]["garch"],
        notes={
            "scale_to_rolling_rv": scale,
            "scale_fitted_on": "validation_only",
            "validation_metrics": validation_metrics,
            "warning": "GARCH predicts conditional daily volatility; a validation-only scalar mapping aligns it to the five-day rolling target.",
        },
        git_commit=git_commit(),
    )
    return save_result(result, y_true, y_pred, dataset.test.timestamps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3_finance.yaml")
    parser.add_argument("--models", default="persistence,ridge_lag,har_rv,esn,tcn,garch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", choices=["smoke", "full"], default="full")
    args = parser.parse_args()

    config = load_yaml(PROJECT_ROOT / args.config)
    dataset = load_phase3_finance_dataset(
        PROJECT_ROOT / config["data"]["csv_path"],
        lookback=int(config["data"]["lookback"]),
    )
    set_global_seed(args.seed)
    outputs = {}
    for name in [x.strip() for x in args.models.split(",") if x.strip()]:
        print(f"\n=== {name} seed={args.seed} ===", flush=True)
        outputs[name] = run_garch(dataset, config, args.seed) if name == "garch" else run_model(
            name, dataset, config, args.seed, args.profile
        )
        print(outputs[name])
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
