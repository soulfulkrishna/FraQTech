import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import subprocess

import yaml

from src.datasets.narma import generate_narma, split_series
from src.metrics.forecasting import regression_metrics
from src.models_classical.persistence import PersistenceModel
from src.training.result_schema import ResultRow


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    config_path = Path("configs") / "datasets" / "narma10.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = 0

    u, y = generate_narma(
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

    splits = split_series(
        x=u,
        y=y,
        train_frac=cfg["split"]["train"],
        val_frac=cfg["split"]["val"],
    )

    model = PersistenceModel()

    train_start = time.perf_counter()
    model.fit(splits["train"]["y"])
    train_time = time.perf_counter() - train_start

    test_y = splits["test"]["y"]

    infer_start = time.perf_counter()
    y_pred = model.predict(test_y)
    infer_time = time.perf_counter() - infer_start

    y_true = test_y[1:]

    metrics = regression_metrics(y_true, y_pred)

    latency_ms = 1000.0 * infer_time / len(y_pred)

    row = ResultRow(
        dataset="narma10",
        model="persistence",
        encoder_type="none",
        readout_type="none",
        seed=seed,
        split_id="60_20_20",

        rmse=metrics["rmse"],
        nrmse_sigma=metrics["nrmse_sigma"],
        mae=metrics["mae"],

        qrc_cache_time_sec=0.0,
        final_train_time_sec=train_time,
        hpo_time_sec=0.0,
        inference_latency_ms_per_sample=latency_ms,

        energy_kwh_cache=0.0,
        energy_kwh_final_train=0.0,
        energy_kwh_hpo=0.0,
        energy_kwh_total=0.0,

        carbon_kgco2e_cache=0.0,
        carbon_kgco2e_final_train=0.0,
        carbon_kgco2e_hpo=0.0,
        carbon_kgco2e_total=0.0,

        peak_ram_gb=0.0,
        peak_gpu_mem_gb=0.0,

        trainable_params=0,
        total_params=0,
        feature_dim=1,

        qubits=None,
        modes=None,
        virtual_nodes=None,
        circuit_depth=None,
        circuit_evals=0,
        shots=0,
        qpu_time_proxy_sec=None,

        backend_type="classical_cpu",
        git_commit=get_git_commit(),
        hardware_id="local_windows",

        extra={
            "note": "First smoke-test run. Energy and memory tracking not yet enabled."
        },
    )

    output_dir = Path("results") / "raw_logs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "narma10_persistence_seed0.json"

    with open(output_path, "w") as f:
        json.dump(row.to_dict(), f, indent=2)

    print(json.dumps(row.to_dict(), indent=2))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()