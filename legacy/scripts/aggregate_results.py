import json
from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "results" / "raw_logs"
    out_dir = root / "results" / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for path in raw_dir.glob("*.json"):
        with open(path, "r") as f:
            row = json.load(f)
        row["_source_file"] = path.name
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No JSON files found in {raw_dir}")

    df = pd.DataFrame(rows)

    preferred_cols = [
        "dataset",
        "model",
        "encoder_type",
        "readout_type",
        "seed",

        "rmse",
        "nrmse_sigma",
        "mae",

        "qrc_cache_time_sec",
        "final_train_time_sec",
        "hpo_time_sec",
        "inference_latency_ms_per_sample",

        "energy_kwh_cache",
        "energy_kwh_final_train",
        "energy_kwh_hpo",
        "energy_kwh_total",

        "carbon_kgco2e_cache",
        "carbon_kgco2e_final_train",
        "carbon_kgco2e_hpo",
        "carbon_kgco2e_total",

        "peak_ram_gb",
        "peak_gpu_mem_gb",

        "trainable_params",
        "total_params",
        "feature_dim",

        "qubits",
        "modes",
        "virtual_nodes",
        "circuit_depth",
        "circuit_evals",
        "shots",
        "qpu_time_proxy_sec",

        "backend_type",
        "git_commit",
        "hardware_id",
        "_source_file",
    ]

    existing_cols = [c for c in preferred_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_cols]
    df = df[existing_cols + other_cols]

    csv_path = out_dir / "all_results.csv"
    df.to_csv(csv_path, index=False)

    summary = (
        df.groupby(["dataset", "model"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),

            nrmse_mean=("nrmse_sigma", "mean"),
            nrmse_std=("nrmse_sigma", "std"),

            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),

            qrc_cache_time_mean=("qrc_cache_time_sec", "mean"),
            train_time_mean=("final_train_time_sec", "mean"),
            hpo_time_mean=("hpo_time_sec", "mean"),
            inference_latency_ms_mean=("inference_latency_ms_per_sample", "mean"),

            energy_kwh_cache_mean=("energy_kwh_cache", "mean"),
            energy_kwh_train_mean=("energy_kwh_final_train", "mean"),
            energy_kwh_hpo_mean=("energy_kwh_hpo", "mean"),
            energy_kwh_total_mean=("energy_kwh_total", "mean"),

            carbon_kgco2e_cache_mean=("carbon_kgco2e_cache", "mean"),
            carbon_kgco2e_train_mean=("carbon_kgco2e_final_train", "mean"),
            carbon_kgco2e_hpo_mean=("carbon_kgco2e_hpo", "mean"),
            carbon_kgco2e_total_mean=("carbon_kgco2e_total", "mean"),

            peak_ram_gb_mean=("peak_ram_gb", "mean"),
            peak_gpu_mem_gb_mean=("peak_gpu_mem_gb", "mean"),

            trainable_params_mean=("trainable_params", "mean"),
            total_params_mean=("total_params", "mean"),
            feature_dim_mean=("feature_dim", "mean"),

            qubits_mean=("qubits", "mean"),
            modes_mean=("modes", "mean"),
            virtual_nodes_mean=("virtual_nodes", "mean"),
            circuit_depth_mean=("circuit_depth", "mean"),
            circuit_evals_mean=("circuit_evals", "mean"),
            shots_mean=("shots", "mean"),
            qpu_time_proxy_sec_mean=("qpu_time_proxy_sec", "mean"),
        )
        .sort_values(["dataset", "nrmse_mean"])
    )

    summary_path = out_dir / "summary_by_model.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Saved: {csv_path}")
    print(f"Saved: {summary_path}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()