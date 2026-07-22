from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "summaries" / "phase3_model_summary.csv"
OUT = ROOT / "results" / "figures"


def main() -> None:
    if not SUMMARY.exists():
        raise FileNotFoundError("Run scripts/aggregate_phase3.py first")
    df = pd.read_csv(SUMMARY)
    OUT.mkdir(parents=True, exist_ok=True)

    forecast = df[df["task"].astype(str).str.contains("vix_volatility", na=False)].copy()
    if not forecast.empty and "metrics.nrmse_sigma.mean" in forecast:
        forecast = forecast.sort_values("metrics.nrmse_sigma.mean")
        plt.figure(figsize=(8, 4.5))
        plt.bar(forecast["model"], forecast["metrics.nrmse_sigma.mean"])
        plt.ylabel("Test NRMSE")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(OUT / "vix_nrmse_by_model.pdf", bbox_inches="tight")
        plt.savefig(OUT / "vix_nrmse_by_model.png", dpi=200, bbox_inches="tight")
        plt.close()

    scaling = forecast[forecast["model"].astype(str).str.contains("temporal_ising_qrc", na=False)].copy()
    if not scaling.empty and "resources.qubits.mean" in scaling:
        scaling = scaling.sort_values("resources.qubits.mean")
        plt.figure(figsize=(6, 4))
        plt.plot(scaling["resources.qubits.mean"], scaling["metrics.nrmse_sigma.mean"], marker="o")
        plt.xlabel("Qubits")
        plt.ylabel("Test NRMSE")
        plt.tight_layout()
        plt.savefig(OUT / "qrc_qubit_scaling.pdf", bbox_inches="tight")
        plt.savefig(OUT / "qrc_qubit_scaling.png", dpi=200, bbox_inches="tight")
        plt.close()

    noise = df[df["task"] == "vix_noise_study"].copy()
    if not noise.empty:
        noise = noise.sort_values("metrics.nrmse_sigma.mean")
        plt.figure(figsize=(7, 4))
        plt.bar(noise["model"], noise["metrics.nrmse_sigma.mean"])
        plt.ylabel("Subset NRMSE")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(OUT / "qrc_noise_robustness.pdf", bbox_inches="tight")
        plt.savefig(OUT / "qrc_noise_robustness.png", dpi=200, bbox_inches="tight")
        plt.close()

    print(f"Figures saved under {OUT}")


if __name__ == "__main__":
    main()
