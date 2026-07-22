from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase3.baselines import ESNWindowRegressor
from src.phase3.config import load_qrc_config, load_yaml
from src.phase3.data import load_phase3_finance_dataset
from src.phase3.metrics import classification_metrics
from src.phase3.pipeline import fit_regime_classifier_final, select_regime_classifier
from src.phase3.results import Phase3Result, save_result
from src.phase3.temporal_ising_qrc import encode_exact
from src.phase3.utils import git_commit, set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-source", choices=["raw", "esn", "qrc"], default="raw")
    parser.add_argument("--finance-config", default="configs/phase3_finance.yaml")
    parser.add_argument("--qrc-config", default="configs/gate_qrc_5q.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = load_yaml(PROJECT_ROOT / args.finance_config)
    dataset = load_phase3_finance_dataset(
        PROJECT_ROOT / cfg["data"]["csv_path"], lookback=int(cfg["data"]["lookback"])
    )
    set_global_seed(args.seed)
    start = time.perf_counter()
    resources = {}
    if args.feature_source == "raw":
        F_train, F_val, F_test = dataset.train.X, dataset.val.X, dataset.test.X
    elif args.feature_source == "esn":
        esn = ESNWindowRegressor(n_reservoir=300, spectral_radius=0.9, leak_rate=0.3, seed=args.seed)
        F_train = esn.transform(dataset.train.X)
        F_val = esn.transform(dataset.val.X)
        F_test = esn.transform(dataset.test.X)
        resources = {"classical_reservoir_nodes": 300}
    else:
        qrc_cfg = load_qrc_config(PROJECT_ROOT / args.qrc_config)
        F_train, meta = encode_exact(dataset.train.X, qrc_cfg, args.seed, batch_size=32, device=args.device)
        F_val, _ = encode_exact(dataset.val.X, qrc_cfg, args.seed, batch_size=32, device=args.device)
        F_test, _ = encode_exact(dataset.test.X, qrc_cfg, args.seed, batch_size=32, device=args.device)
        resources = {
            "qubits": qrc_cfg.n_qubits,
            "logical_depth_proxy": meta["logical_depth_proxy_full_window"],
            "flattened_feature_dim": meta["flattened_feature_dim"],
        }
    feature_sec = time.perf_counter() - start

    best_C, val_metrics = select_regime_classifier(
        F_train, dataset.train.target_regime,
        F_val, dataset.val.target_regime,
        cfg["qrc"]["classification_C"],
    )
    prob = fit_regime_classifier_final(
        F_train, dataset.train.target_regime,
        F_val, dataset.val.target_regime,
        F_test, best_C,
    )
    metrics = classification_metrics(
        dataset.test.target_regime, prob, dataset.test.enter_high_regime
    )
    pred = (prob >= 0.5).astype(int)
    result = Phase3Result(
        task="vix_regime_transition_classification",
        dataset=cfg["data"]["dataset_name"],
        model=f"{args.feature_source}_features_logistic",
        seed=args.seed,
        split="test",
        metrics=metrics,
        runtime={"feature_generation_sec": feature_sec},
        resources=resources,
        configuration={"feature_source": args.feature_source, "classifier_C": best_C},
        notes={"validation_metrics": val_metrics},
        git_commit=git_commit(),
    )
    paths = save_result(
        result,
        y_true=dataset.test.target_regime.astype(float),
        y_pred=pred.astype(float),
        timestamps=dataset.test.timestamps,
        probability=prob,
        target_regime=dataset.test.target_regime,
        transition=dataset.test.enter_high_regime,
    )
    print(json.dumps({"result": result.to_dict(), "paths": paths}, indent=2, default=str))


if __name__ == "__main__":
    main()
