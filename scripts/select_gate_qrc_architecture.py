from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase3.baselines import QRCFeatureReadout
from src.phase3.config import load_qrc_config, load_yaml
from src.phase3.data import ForecastSplit, load_phase3_finance_dataset
from src.phase3.metrics import regression_metrics
from src.phase3.temporal_ising_qrc import encode_exact
from src.phase3.utils import atomic_json_dump, git_commit, set_global_seed


def deterministic_subset(split: ForecastSplit, maximum: int) -> ForecastSplit:
    if maximum <= 0 or len(split.y) <= maximum:
        return split
    idx = np.linspace(0, len(split.y) - 1, maximum, dtype=int)
    return ForecastSplit(
        X=split.X[idx], y=split.y[idx], timestamps=split.timestamps[idx],
        target_regime=split.target_regime[idx] if split.target_regime is not None else None,
        enter_high_regime=split.enter_high_regime[idx] if split.enter_high_regime is not None else None,
        target_regime_threshold=split.target_regime_threshold[idx] if split.target_regime_threshold is not None else None,
    )


def score_candidate(dataset, train, val, cfg, seed: int, alphas, batch_size: int, device: str):
    started = time.perf_counter()
    Z_train, meta = encode_exact(train.X, cfg, seed, batch_size=batch_size, device=device)
    Z_val, _ = encode_exact(val.X, cfg, seed, batch_size=batch_size, device=device)
    encode_sec = time.perf_counter() - started
    y_val = dataset.scaler.inverse_transform(val.y)
    best = None
    for alpha in alphas:
        model = QRCFeatureReadout(alpha=float(alpha)).fit(Z_train, train.y)
        pred = np.maximum(dataset.scaler.inverse_transform(model.predict(Z_val)), 1e-10)
        metrics = regression_metrics(y_val, pred)
        if best is None or metrics["qlike"] < best["metrics"]["qlike"]:
            best = {"alpha": float(alpha), "metrics": metrics}
    assert best is not None
    return {
        **best,
        "encoding_sec": encode_sec,
        "resources": meta,
        "config": cfg.__dict__,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finance-config", default="configs/phase3_finance.yaml")
    parser.add_argument("--base-qrc-config", default="configs/gate_qrc_5q.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", choices=["smoke", "full"], default="full")
    parser.add_argument("--max-train", type=int, default=600)
    parser.add_argument("--max-val", type=int, default=250)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    finance = load_yaml(PROJECT_ROOT / args.finance_config)
    base = load_qrc_config(PROJECT_ROOT / args.base_qrc_config)
    dataset = load_phase3_finance_dataset(
        PROJECT_ROOT / finance["data"]["csv_path"],
        lookback=int(finance["data"]["lookback"]),
    )
    train = deterministic_subset(dataset.train, args.max_train)
    val = deterministic_subset(dataset.val, args.max_val)
    set_global_seed(args.seed)

    if args.profile == "smoke":
        search = {
            "topology": ["ring", "ring_chord"],
            "temporal_bins": [6],
            "virtual_nodes": [1],
            "reservoir_layers": [1],
            "input_reupload": [True],
            "interaction_scale": [0.25, 0.38],
        }
    else:
        search = {
            "topology": ["line", "ring", "ring_chord"],
            "temporal_bins": [4, 6, 8],
            "virtual_nodes": [1, 2],
            "reservoir_layers": [1, 2],
            "input_reupload": [False, True],
            "interaction_scale": [0.20, 0.32, 0.45],
        }

    rows = []
    keys = list(search)
    combos = list(itertools.product(*(search[k] for k in keys)))
    print(f"Architecture candidates: {len(combos)}")
    for i, values in enumerate(combos, start=1):
        updates = dict(zip(keys, values))
        cfg = replace(base, **updates)
        print(f"[{i}/{len(combos)}] {updates}", flush=True)
        try:
            result = score_candidate(
                dataset, train, val, cfg, args.seed,
                finance["qrc"]["ridge_alphas"], args.batch_size, args.device,
            )
            rows.append({
                **updates,
                "alpha": result["alpha"],
                "encoding_sec": result["encoding_sec"],
                **result["metrics"],
                "logical_depth_proxy": result["resources"]["logical_depth_proxy_full_window"],
                "logical_gate_count": result["resources"]["logical_gate_count_full_window"],
                "flattened_feature_dim": result["resources"]["flattened_feature_dim"],
                "status": "ok",
            })
        except Exception as exc:
            rows.append({**updates, "status": "failed", "error": repr(exc)})

    frame = pd.DataFrame(rows)
    out_dir = PROJECT_ROOT / "results" / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"gate_qrc_architecture_search_seed{args.seed}_{args.profile}.csv"
    frame.to_csv(csv_path, index=False)
    valid = frame[frame["status"] == "ok"].sort_values(["qlike", "nrmse_sigma"])
    if valid.empty:
        raise RuntimeError("All architecture candidates failed")
    best = valid.iloc[0].to_dict()
    best_path = out_dir / f"gate_qrc_architecture_best_seed{args.seed}_{args.profile}.json"
    atomic_json_dump(
        {
            "selection_split": "validation",
            "selection_metric": "qlike",
            "seed": args.seed,
            "profile": args.profile,
            "train_samples": len(train.y),
            "validation_samples": len(val.y),
            "best": best,
            "git_commit": git_commit(),
        },
        best_path,
    )
    print(json.dumps({"csv": str(csv_path), "best": best, "best_json": str(best_path)}, indent=2, default=str))


if __name__ == "__main__":
    main()
