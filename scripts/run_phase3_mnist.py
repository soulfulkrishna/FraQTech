from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase3.config import load_qrc_config
from src.phase3.results import Phase3Result, save_result
from src.phase3.temporal_ising_qrc import encode_exact
from src.phase3.utils import git_commit, set_global_seed


def load_mnist(train_size: int, test_size: int, seed: int):
    try:
        from torchvision.datasets import MNIST
    except ImportError as exc:
        raise RuntimeError("Install torchvision to run MNIST") from exc
    root = PROJECT_ROOT / "data" / "mnist"
    train = MNIST(root=root, train=True, download=True)
    test = MNIST(root=root, train=False, download=True)
    rng = np.random.default_rng(seed)
    tr_idx = rng.choice(len(train), size=min(train_size, len(train)), replace=False)
    te_idx = rng.choice(len(test), size=min(test_size, len(test)), replace=False)

    def prepare(dataset, indices):
        images = dataset.data[indices].numpy().astype(np.float32) / 255.0
        labels = dataset.targets[indices].numpy().astype(int)
        # Deterministic 28x28 -> 8x8 average pooling. Each row is one temporal input step.
        edges = np.linspace(0, 28, 9, dtype=int)
        pooled = np.empty((len(images), 8, 8), dtype=np.float32)
        for i in range(8):
            for j in range(8):
                pooled[:, i, j] = images[:, edges[i]:edges[i+1], edges[j]:edges[j+1]].mean(axis=(1, 2))
        return (2.0 * pooled - 1.0).astype(np.float32), labels

    return (*prepare(train, tr_idx), *prepare(test, te_idx))


def fit_logistic_selected(X_train, y_train, X_val, y_val, C_values):
    best_C, best_score = None, -np.inf
    for C in C_values:
        clf = LogisticRegression(C=float(C), max_iter=3000, solver="lbfgs").fit(X_train, y_train)
        score = f1_score(y_val, clf.predict(X_val), average="macro")
        if score > best_score:
            best_C, best_score = float(C), float(score)
    final = LogisticRegression(C=best_C, max_iter=3000, solver="lbfgs").fit(
        np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val])
    )
    return final, best_C, best_score


def save_classification_result(model_name, seed, metrics, runtime, resources, configuration, notes, y_test, pred):
    result = Phase3Result(
        task="mnist_qrc_expressivity",
        dataset="MNIST_8x8_seeded_subset",
        model=model_name,
        seed=seed,
        split="test",
        metrics=metrics,
        runtime=runtime,
        resources=resources,
        configuration=configuration,
        notes=notes,
        git_commit=git_commit(),
    )
    paths = save_result(
        result, y_true=y_test.astype(float), y_pred=pred.astype(float),
        timestamps=np.asarray([f"mnist_test_{i}" for i in range(len(y_test))]),
    )
    return {"result": result.to_dict(), "paths": paths}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrc-config", default="configs/gate_qrc_5q.yaml")
    parser.add_argument("--qubits", type=int, default=5)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()
    set_global_seed(args.seed)

    base = load_qrc_config(PROJECT_ROOT / args.qrc_config)
    cfg = replace(
        base, n_qubits=args.qubits, input_dim=8, temporal_bins=8,
        virtual_nodes=1, washout_bins=0,
    )
    X_train_all, y_train_all, X_test, y_test = load_mnist(
        args.train_size, args.test_size, args.seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all, test_size=0.2, random_state=args.seed, stratify=y_train_all
    )
    C_values = [0.01, 0.1, 1.0, 10.0]
    outputs = {}

    # Main QRC result.
    start = time.perf_counter()
    Z_train, meta = encode_exact(X_train, cfg, args.seed, args.batch_size, args.device)
    Z_val, _ = encode_exact(X_val, cfg, args.seed, args.batch_size, args.device)
    Z_test, _ = encode_exact(X_test, cfg, args.seed, args.batch_size, args.device)
    encode_sec = time.perf_counter() - start
    Z_train = Z_train.reshape(len(Z_train), -1)
    Z_val = Z_val.reshape(len(Z_val), -1)
    Z_test = Z_test.reshape(len(Z_test), -1)
    clf, best_C, best_score = fit_logistic_selected(Z_train, y_train, Z_val, y_val, C_values)
    pred = clf.predict(Z_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "validation_macro_f1": float(best_score),
    }
    outputs["qrc"] = save_classification_result(
        f"temporal_ising_qrc_{args.qubits}q_logistic", args.seed, metrics,
        {"feature_encoding_sec": encode_sec},
        {
            "qubits": args.qubits, "shots": 0, "backend": "custom_exact_statevector",
            "logical_depth_proxy": meta["logical_depth_proxy_full_window"],
            "logical_gate_count": meta["logical_gate_count_full_window"],
            "flattened_feature_dim": meta["flattened_feature_dim"],
        },
        {"qrc": cfg.__dict__, "classifier_C": best_C},
        {"train_size": len(X_train_all), "test_size": len(X_test), "quantum_parameters_trainable": 0},
        y_test, pred,
    )

    if not args.skip_baselines:
        raw_train = X_train.reshape(len(X_train), -1)
        raw_val = X_val.reshape(len(X_val), -1)
        raw_test = X_test.reshape(len(X_test), -1)
        start = time.perf_counter()
        raw_clf, raw_C, raw_val_f1 = fit_logistic_selected(
            raw_train, y_train, raw_val, y_val, C_values
        )
        raw_fit_sec = time.perf_counter() - start
        raw_pred = raw_clf.predict(raw_test)
        outputs["raw_logistic"] = save_classification_result(
            "raw_8x8_logistic", args.seed,
            {
                "accuracy": float(accuracy_score(y_test, raw_pred)),
                "macro_f1": float(f1_score(y_test, raw_pred, average="macro")),
                "validation_macro_f1": raw_val_f1,
            },
            {"fit_and_inference_sec": raw_fit_sec},
            {"feature_dim": raw_train.shape[1], "backend": "classical_cpu", "shots": 0},
            {"classifier_C": raw_C},
            {"matched_data_subset": True}, y_test, raw_pred,
        )

        # Matched-dimensional fixed random nonlinear feature map: controls for dimension.
        rng = np.random.default_rng(args.seed + 90000 + args.qubits)
        target_dim = Z_train.shape[1]
        W = rng.normal(0.0, 1.0 / np.sqrt(raw_train.shape[1]), size=(raw_train.shape[1], target_dim))
        b = rng.uniform(-np.pi, np.pi, size=target_dim)
        rf_train = np.tanh(raw_train @ W + b)
        rf_val = np.tanh(raw_val @ W + b)
        rf_test = np.tanh(raw_test @ W + b)
        start = time.perf_counter()
        rf_clf, rf_C, rf_val_f1 = fit_logistic_selected(
            rf_train, y_train, rf_val, y_val, C_values
        )
        rf_fit_sec = time.perf_counter() - start
        rf_pred = rf_clf.predict(rf_test)
        outputs["random_features"] = save_classification_result(
            f"matched_random_features_{target_dim}d_logistic", args.seed,
            {
                "accuracy": float(accuracy_score(y_test, rf_pred)),
                "macro_f1": float(f1_score(y_test, rf_pred, average="macro")),
                "validation_macro_f1": rf_val_f1,
            },
            {"fit_and_inference_sec": rf_fit_sec},
            {"feature_dim": target_dim, "backend": "classical_cpu", "shots": 0},
            {"classifier_C": rf_C, "random_feature_seed": args.seed + 90000 + args.qubits},
            {"matched_qrc_feature_dimension": True}, y_test, rf_pred,
        )

    print(json.dumps(outputs, indent=2, default=str))


if __name__ == "__main__":
    main()
