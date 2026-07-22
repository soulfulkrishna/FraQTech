from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase3.data import load_phase3_finance_dataset
from src.phase3.metrics import classification_metrics, regression_metrics
from src.phase3.qrc_parameters import TemporalIsingQRCConfig
from src.phase3.temporal_ising_qrc import encode_exact


def main() -> None:
    rng = np.random.default_rng(7)
    n = 360
    value = np.abs(0.02 + 0.005 * np.sin(np.arange(n) / 11.0) + rng.normal(0, 0.001, n))
    target = np.roll(value, -1)
    target[-1] = target[-2]
    threshold = pd.Series(value).rolling(60, min_periods=20).quantile(0.8).bfill().to_numpy()
    regime = (target > np.roll(threshold, -1)).astype(int)
    split = np.array(["train"] * 216 + ["val"] * 72 + ["test"] * 72)
    frame = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="D").astype(str),
        "value": value,
        "target": target,
        "target_regime": regime,
        "target_regime_threshold": np.roll(threshold, -1),
        "enter_high_regime": regime,
        "split": split,
    })
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "tiny.csv"
        frame.to_csv(path, index=False)
        data = load_phase3_finance_dataset(path, lookback=20)
        cfg = TemporalIsingQRCConfig(
            n_qubits=3, temporal_bins=4, virtual_nodes=1, reservoir_layers=1,
            topology="ring", include_z=True, include_zz=True, include_global_features=True,
        )
        Z_train, meta = encode_exact(data.train.X[:24], cfg, seed=0, batch_size=8, device="cpu")
        Z_test, _ = encode_exact(data.test.X[:12], cfg, seed=0, batch_size=4, device="cpu")
        model = Ridge(alpha=1e-3).fit(Z_train.reshape(len(Z_train), -1), data.train.y[:24])
        pred = model.predict(Z_test.reshape(len(Z_test), -1))
        y_true = data.scaler.inverse_transform(data.test.y[:12])
        y_pred = data.scaler.inverse_transform(pred)
        reg = regression_metrics(y_true, np.maximum(y_pred, 1e-8))
        cls = classification_metrics(np.array([0, 1, 0, 1]), np.array([0.1, 0.8, 0.3, 0.7]))
        assert Z_train.shape[0] == 24
        assert np.isfinite(reg["nrmse_sigma"])
        assert cls["macro_f1"] > 0.9
        print(json.dumps({
            "status": "PASS",
            "train_shape": list(Z_train.shape),
            "test_shape": list(Z_test.shape),
            "feature_dim": meta["flattened_feature_dim"],
            "metrics": reg,
        }, indent=2))


if __name__ == "__main__":
    main()
