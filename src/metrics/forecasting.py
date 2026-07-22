import numpy as np
from typing import Dict


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")

    if not np.all(np.isfinite(y_true)):
        raise ValueError("y_true contains non-finite values.")

    if not np.all(np.isfinite(y_pred)):
        raise ValueError("y_pred contains non-finite values.")

    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))

    sigma = float(np.std(y_true))
    nrmse_sigma = rmse / sigma if sigma > 0 else float("nan")

    return {
        "rmse": rmse,
        "nrmse_sigma": nrmse_sigma,
        "mae": mae,
    }