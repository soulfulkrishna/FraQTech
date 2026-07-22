from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


EPS = 1e-10


def qlike_volatility(y_true_vol: np.ndarray, y_pred_vol: np.ndarray) -> float:
    """QLIKE on variance when inputs are volatility (standard-deviation) estimates."""
    y_true_var = np.maximum(np.asarray(y_true_vol, dtype=float).reshape(-1) ** 2, EPS)
    y_pred_var = np.maximum(np.asarray(y_pred_vol, dtype=float).reshape(-1) ** 2, EPS)
    return float(np.mean(np.log(y_pred_var) + y_true_var / y_pred_var))


def mincer_zarnowitz(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    x = np.asarray(y_pred, dtype=float).reshape(-1)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    n, k = X.shape
    sigma2 = float(residuals @ residuals / max(n - k, 1))
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    restrictions = np.array([beta[0] - 0.0, beta[1] - 1.0])
    r_cov = cov
    stat = float(restrictions.T @ np.linalg.pinv(r_cov) @ restrictions / 2.0)
    pvalue = float(1.0 - stats.f.cdf(stat, 2, max(n - k, 1)))
    return {
        "mz_intercept": float(beta[0]),
        "mz_slope": float(beta[1]),
        "mz_joint_f": stat,
        "mz_joint_pvalue": pvalue,
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        raise ValueError("Metrics require finite arrays")
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    sigma = float(np.std(y_true))
    out = {
        "rmse": rmse,
        "nrmse_sigma": rmse / sigma if sigma > EPS else float("nan"),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "qlike": qlike_volatility(y_true, np.maximum(y_pred, EPS)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    out.update(mincer_zarnowitz(y_true, y_pred))
    return out


def classification_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    transition_mask: np.ndarray | None = None,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probability = np.clip(np.asarray(probability, dtype=float).reshape(-1), 0.0, 1.0)
    pred = (probability >= 0.5).astype(int)
    out = {
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "brier": float(brier_score_loss(y_true, probability)),
    }
    try:
        out["auroc"] = float(roc_auc_score(y_true, probability))
    except ValueError:
        out["auroc"] = float("nan")
    if transition_mask is not None:
        transition_mask = np.asarray(transition_mask, dtype=int).reshape(-1) == 1
        if transition_mask.any():
            out["transition_recall"] = float(np.mean(pred[transition_mask] == 1))
        else:
            out["transition_recall"] = float("nan")
    return out


def diebold_mariano(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    loss: str = "squared",
    horizon: int = 1,
) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    a = np.asarray(pred_a, dtype=float).reshape(-1)
    b = np.asarray(pred_b, dtype=float).reshape(-1)
    if loss == "squared":
        d = (y - a) ** 2 - (y - b) ** 2
    elif loss == "absolute":
        d = np.abs(y - a) - np.abs(y - b)
    elif loss == "qlike":
        ya = np.maximum(y**2, EPS)
        va = np.maximum(a**2, EPS)
        vb = np.maximum(b**2, EPS)
        d = (np.log(va) + ya / va) - (np.log(vb) + ya / vb)
    else:
        raise ValueError(f"Unknown loss={loss}")
    n = len(d)
    mean_d = float(np.mean(d))
    centered = d - mean_d
    gamma0 = float(np.dot(centered, centered) / n)
    long_run = gamma0
    for lag in range(1, max(horizon, 1)):
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
        long_run += 2.0 * gamma
    variance = max(long_run / n, EPS)
    statistic = mean_d / np.sqrt(variance)
    pvalue = 2.0 * (1.0 - stats.norm.cdf(abs(statistic)))
    return {"dm_statistic": float(statistic), "dm_pvalue": float(pvalue), "mean_loss_difference": mean_d}
