import numpy as np
from typing import Tuple


def make_supervised_windows(
    series: np.ndarray,
    lookback: int,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a scalar series to supervised windows.

    X[i] = series[i : i + lookback]
    y[i] = series[i + lookback + horizon - 1]

    Returns:
        X: shape (N, lookback, 1)
        y: shape (N,)
    """
    series = np.asarray(series, dtype=float).reshape(-1)

    if lookback < 1:
        raise ValueError("lookback must be >= 1")

    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    n = len(series) - lookback - horizon + 1

    if n <= 0:
        raise ValueError(
            f"Series too short for lookback={lookback}, horizon={horizon}, len={len(series)}"
        )

    X = np.zeros((n, lookback, 1), dtype=float)
    y = np.zeros(n, dtype=float)

    for i in range(n):
        X[i, :, 0] = series[i : i + lookback]
        y[i] = series[i + lookback + horizon - 1]

    return X, y


def make_test_windows_with_context(
    train_val_series: np.ndarray,
    test_series: np.ndarray,
    lookback: int,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build test windows using the last lookback values from train+val as context.

    For horizon=1, this returns exactly len(test_series) test targets.
    """
    train_val_series = np.asarray(train_val_series, dtype=float).reshape(-1)
    test_series = np.asarray(test_series, dtype=float).reshape(-1)

    if len(train_val_series) < lookback:
        raise ValueError("train_val_series is shorter than lookback.")

    context = np.concatenate([train_val_series[-lookback:], test_series])
    X, y = make_supervised_windows(context, lookback=lookback, horizon=horizon)

    return X, y