import numpy as np
from typing import Dict, Tuple


def _is_valid_sequence(y: np.ndarray, max_abs_value: float = 1e6) -> bool:
    if not np.all(np.isfinite(y)):
        return False
    if np.max(np.abs(y)) > max_abs_value:
        return False
    return True


def generate_narma(
    order: int,
    sequence_length: int,
    discard_transient: int,
    input_low: float,
    input_high: float,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    seed: int,
    max_retries: int = 100,
    max_abs_value: float = 1e6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a stable NARMA-order sequence.

    Some NARMA settings can diverge for particular seeds, especially for
    higher order. We deterministically retry with seed + attempt * 100000.
    """
    total_length = sequence_length + discard_transient + order + 1

    for attempt in range(max_retries):
        actual_seed = seed + attempt * 100_000
        rng = np.random.default_rng(actual_seed)

        u = rng.uniform(input_low, input_high, size=total_length)
        y = np.zeros(total_length, dtype=float)

        valid = True

        for t in range(order, total_length - 1):
            y_sum = np.sum(y[t - order + 1 : t + 1])
            y_next = (
                alpha * y[t]
                + beta * y[t] * y_sum
                + gamma * u[t - order + 1] * u[t]
                + delta
            )

            if not np.isfinite(y_next) or abs(y_next) > max_abs_value:
                valid = False
                break

            y[t + 1] = y_next

        if not valid:
            continue

        start = discard_transient + order
        end = start + sequence_length

        u_out = u[start:end]
        y_out = y[start:end]

        if _is_valid_sequence(y_out, max_abs_value=max_abs_value):
            return u_out, y_out

    raise RuntimeError(
        f"Could not generate stable NARMA-{order} sequence after "
        f"{max_retries} attempts for seed={seed}."
    )


def split_series(
    x: np.ndarray,
    y: np.ndarray,
    train_frac: float,
    val_frac: float,
) -> Dict[str, Dict[str, np.ndarray]]:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    if len(x) != len(y):
        raise ValueError(f"x and y lengths differ: {len(x)} vs {len(y)}")

    if not np.all(np.isfinite(x)):
        raise ValueError("Input sequence x contains non-finite values.")

    if not np.all(np.isfinite(y)):
        raise ValueError("Target sequence y contains non-finite values.")

    n = len(y)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_end = n_train
    val_end = n_train + n_val

    return {
        "train": {"x": x[:train_end], "y": y[:train_end]},
        "val": {"x": x[train_end:val_end], "y": y[train_end:val_end]},
        "test": {"x": x[val_end:], "y": y[val_end:]},
    }