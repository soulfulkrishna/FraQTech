import numpy as np
from typing import Tuple, Sequence


def _lorenz_rhs(state: np.ndarray, sigma: float, rho: float, beta: float) -> np.ndarray:
    x, y, z = state
    return np.array(
        [
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z,
        ],
        dtype=float,
    )


def _rk4_step(state: np.ndarray, dt: float, sigma: float, rho: float, beta: float) -> np.ndarray:
    k1 = _lorenz_rhs(state, sigma, rho, beta)
    k2 = _lorenz_rhs(state + 0.5 * dt * k1, sigma, rho, beta)
    k3 = _lorenz_rhs(state + 0.5 * dt * k2, sigma, rho, beta)
    k4 = _lorenz_rhs(state + dt * k3, sigma, rho, beta)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def generate_lorenz63(
    sequence_length: int,
    discard_transient: int,
    sigma: float,
    rho: float,
    beta: float,
    dt: float,
    sample_every: int,
    initial_state: Sequence[float],
    initial_jitter: float,
    target_variable: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Lorenz-63 and return a scalar target series.

    target_variable: one of {"x", "y", "z"}.
    """
    rng = np.random.default_rng(seed)

    state = np.asarray(initial_state, dtype=float).reshape(3)
    state = state + rng.normal(0.0, initial_jitter, size=3)

    total_samples = sequence_length + discard_transient
    total_steps = total_samples * sample_every + 1

    sampled = []

    for step in range(total_steps):
        if step % sample_every == 0:
            sampled.append(state.copy())

        state = _rk4_step(state, dt=dt, sigma=sigma, rho=rho, beta=beta)

        if not np.all(np.isfinite(state)):
            raise RuntimeError(f"Lorenz-63 produced non-finite state at step {step}.")

    arr = np.asarray(sampled, dtype=float)

    start = discard_transient
    end = start + sequence_length
    arr = arr[start:end]

    if target_variable == "x":
        y = arr[:, 0]
    elif target_variable == "y":
        y = arr[:, 1]
    elif target_variable == "z":
        y = arr[:, 2]
    else:
        raise ValueError(f"Unknown target_variable: {target_variable}")

    if len(y) != sequence_length:
        raise RuntimeError(f"Expected length {sequence_length}, got {len(y)}.")

    return y.copy(), y.copy()