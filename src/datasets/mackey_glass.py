import numpy as np
from typing import Tuple


def generate_mackey_glass(
    sequence_length: int,
    discard_transient: int,
    beta: float,
    gamma: float,
    n: int,
    tau: float,
    dt: float,
    sample_every: int,
    initial_value: float,
    initial_jitter: float,
    noise_std: float,
    seed: int,
    drift_amplitude: float = 0.0,
    drift_period: float = 500.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a harder Mackey-Glass time series.

    Equation:
        dx/dt = beta * x(t - tau) / (1 + x(t - tau)^n) - gamma * x(t)

    Then optionally add:
        additive Gaussian observation noise
        slow sinusoidal drift

    Returns:
        x: same as y for compatibility, shape (T,)
        y: generated series, shape (T,)
    """
    rng = np.random.default_rng(seed)

    delay_steps = int(round(tau / dt))
    total_samples = sequence_length + discard_transient + 1
    total_steps = total_samples * sample_every + delay_steps + 1

    series = np.zeros(total_steps, dtype=float)

    history = initial_value + rng.normal(
        loc=0.0,
        scale=initial_jitter,
        size=delay_steps + 1,
    )
    history = np.maximum(history, 1e-6)

    series[: delay_steps + 1] = history

    for t in range(delay_steps, total_steps - 1):
        x_t = series[t]
        x_tau = series[t - delay_steps]

        dx = beta * x_tau / (1.0 + x_tau ** n) - gamma * x_t
        x_next = x_t + dt * dx

        if not np.isfinite(x_next):
            raise RuntimeError(f"Mackey-Glass produced non-finite value at step {t}.")

        series[t + 1] = x_next

    sampled = series[delay_steps::sample_every]

    start = discard_transient
    end = start + sequence_length

    y = sampled[start:end].astype(float)

    if len(y) != sequence_length:
        raise RuntimeError(f"Expected length {sequence_length}, got {len(y)}.")

    if drift_amplitude != 0.0:
        t = np.arange(len(y), dtype=float)
        y = y + drift_amplitude * np.sin(2.0 * np.pi * t / float(drift_period))

    if noise_std > 0.0:
        y = y + rng.normal(0.0, noise_std, size=len(y))

    if not np.all(np.isfinite(y)):
        raise RuntimeError("Generated Mackey-Glass sequence contains non-finite values.")

    return y.copy(), y.copy()