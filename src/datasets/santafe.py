from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def load_santafe_series(
    file_path: str,
    sequence_length: int,
    discard_transient: int,
    value_column: Optional[int],
    skip_header: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Santa Fe laser data from a local text/CSV file.

    Expected formats:
      1. one scalar value per line
      2. CSV/whitespace table with the target value in value_column
      3. if value_column is null and table has multiple columns, use the last column
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Santa Fe data file not found: {path}\n"
            "Place the Santa Fe laser series at data/raw/santafe_laser.txt "
            "with one scalar value per line or CSV row."
        )

    try:
        data = np.loadtxt(path, delimiter=",", skiprows=skip_header)
    except Exception:
        data = np.loadtxt(path, skiprows=skip_header)

    data = np.asarray(data, dtype=float)

    if data.ndim == 1:
        series = data
    elif data.ndim == 2:
        if value_column is None:
            series = data[:, -1]
        else:
            series = data[:, int(value_column)]
    else:
        raise ValueError(f"Unsupported Santa Fe data shape: {data.shape}")

    series = series.reshape(-1)

    start = discard_transient
    end = start + sequence_length

    if len(series) < end:
        raise ValueError(
            f"Santa Fe file contains only {len(series)} values, "
            f"but need at least {end}."
        )

    y = series[start:end]

    if not np.all(np.isfinite(y)):
        raise ValueError("Santa Fe series contains non-finite values.")

    return y.copy(), y.copy()