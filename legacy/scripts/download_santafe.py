from pathlib import Path

import numpy as np
from reservoirpy.datasets import santafe_laser


def main() -> None:
    out_path = Path("data") / "raw" / "santafe_laser.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.asarray(santafe_laser(), dtype=float).reshape(-1)

    if len(data) < 10000:
        raise RuntimeError(f"Santa Fe series too short: got {len(data)} values.")

    np.savetxt(out_path, data, fmt="%.10f")

    print(f"Saved {len(data)} Santa Fe laser values to {out_path}")
    print("First 10 values:")
    print(data[:10])


if __name__ == "__main__":
    main()