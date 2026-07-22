import numpy as np
from typing import Dict, Optional


class CVGaussianQRC:
    """
    Lightweight CV/Gaussian QRC simulator.

    This is a moment-based reservoir:
      - mean vector m_t
      - covariance matrix V_t

    It is not a full Hilbert-space simulator. It is intended as a
    sustainable CV/Gaussian reservoir encoder for time-series benchmarks.

    Input:
      scalar sequence x_t

    Output:
      feature sequence z_t with shape (T, feature_dim)
    """

    def __init__(
        self,
        modes: int = 8,
        virtual_nodes: int = 4,
        washout: int = 2,
        spectral_radius: float = 0.99,
        leak_rate: float = 0.1,
        input_scale: float = 0.5,
        noise_scale: float = 0.001,
        include_means: bool = True,
        include_variances: bool = True,
        include_covariances: bool = False,
        include_squared_means: bool = False,
        include_abs_means: bool = False,
        seed: int = 0,
    ) -> None:
        self.modes = int(modes)
        self.virtual_nodes = int(virtual_nodes)
        self.washout = int(washout)

        self.spectral_radius = float(spectral_radius)
        self.leak_rate = float(leak_rate)
        self.input_scale = float(input_scale)
        self.noise_scale = float(noise_scale)

        self.include_means = bool(include_means)
        self.include_variances = bool(include_variances)
        self.include_covariances = bool(include_covariances)
        self.include_squared_means = bool(include_squared_means)
        self.include_abs_means = bool(include_abs_means)

        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)

        self.A = None
        self.B = None
        self.C = None
        self.Q = None

        self.m = None
        self.V = None

        self._initialize_reservoir()

    def _initialize_reservoir(self) -> None:
        A = self.rng.normal(0.0, 1.0, size=(self.modes, self.modes))

        eigvals = np.linalg.eigvals(A)
        radius = np.max(np.abs(eigvals))

        if radius > 0:
            A = A * (self.spectral_radius / radius)

        self.A = A

        self.B = self.rng.normal(
            0.0,
            self.input_scale,
            size=(self.modes,),
        )

        # Extra weak nonlinear modulation vector.
        self.C = self.rng.normal(
            0.0,
            self.input_scale,
            size=(self.modes,),
        )

        self.Q = np.eye(self.modes) * self.noise_scale

        self.reset_state()

    def reset_state(self) -> None:
        self.m = np.zeros(self.modes, dtype=float)
        self.V = np.eye(self.modes, dtype=float)

    def _step_once(self, u: float) -> None:
        u = float(u)

        if not np.isfinite(u):
            u = 0.0

        # Input-dependent bounded drive.
        drive = self.B * u + self.C * np.tanh(u)

        m_candidate = self.A @ self.m + drive

        # Stable covariance update.
        V_candidate = self.A @ self.V @ self.A.T + self.Q
        V_candidate = 0.5 * (V_candidate + V_candidate.T)

        self.m = (1.0 - self.leak_rate) * self.m + self.leak_rate * np.tanh(m_candidate)
        self.V = (1.0 - self.leak_rate) * self.V + self.leak_rate * V_candidate

        # Stabilize covariance numerically.
        diag = np.diag(self.V)
        diag = np.clip(diag, 1e-8, 1e6)
        np.fill_diagonal(self.V, diag)

        self.m = np.nan_to_num(self.m, nan=0.0, posinf=0.0, neginf=0.0)
        self.V = np.nan_to_num(self.V, nan=0.0, posinf=0.0, neginf=0.0)

    def _extract_single_feature(self) -> np.ndarray:
        features = []

        if self.include_means:
            features.append(self.m.copy())

        if self.include_variances:
            features.append(np.diag(self.V).copy())

        if self.include_covariances:
            cov_entries = []
            for i in range(self.modes):
                for j in range(i + 1, self.modes):
                    cov_entries.append(self.V[i, j])
            features.append(np.asarray(cov_entries, dtype=float))

        if self.include_squared_means:
            features.append(self.m ** 2)

        if self.include_abs_means:
            features.append(np.abs(self.m))

        if not features:
            raise ValueError("At least one feature type must be enabled.")

        out = np.concatenate(features)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        return out

    def encode(self, series: np.ndarray) -> np.ndarray:
        """
        Encode a scalar time series into QRC features.

        Returns:
            features: shape (T - washout, virtual_nodes * per_node_feature_dim)
        """
        series = np.asarray(series, dtype=float).reshape(-1)

        self.reset_state()

        all_features = []

        for t, u in enumerate(series):
            substep_features = []

            for _ in range(self.virtual_nodes):
                self._step_once(u)
                substep_features.append(self._extract_single_feature())

            z_t = np.concatenate(substep_features)

            if t >= self.washout:
                all_features.append(z_t)

        if not all_features:
            raise ValueError(
                f"Series length {len(series)} is too short for washout={self.washout}."
            )

        features = np.asarray(all_features, dtype=float)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features

    def feature_dim(self) -> int:
        dummy = self._extract_single_feature()
        return int(dummy.size * self.virtual_nodes)

    def total_params(self) -> int:
        return int(self.A.size + self.B.size + self.C.size + self.Q.size)

    def trainable_params(self) -> int:
        return 0

    def metadata(self) -> Dict[str, Optional[int]]:
        return {
            "modes": self.modes,
            "virtual_nodes": self.virtual_nodes,
            "washout": self.washout,
            "feature_dim": self.feature_dim(),
            "total_params": self.total_params(),
        }