import numpy as np


class ESNModel:
    """
    Simple Echo State Network for one-step-ahead forecasting.

    Input:
        y_t

    Target:
        y_{t+1}
    """

    def __init__(
        self,
        n_reservoir: int = 300,
        spectral_radius: float = 0.8,
        leak_rate: float = 0.2,
        input_scale: float = 0.2,
        ridge_alpha: float = 1e-3,
        washout: int = 100,
        seed: int = 0,
        clip_state: float = 1.0,
        clip_prediction: float = 10.0,
    ) -> None:
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.input_scale = input_scale
        self.ridge_alpha = ridge_alpha
        self.washout = washout
        self.seed = seed
        self.clip_state = clip_state
        self.clip_prediction = clip_prediction

        self.rng = np.random.default_rng(seed)

        self.Win = None
        self.W = None
        self.Wout = None
        self.state = None

        self.prediction_guard_count = 0

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        self.Win = self.rng.uniform(
            low=-self.input_scale,
            high=self.input_scale,
            size=(self.n_reservoir, 2),
        )

        W = self.rng.uniform(
            low=-0.5,
            high=0.5,
            size=(self.n_reservoir, self.n_reservoir),
        )

        eigvals = np.linalg.eigvals(W)
        radius = np.max(np.abs(eigvals))

        if radius > 0:
            W = W * (self.spectral_radius / radius)

        self.W = W
        self.state = np.zeros(self.n_reservoir, dtype=float)

    def reset_state(self) -> None:
        self.state = np.zeros(self.n_reservoir, dtype=float)

    def _step(self, u: float) -> np.ndarray:
        u = float(u)

        if not np.isfinite(u):
            u = 0.0

        pre_activation = (
            self.W @ self.state
            + self.Win[:, 0]
            + self.Win[:, 1] * u
        )

        new_state = np.tanh(pre_activation)
        self.state = (1.0 - self.leak_rate) * self.state + self.leak_rate * new_state

        if self.clip_state is not None:
            self.state = np.clip(self.state, -self.clip_state, self.clip_state)

        return self.state.copy()

    def fit(self, y_train: np.ndarray) -> "ESNModel":
        y_train = np.asarray(y_train, dtype=float).reshape(-1)

        if len(y_train) <= self.washout + 2:
            raise ValueError(
                f"Training series too short for washout={self.washout}: "
                f"len(y_train)={len(y_train)}"
            )

        self.reset_state()

        states = []
        targets = []

        for t in range(len(y_train) - 1):
            state_t = self._step(y_train[t])

            if t >= self.washout:
                states.append(np.concatenate([[1.0], state_t]))
                targets.append(y_train[t + 1])

        X = np.asarray(states, dtype=float)
        Y = np.asarray(targets, dtype=float).reshape(-1, 1)

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

        identity = np.eye(X.shape[1], dtype=float)
        identity[0, 0] = 0.0

        lhs = X.T @ X + self.ridge_alpha * identity
        rhs = X.T @ Y

        try:
            self.Wout = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            self.Wout = np.linalg.pinv(lhs) @ rhs

        self.Wout = np.nan_to_num(self.Wout, nan=0.0, posinf=0.0, neginf=0.0)

        return self

    def predict_one_step(self, y_test: np.ndarray, previous_value: float) -> np.ndarray:
        if self.Wout is None:
            raise RuntimeError("ESNModel must be fitted before prediction.")

        y_test = np.asarray(y_test, dtype=float).reshape(-1)

        inputs = np.concatenate([[float(previous_value)], y_test[:-1]])
        preds = []

        self.prediction_guard_count = 0

        for u in inputs:
            state_t = self._step(u)
            x_t = np.concatenate([[1.0], state_t])

            pred = float(np.asarray(x_t @ self.Wout).reshape(-1)[0])

            if not np.isfinite(pred):
                pred = 0.0
                self.prediction_guard_count += 1

            if self.clip_prediction is not None:
                if pred > self.clip_prediction:
                    pred = self.clip_prediction
                    self.prediction_guard_count += 1
                elif pred < -self.clip_prediction:
                    pred = -self.clip_prediction
                    self.prediction_guard_count += 1

            preds.append(pred)

        return np.asarray(preds, dtype=float)

    def trainable_params(self) -> int:
        if self.Wout is None:
            return 0
        return int(self.Wout.size)

    def total_params(self) -> int:
        total = int(self.Win.size + self.W.size)
        if self.Wout is not None:
            total += int(self.Wout.size)
        return total