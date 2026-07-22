import numpy as np


class PersistenceModel:
    """
    One-step persistence baseline:
        y_hat[t] = y[t-1]

    For the first test point, use the last train/validation value.
    """

    def fit(self, y_train: np.ndarray) -> "PersistenceModel":
        return self

    def predict_one_step(self, y_test: np.ndarray, previous_value: float) -> np.ndarray:
        y_test = np.asarray(y_test, dtype=float).reshape(-1)
        return np.concatenate([[float(previous_value)], y_test[:-1]])