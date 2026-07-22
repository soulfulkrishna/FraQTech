import numpy as np
from sklearn.linear_model import Ridge


class RidgeLagModel:
    """
    Ridge regression on flattened lag windows.
    """

    def __init__(self, alpha: float = 1e-3) -> None:
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RidgeLagModel":
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train, dtype=float).reshape(-1)

        X_flat = X.reshape(X.shape[0], -1)
        self.model.fit(X_flat, y)

        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X = np.asarray(X_test, dtype=float)
        X_flat = X.reshape(X.shape[0], -1)
        pred = self.model.predict(X_flat)
        return np.asarray(pred, dtype=float).reshape(-1)

    def trainable_params(self) -> int:
        if not hasattr(self.model, "coef_"):
            return 0
        return int(self.model.coef_.size + 1)

    def total_params(self) -> int:
        return self.trainable_params()