import warnings
from typing import Tuple

import numpy as np
from statsmodels.tsa.arima.model import ARIMA


class ARIMAModel:
    """
    Simple ARIMA baseline.

    Default order is ARIMA(5, 0, 0), effectively an AR baseline.
    """

    def __init__(self, order: Tuple[int, int, int] = (5, 0, 0)) -> None:
        self.order = order
        self.result = None
        self.train_y = None

    def fit(self, y_train: np.ndarray) -> "ARIMAModel":
        self.train_y = np.asarray(y_train, dtype=float).reshape(-1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.result = ARIMA(self.train_y, order=self.order).fit()

        return self

    def predict_one_step(self, y_test: np.ndarray) -> np.ndarray:
        if self.result is None or self.train_y is None:
            raise RuntimeError("ARIMAModel must be fitted before prediction.")

        y_test = np.asarray(y_test, dtype=float).reshape(-1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            extended = self.result.append(y_test, refit=False)
            start = len(self.train_y)
            end = len(self.train_y) + len(y_test) - 1
            pred = extended.predict(start=start, end=end)

        pred = np.asarray(pred, dtype=float).reshape(-1)
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

        return pred

    def trainable_params(self) -> int:
        if self.result is None:
            return 0
        return int(len(self.result.params))

    def total_params(self) -> int:
        return self.trainable_params()