from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .baselines import RegimeClassifier
from .data import ForecastDataset, ForecastSplit, concatenate_splits
from .metrics import classification_metrics, regression_metrics


@dataclass
class SelectedModel:
    model: object
    parameter: float | int | str | None
    validation_metrics: Dict[str, float]


def inverse_predictions(dataset: ForecastDataset, y_scaled: np.ndarray) -> np.ndarray:
    return dataset.scaler.inverse_transform(np.asarray(y_scaled, dtype=float).reshape(-1))


def score_scaled_predictions(
    dataset: ForecastDataset,
    split: ForecastSplit,
    pred_scaled: np.ndarray,
) -> Dict[str, float]:
    y_true = inverse_predictions(dataset, split.y)
    y_pred = np.maximum(inverse_predictions(dataset, pred_scaled), 1e-10)
    return regression_metrics(y_true, y_pred)


def select_model_on_validation(
    dataset: ForecastDataset,
    candidates: Sequence[Tuple[object, object]],
    selection_metric: str = "qlike",
) -> SelectedModel:
    best: SelectedModel | None = None
    for parameter, model in candidates:
        model.fit(dataset.train.X, dataset.train.y)
        pred = model.predict(dataset.val.X)
        metrics = score_scaled_predictions(dataset, dataset.val, pred)
        if best is None or metrics[selection_metric] < best.validation_metrics[selection_metric]:
            best = SelectedModel(model=model, parameter=parameter, validation_metrics=metrics)
    if best is None:
        raise ValueError("No candidates supplied")
    return best


def refit_and_predict(
    dataset: ForecastDataset,
    model_factory: Callable[[], object],
) -> Tuple[object, np.ndarray]:
    train_val = concatenate_splits(dataset.train, dataset.val)
    model = model_factory()
    model.fit(train_val.X, train_val.y)
    return model, np.asarray(model.predict(dataset.test.X), dtype=float).reshape(-1)


def select_regime_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    C_values: Iterable[float],
) -> Tuple[float, Dict[str, float]]:
    best_C, best_metrics = None, None
    for C in C_values:
        model = RegimeClassifier(C=C).fit(train_features, train_labels)
        prob = model.predict_proba(val_features)
        metrics = classification_metrics(val_labels, prob)
        if best_metrics is None or metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_C, best_metrics = float(C), metrics
    if best_C is None or best_metrics is None:
        raise ValueError("No classifier candidates")
    return best_C, best_metrics


def fit_regime_classifier_final(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    test_features: np.ndarray,
    C: float,
) -> np.ndarray:
    X = np.concatenate([train_features, val_features])
    y = np.concatenate([train_labels, val_labels])
    model = RegimeClassifier(C=C).fit(X, y)
    return model.predict_proba(test_features)


def select_qrc_ridge_alpha(
    dataset: ForecastDataset,
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_val: np.ndarray,
    y_val: np.ndarray,
    alphas: Iterable[float],
    selection_metric: str = "qlike",
) -> Tuple[float, Dict[str, float]]:
    """Select a frozen-QRC ridge readout on validation only."""
    from .baselines import QRCFeatureReadout

    y_val_raw = dataset.scaler.inverse_transform(np.asarray(y_val, dtype=float).reshape(-1))
    best_alpha: float | None = None
    best_metrics: Dict[str, float] | None = None
    for alpha in alphas:
        model = QRCFeatureReadout(alpha=float(alpha)).fit(Z_train, y_train)
        pred_scaled = model.predict(Z_val)
        pred_raw = np.maximum(dataset.scaler.inverse_transform(pred_scaled), 1e-10)
        metrics = regression_metrics(y_val_raw, pred_raw)
        if best_metrics is None or metrics[selection_metric] < best_metrics[selection_metric]:
            best_alpha = float(alpha)
            best_metrics = metrics
    if best_alpha is None or best_metrics is None:
        raise ValueError("No ridge alpha candidates supplied")
    return best_alpha, best_metrics


def fit_qrc_ridge_final(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_val: np.ndarray,
    y_val: np.ndarray,
    alpha: float,
):
    """Refit a QRC ridge readout on train+validation after alpha selection."""
    from .baselines import QRCFeatureReadout

    return QRCFeatureReadout(alpha=float(alpha)).fit(
        np.concatenate([Z_train, Z_val], axis=0),
        np.concatenate([y_train, y_val], axis=0),
    )
