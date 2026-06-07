"""Classical fingerprint baselines: XGBoost and RandomForest.

For classification we return the positive-class probability (so AUROC/AUPRC and
calibration are well defined). For regression we return the predicted value.
"""

from __future__ import annotations

import numpy as np

BASELINES = ("xgboost", "rf")


def fit_model(model: str, task: str, X: np.ndarray, y: np.ndarray, seed: int = 0):
    """Fit and return an estimator (so callers can predict on several sets, e.g.
    a conformal calibration fold and the test fold)."""
    est = _make_classifier(model, seed) if task == "classification" else _make_regressor(model, seed)
    est.fit(X, y)
    return est


def predict(estimator, task: str, X: np.ndarray) -> np.ndarray:
    """Classification -> P(y=1); regression -> predicted value."""
    if task == "classification":
        return estimator.predict_proba(X)[:, 1]
    return estimator.predict(X)


def fit_predict(
    model: str,
    task: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """Train ``model`` on (X_train, y_train) and return predictions on X_test.
    Classification -> P(y=1); regression -> predicted value."""
    return predict(fit_model(model, task, X_train, y_train, seed), task, X_test)


def ensemble_predict(
    model: str, task: str, X_train: np.ndarray, y_train: np.ndarray,
    X_eval: np.ndarray, n_members: int = 5,
):
    """Deep-ensemble surrogate: ``n_members`` seed-perturbed models. Returns
    (mean_prediction, disagreement_std). The std is a label-free confidence signal
    for selective prediction (higher std = less confident)."""
    preds = np.stack([
        predict(fit_model(model, task, X_train, y_train, seed=s), task, X_eval)
        for s in range(n_members)
    ])
    return preds.mean(axis=0), preds.std(axis=0)


def _make_classifier(model: str, seed: int):
    if model == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=seed,
        )
    if model == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=500, n_jobs=-1, random_state=seed
        )
    raise ValueError(f"unknown baseline {model!r}")


def _make_regressor(model: str, seed: int):
    if model == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=seed,
        )
    if model == "rf":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=500, n_jobs=-1, random_state=seed
        )
    raise ValueError(f"unknown baseline {model!r}")
