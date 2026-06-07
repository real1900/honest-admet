"""Metrics and calibration for Honest ADMET.

Regression metrics: MAE, Spearman. Classification: AUROC, AUPRC. These match the
official TDC ADMET leaderboard conventions (see data.ADMET_METRICS). Calibration:
predicted-class ECE (equal-width + adaptive equal-mass) and Brier score.

All scorers are guarded against degenerate folds (single-class labels, constant
predictions) so a bad fold yields NaN with a reason rather than crashing the sweep.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    roc_auc_score,
)

# whether a higher metric value is better (for computing the generalization gap)
HIGHER_IS_BETTER = {"auroc": True, "auprc": True, "spearman": True, "mae": False}


def score(metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute an official TDC metric. For classification, ``y_pred`` is the
    predicted probability of the positive class. Returns NaN on degenerate folds."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if metric == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    if metric == "spearman":
        if np.unique(y_pred).size < 2 or np.unique(y_true).size < 2:
            return float("nan")  # Spearman undefined for constant inputs
        return float(spearmanr(y_true, y_pred).statistic)
    if metric in ("auroc", "auprc"):
        if np.unique(y_true).size < 2:
            return float("nan")  # AUROC/AUPRC undefined when test fold is single-class
        if metric == "auroc":
            return float(roc_auc_score(y_true, y_pred))
        return float(average_precision_score(y_true, y_pred))
    raise ValueError(f"unknown metric {metric!r}")


def normalized_auprc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """AUPRC minus the positive base rate — skill above chance, robust to the
    prevalence shifts that grouped splits induce on the 5 AUPRC CYP endpoints."""
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_pred) - y_true.mean())


def generalization_gap(metric: str, optimistic: float, realistic: float) -> float:
    """Signed drop from an optimistic (random) split to a realistic split,
    expressed so that a POSITIVE number always means 'performance got worse'."""
    return (optimistic - realistic) if HIGHER_IS_BETTER[metric] else (realistic - optimistic)


def _ece(y_true, conf, correct, n_bins, adaptive):
    """Core ECE: weighted gap between confidence and accuracy over bins."""
    n = len(y_true)
    if adaptive:  # equal-mass bins (each ~n/n_bins points)
        order = np.argsort(conf)
        edges_idx = np.linspace(0, n, n_bins + 1).astype(int)
        bins = [order[edges_idx[i]:edges_idx[i + 1]] for i in range(n_bins)]
    else:  # equal-width bins, first bin left-closed at 0
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
        bins = [np.where(idx == b)[0] for b in range(n_bins)]
    ece, w_sum = 0.0, 0.0
    for b in bins:
        if len(b) == 0:
            continue
        w = len(b) / n
        ece += w * abs(conf[b].mean() - correct[b].mean())
        w_sum += w
    assert abs(w_sum - 1.0) < 1e-9, f"bin weights sum to {w_sum}, not 1"
    return float(ece)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                               n_bins: int = 15, adaptive: bool = False) -> float:
    """Predicted-class ECE for binary classification: confidence = max(p, 1-p),
    accuracy = fraction predicted correctly. Set ``adaptive=True`` for equal-mass
    bins (less biased under class imbalance)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    pred = (y_prob >= 0.5).astype(int)
    conf = np.maximum(y_prob, 1.0 - y_prob)
    correct = (pred == y_true).astype(float)
    return _ece(y_true, conf, correct, n_bins, adaptive)


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score (lower is better); prevalence-aware complement to ECE."""
    return float(brier_score_loss(np.asarray(y_true), np.asarray(y_prob)))
