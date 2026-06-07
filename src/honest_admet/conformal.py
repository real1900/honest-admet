"""Conformal prediction + covariate-shift weighting for Honest ADMET.

Split (inductive) conformal prediction gives finite-sample marginal coverage 1-alpha
under EXCHANGEABILITY of calibration and test points. Realistic scaffold/cluster splits
deliberately break exchangeability (covariate shift), so standard split-conformal
under-covers on the shifted test set — which is exactly the failure we want to measure.

We then apply weighted conformal prediction (Tibshirani et al. 2019), with importance
weights w(x) ∝ P(test|x)/P(cal|x) estimated by a train-vs-test domain classifier (the
CoDrug, Laghuvarapu et al. 2023, setting), and show whether it restores coverage.

Calibration uses the validation fold; the model is fit on train; everything is evaluated
on test — no label from the calibration or test fold leaks into model fitting.
"""

from __future__ import annotations

import numpy as np


def _conformal_quantile(scores: np.ndarray, alpha: float, weights=None) -> float:
    """Weighted split-conformal threshold (Tibshirani et al. 2019).

    Puts mass w_i/(sum_j w_j + w_test) on each calibration score and mass
    w_test/(sum_j w_j + w_test) on a +inf atom for the test point (w_test = mean
    calibration weight). Returns the smallest score whose normalized cumulative weight
    reaches 1-alpha, or +inf when the calibration mass cannot reach it (the honest
    "cannot certify coverage under this much shift" outcome).

    With uniform weights this reduces EXACTLY to textbook split-conformal: the
    ceil((n+1)(1-alpha))-th smallest score. So the unweighted path (weights=None) and
    the weighted path share one code path and agree under uniform weights by construction.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        return float("inf")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    w_test = float(w.mean())
    total = w.sum() + w_test
    if total <= 0:
        return float("inf")
    order = np.argsort(scores)
    s, cw = scores[order], np.cumsum(w[order]) / total
    target = 1.0 - alpha
    idx = int(np.searchsorted(cw, target, side="left"))
    if idx < n and cw[idx] < target:  # tie-safety: never under-shoot the >= rule
        idx += 1
    return float("inf") if idx >= n else float(s[idx])


def kish_ess(weights: np.ndarray) -> float:
    """Kish effective sample size fraction (sum w)^2 / (n * sum w^2) in [0,1].
    Low values mean a few extreme importance weights dominate -> weighting is unreliable."""
    w = np.asarray(weights, dtype=float)
    denom = len(w) * np.sum(w**2)
    return float(w.sum() ** 2 / denom) if denom > 0 else 0.0


def domain_classifier_weights(X_cal: np.ndarray, X_test: np.ndarray, seed: int = 0,
                              C: float = 0.1, clip_pct: float = 99.0) -> np.ndarray:
    """Importance weights for calibration points under covariate shift:
    w(x) ∝ P(test|x)/(1-P(test|x)), via an isotonic-CALIBRATED, L2-regularized,
    class-balanced domain classifier discriminating calibration from test. Weights are
    clipped at the ``clip_pct`` percentile to tame the heavy tails that high-dimensional
    near-separable fingerprints produce under strong shift."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression

    X = np.vstack([X_cal, X_test]).astype(np.float32)
    y = np.concatenate([np.zeros(len(X_cal)), np.ones(len(X_test))])
    base = LogisticRegression(max_iter=1000, C=C, class_weight="balanced")
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3).fit(X, y)
    p = np.clip(clf.predict_proba(X_cal.astype(np.float32))[:, 1], 1e-6, 1 - 1e-6)
    w = p / (1.0 - p)
    cap = np.percentile(w, clip_pct)
    return np.minimum(w, cap)


# --- regression: symmetric absolute-residual intervals ------------------------
def conformal_regression_q(cal_pred, cal_y, alpha: float = 0.1, weights=None) -> float:
    """Half-width qhat such that [pred-qhat, pred+qhat] has ~1-alpha coverage.
    qhat may be +inf (interval = whole line) when coverage cannot be certified."""
    resid = np.abs(np.asarray(cal_y) - np.asarray(cal_pred))
    return _conformal_quantile(resid, alpha, weights)


def regression_coverage(test_pred, test_y, qhat: float):
    """Returns (empirical coverage, mean interval width)."""
    test_pred, test_y = np.asarray(test_pred), np.asarray(test_y)
    covered = (test_y >= test_pred - qhat) & (test_y <= test_pred + qhat)
    return float(covered.mean()), float(2 * qhat)


# --- binary classification: LAC (nonconformity = 1 - p[true class]) -----------
def conformal_lac_q(cal_prob, cal_y, alpha: float = 0.1, weights=None) -> float:
    cal_prob, cal_y = np.asarray(cal_prob), np.asarray(cal_y).astype(int)
    p_true = np.where(cal_y == 1, cal_prob, 1 - cal_prob)
    s = 1.0 - p_true  # nonconformity score
    return _conformal_quantile(s, alpha, weights)


def classification_coverage(test_prob, test_y, qhat: float):
    """LAC prediction sets {y : p(y) >= 1-qhat}. Returns (coverage, mean set size)."""
    test_prob, test_y = np.asarray(test_prob), np.asarray(test_y).astype(int)
    p = np.stack([1 - test_prob, test_prob], axis=1)  # P(y=0), P(y=1)
    in_set = p >= (1 - qhat)
    covered = in_set[np.arange(len(test_y)), test_y]
    return float(covered.mean()), float(in_set.sum(axis=1).mean())
