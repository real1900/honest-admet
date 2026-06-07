"""Selective prediction: risk-coverage curves and AURC.

Given a label-free confidence score (higher = more confident) and a per-point error,
sweep coverage (the fraction of most-confident predictions we keep) and compute risk
(mean error on the kept set). AURC = area under that curve (lower is better). The
confidence signal can be ensemble disagreement, model probability margin, or
applicability-domain distance (nearest-neighbour Tanimoto to train) — letting us ask
which abstention rule best recovers the optimism gap under scaffold/cluster shift.
"""

from __future__ import annotations

import numpy as np


def risk_coverage(confidence, error):
    """Returns (coverages, risks): risk at each coverage level, keeping the most
    confident points first."""
    confidence, error = np.asarray(confidence, float), np.asarray(error, float)
    order = np.argsort(-confidence)  # most confident first
    err = error[order]
    n = len(err)
    coverages = np.arange(1, n + 1) / n
    risks = np.cumsum(err) / np.arange(1, n + 1)
    return coverages, risks


_trapz = getattr(np, "trapezoid", np.trapz)  # np.trapz deprecated in numpy 2.0


def aurc(confidence, error) -> float:
    """Area under the risk-coverage curve (lower is better). The curve is anchored at
    (coverage=0, risk=0) for a stable left tail."""
    cov, risk = risk_coverage(confidence, error)
    cov = np.concatenate([[0.0], cov])
    risk = np.concatenate([[0.0], risk])
    return float(_trapz(risk, cov))


def excess_aurc(confidence, error) -> float:
    """AURC minus the oracle AURC (ordering errors last). Isolates ordering quality and
    is comparable across folds with different base error rates (lower is better)."""
    return aurc(confidence, error) - aurc(-np.asarray(error, float), error)


def risk_at_coverage(confidence, error, coverage: float) -> float:
    """Selective risk when retaining the top ``coverage`` fraction by confidence."""
    cov, risk = risk_coverage(confidence, error)
    idx = min(int(np.ceil(coverage * len(risk))) - 1, len(risk) - 1)
    return float(risk[max(idx, 0)])


def coverage_at_risk(confidence, error, target_risk: float) -> float:
    """Largest coverage whose selective risk stays <= ``target_risk``."""
    cov, risk = risk_coverage(confidence, error)
    ok = np.where(risk <= target_risk)[0]
    return float(cov[ok.max()]) if ok.size else 0.0


def errors(task: str, y_true, y_pred) -> np.ndarray:
    """Per-point error: 0/1 misclassification (classification) or absolute error
    (regression). For classification ``y_pred`` is P(y=1)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if task == "classification":
        return (y_true.astype(int) != (y_pred >= 0.5).astype(int)).astype(float)
    return np.abs(y_true - y_pred)
