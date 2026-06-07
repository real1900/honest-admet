"""Statistical rigor for the generalization gap.

A gap is never reported as a bare point estimate. Given a vector of per-matched-
replicate gaps {g_r} (one per split_seed x model_seed pair, with the SAME seed used
for the random and realistic split so they are paired), we report:
  * mean gap
  * 95% bootstrap CI over replicates
  * paired Wilcoxon signed-rank p-value (primary) and paired t p-value (secondary)
  * an 'established' flag = CI excludes 0
Across the 22-endpoint family we apply Benjamini-Hochberg FDR (q-values), and we
give an aggregate 'gap positive on average' test over per-dataset mean gaps.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


def gap_stats(gaps, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Summarize a vector of paired per-replicate gaps {g_r}."""
    g = np.asarray([x for x in gaps if np.isfinite(x)], dtype=float)
    out = {"n": int(g.size), "mean": float("nan"), "ci_lo": float("nan"),
           "ci_hi": float("nan"), "p_wilcoxon": float("nan"), "p_ttest": float("nan"),
           "established": False}
    if g.size == 0:
        return out
    out["mean"] = float(g.mean())
    rng = np.random.default_rng(seed)
    boot = rng.choice(g, size=(n_boot, g.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    out["ci_lo"], out["ci_hi"] = float(lo), float(hi)
    # paired tests against a zero gap (g_r are already paired differences)
    if g.size >= 2 and np.any(g != 0):
        try:
            out["p_wilcoxon"] = float(wilcoxon(g).pvalue)
        except ValueError:
            out["p_wilcoxon"] = float("nan")
        out["p_ttest"] = float(ttest_rel(g, np.zeros_like(g)).pvalue)
    out["established"] = bool(lo > 0 or hi < 0)  # CI excludes zero
    return out


def standardized_effect(gaps, metric_values) -> float:
    """Gap as a multiple of the metric's own across-replicate SD (Cohen's-d-like),
    putting every endpoint on a comparable scale."""
    g = np.asarray([x for x in gaps if np.isfinite(x)], dtype=float)
    sd = np.std(np.asarray([x for x in metric_values if np.isfinite(x)], dtype=float))
    if g.size == 0 or sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float(g.mean() / sd)


def benjamini_hochberg(pvals) -> np.ndarray:
    """BH-FDR q-values (NaNs preserved)."""
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return q
    order = idx[np.argsort(p[idx])]
    m = idx.size
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * m / (rank + 1))
        q[i] = prev
    return q


def aggregate_gap_test(per_dataset_gaps) -> dict:
    """Is the gap positive on average across endpoints? One-sample Wilcoxon
    signed-rank on per-dataset mean gaps — the defensible headline claim."""
    g = np.asarray([x for x in per_dataset_gaps if np.isfinite(x)], dtype=float)
    res = {"n_datasets": int(g.size), "mean_gap": float(g.mean()) if g.size else float("nan"),
           "median_gap": float(np.median(g)) if g.size else float("nan"),
           "p_wilcoxon": float("nan"), "n_positive": int((g > 0).sum())}
    if g.size >= 2 and np.any(g != 0):
        try:
            res["p_wilcoxon"] = float(wilcoxon(g).pvalue)
        except ValueError:
            pass
    return res
