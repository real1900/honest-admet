"""Self-test / regression test for the conformal core (no TDC data needed).

Verifies the two properties the correctness review demanded:
  1. Uniform-weight reduction: the weighted quantile with equal weights equals the
     textbook split-conformal threshold (the ceil((n+1)(1-alpha))-th smallest score),
     so turning weighting ON is a no-op when there is no shift.
  2. Marginal coverage: on exchangeable (no-shift) data, split-conformal regression
     covers at >= 1-alpha and not much above, as the theory guarantees.

Run anywhere:
    python experiments/05_conformal_selftest.py
"""

import numpy as np

from honest_admet.conformal import _conformal_quantile, conformal_regression_q, regression_coverage


def textbook_q(scores, alpha):
    """The ceil((n+1)(1-alpha))-th smallest score, or +inf if that exceeds n."""
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float("inf") if k > n else float(s[k - 1])


def test_uniform_reduces_to_textbook():
    rng = np.random.default_rng(0)
    alpha = 0.1
    for n in (10, 20, 50, 90, 200, 877, 1300):
        scores = rng.gamma(2.0, 1.0, size=n)
        q_unweighted = _conformal_quantile(scores, alpha, weights=None)
        q_uniform = _conformal_quantile(scores, alpha, weights=np.ones(n))
        q_book = textbook_q(scores, alpha)
        assert q_unweighted == q_uniform == q_book, (
            f"n={n}: {q_unweighted} / {q_uniform} / {q_book}")
    print("✅ uniform weights == unweighted == textbook ceil((n+1)(1-alpha)) quantile")


def test_marginal_coverage():
    rng = np.random.default_rng(1)
    alpha, n_cal, n_test, trials = 0.1, 200, 500, 400
    covs = []
    for _ in range(trials):
        cal_pred = rng.normal(0, 1, n_cal)
        cal_y = cal_pred + rng.normal(0, 1, n_cal)      # exchangeable residuals
        te_pred = rng.normal(0, 1, n_test)
        te_y = te_pred + rng.normal(0, 1, n_test)
        q = conformal_regression_q(cal_pred, cal_y, alpha)
        cov, _ = regression_coverage(te_pred, te_y, q)
        covs.append(cov)
    mean_cov = float(np.mean(covs))
    assert 1 - alpha - 0.01 <= mean_cov <= 1 - alpha + 0.03, mean_cov
    print(f"✅ marginal coverage = {mean_cov:.3f} (target {1 - alpha:.2f}, valid range "
          f"[{1 - alpha:.2f}, {1 - alpha + 1 / (n_cal + 1):.3f}])")


def test_inf_when_unreachable():
    # alpha so small that ceil((n+1)(1-alpha)) > n -> cannot certify -> +inf
    assert _conformal_quantile(np.arange(5.0), alpha=0.01) == float("inf")
    print("✅ returns +inf when coverage cannot be certified (small n, tiny alpha)")


if __name__ == "__main__":
    test_uniform_reduces_to_textbook()
    test_marginal_coverage()
    test_inf_when_unreachable()
    print("\nAll conformal self-tests passed.")
