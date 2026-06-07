"""Experiment 3b: the unifying analysis — does the gap predict abstention benefit?

The thesis: the endpoints with the largest random->cluster optimism gap are exactly the
ones where abstaining on the out-of-distribution tail (by applicability-domain distance)
removes the most risk. We correlate, per endpoint under the cluster split:
    relative optimism gap (from the atlas)   vs.   abstention benefit
where abstention benefit = fraction of risk removed by dropping the 20% least in-domain
test molecules, (full_risk - selective_risk@80%_AD) / full_risk, averaged over seeds.

Run (after results/atlas_full_gaps.csv and a full-22 results/uncertainty.csv exist):
    cd ~/research/honest-admet && .venv/bin/python experiments/07_unifying.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


def main():
    gaps = pd.read_csv("results/atlas_full_gaps.csv")
    gaps = gaps[(gaps.realistic == "cluster") & (gaps.model == "xgboost")].copy()
    gaps["rel_gap"] = gaps["mean"] / gaps["random_mean"].abs().clip(lower=1e-9)
    gaps = gaps.set_index("dataset")["rel_gap"]

    unc = pd.read_csv("results/uncertainty.csv")
    cl = unc[unc.split == "cluster"].copy()
    cl["benefit"] = (cl["full_risk"] - cl["sel_risk80_ad"]) / cl["full_risk"].clip(lower=1e-9)
    per_ep = cl.groupby("dataset").agg(
        benefit=("benefit", "mean"),
        eaurc_ad=("eaurc_ad", "mean"),
        eaurc_ens=("eaurc_ens", "mean")).reset_index().set_index("dataset")

    df = per_ep.join(gaps.rename("rel_gap"), how="inner").dropna(subset=["rel_gap", "benefit"])
    df.to_csv("results/unifying.csv")

    print("=" * 74)
    print("UNIFYING ANALYSIS (cluster split): does a bigger optimism gap predict bigger")
    print("abstention benefit?   [n =", len(df), "endpoints]")
    print("=" * 74)
    rho, p = spearmanr(df["rel_gap"], df["benefit"])
    print(f"  Spearman(rel_gap, abstention_benefit) rho={rho:+.3f} (p={p:.4f})")
    print("  [thesis: POSITIVE — high-gap endpoints gain most from abstaining on the OOD tail]\n")
    print(f"  {'dataset':<26}{'rel_gap':>9}{'benefit':>9}{'E-AURC ad':>11}{'E-AURC ens':>12}")
    print("-" * 67)
    for r in df.sort_values("rel_gap", ascending=False).itertuples():
        print(f"  {r.Index:<26}{r.rel_gap:>9.3f}{r.benefit:>9.3f}{r.eaurc_ad:>11.3f}{r.eaurc_ens:>12.3f}")

    # is AD a better abstention signal than ensemble disagreement, across endpoints?
    d = df["eaurc_ens"] - df["eaurc_ad"]
    try:
        pp = wilcoxon(d).pvalue
    except ValueError:
        pp = float("nan")
    print(f"\n  AD vs ensemble across {len(df)} endpoints (cluster): mean E-AURC(ens-ad)="
          f"{d.mean():+.3f}, Wilcoxon p={pp:.4f}  [>0 => AD better]")
    print("\n✅ unifying analysis done -> results/unifying.csv (thesis figure data)")


if __name__ == "__main__":
    main()
