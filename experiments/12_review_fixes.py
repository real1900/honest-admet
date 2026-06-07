"""Recompute the numbers the draft-review flagged, and rewrite atlas_full_gaps.csv with
block-bootstrap CIs (resampling split seeds, the unit of independence) + established flags.

Fixes: (1) true RELATIVE median gaps; (2) non-independent bootstrap -> split-seed block
bootstrap; (3) aggregate conformal-coverage distribution; (4) CIs/power on the nulls.

    cd ~/research/honest-admet && .venv/bin/python experiments/12_review_fixes.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from honest_admet.eval import generalization_gap

RNG = np.random.default_rng(0)


def block_bootstrap_relgap(sub, metric, B=2000):
    """Per-endpoint relative gap (gap/|random|) with a CI that resamples SPLIT SEEDS
    (blocks), not individual replicates. Returns (relgap, lo, hi)."""
    piv = sub.pivot_table(index=["split_seed", "model_seed"], columns="split", values="score")
    if "random" not in piv:
        return np.nan, np.nan, np.nan
    seeds = piv.index.get_level_values("split_seed").unique().to_numpy()
    out = {}
    for realistic in ("scaffold", "cluster"):
        if realistic not in piv:
            continue
        boots = []
        for _ in range(B):
            chosen = RNG.choice(seeds, size=len(seeds), replace=True)
            gaps, rands = [], []
            for s in chosen:
                blk = piv.xs(s, level="split_seed")[["random", realistic]].dropna()
                for r in blk.itertuples():
                    gaps.append(generalization_gap(metric, r.random, getattr(r, realistic)))
                    rands.append(abs(r.random))
            if gaps:
                boots.append(np.mean(gaps) / max(1e-9, np.mean(rands)))
        boots = np.asarray(boots)
        paired = piv[["random", realistic]].dropna()
        point = np.mean([generalization_gap(metric, r.random, getattr(r, realistic))
                         for r in paired.itertuples()]) / max(1e-9, paired["random"].abs().mean())
        out[realistic] = (point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    return out


def main():
    atlas = pd.read_csv("results/atlas_full.csv").query("model == 'xgboost'")
    rows = []
    for dataset, sub in atlas.groupby("dataset"):
        metric = sub["metric"].iloc[0]
        res = block_bootstrap_relgap(sub, metric)
        sc, cl = res.get("scaffold"), res.get("cluster")
        rows.append(dict(dataset=dataset, metric=metric,
                         scaffold_relgap=sc[0], scaffold_lo=sc[1], scaffold_hi=sc[2],
                         scaffold_est=(sc[1] > 0 or sc[2] < 0),
                         cluster_relgap=cl[0], cluster_lo=cl[1], cluster_hi=cl[2],
                         cluster_est=(cl[1] > 0 or cl[2] < 0)))
    g = pd.DataFrame(rows)
    g.to_csv("results/atlas_relgap_blockci.csv", index=False)

    print("=== RELATIVE-GAP MEDIANS (block-bootstrap CIs, resampling split seeds) ===")
    print(f"  scaffold median relgap = {g.scaffold_relgap.median():.3f}  "
          f"(established {int(g.scaffold_est.sum())}/22)")
    print(f"  cluster  median relgap = {g.cluster_relgap.median():.3f}  "
          f"(established {int(g.cluster_est.sum())}/22)")
    print(f"  ratio cluster/scaffold (median relgap) = "
          f"{g.cluster_relgap.median()/g.scaffold_relgap.median():.2f}x")
    print(f"  hERG: scaffold relgap {g.set_index('dataset').loc['herg','scaffold_relgap']:+.3f} "
          f"(est={g.set_index('dataset').loc['herg','scaffold_est']}), "
          f"cluster {g.set_index('dataset').loc['herg','cluster_relgap']:+.3f}")
    print(f"  widest cluster CI example: "
          f"{g.loc[(g.cluster_hi-g.cluster_lo).idxmax(),'dataset']} "
          f"[{g.loc[(g.cluster_hi-g.cluster_lo).idxmax(),'cluster_lo']:+.3f},"
          f"{g.loc[(g.cluster_hi-g.cluster_lo).idxmax(),'cluster_hi']:+.3f}]")

    print("\n=== CONFORMAL CLUSTER-COVERAGE DISTRIBUTION (all 22) ===")
    u = pd.read_csv("results/uncertainty.csv")
    cov = u[u.split == "cluster"].groupby("dataset")["coverage"].mean()
    covr = u[u.split == "random"].groupby("dataset")["coverage"].mean()
    print(f"  random  mean coverage = {covr.mean():.3f}")
    print(f"  cluster mean coverage = {cov.mean():.3f}  (target 0.90)")
    print(f"  cluster range = [{cov.min():.3f}, {cov.max():.3f}];  "
          f"#endpoints <0.88 = {(cov<0.88).sum()}/22;  worst = {cov.idxmin()} {cov.min():.3f}")

    print("\n=== NULLS: CIs / power ===")
    un = pd.read_csv("results/unifying.csv")
    rho, p = spearmanr(un["rel_gap"], un["benefit"])
    n = len(un); z = np.arctanh(rho); se = 1/np.sqrt(n-3)
    lo, hi = np.tanh(z-1.96*se), np.tanh(z+1.96*se)
    print(f"  gap<->benefit: rho={rho:.3f}, p={p:.3f}, 95% CI [{lo:+.2f},{hi:+.2f}] (n={n}, under-powered)")
    cl = u[u.split == "cluster"].groupby("dataset").agg(
        ens=("eaurc_ens", "mean"), ad=("eaurc_ad", "mean"),
        fr=("full_risk", "mean"), s80=("sel_risk80_ad", "mean"))
    ens_wins = (cl["ens"] < cl["ad"]).sum()
    ad_hurts = (cl["s80"] > cl["fr"]).sum()
    dd = cl["ens"] - cl["ad"]
    print(f"  AD vs ensemble (cluster): ensemble wins {ens_wins}/22; AD abstention hurts {ad_hurts}/22; "
          f"paired Wilcoxon p={wilcoxon(dd).pvalue:.4f}")
    print("\nwrote results/atlas_relgap_blockci.csv")


if __name__ == "__main__":
    main()
