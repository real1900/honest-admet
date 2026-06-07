"""Generate the paper figures from the result CSVs. Saves PNGs to paper/figures/.

Run (on the M4 Pro), after the analyses have produced their CSVs:
    cd ~/research/honest-admet && .venv/bin/python experiments/10_figures.py
Missing inputs are skipped with a note, so it is safe to run incrementally.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)


def _relgap(df):
    df = df.copy()
    df["relgap"] = df["mean"] / df["random_mean"].abs().clip(lower=1e-9)
    return df


def fig_atlas():
    """Fig 1: per-endpoint relative gap, scaffold vs cluster (sorted)."""
    p = Path("results/atlas_full_gaps.csv")
    if not p.exists():
        return print("skip fig1 (no atlas)")
    g = _relgap(pd.read_csv(p).query("model == 'xgboost'"))
    piv = g.pivot_table(index="dataset", columns="realistic", values="relgap")
    piv = piv.sort_values("cluster")
    y = np.arange(len(piv))
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.scatter(piv["scaffold"], y, color="#4C72B0", label="scaffold", s=36)
    ax.scatter(piv["cluster"], y, color="#C44E52", label="cluster", s=36)
    for yi, (_, r) in zip(y, piv.iterrows()):
        ax.plot([r["scaffold"], r["cluster"]], [yi, yi], color="0.7", lw=1, zorder=0)
    ax.set_yticks(y); ax.set_yticklabels(piv.index, fontsize=7)
    ax.set_xlabel("relative generalization gap  (gap / random-split score)")
    ax.set_title("Cluster splits reveal a larger gap than scaffold\n(all 22 endpoints)", fontsize=11)
    ax.legend(); ax.axvline(0, color="0.5", lw=0.8)
    fig.tight_layout(); fig.savefig(OUT / "fig1_atlas.png", dpi=150); plt.close(fig)
    print("wrote fig1_atlas.png")


def fig_applicability():
    """Fig 2: normalized error vs nearest-neighbour similarity bin."""
    p = Path("results/ad_deciles.csv")
    if not p.exists():
        return print("skip fig2 (no ad_deciles)")
    d = pd.read_csv(p)
    centers = (d["sim_bin"] + 0.5) / 10
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(centers, d["mean_norm_err"], "o-", color="#C44E52")
    ax.axhline(1.0, color="0.5", lw=0.8, ls="--", label="endpoint mean error")
    ax.set_xlabel("nearest-neighbour Tanimoto similarity to training")
    ax.set_ylabel("normalized error (1 = endpoint mean)")
    ax.set_title("Error rises sharply as molecules leave the training domain")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "fig2_applicability.png", dpi=150); plt.close(fig)
    print("wrote fig2_applicability.png")


def fig_unifying():
    """Fig 3: gap vs abstention benefit (the honest null)."""
    p = Path("results/unifying.csv")
    if not p.exists():
        return print("skip fig3 (no unifying)")
    u = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(u["rel_gap"], u["benefit"], color="#55A868")
    for _, r in u.iterrows():
        ax.annotate(r["dataset"], (r["rel_gap"], r["benefit"]), fontsize=5, alpha=0.6)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("relative optimism gap (cluster)")
    ax.set_ylabel("abstention benefit (risk removed @80% coverage, AD)")
    ax.set_title("Gap does NOT predict abstention benefit (Spearman ρ=0.07, n.s.)")
    fig.tight_layout(); fig.savefig(OUT / "fig3_unifying.png", dpi=150); plt.close(fig)
    print("wrote fig3_unifying.png")


def fig_foundation():
    """Fig 4: foundation-model vs fingerprint relative cluster gap."""
    fp_p, fm_p = Path("results/atlas_full_gaps.csv"), Path("results/foundation_gaps.csv")
    if not (fp_p.exists() and fm_p.exists()):
        return print("skip fig4 (no foundation results yet)")
    fp = _relgap(pd.read_csv(fp_p).query("model == 'xgboost' and realistic == 'cluster'"))
    fp = fp.set_index("dataset")["relgap"]
    fm = pd.read_csv(fm_p)
    fm["relgap"] = fm["gap_fm"] / fm["random_mean"].abs().clip(lower=1e-9)
    fm = fm.set_index("dataset")["relgap"]
    m = pd.concat([fp.rename("fingerprint"), fm.rename("chemberta")], axis=1).dropna()
    lim = [0, max(m.max()) * 1.05]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(m["fingerprint"], m["chemberta"], color="#8172B3")
    ax.plot(lim, lim, color="0.5", ls="--", lw=0.8, label="equal gap")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("ECFP4 fingerprint relative cluster gap")
    ax.set_ylabel("ChemBERTa relative cluster gap")
    ax.set_title("Does the foundation model close the gap?\n(points below diagonal = smaller gap)")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "fig4_foundation.png", dpi=150); plt.close(fig)
    print("wrote fig4_foundation.png")


if __name__ == "__main__":
    fig_atlas(); fig_applicability(); fig_unifying(); fig_foundation()
    print("figures ->", OUT)
