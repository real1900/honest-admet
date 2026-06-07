"""Extra data-viz for the README (not in the paper):
  fig5 — split-leakage gradient: mean nearest-neighbour Tanimoto (test->train) across all 22
         endpoints for random / scaffold / cluster splits (the core premise, visualized).
  fig6 — conformal-coverage distribution across 22 endpoints per split (near-nominal on average,
         tail degrades under cluster shift).

Run (on the M4 Pro): cd ~/research/honest-admet && .venv/bin/python experiments/13_extra_figs.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from honest_admet.data import (  # noqa: E402
    SPLIT_METHODS, list_admet, load_admet, make_split, nn_tanimoto_leakage,
)

OUT = Path("paper/figures"); OUT.mkdir(parents=True, exist_ok=True)
COL = {"random": "#4C72B0", "scaffold": "#DD8452", "cluster": "#C44E52"}


def _box_strip(ax, values_by_split, seed0=0):
    for i, m in enumerate(SPLIT_METHODS):
        y = np.asarray(values_by_split[m], float)
        x = np.random.default_rng(seed0 + i).normal(i, 0.06, len(y))
        ax.scatter(x, y, color=COL[m], alpha=0.65, s=24, zorder=3, edgecolor="white", linewidth=0.4)
        ax.boxplot([y], positions=[i], widths=0.55, showfliers=False,
                   medianprops=dict(color="black", lw=1.5),
                   boxprops=dict(color="0.4"), whiskerprops=dict(color="0.4"),
                   capprops=dict(color="0.4"))
    ax.set_xticks(range(len(SPLIT_METHODS)))
    ax.set_xticklabels(list(SPLIT_METHODS))


def leakage_gradient():
    rows = []
    for name in list_admet():
        ds = load_admet(name)
        for m in SPLIT_METHODS:
            sp = make_split(m, ds.smiles, seed=0)
            rows.append(dict(dataset=name, split=m,
                             sim=nn_tanimoto_leakage(ds.smiles, sp)["mean_max_sim"]))
        print("  leakage:", name, flush=True)
    df = pd.DataFrame(rows); df.to_csv("results/leakage_gradient.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    _box_strip(ax, {m: df[df.split == m]["sim"].values for m in SPLIT_METHODS})
    ax.set_ylabel("mean nearest-neighbour Tanimoto\n(test → nearest training molecule)")
    ax.set_title("Realistic splits push test chemistry away from training\n(each point = one of 22 endpoints)")
    fig.tight_layout(); fig.savefig(OUT / "fig5_leakage_gradient.png", dpi=150); plt.close(fig)
    print("wrote fig5_leakage_gradient.png")


def coverage_distribution():
    u = pd.read_csv("results/uncertainty.csv")
    g = u.groupby(["dataset", "split"])["coverage"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    _box_strip(ax, {m: g[g.split == m]["coverage"].values for m in SPLIT_METHODS}, seed0=9)
    ax.axhline(0.90, color="0.35", ls="--", lw=1.2, zorder=1, label="target 90%")
    ax.set_ylabel("split-conformal coverage")
    ax.set_title("Coverage is near-nominal on average,\nbut the tail degrades under cluster shift")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "fig6_coverage.png", dpi=150); plt.close(fig)
    print("wrote fig6_coverage.png")


if __name__ == "__main__":
    leakage_gradient(); coverage_distribution()
    print("extra figures ->", OUT)
