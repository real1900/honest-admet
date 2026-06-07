"""Experiment 3a: the applicability-domain MECHANISM behind the gap.

Two views, both reusing cached fingerprints:
  (A) Across endpoints: regress the per-endpoint relative cluster gap (from the atlas)
      on how out-of-domain the cluster test set is (mean nearest-neighbour Tanimoto to
      train). If lower similarity -> bigger gap, the gap is an applicability-domain effect.
  (B) Within endpoints: per-test error binned by nearest-neighbour similarity. If error
      rises monotonically as similarity falls, distance-to-training IS the error driver
      (and hence a good abstention signal).

Run (on the M4 Pro), after the atlas (results/atlas_full_gaps.csv) exists:
    cd ~/research/honest-admet && .venv/bin/python experiments/06_applicability_domain.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from honest_admet import selective
from honest_admet.data import list_admet, load_admet, make_split, nn_tanimoto_leakage, test_train_max_sim
from honest_admet.features import morgan_fingerprints
from honest_admet.models.baselines import fit_predict

SIM_BINS = np.linspace(0.0, 1.0, 11)  # 10 similarity bins


def main():
    gaps = pd.read_csv("results/atlas_full_gaps.csv")
    gaps = gaps[(gaps.realistic == "cluster") & (gaps.model == "xgboost")].set_index("dataset")

    per_ep, decile_rows = [], []
    for name in list_admet():
        ds = load_admet(name)
        X = morgan_fingerprints(ds.smiles)
        sp_r = make_split("random", ds.smiles, seed=0)
        sp_c = make_split("cluster", ds.smiles, seed=0)
        sim_r = nn_tanimoto_leakage(ds.smiles, sp_r)["mean_max_sim"]
        sim_c = nn_tanimoto_leakage(ds.smiles, sp_c)["mean_max_sim"]
        if name in gaps.index:
            g = gaps.loc[name]
            rel_gap = float(g["mean"]) / max(1e-9, abs(float(g["random_mean"])))
            per_ep.append(dict(dataset=name, metric=g["metric"], gap=float(g["mean"]),
                               rel_gap=rel_gap, sim_random=sim_r, sim_cluster=sim_c,
                               sim_drop=sim_r - sim_c))

        # (B) per-similarity-bin error on a random split (in-distribution model)
        fit_idx = np.concatenate([sp_r["train"], sp_r["valid"]])
        pred = fit_predict("xgboost", ds.task, X[fit_idx], ds.y[fit_idx], X[sp_r["test"]], seed=0)
        err = selective.errors(ds.task, ds.y[sp_r["test"]], pred)
        sim = test_train_max_sim(ds.smiles, sp_r)
        norm_err = err / max(1e-9, err.mean())  # normalize so endpoints are comparable
        which = np.clip(np.digitize(sim, SIM_BINS[1:-1]), 0, 9)
        for b in range(10):
            m = which == b
            if m.any():
                decile_rows.append(dict(dataset=name, sim_bin=b, mean_norm_err=float(norm_err[m].mean()),
                                        n=int(m.sum())))
        print(f"  {name:<26} sim random={sim_r:.3f} cluster={sim_c:.3f}", flush=True)

    ep = pd.DataFrame(per_ep)
    Path("results").mkdir(exist_ok=True)
    ep.to_csv("results/applicability_domain.csv", index=False)
    dec = pd.DataFrame(decile_rows)
    dec.groupby("sim_bin")["mean_norm_err"].mean().rename("mean_norm_err").to_csv(
        "results/ad_deciles.csv")  # for the error-vs-similarity figure

    print("\n" + "=" * 70)
    print("(A) ACROSS ENDPOINTS: does a more out-of-domain cluster test => bigger gap?")
    print("=" * 70)
    rho1, p1 = spearmanr(ep["rel_gap"], ep["sim_cluster"])
    rho2, p2 = spearmanr(ep["rel_gap"], ep["sim_drop"])
    print(f"  rel_gap vs cluster mean-NN-sim : Spearman rho={rho1:+.3f} (p={p1:.4f})  "
          f"[expect NEGATIVE: less similar -> bigger gap]")
    print(f"  rel_gap vs sim drop (rand-clus): Spearman rho={rho2:+.3f} (p={p2:.4f})  "
          f"[expect POSITIVE: bigger similarity drop -> bigger gap]")

    print("\n(B) WITHIN ENDPOINTS: normalized error by NN-similarity bin (mean over 22 endpoints)")
    agg = dec.groupby("sim_bin")["mean_norm_err"].mean()
    for b in range(10):
        if b in agg.index:
            bar = "#" * int(agg[b] * 20)
            print(f"  sim[{SIM_BINS[b]:.1f}-{SIM_BINS[b+1]:.1f})  norm_err={agg[b]:.2f}  {bar}")
    print("\nError should be HIGH at low similarity and fall toward 1.0 as similarity rises — "
          "i.e.\ndistance-to-training drives error, so it is a principled abstention signal.")
    print("\n✅ applicability-domain analysis done -> results/applicability_domain.csv")


if __name__ == "__main__":
    main()
