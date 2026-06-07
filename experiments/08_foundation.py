"""Experiment 4: the model-family ladder — do foundation models close the gap?

Runs the same generalization-gap machinery on FROZEN molecular foundation-model
embeddings (ChemBERTa / MolFormer) + an XGBoost head, then compares the per-endpoint
random->cluster gap to the ECFP4 fingerprint atlas. The crisp question: does a pretrained
chemical representation SHRINK the optimism gap relative to fingerprints, or just shift the
absolute scores? We compare RELATIVE gaps (gap / random-split score) so representations with
different absolute performance are comparable.

Run (on the M4 Pro; needs results/atlas_full_gaps.csv for the comparison):
    cd ~/research/honest-admet && .venv/bin/python experiments/08_foundation.py \
        --datasets all --fm chemberta --split-seeds 5 --model-seeds 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from honest_admet.data import list_admet, load_admet, make_split
from honest_admet.eval import generalization_gap, score
from honest_admet.models.baselines import fit_predict
from honest_admet.models.foundation import embed_smiles
from honest_admet.stats import gap_stats

SPLITS = ("random", "scaffold", "cluster")


def run(datasets, fm, split_seeds, model_seeds, head, out_path: Path) -> pd.DataFrame:
    rows = []
    for name in datasets:
        ds = load_admet(name)
        X = embed_smiles(ds.smiles, model=fm)  # frozen, cached
        print(f"=== {name} ({ds.task}/{ds.metric}, N={len(ds)}, emb={X.shape[1]}d) ===", flush=True)
        for ss in split_seeds:
            sp = {m: make_split(m, ds.smiles, seed=ss) for m in SPLITS}
            for m in SPLITS:
                fit_idx = np.concatenate([sp[m]["train"], sp[m]["valid"]])
                te = sp[m]["test"]
                for msd in model_seeds:
                    pred = fit_predict(head, ds.task, X[fit_idx], ds.y[fit_idx], X[te], seed=msd)
                    rows.append(dict(dataset=name, task=ds.task, metric=ds.metric, fm=fm,
                                     split=m, split_seed=ss, model_seed=msd,
                                     score=score(ds.metric, ds.y[te], pred)))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)  # incremental
        print(f"  done {name}", flush=True)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, fm: str, out_path: Path) -> None:
    recs = []
    for dataset, sub in df.groupby("dataset"):
        metric = sub["metric"].iloc[0]
        piv = sub.pivot_table(index=["split_seed", "model_seed"], columns="split", values="score")
        if "random" not in piv or "cluster" not in piv:
            continue
        paired = piv[["random", "cluster"]].dropna()
        gaps = [generalization_gap(metric, r.random, r.cluster) for r in paired.itertuples()]
        st = gap_stats(gaps)
        recs.append(dict(dataset=dataset, metric=metric, random_mean=float(paired["random"].mean()),
                         cluster_mean=float(paired["cluster"].mean()), gap_fm=st["mean"],
                         established=st["established"]))
    fmdf = pd.DataFrame(recs)
    fmdf.to_csv(out_path, index=False)

    try:
        fp = pd.read_csv("results/atlas_full_gaps.csv")
        fp = fp[(fp.realistic == "cluster") & (fp.model == "xgboost")][
            ["dataset", "mean", "random_mean"]].rename(columns={"mean": "gap_fp", "random_mean": "fp_random"})
    except FileNotFoundError:
        print("(no fingerprint atlas to compare against)")
        return
    m = fmdf.merge(fp, on="dataset")
    m["relgap_fm"] = m["gap_fm"] / m["random_mean"].abs().clip(lower=1e-9)
    m["relgap_fp"] = m["gap_fp"] / m["fp_random"].abs().clip(lower=1e-9)
    d = (m["relgap_fm"] - m["relgap_fp"]).values

    print("\n" + "=" * 84)
    print(f"DOES {fm.upper()} CLOSE THE GAP?  relative random->cluster gap: foundation model vs ECFP4")
    print("=" * 84)
    print(f"{'dataset':<26}{'metric':<9}{'relgap_FM':>11}{'relgap_FP':>11}{'Δ(FM-FP)':>11}")
    print("-" * 68)
    for r in m.sort_values("relgap_fp", ascending=False).itertuples():
        print(f"{r.dataset:<26}{r.metric:<9}{r.relgap_fm:>11.3f}{r.relgap_fp:>11.3f}"
              f"{r.relgap_fm - r.relgap_fp:>+11.3f}")
    try:
        p = wilcoxon(d).pvalue
    except ValueError:
        p = float("nan")
    verdict = "SHRINKS" if d.mean() < 0 else "WIDENS"
    print(f"\nAcross {len(m)} endpoints: mean Δ(FM-FP) = {d.mean():+.3f}, Wilcoxon p={p:.4f}  "
          f"=> foundation model {verdict} the gap on average (negative Δ = smaller gap).")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="bbb_martins,caco2_wang,herg,bioavailability_ma")
    p.add_argument("--fm", default="chemberta", choices=["chemberta", "molformer"])
    p.add_argument("--head", default="xgboost")
    p.add_argument("--split-seeds", type=int, default=5)
    p.add_argument("--model-seeds", type=int, default=3)
    p.add_argument("--out", default="results/foundation.csv")
    a = p.parse_args()
    datasets = list_admet() if a.datasets == "all" else a.datasets.split(",")
    df = run(datasets, a.fm, list(range(a.split_seeds)), list(range(a.model_seeds)), a.head, Path(a.out))
    summarize(df, a.fm, Path(a.out).with_name(Path(a.out).stem + "_gaps.csv"))


if __name__ == "__main__":
    main()
