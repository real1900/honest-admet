"""Experiment 1: the generalization-gap atlas (fingerprint baselines).

Measures how much test performance drops moving from an optimistic random split to
realistic scaffold / cluster splits, across datasets and models. Every gap is a
PAIRED, matched-replicate quantity with a bootstrap CI, a Wilcoxon p-value, and a
Benjamini-Hochberg q-value across the dataset family. Gaps whose CI crosses 0 are
labelled 'not established'. cluster is the primary stressor (scaffold ~ random for
ADMET per Fooladi et al. 2025).

Randomness is decoupled: ``split_seed`` controls the partition, ``model_seed`` the
model — so split-variance and model-variance are not confounded.

Examples (on the M4 Pro):
    .venv/bin/python experiments/02_baselines.py \
        --datasets bbb_martins,caco2_wang,herg,dili,bioavailability_ma \
        --split-seeds 5 --model-seeds 3
    .venv/bin/python experiments/02_baselines.py --datasets all \
        --split-seeds 10 --model-seeds 5 --out results/baselines_full.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from honest_admet.data import fold_label_stats, list_admet, load_admet, make_split
from honest_admet.eval import generalization_gap, score
from honest_admet.features import morgan_fingerprints
from honest_admet.models.baselines import BASELINES, fit_predict
from honest_admet.stats import aggregate_gap_test, benjamini_hochberg, gap_stats


def run(datasets, splits, split_seeds, model_seeds, models, out_path: Path) -> pd.DataFrame:
    rows = []
    for name in datasets:
        ds = load_admet(name)
        X = morgan_fingerprints(ds.smiles)
        print(f"\n=== {name} ({ds.task}/{ds.metric}, N={len(ds)}, "
              f"dropped={ds.n_dropped}, dups={ds.n_dups}) ===")
        for ss in split_seeds:
            sp = {m: make_split(m, ds.smiles, seed=ss) for m in splits}
            for m in splits:
                fit_idx = np.concatenate([sp[m]["train"], sp[m]["valid"]])
                te = sp[m]["test"]
                lab = fold_label_stats(ds.y, sp[m], ds.task)["test"]
                shift = lab.get("pos_rate", lab.get("y_mean", float("nan")))
                for model in models:
                    for ms in model_seeds:
                        t0 = time.time()
                        pred = fit_predict(model, ds.task, X[fit_idx], ds.y[fit_idx],
                                           X[te], seed=ms)
                        rows.append(dict(
                            dataset=name, task=ds.task, metric=ds.metric, model=model,
                            split=m, split_seed=ss, model_seed=ms,
                            score=score(ds.metric, ds.y[te], pred),
                            test_label=round(float(shift), 4), n_test=len(te),
                            secs=round(time.time() - t0, 1)))
        n = len(split_seeds) * len(splits) * len(models) * len(model_seeds)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)  # incremental: crash-safe + monitorable
        print(f"  {n} fits done — {len(rows)} rows written", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path} ({len(df)} rows)")
    return df


def summarize(df: pd.DataFrame, out_path: Path) -> None:
    """Matched-replicate gaps with CIs, Wilcoxon p, BH-FDR q, and an aggregate test."""
    records = []
    for (dataset, model), sub in df.groupby(["dataset", "model"]):
        metric = sub["metric"].iloc[0]
        piv = sub.pivot_table(index=["split_seed", "model_seed"], columns="split", values="score")
        if "random" not in piv:
            continue
        for realistic in ("scaffold", "cluster"):
            if realistic not in piv:
                continue
            paired = piv[["random", realistic]].dropna()
            gaps = [generalization_gap(metric, r.random, getattr(r, realistic))
                    for r in paired.itertuples()]
            st = gap_stats(gaps)
            records.append(dict(dataset=dataset, model=model, metric=metric, realistic=realistic,
                                random_mean=float(paired["random"].mean()),
                                realistic_mean=float(paired[realistic].mean()), **st))
    gaps_df = pd.DataFrame(records)
    if gaps_df.empty:
        return
    # BH-FDR within each (model, realistic) family across datasets
    gaps_df["q_value"] = np.nan
    for _, idx in gaps_df.groupby(["model", "realistic"]).groups.items():
        gaps_df.loc[idx, "q_value"] = benjamini_hochberg(gaps_df.loc[idx, "p_wilcoxon"].values)
    gaps_df.to_csv(out_path, index=False)

    for realistic in ("scaffold", "cluster"):
        block = gaps_df[gaps_df["realistic"] == realistic]
        if block.empty:
            continue
        print("\n" + "=" * 86)
        print(f"GENERALIZATION GAP: random -> {realistic.upper()}  (+gap = worse; "
              f"'est' = 95% CI excludes 0)")
        print("=" * 86)
        print(f"{'dataset':<22}{'model':<8}{'metric':<8}{'random':>8}{realistic:>9}"
              f"{'gap':>8}{'95% CI':>16}{'q':>7}{'est':>5}")
        print("-" * 86)
        for r in block.sort_values(["dataset", "model"]).itertuples():
            ci = f"[{r.ci_lo:+.3f},{r.ci_hi:+.3f}]"
            est = "yes" if r.established else " no"
            print(f"{r.dataset:<22}{r.model:<8}{r.metric:<8}{r.random_mean:>8.3f}"
                  f"{r.realistic_mean:>9.3f}{r.mean:>+8.3f}{ci:>16}{r.q_value:>7.3f}{est:>5}")
        agg = aggregate_gap_test(block.groupby("dataset")["mean"].mean().values)
        print(f"\nAGGREGATE ({realistic}): mean per-dataset gap={agg['mean_gap']:+.3f}, "
              f"median={agg['median_gap']:+.3f}, positive on {agg['n_positive']}/"
              f"{agg['n_datasets']} datasets, Wilcoxon p={agg['p_wilcoxon']:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="bbb_martins,caco2_wang,herg,dili,bioavailability_ma")
    p.add_argument("--splits", default="random,scaffold,cluster")
    p.add_argument("--split-seeds", type=int, default=5)
    p.add_argument("--model-seeds", type=int, default=3)
    p.add_argument("--models", default=",".join(BASELINES))
    p.add_argument("--out", default="results/baselines.csv")
    a = p.parse_args()
    datasets = list_admet() if a.datasets == "all" else a.datasets.split(",")
    df = run(datasets, a.splits.split(","), list(range(a.split_seeds)),
             list(range(a.model_seeds)), a.models.split(","), Path(a.out))
    summarize(df, Path(a.out).with_name(Path(a.out).stem + "_gaps.csv"))


if __name__ == "__main__":
    main()
