"""Experiment 0 (credibility): reproduce TDC ADMET leaderboard numbers.

Runs our exact featurizer (ECFP4) + baselines (XGBoost / RF) on TDC's OFFICIAL,
unpooled split using the admet_group 5-seed evaluate protocol, so the numbers are
directly comparable to the public leaderboard at:
    https://tdcommons.ai/benchmark/admet_group/

This proves the harness is correct BEFORE we introduce the pooled re-splits used for
the controlled split-type comparison (which are deliberately not leaderboard-
comparable). evaluate_many() applies each dataset's official metric automatically.

Run (on the M4 Pro):
    cd ~/research/honest-admet && .venv/bin/python experiments/03_leaderboard_repro.py \
        --datasets caco2_wang,bbb_martins,herg,solubility_aqsoldb
"""

from __future__ import annotations

import argparse

import numpy as np
from rdkit import Chem

from honest_admet.data import ADMET_METRICS, list_admet
from honest_admet.features import morgan_fingerprints
from honest_admet.models.baselines import fit_predict


def _feat(df):
    """Featurize a TDC benchmark frame; assumes curated (parseable) SMILES."""
    smi = df["Drug"].to_numpy()
    bad = [s for s in smi if Chem.MolFromSmiles(s) is None]
    if bad:
        raise ValueError(f"{len(bad)} unparseable official SMILES, e.g. {bad[0]!r}")
    return morgan_fingerprints(smi), df["Y"].to_numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="caco2_wang,bbb_martins,herg,solubility_aqsoldb")
    p.add_argument("--models", default="xgboost,rf")
    p.add_argument("--path", default="data/")
    a = p.parse_args()
    datasets = list_admet() if a.datasets == "all" else a.datasets.split(",")
    models = a.models.split(",")

    from tdc.benchmark_group import admet_group

    import pandas as pd

    group = admet_group(path=a.path)
    print(f"{'dataset':<24}{'metric':<9}" + "".join(f"{m+' (mean±std)':>22}" for m in models))
    print("-" * (33 + 22 * len(models)))
    rows = []
    for name in datasets:
        task, metric = ADMET_METRICS[name.lower()]
        bench = group.get(name)
        Xtr, ytr = _feat(bench["train_val"])
        Xte, _ = _feat(bench["test"])
        cells = []
        for model in models:
            preds = []
            for seed in (1, 2, 3, 4, 5):  # official 5-seed protocol
                pred = fit_predict(model, task, Xtr, ytr, Xte, seed=seed)
                preds.append({name: pred})
            mean, std = group.evaluate_many(preds)[name]
            cells.append(f"{mean:.3f} ± {std:.3f}")
            rows.append(dict(dataset=name, metric=metric, model=model, mean=mean, std=std))
        print(f"{name:<24}{metric:<9}" + "".join(f"{c:>22}" for c in cells), flush=True)
    pd.DataFrame(rows).to_csv("results/leaderboard.csv", index=False)

    print("\nCompare the above to https://tdcommons.ai/benchmark/admet_group/ — our ECFP4+tree "
          "baselines\nshould land near the published fingerprint/tree baselines, confirming the "
          "harness is sound.")


if __name__ == "__main__":
    main()
