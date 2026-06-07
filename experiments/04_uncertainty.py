"""Experiment 2: trustworthy uncertainty under distribution shift (multi-seed).

Two questions, evaluated SEPARATELY on random / scaffold / cluster test sets, over many
split seeds so every number has error bars:

  (A) Calibration under shift. Does split-conformal (target 1-alpha coverage) still cover
      under realistic splits? Does covariate-shift weighting (ESS-guarded, calibrated
      domain-classifier importance weights) restore coverage — and is it a no-op on random
      splits (the control it must pass)?

  (B) Selective prediction. Abstaining on the least-confident predictions, how much
      ordering quality do two label-free signals have — deep-ensemble disagreement vs
      applicability-domain similarity (NN Tanimoto to train)? Measured by Excess-AURC
      (AURC - oracle), which is comparable across folds with different base error rates,
      with a PAIRED Wilcoxon test across seeds.

Run (on the M4 Pro):
    cd ~/research/honest-admet && .venv/bin/python experiments/04_uncertainty.py \
        --datasets caco2_wang,bbb_martins,herg,solubility_aqsoldb --split-seeds 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from honest_admet import selective
from honest_admet.conformal import (
    classification_coverage,
    conformal_lac_q,
    conformal_regression_q,
    domain_classifier_weights,
    kish_ess,
    regression_coverage,
)
from honest_admet.data import (
    SPLIT_METHODS,
    fold_label_stats,
    list_admet,
    load_admet,
    make_split,
    test_train_max_sim,
)
from honest_admet.features import morgan_fingerprints
from honest_admet.models.baselines import fit_model, predict

ESS_FLOOR = 0.1  # fall back to unweighted conformal when Kish ESS fraction < this


def run(datasets, splits, split_seeds, model, alpha, n_members, out_path: Path) -> pd.DataFrame:
    rows = []
    for name in datasets:
        ds = load_admet(name)
        X = morgan_fingerprints(ds.smiles)
        print(f"\n=== {name} ({ds.task}/{ds.metric}, N={len(ds)}) target cov={1 - alpha:.2f} ===")
        for split in splits:
            for seed in split_seeds:
                sp = make_split(split, ds.smiles, seed=seed)
                tr, cal, te = sp["train"], sp["valid"], sp["test"]
                members = [fit_model(model, ds.task, X[tr], ds.y[tr], seed=s) for s in range(n_members)]
                P_cal = np.stack([predict(m, ds.task, X[cal]) for m in members])
                P_te = np.stack([predict(m, ds.task, X[te]) for m in members])
                mean_cal, mean_te, std_te = P_cal.mean(0), P_te.mean(0), P_te.std(0)

                # (A) conformal coverage: unweighted vs ESS-guarded weighted
                w = domain_classifier_weights(X[cal], X[te])
                ess = kish_ess(w)
                use_w = w if ess >= ESS_FLOOR else None  # fall back when weights collapse
                cov_fn = regression_coverage if ds.task == "regression" else classification_coverage
                q_fn = conformal_regression_q if ds.task == "regression" else conformal_lac_q
                q = q_fn(mean_cal, ds.y[cal], alpha)
                qw = q_fn(mean_cal, ds.y[cal], alpha, weights=use_w)
                cov, width = cov_fn(mean_te, ds.y[te], q)
                cov_w, _ = cov_fn(mean_te, ds.y[te], qw)

                # (B) selective prediction: Excess-AURC (base-rate comparable) + the
                # risk removed by abstaining on the 20% least-confident (for the unifying
                # gap<->abstention analysis). AD = applicability-domain distance signal.
                err = selective.errors(ds.task, ds.y[te], mean_te)
                conf_ad = test_train_max_sim(ds.smiles, sp)
                lab = fold_label_stats(ds.y, sp, ds.task)["test"]
                full_risk = float(err.mean())
                rows.append(dict(
                    dataset=name, task=ds.task, split=split, seed=seed,
                    coverage=cov, coverage_weighted=cov_w, ess_frac=ess,
                    ess_fallback=(use_w is None), conf_width=width, full_risk=full_risk,
                    eaurc_ens=selective.excess_aurc(-std_te, err),
                    eaurc_ad=selective.excess_aurc(conf_ad, err),
                    sel_risk80_ad=selective.risk_at_coverage(conf_ad, err, 0.8),
                    sel_risk80_ens=selective.risk_at_coverage(-std_te, err, 0.8),
                    test_pos_or_ymean=round(float(lab.get("pos_rate", lab.get("y_mean", np.nan))), 4),
                    n_test=len(te)))
            sub = [r for r in rows if r["dataset"] == name and r["split"] == split]
            cov_m = np.mean([r["coverage"] for r in sub])
            print(f"  {split:<9} cov={cov_m:.3f} over {len(sub)} seeds")
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path} ({len(df)} rows)")
    return df


def summarize(df: pd.DataFrame, alpha: float) -> None:
    print("\n" + "=" * 100)
    print(f"COVERAGE UNDER SHIFT (target {1 - alpha:.0%}) + ABSTENTION (Excess-AURC, lower=better; "
          f"mean±std over seeds)")
    print("=" * 100)
    print(f"{'dataset':<20}{'split':<9}{'cov':>12}{'cov_wtd':>12}{'ESS':>6}"
          f"{'E-AURC ens':>14}{'E-AURC ad':>14}{'Δ(ens-ad)':>12}{'p':>7}")
    print("-" * 100)
    for (dataset, split), g in df.groupby(["dataset", "split"]):
        delta = g["eaurc_ens"].values - g["eaurc_ad"].values
        try:
            p = wilcoxon(delta).pvalue if np.any(delta != 0) else float("nan")
        except ValueError:
            p = float("nan")
        adwin = "ad" if delta.mean() > 0 else "ens"
        print(f"{dataset:<20}{split:<9}"
              f"{g['coverage'].mean():>7.3f}±{g['coverage'].std():>3.2f}"
              f"{g['coverage_weighted'].mean():>7.3f}±{g['coverage_weighted'].std():>3.2f}"
              f"{g['ess_frac'].mean():>6.2f}"
              f"{g['eaurc_ens'].mean():>10.3f}±{g['eaurc_ens'].std():>3.2f}"
              f"{g['eaurc_ad'].mean():>10.3f}±{g['eaurc_ad'].std():>3.2f}"
              f"{delta.mean():>+12.3f}{p:>7.3f}")
    print("\ncov should sit near target on random/scaffold and DROP under cluster (the honest "
          "failure).\ncov_wtd: weighting should restore cluster coverage and be a no-op on random "
          "(control).\nΔ(ens-ad)>0 with small p ⇒ AD-distance is the better abstention signal "
          "(ensemble degrades under shift).")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="caco2_wang,bbb_martins,herg,solubility_aqsoldb")
    p.add_argument("--splits", default=",".join(SPLIT_METHODS))
    p.add_argument("--split-seeds", type=int, default=10)
    p.add_argument("--model", default="rf")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--members", type=int, default=5)
    p.add_argument("--out", default="results/uncertainty.csv")
    a = p.parse_args()
    datasets = list_admet() if a.datasets == "all" else a.datasets.split(",")
    df = run(datasets, a.splits.split(","), list(range(a.split_seeds)),
             a.model, a.alpha, a.members, Path(a.out))
    summarize(df, a.alpha)


if __name__ == "__main__":
    main()
