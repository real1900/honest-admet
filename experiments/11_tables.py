"""Emit LaTeX table fragments (booktabs) from the result CSVs into paper/tables/.

Guarantees the manuscript's numbers exactly match the experiments. Run after the
analyses; missing inputs are skipped.
    cd ~/research/honest-admet && .venv/bin/python experiments/11_tables.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path("paper/tables")
OUT.mkdir(parents=True, exist_ok=True)


def esc(s):
    return str(s).replace("_", r"\_")


def write(name, lines):
    (OUT / name).write_text("\n".join(lines) + "\n")
    print("wrote", name)


def tab_leaderboard():
    p = Path("results/leaderboard.csv")
    if not p.exists():
        return print("skip leaderboard (no csv)")
    d = pd.read_csv(p).query("model == 'xgboost'").sort_values("dataset")
    rows = [r"\begin{tabular}{llr}", r"\toprule",
            r"Endpoint & Metric & ECFP4+XGBoost (mean$\pm$std) \\", r"\midrule"]
    for r in d.itertuples():
        rows.append(f"{esc(r.dataset)} & {r.metric} & {r.mean:.3f} $\\pm$ {r.std:.3f} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("tab_leaderboard.tex", rows)


def _relgap(df):
    df = df.copy()
    df["relgap"] = df["mean"] / df["random_mean"].abs().clip(lower=1e-9)
    return df


def tab_atlas():
    """Relative gaps with split-seed block-bootstrap CIs and established flags."""
    p = Path("results/atlas_relgap_blockci.csv")
    if not p.exists():
        return print("skip atlas (no block-ci csv)")
    g = pd.read_csv(p).sort_values("cluster_relgap", ascending=False)
    rows = [r"\begin{tabular}{llrcrl}", r"\toprule",
            r"Endpoint & Metric & Scaffold & Sc.\ est? & Cluster & Cluster 95\% CI \\",
            r"\midrule"]
    for r in g.itertuples():
        ns = "" if r.cluster_est else r" \textit{(n.s.)}"
        rows.append(f"{esc(r.dataset)} & {r.metric} & {r.scaffold_relgap:+.3f} & "
                    f"{'Y' if r.scaffold_est else 'N'} & {r.cluster_relgap:+.3f} & "
                    f"[{r.cluster_lo:+.3f}, {r.cluster_hi:+.3f}]{ns} \\\\")
    rows += [r"\midrule",
             f"\\textit{{Median}} & & {g.scaffold_relgap.median():+.3f} & "
             f"{int(g.scaffold_est.sum())}/22 & {g.cluster_relgap.median():+.3f} & "
             f"{int(g.cluster_est.sum())}/22 est. \\\\",
             r"\bottomrule", r"\end{tabular}"]
    write("tab_atlas.tex", rows)


def tab_foundation():
    fp_p, fm_p = Path("results/atlas_full_gaps.csv"), Path("results/foundation_gaps.csv")
    if not (fp_p.exists() and fm_p.exists()):
        return print("skip foundation (no csv)")
    fp = _relgap(pd.read_csv(fp_p).query("model == 'xgboost' and realistic == 'cluster'"))
    fp = fp.set_index("dataset")["relgap"]
    fm = pd.read_csv(fm_p)
    fm["relgap"] = fm["gap_fm"] / fm["random_mean"].abs().clip(lower=1e-9)
    fm = fm.set_index("dataset")["relgap"]
    m = pd.concat([fp.rename("fp"), fm.rename("cb")], axis=1).dropna()
    m["delta"] = m["cb"] - m["fp"]
    m = m.sort_values("delta")
    rows = [r"\begin{tabular}{lrrr}", r"\toprule",
            r"Endpoint & ECFP4 & ChemBERTa & $\Delta$ \\", r"\midrule"]
    for name, r in m.iterrows():
        rows.append(f"{esc(name)} & {r.fp:.3f} & {r.cb:.3f} & {r.delta:+.3f} \\\\")
    rows += [r"\midrule",
             f"Mean & {m.fp.mean():.3f} & {m.cb.mean():.3f} & {m.delta.mean():+.3f} \\\\",
             r"\bottomrule", r"\end{tabular}"]
    write("tab_foundation.tex", rows)


if __name__ == "__main__":
    tab_leaderboard(); tab_atlas(); tab_foundation()
    print("tables ->", OUT)
