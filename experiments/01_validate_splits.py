"""Validate the split layer and quantify leakage with a SPLITTER-INDEPENDENT metric.

The premise of Honest ADMET: under random splits the test set is chemically close to
training (inflating apparent performance); realistic splits push it away. We measure
that with nearest-neighbour ECFP4 Tanimoto similarity (test->train), which scores all
three split types on the same footing — unlike scaffold-overlap, which is ~0 by
construction for the scaffold split and so cannot fairly compare across splits.

Run (on the M4 Pro):
    cd ~/research/honest-admet && .venv/bin/python experiments/01_validate_splits.py
"""

import numpy as np

from honest_admet.data import (
    SPLIT_METHODS,
    fold_label_stats,
    load_admet,
    make_split,
    nn_tanimoto_leakage,
)


def check_partition(split, n):
    all_idx = np.concatenate([split["train"], split["valid"], split["test"]])
    assert len(all_idx) == n, f"coverage {len(all_idx)} != {n}"
    assert len(np.unique(all_idx)) == n, "folds overlap"


def main():
    datasets = ["bbb_martins", "caco2_wang", "herg", "solubility_aqsoldb", "cyp2c9_veith"]
    print(f"{'dataset':<20}{'split':<9}{'tr/va/te':>16}{'NN-leak%':>10}"
          f"{'meanMaxSim':>12}{'test pos/y':>12}")
    print("-" * 79)
    for name in datasets:
        ds = load_admet(name)
        print(f"# {name}: N={len(ds)}  task={ds.task}/{ds.metric}  "
              f"dropped={ds.n_dropped} dups_merged={ds.n_dups}")
        for method in SPLIT_METHODS:
            sp = make_split(method, ds.smiles, seed=0)
            check_partition(sp, len(ds))
            leak = nn_tanimoto_leakage(ds.smiles, sp, threshold=0.4)
            lab = fold_label_stats(ds.y, sp, ds.task)["test"]
            tag = (f"{lab['pos_rate']:.2f}" if ds.task == "classification"
                   else f"{lab['y_mean']:.2f}")
            sizes = f"{len(sp['train'])}/{len(sp['valid'])}/{len(sp['test'])}"
            print(f"{'':<20}{method:<9}{sizes:>16}{100 * leak['frac_above']:>9.1f}%"
                  f"{leak['mean_max_sim']:>12.3f}{tag:>12}")
        print("-" * 79)

    print("\nNN-leak% = fraction of test molecules with ECFP4 Tanimoto > 0.4 to some train "
          "molecule.\nExpect: random > cluster, and cluster the lowest (hardest OOD). "
          "Watch test pos/y\nshift across splits — that is the label-distribution confound "
          "the stats stage controls for.")
    print("\n✅ split validation passed")


if __name__ == "__main__":
    main()
