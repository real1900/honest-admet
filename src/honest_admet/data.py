"""Data layer for Honest ADMET.

Loads the 22 datasets of the TDC ADMET benchmark group and re-splits each one
under controlled regimes so we can measure the *generalization gap* between an
optimistic random split and realistic chemistry-aware splits.

Two loading modes:
  * ``load_admet``         — pools TDC's official train_val + test into one set and
                             lets us re-split it under controlled regimes (random /
                             scaffold / cluster) with explicit seeds. Pooled numbers
                             are for internal split-type comparison only and are NOT
                             leaderboard-comparable.
  * ``load_admet_official``— returns TDC's native train_val / test partition,
                             unpooled, for reproducing published leaderboard numbers.

Design notes addressing review feedback:
  * Murcko scaffolds distinguish parse-failure (dropped at load) from genuinely
    acyclic molecules (treated as singleton groups, not lumped into one bucket).
  * Cluster split uses scalable sphere-exclusion (RDKit LeaderPicker), O(N*L), so it
    does not build an O(N^2) distance matrix and will not OOM on the ~13k CYP sets.
  * Canonical-SMILES de-duplication prevents identical molecules leaking across folds.
  * Leakage is reported with a splitter-INDEPENDENT nearest-neighbour Tanimoto metric.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")  # silence RDKit parse chatter

# --- TDC ADMET benchmark group: task type + official leaderboard metric -------
# Keys match tdc admet_group.dataset_names (lowercase). 9 regression, 13 classification.
ADMET_METRICS: dict[str, tuple[str, str]] = {
    # regression
    "caco2_wang": ("regression", "mae"),
    "lipophilicity_astrazeneca": ("regression", "mae"),
    "solubility_aqsoldb": ("regression", "mae"),
    "ppbr_az": ("regression", "mae"),
    "ld50_zhu": ("regression", "mae"),
    "vdss_lombardo": ("regression", "spearman"),
    "half_life_obach": ("regression", "spearman"),
    "clearance_hepatocyte_az": ("regression", "spearman"),
    "clearance_microsome_az": ("regression", "spearman"),
    # classification
    "hia_hou": ("classification", "auroc"),
    "pgp_broccatelli": ("classification", "auroc"),
    "bioavailability_ma": ("classification", "auroc"),
    "bbb_martins": ("classification", "auroc"),
    "cyp3a4_substrate_carbonmangels": ("classification", "auroc"),
    "herg": ("classification", "auroc"),
    "ames": ("classification", "auroc"),
    "dili": ("classification", "auroc"),
    "cyp2c9_veith": ("classification", "auprc"),
    "cyp2d6_veith": ("classification", "auprc"),
    "cyp3a4_veith": ("classification", "auprc"),
    "cyp2c9_substrate_carbonmangels": ("classification", "auprc"),
    "cyp2d6_substrate_carbonmangels": ("classification", "auprc"),
}

SPLIT_METHODS = ("random", "scaffold", "cluster")

# shared Morgan generator (ECFP4) — one featurization path for splitter + features
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_FP_CACHE: dict[str, list] = {}  # smiles-set hash -> list[ExplicitBitVect]


@dataclass
class Dataset:
    """A pooled ADMET dataset ready for re-splitting."""

    name: str
    task: str  # "classification" | "regression"
    metric: str  # official TDC metric: mae | spearman | auroc | auprc
    smiles: np.ndarray  # (N,) canonical SMILES strings (deduplicated)
    y: np.ndarray  # (N,) labels (float; 0/1 for classification)
    n_dropped: int = 0  # parse failures dropped at load
    n_dups: int = 0  # duplicate canonical SMILES merged at load

    def __len__(self) -> int:
        return len(self.y)


# --- loading ------------------------------------------------------------------
def _clean(drug: np.ndarray, y: np.ndarray, task: str):
    """Canonicalize, drop parse failures, and merge duplicate canonical SMILES."""
    canon: dict[str, list[float]] = defaultdict(list)
    n_dropped = 0
    for smi, label in zip(drug, y):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_dropped += 1
            continue
        canon[Chem.MolToSmiles(mol)].append(float(label))
    smiles, labels, n_dups = [], [], 0
    for smi, vals in canon.items():
        if len(vals) > 1:
            n_dups += len(vals) - 1
        agg = float(np.mean(vals))
        labels.append(1.0 if (task == "classification" and agg >= 0.5) else
                      (0.0 if task == "classification" else agg))
        smiles.append(smi)
    return np.asarray(smiles), np.asarray(labels, dtype=float), n_dropped, n_dups


def load_admet(name: str, path: str = "data/") -> Dataset:
    """Load one ADMET dataset, pooling TDC's official train_val + test, then
    canonicalizing and de-duplicating. For the controlled split-type comparison."""
    name = name.lower()
    if name not in ADMET_METRICS:
        raise KeyError(f"Unknown ADMET dataset {name!r}. Known: {sorted(ADMET_METRICS)}")
    from tdc.benchmark_group import admet_group

    group = admet_group(path=path)
    bench = group.get(name)
    pooled = pd.concat([bench["train_val"], bench["test"]], ignore_index=True)
    task, metric = ADMET_METRICS[name]
    smiles, y, n_dropped, n_dups = _clean(pooled["Drug"].to_numpy(), pooled["Y"].to_numpy(), task)
    return Dataset(name, task, metric, smiles, y, n_dropped, n_dups)


def load_admet_official(name: str, path: str = "data/"):
    """TDC's native train_val / test partition (unpooled) for leaderboard repro.
    Returns (train_val_df, test_df) with columns Drug_ID, Drug, Y plus task/metric."""
    name = name.lower()
    from tdc.benchmark_group import admet_group

    group = admet_group(path=path)
    bench = group.get(name)
    task, metric = ADMET_METRICS[name]
    return bench["train_val"], bench["test"], task, metric


def list_admet() -> list[str]:
    """All 22 ADMET dataset names."""
    return sorted(ADMET_METRICS)


# --- fingerprints (shared, cached) --------------------------------------------
def _fps(smiles: np.ndarray) -> list:
    """Cached ECFP4 ExplicitBitVects for a SMILES set (seed-independent)."""
    key = hashlib.sha1("\n".join(smiles.tolist()).encode()).hexdigest()
    if key not in _FP_CACHE:
        _FP_CACHE[key] = [_MORGAN.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]
    return _FP_CACHE[key]


# --- scaffolds ----------------------------------------------------------------
def murcko_scaffold(smiles: str):
    """Bemis-Murcko scaffold SMILES. Returns None on parse failure, "" for a
    genuinely acyclic molecule (no ring system) — callers must NOT lump the two."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def _scaffold_groups(smiles: np.ndarray) -> dict[str, list[int]]:
    """Group indices by Murcko scaffold; acyclic/parse-fail molecules become
    singleton groups (unique key) so they distribute across folds, not one bucket."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(smiles):
        scaf = murcko_scaffold(smi)
        key = scaf if scaf else f"__singleton_{i}"  # "" or None -> unique singleton
        groups[key].append(i)
    return groups


# --- generic grouped greedy packing (independent fold caps; DeepChem-style) ---
def _pack_groups(groups: list[list[int]], n: int, frac, seed: int) -> dict[str, np.ndarray]:
    """Assign whole groups to train/valid/test with INDEPENDENT caps. Largest
    groups first; ``seed`` shuffles tie order (and, for genuine stochasticity,
    perturbs the ordering of equal-priority groups)."""
    rng = np.random.default_rng(seed)
    n_train = int(np.floor(frac[0] * n))
    n_valid = int(np.floor(frac[1] * n))
    order = sorted(range(len(groups)), key=lambda g: (-len(groups[g]), rng.random()))
    train, valid, test = [], [], []
    for g in order:
        idx = groups[g]
        if len(train) + len(idx) <= n_train:
            train += idx
        elif len(valid) + len(idx) <= n_valid:
            valid += idx
        else:
            test += idx
    return {k: np.asarray(v, dtype=int) for k, v in
            (("train", train), ("valid", valid), ("test", test))}


# --- splitters ----------------------------------------------------------------
def random_split(smiles: np.ndarray, frac=(0.7, 0.1, 0.2), seed: int = 0) -> dict[str, np.ndarray]:
    """Uniform random split — the optimistic baseline."""
    n = len(smiles)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(np.floor(frac[0] * n))
    n_valid = int(np.floor(frac[1] * n))
    return {"train": perm[:n_train], "valid": perm[n_train:n_train + n_valid],
            "test": perm[n_train + n_valid:]}


def scaffold_split(smiles: np.ndarray, frac=(0.7, 0.1, 0.2), seed: int = 0) -> dict[str, np.ndarray]:
    """Bemis-Murcko scaffold split: whole scaffolds go to one fold only. Acyclic
    molecules are singleton groups (not lumped). NOTE: the community-default
    scaffold split is a relatively weak OOD stressor (Fooladi et al. 2025); we
    report it but treat ``cluster`` as the primary stressor."""
    groups = list(_scaffold_groups(smiles).values())
    return _pack_groups(groups, len(smiles), frac, seed)


def cluster_split(smiles: np.ndarray, frac=(0.7, 0.1, 0.2), seed: int = 0,
                  cutoff: float = 0.65) -> dict[str, np.ndarray]:
    """Sphere-exclusion (Leader) clustering split on ECFP4 — a stricter OOD test
    than scaffold split. O(N*L) via RDKit LeaderPicker; no O(N^2) matrix, so it is
    memory-safe on the ~13k-molecule CYP datasets. ``cutoff`` is Tanimoto DISTANCE
    between cluster leaders."""
    from rdkit.SimDivFilters import rdSimDivPickers

    fps = _fps(smiles)
    picker = rdSimDivPickers.LeaderPicker()
    leaders = list(picker.LazyBitVectorPick(fps, len(fps), cutoff))
    leader_fps = [fps[i] for i in leaders]
    clusters: dict[int, list[int]] = defaultdict(list)
    for i, fp in enumerate(fps):
        sims = DataStructs.BulkTanimotoSimilarity(fp, leader_fps)
        clusters[int(np.argmax(sims))].append(i)
    return _pack_groups(list(clusters.values()), len(smiles), frac, seed)


SPLITTERS = {"random": random_split, "scaffold": scaffold_split, "cluster": cluster_split}


def make_split(method: str, smiles: np.ndarray, frac=(0.7, 0.1, 0.2), seed: int = 0) -> dict[str, np.ndarray]:
    if method not in SPLITTERS:
        raise ValueError(f"method must be one of {SPLIT_METHODS}, got {method!r}")
    return SPLITTERS[method](smiles, frac=frac, seed=seed)


# --- diagnostics --------------------------------------------------------------
def nn_tanimoto_leakage(smiles: np.ndarray, split: dict[str, np.ndarray],
                        threshold: float = 0.4) -> dict[str, float]:
    """Splitter-INDEPENDENT leakage: for each test molecule, max ECFP4 Tanimoto to
    any train molecule. Returns fraction above ``threshold`` and the mean max-sim.
    Fairly characterizes random/scaffold/cluster splits on the same footing."""
    fps = _fps(smiles)
    train_fps = [fps[i] for i in split["train"]]
    maxes = []
    for i in split["test"]:
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], train_fps)
        maxes.append(max(sims) if sims else 0.0)
    maxes = np.asarray(maxes)
    return {"frac_above": float((maxes > threshold).mean()),
            "mean_max_sim": float(maxes.mean()), "threshold": threshold}


def test_train_max_sim(smiles: np.ndarray, split: dict[str, np.ndarray]) -> np.ndarray:
    """Per-test-molecule maximum ECFP4 Tanimoto to any training molecule — an
    applicability-domain confidence signal (higher = more in-domain) for selective
    prediction. Label-free, so usable as an abstention rule at inference time."""
    fps = _fps(smiles)
    train_fps = [fps[i] for i in split["train"]]
    out = np.empty(len(split["test"]))
    for j, i in enumerate(split["test"]):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], train_fps)
        out[j] = max(sims) if sims else 0.0
    return out


def fold_label_stats(y: np.ndarray, split: dict[str, np.ndarray], task: str) -> dict:
    """Per-fold label statistics to diagnose label-distribution shift across splits."""
    out = {}
    for fold in ("train", "valid", "test"):
        yi = y[split[fold]]
        if len(yi) == 0:
            out[fold] = {}
        elif task == "classification":
            out[fold] = {"n": int(len(yi)), "pos_rate": float(yi.mean())}
        else:
            out[fold] = {"n": int(len(yi)), "y_mean": float(yi.mean()),
                         "y_std": float(yi.std()), "y_min": float(yi.min()),
                         "y_max": float(yi.max())}
    return out
