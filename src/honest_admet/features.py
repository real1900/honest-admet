"""Molecular featurization. Morgan/ECFP4 fingerprints for the classical baselines.

Uses the same shared MorganGenerator as the splitter (data._MORGAN) so the loader,
the cluster splitter, and the model features never disagree on featurization.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs

from .data import _MORGAN


def morgan_fingerprints(smiles: np.ndarray, n_bits: int = 2048) -> np.ndarray:
    """ECFP4 Morgan fingerprints as a dense (N, n_bits) uint8 array.

    Raises if any SMILES fails to parse — callers should pass already-cleaned
    SMILES (load_admet drops parse failures), so a failure here is a real bug
    rather than something to silently turn into an all-zero row.
    """
    out = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"unparseable SMILES reached featurization: {smi!r}")
        DataStructs.ConvertToNumpyArray(_MORGAN.GetFingerprint(mol), out[i])
    return out
