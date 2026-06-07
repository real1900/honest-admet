"""Smoke test: confirm the environment, GPU backend, and live TDC data access.

Run (on the M4 Pro):
    cd ~/research/honest-admet && .venv/bin/python experiments/00_smoke_test.py
"""

import platform


def main() -> None:
    print(f"python      : {platform.python_version()} ({platform.machine()})")

    # --- core libs import cleanly ---
    import numpy as np
    import sklearn
    import xgboost
    import rdkit
    from rdkit import Chem

    print(f"numpy       : {np.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"xgboost     : {xgboost.__version__}")
    print(f"rdkit       : {rdkit.__version__}")

    # --- torch + Apple Metal (MPS) GPU backend ---
    import torch

    mps = torch.backends.mps.is_available()
    print(f"torch       : {torch.__version__}  | MPS (Metal GPU): {mps}")
    device = "mps" if mps else "cpu"
    # tiny op on the chosen device to prove it works end-to-end
    x = torch.randn(1024, 1024, device=device)
    _ = (x @ x).sum().item()
    print(f"device check: matmul on '{device}' OK")

    # --- RDKit can parse a molecule ---
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    print(f"rdkit parse : aspirin -> {mol.GetNumAtoms()} atoms")

    # --- live TDC ADMET benchmark group: list datasets, load one ---
    from tdc.benchmark_group import admet_group

    group = admet_group(path="data/")
    names = group.dataset_names
    print(f"\nTDC ADMET group: {len(names)} datasets")
    print("  e.g.:", ", ".join(sorted(names)[:6]), "...")

    bench = group.get("Caco2_Wang")
    train_val, test = bench["train_val"], bench["test"]
    print(f"\nCaco2_Wang  : train_val={train_val.shape}, test={test.shape}")
    print("columns     :", list(train_val.columns))
    print(train_val.head(3).to_string(index=False))

    print("\n✅ smoke test passed")


if __name__ == "__main__":
    main()
