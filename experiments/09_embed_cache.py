"""Pre-extract and cache foundation-model embeddings for all ADMET datasets.

Run this FIRST, in its own (torch-only) process. The downstream gap experiment
(08_foundation.py) then reads embeddings from cache and never imports torch, avoiding
the macOS OpenMP segfault that occurs when torch and XGBoost are loaded together.

Run (on the M4 Pro):
    cd ~/research/honest-admet && .venv/bin/python -u experiments/09_embed_cache.py --fm chemberta
"""

from __future__ import annotations

import argparse

from honest_admet.data import list_admet, load_admet
from honest_admet.models.foundation import embed_smiles


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fm", default="chemberta", choices=["chemberta", "molformer"])
    a = p.parse_args()
    for name in list_admet():
        ds = load_admet(name)
        emb = embed_smiles(ds.smiles, model=a.fm)
        print(f"cached {name:<28} {emb.shape}", flush=True)
    print("✅ all embeddings cached")


if __name__ == "__main__":
    main()
