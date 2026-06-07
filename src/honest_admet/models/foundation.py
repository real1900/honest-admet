"""Frozen molecular foundation-model embeddings (MolFormer / ChemBERTa).

We use FROZEN embeddings + a simple head as the primary foundation-model arm: it is
MPS-safe, fast, and isolates the question "do pretrained chemical representations close
the random->cluster generalization gap relative to ECFP4 fingerprints?" from the
confounds of full fine-tuning. Embeddings are cached to disk per (model, dataset).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

# Reliable public checkpoints on the HuggingFace Hub.
MODELS = {
    "chemberta": "DeepChem/ChemBERTa-77M-MLM",
    "molformer": "ibm/MoLFormer-XL-both-10pct",
}


def embed_smiles(smiles, model: str = "chemberta", batch_size: int = 64,
                 device: str | None = None, max_len: int = 256, cache_dir: str = "data/emb") -> np.ndarray:
    """Mean-pooled frozen embeddings for a list of SMILES, cached per (model, smiles-set)."""
    name = MODELS.get(model, model)
    key = hashlib.sha1((name + "\n" + "\n".join(map(str, smiles))).encode()).hexdigest()[:16]
    cache = Path(cache_dir) / f"{model}_{key}.npy"
    if cache.exists():
        return np.load(cache)

    import torch
    from transformers import AutoModel, AutoTokenizer

    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    kwargs = {"trust_remote_code": True}
    if model == "molformer":
        kwargs["deterministic_eval"] = True  # MolFormer-only flag
    net = AutoModel.from_pretrained(name, **kwargs).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(smiles), batch_size):
            batch = [str(s) for s in smiles[i:i + batch_size]]
            enc = tok(batch, padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt").to(device)
            h = net(**enc).last_hidden_state               # (B, L, D)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1)  # masked mean pool
            out.append(emb.float().cpu().numpy())
    embs = np.vstack(out)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, embs)
    return embs
