"""Cache frozen ESM-2 150M embeddings for every distinct sequence in the dataset.

The saving
----------
28,166 rows contain only 5,633 distinct peptides and 75 distinct HLA domains. Encoding per row
would be 28,166 forward passes; encoding per distinct sequence is 5,708 -- **4.9x fewer**.

This is valid ONLY because the two chains are encoded separately, with no cross-attention between
them. The moment an adapter trains (L150) or the chains attend to each other, an HLA embedding
depends on which peptide it is paired with and this cache is void. That is recorded in the manifest
so a later run cannot silently reuse it.

What is stored
--------------
Per-residue embeddings, not just pooled vectors:
    peptides   (5633,   9, 640)  float16   ~65 MB
    hla        (  75, 182, 640)  float16   ~17 MB

X150 needs the per-residue tensors for its interaction head; F150 only needs the mean. Storing
per-residue once serves both and avoids a second pass.

    python src/embed.py
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "cache"
MODEL_ID = "facebook/esm2_t30_150M_UR50D"   # 150M, 30 layers, d=640 -- the plan of record's choice
BATCH = 64


def sequences():
    df = pd.read_csv(ROOT / "context" / "dataset.csv")
    peptides = sorted(df.peptide.unique())
    # one row per allele carries its domain; allele -> hla_seq is 1:1 (verified in context/NOTES.md)
    hla = df.drop_duplicates("allele")[["allele", "hla_seq"]].sort_values("allele")
    return peptides, hla


def encode(model, tok, seqs, device, desc):
    """Mean-pooled and per-residue embeddings, special tokens excluded."""
    import torch

    per_res, pooled = [], []
    t0 = time.time()
    for i in range(0, len(seqs), BATCH):
        chunk = seqs[i : i + BATCH]
        enc = tok(chunk, return_tensors="pt", padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc).last_hidden_state        # (b, L+2, 640)
        mask = enc["attention_mask"].bool()
        for row in range(out.shape[0]):
            # drop <cls> and <eos>; keep only real residues
            valid = out[row][mask[row]][1:-1]
            per_res.append(valid.cpu().numpy().astype(np.float16))
            pooled.append(valid.mean(0).cpu().numpy().astype(np.float16))
        if i % (BATCH * 10) == 0:
            done = min(i + BATCH, len(seqs))
            rate = done / max(time.time() - t0, 1e-9)
            print(f"    {desc}: {done:,}/{len(seqs):,}  ({rate:.0f} seq/s)", flush=True)
    return np.stack(per_res), np.stack(pooled)


def main():
    import torch
    from transformers import AutoTokenizer, EsmModel

    OUT.mkdir(exist_ok=True)
    peptides, hla = sequences()
    print(f"distinct peptides: {len(peptides):,}")
    print(f"distinct HLA domains: {len(hla):,}")
    print(f"-> {len(peptides) + len(hla):,} encodes instead of 28,166 "
          f"({28166 / (len(peptides) + len(hla)):.1f}x fewer)\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
    print(f"loading {MODEL_ID} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = EsmModel.from_pretrained(MODEL_ID).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params/1e6:.0f}M parameters, hidden {model.config.hidden_size}\n", flush=True)

    t0 = time.time()
    pep_res, pep_pool = encode(model, tok, peptides, device, "peptides")
    hla_res, hla_pool = encode(model, tok, hla.hla_seq.tolist(), device, "hla")
    elapsed = time.time() - t0

    np.savez_compressed(
        OUT / "esm2_150m.npz",
        peptide_residue=pep_res, peptide_pooled=pep_pool,
        hla_residue=hla_res, hla_pooled=hla_pool,
        peptides=np.array(peptides), alleles=hla.allele.to_numpy(),
    )

    manifest = {
        "model": MODEL_ID,
        "parameters": int(n_params),
        "hidden": int(model.config.hidden_size),
        "device": device,
        "n_peptides": len(peptides),
        "n_alleles": len(hla),
        "encodes": len(peptides) + len(hla),
        "encodes_saved_vs_per_row": 28166 - (len(peptides) + len(hla)),
        "peptide_residue_shape": list(pep_res.shape),
        "hla_residue_shape": list(hla_res.shape),
        "dtype": "float16",
        "seconds": round(elapsed, 1),
        "VALID_ONLY_IF": "the encoder is frozen and the two chains are encoded separately. "
                         "Any adapter training (L150) or cross-chain attention voids this cache.",
    }
    (OUT / "esm2_150m.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    size_mb = (OUT / "esm2_150m.npz").stat().st_size / 1e6
    print(f"\nencoded {len(peptides)+len(hla):,} sequences in {elapsed:.0f}s")
    print(f"wrote {OUT/'esm2_150m.npz'}  ({size_mb:.0f} MB)")
    print(f"  peptide residue {pep_res.shape}, hla residue {hla_res.shape}")


if __name__ == "__main__":
    main()
