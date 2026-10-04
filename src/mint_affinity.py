"""MINT Stage-1: a published pMHC foundation model that has never seen a half-life label.

Why this one, and why it is a fair comparator
---------------------------------------------
The challenge asks whether *existing* foundation models improve peptide-HLA stability prediction.
Until now this project answered with models we assembled ourselves -- ESM-2, ProtBERT and ProtT5
behind our own head. `dkarthikeyan1/mint-stage1-affinity` is the other kind of answer: a released,
published model for exactly this molecular system, used exactly as its authors shipped it.

It is the MINT backbone (ESM2-650M with cross-chain multimer attention, pretrained on 96M STRING
protein-protein interactions) fine-tuned on **NetMHCpan 4.1 binding affinity**, ~126K samples,
against `1 - log(IC50)/log(50000)`.

**It has never seen a half-life label.** That is what makes it usable here when two closely
related models are not:

    NetMHCstabpan            trained on THIS dataset            -> excluded, leakage
    SPEARMINT (Stage 3)      fine-tuned on pMHC STABILITY       -> excluded, same endpoint
    mint-stage2-stability    fine-tuned on pMHC STABILITY       -> excluded, same endpoint
    mint-stage1-affinity     fine-tuned on pMHC AFFINITY only   -> usable

This also closes two items this project had listed as not attempted: MINT itself, which Run 21
identified as the missing arm (the only model here built for two chains), and affinity
pre-training, which was declined as several hours of decontamination work. The decontamination is
not needed precisely because the endpoint differs.

What is measured
----------------
Zero-shot. No training, no fitting, no head. The model's released affinity score is correlated
with measured half-life on our frozen validation split, pooled and within-allele, against:

    B0b  0.563   per-allele median, no peptide information
    X150 0.754   our best frozen-PLM configuration
    B1   0.780   the one-hot baseline that beats everything so far

A caveat that is measured rather than assumed
----------------------------------------------
The model documents its MHC input as a **full ~365-residue heavy chain**. Our dataset carries the
182-residue alpha-1/alpha-2 groove domain. Every groove contact lives in those 182 residues, but
the input is still out of the distribution the model was trained on, and a truncation penalty
would masquerade as "MINT does not work here".

So both are run: the 182 residues we have, and the same sequence extended with a constant alpha-3
domain to bring the length into range. The alpha-3 domain is near-invariant across class I
alleles, so a shared constant suffix cannot carry allele-specific signal -- it can only change
whether the input looks like a heavy chain. If the two conditions agree, truncation is not the
explanation for whatever we see.

    python -m modal run modal_app.py::mint
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
from pathlib import Path

import numpy as np

import features
import metrics

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
MODEL_ID = "dkarthikeyan1/mint-stage1-affinity"
# Pinned to an exact commit. Two reasons, and the second is the one this project cares about:
#
#   1. `trust_remote_code=True` executes Python fetched from the Hub. Unpinned, that is whatever
#      the repository's main branch holds at the moment the job runs -- a supply-chain surface we
#      do not control.
#   2. Reproducibility. This repository content-addresses its data split by sha256 and refuses to
#      read a split whose hash has changed. Loading a model from a moving reference and then
#      reporting a number against it holds the model to a visibly lower standard than the data.
#
# The checkpoint was last modified 2026-07-01, months before this run, so this SHA is what Runs 29
# and 30 executed. Pinning records that rather than assuming it stays true.
REVISION = "8bf8e51906cf63336706d6cfc4f85e95a8109c86"

# HLA class I alpha-3 domain (residues ~183-274), near-invariant across alleles. Taken from the
# HLA-A*02:01 consensus. Used ONLY as a constant suffix in the length-sensitivity condition: being
# identical for every allele, it cannot introduce allele-specific signal, only length.
ALPHA3 = ("DPPKTHMTHHPISDHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYT"
          "CHVQHEGLPKPLTLRWE")

B0B, X150, B1_POOLED, B1_WITHIN = 0.563, 0.754, 0.780, 0.633


def score(torch, model, tokenizer, peptides, mhcs, device, batch, tag):
    out, t0 = [], time.time()
    for i in range(0, len(peptides), batch):
        chains, chain_ids = tokenizer.prepare_batch(peptides[i:i + batch], mhcs[i:i + batch])
        with torch.no_grad():
            o = model(chains.to(device), chain_ids.to(device))
        out.append(o["logits"].squeeze(-1).float().cpu().numpy())
        if i % (batch * 20) == 0:
            print(f"    {tag} {min(i + batch, len(peptides)):,}/{len(peptides):,} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    v = np.concatenate(out)
    assert v.shape[0] == len(peptides), f"{tag}: got {v.shape[0]} scores for {len(peptides)} rows"
    assert np.isfinite(v).all(), f"{tag}: non-finite affinity scores returned"
    return v


def main():
    import torch
    from transformers import AutoModel
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("refusing to run an 814M model on CPU")
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    OUT.mkdir(exist_ok=True)
    t_start = time.time()

    m = features.load()
    va = m[m.split == "val"].reset_index(drop=True)
    y = features.target(va).astype(float)
    peptides = va.peptide.tolist()
    allele = va.allele.to_numpy()
    print(f"{len(va):,} validation rows, {len(set(allele))} alleles", flush=True)

    print(f"loading {MODEL_ID} ...", flush=True)
    model = AutoModel.from_pretrained(MODEL_ID, revision=REVISION,
                                  trust_remote_code=True).to(device).eval()
    Tok = get_class_from_dynamic_module("modeling_mint_stability.MintTokenizer",
                                        MODEL_ID, revision=REVISION,
                                        trust_remote_code=True)
    tokenizer = Tok()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params / 1e6:.0f}M parameters", flush=True)

    conditions = {
        "groove_182": va.hla_seq.tolist(),
        "groove_plus_alpha3": [s + ALPHA3 for s in va.hla_seq],
    }
    results = {}
    for tag, mhcs in conditions.items():
        print(f"\n=== {tag} (MHC length {len(mhcs[0])}) ===", flush=True)
        t0 = time.time()
        pred = score(torch, model, tokenizer, peptides, mhcs, device, 16, tag)
        # Higher affinity score means stronger binding; stability should rise with it, so the
        # score is used as-is and a NEGATIVE correlation would itself be the finding.
        r = metrics.evaluate(y, pred, allele)
        r.update({"mhc_length": len(mhcs[0]), "seconds": round(time.time() - t0, 1),
                  "score_mean": float(pred.mean()), "score_std": float(pred.std())})
        results[tag] = r
        w = r["spearman_within_allele"]
        print(f"  -> pooled {r['spearman_pooled']:.3f}   "
              f"within {(w if w is not None else float('nan')):.3f}   "
              f"(score mean {pred.mean():.3f} sd {pred.std():.3f})", flush=True)
        np.save(OUT / f"mint_affinity_{tag}.npy", pred)

    a, b = results["groove_182"], results["groove_plus_alpha3"]
    delta = abs(a["spearman_pooled"] - b["spearman_pooled"])
    print("\nMINT STAGE-1 AFFINITY, ZERO-SHOT ON STABILITY (validation)\n")
    print(f"  {'condition':<22}{'MHC len':>9}{'pooled':>9}{'within':>9}")
    for tag in ("groove_182", "groove_plus_alpha3"):
        r = results[tag]
        w = r["spearman_within_allele"]
        print(f"  {tag:<22}{r['mhc_length']:>9}{r['spearman_pooled']:>9.3f}"
              f"{(w if w is not None else float('nan')):>9.3f}")
    print(f"  {'B0b no-peptide floor':<22}{'-':>9}{B0B:>9.3f}{'-':>9}")
    print(f"  {'X150 ours, trained':<22}{'-':>9}{X150:>9.3f}{0.558:>9.3f}")
    print(f"  {'B1 one-hot, trained':<22}{'-':>9}{B1_POOLED:>9.3f}{B1_WITHIN:>9.3f}")
    print(f"\n  truncation sensitivity: |182 - (182+alpha3)| = {delta:.3f}")
    print(f"    -> {'LENGTH MATTERS, caveat stands' if delta > 0.03 else 'length is not the explanation'}")
    best = max(a["spearman_pooled"], b["spearman_pooled"])
    print(f"\n  zero-shot best {best:.3f} vs the no-peptide floor {B0B:.3f}: "
          f"{'ABOVE the floor' if best > B0B else 'BELOW the floor'}")
    print(f"  zero-shot best {best:.3f} vs trained one-hot {B1_POOLED:.3f}: "
          f"{'BEATS B1' if best > B1_POOLED else 'below B1'}")

    dest = OUT / "mint_affinity.json"
    dest.write_text(json.dumps({
        "model": MODEL_ID, "params_millions": round(n_params / 1e6, 1),
        "trained_on": "NetMHCpan 4.1 binding affinity (~126K), never half-life",
        "zero_shot": True, "results": results,
        "truncation_delta_pooled": delta,
        "reference": {"B0b": B0B, "X150": X150, "B1_pooled": B1_POOLED, "B1_within": B1_WITHIN},
        "seconds": round(time.time() - t_start, 1)}, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
