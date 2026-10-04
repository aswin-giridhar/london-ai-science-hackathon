"""Does either model know the biology? In-silico mutagenesis against published binding motifs.

The question
------------
Every comparison so far is a correlation on held-out rows. None of them asks whether a model has
internalised *the right biology* - and a model can rank peptides well while having learned
something that is not what immunologists would call binding.

Published anchor motifs give an external answer. They were established by peptide elution and
crystallography, neither model has seen them, and they are specific enough to be falsifiable.

    HLA-A*02:01   P2 prefers L, M, I (L dominant)      P9 prefers V, L, I, A (V most frequent)
    HLA-B*27:05   P2 strictly prefers R (alt Q, K)

Sources: MHC Motif Atlas (NAR 2023, doi 10.1093/nar/gkac965) and "The pockets guide to HLA class I
molecules" (Biochem Soc Trans 2021). Only alleles whose motif was verifiable are used as external
ground truth; the rest are scored against a motif derived from the training split.

Method
------
In-silico saturation mutagenesis. For each allele, take reference peptides from training, mutate
each of the 9 positions to each of the 20 amino acids, predict the half-life, and average the
change. That yields a position weight matrix **derived from the model**, not from the data.

Then ask: at the anchor positions, do the published preferred residues rank near the top of the
model's predictions? Random expectation is a mean rank of 10.5 out of 20.

Why this is a fair test of the foundation model specifically
-------------------------------------------------------------
Mutants are sequences that do **not** appear in the dataset, so they are absent from the embedding
cache and must be re-encoded. Generalising to unseen sequence is precisely the capability
pretraining is supposed to provide, so if ESM-2 has an advantage anywhere it should be here.

    python -m modal run modal_app.py::motifs
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import baselines
import esm_heads
import features
import metrics

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
MODEL_ID = "facebook/esm2_t30_150M_UR50D"
AA = features.AMINO_ACIDS
N_REF = 6                 # reference peptides per allele
N_ALLELES = 10            # the two with external motifs, plus the largest by row count
SEEDS = baselines.SEEDS
BLOCKS = ("peptide", "pseudoseq", "allele")

# Published anchor preferences. Only alleles whose motif was independently verifiable are here.
KNOWN = {
    "HLA-A*02:01": {2: set("LMI"), 9: set("VLIA")},
    "HLA-B*27:05": {2: set("R")},
}


def pick_alleles(m):
    counts = m[m.split == "train"].allele.value_counts()
    chosen = [a for a in KNOWN if a in counts.index]
    for a in counts.index:
        if len(chosen) >= N_ALLELES:
            break
        if a not in chosen:
            chosen.append(a)
    return chosen


def mutants(peptide):
    """All 9 x 20 single substitutions, including the wild-type residue (a no-op mutation)."""
    out = []
    for pos in range(9):
        for aa in AA:
            out.append((pos, aa, peptide[:pos] + aa + peptide[pos + 1:]))
    return out


def rank_of(preferred, order):
    """Mean 1-based rank of the preferred residues in a model's ordering (1 = best)."""
    pos = {aa: i + 1 for i, aa in enumerate(order)}
    return float(np.mean([pos[a] for a in preferred if a in pos]))


def main():
    import torch
    from torch import nn
    from transformers import AutoTokenizer, EsmModel

    esm_heads.DEVICE = device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    OUT.mkdir(exist_ok=True)
    t0 = time.time()

    m = features.load()
    tr = m[m.split == "train"].reset_index(drop=True)
    va = m[m.split == "val"].reset_index(drop=True)
    vocab = sorted(tr.allele.unique())
    y_tr, y_va = features.target(tr), features.target(va)

    alleles = pick_alleles(m)
    print(f"alleles: {alleles}")
    print(f"external motifs available for: {sorted(KNOWN)}\n")

    # reference peptides: spread across the half-life range so the PWM is not anchored on one regime
    refs = {}
    for a in alleles:
        sub = tr[(tr.allele == a) & (tr.thalf_hours > 0)].sort_values("thalf_hours")
        if len(sub) < N_REF:
            continue
        idx = np.linspace(0, len(sub) - 1, N_REF).astype(int)
        refs[a] = sub.iloc[idx].peptide.tolist()
    alleles = [a for a in alleles if a in refs]

    # every sequence either model will need to score
    needed = sorted({seq for a in alleles for p in refs[a] for _, _, seq in mutants(p)}
                    | {p for a in alleles for p in refs[a]})
    print(f"{len(needed):,} distinct peptide sequences to score "
          f"({len(alleles)} alleles x {N_REF} refs x 180 mutants)")
    novel = [s for s in needed if s not in set(m.peptide)]
    print(f"  of which {len(novel):,} ({100 * len(novel) / len(needed):.0f}%) never appear in the "
          f"dataset -- so ESM-2 must generalise, not look up\n")

    # ---------------------------------------------------------------- train both models
    Xtr = features.build(tr, BLOCKS, vocab)
    b1_models = [baselines.fit_mlp(Xtr, y_tr, features.build(va, BLOCKS, vocab), y_va, seed=s)[0]
                 for s in SEEDS]
    print(f"B1 trained ({time.time() - t0:.0f}s)", flush=True)

    z, pep_index, all_index = esm_heads.load_cache()
    banks = {}
    for k, src in (("pep_res", "peptide_residue"), ("hla_res", "hla_residue")):
        t = torch.tensor(z[src].astype(np.float32)).to(device)
        banks[k] = torch.nn.functional.layer_norm(t, (t.shape[-1],))
    banks["pep_pool"] = banks["pep_res"].mean(1)
    banks["hla_pool"] = banks["hla_res"].mean(1)
    d_res = banks["pep_res"].shape[-1]
    X150cls = esm_heads.build_models(torch, nn, d_in_pooled=2 * d_res,
                                     d_res=d_res)["X150_interaction_head"]

    tensors = {}
    for nm, sub in (("train", tr), ("val", va)):
        pi, ai = esm_heads.row_indices(sub, pep_index, all_index)
        tensors[nm] = {"pi": torch.from_numpy(pi).to(device),
                       "ai": torch.from_numpy(ai).to(device),
                       "y": torch.tensor(features.target(sub).astype(np.float32)).to(device),
                       "allele": sub.allele.to_numpy(), "meta": sub}
    x150_models = [esm_heads.train_one(torch, nn, X150cls, banks, tensors, s)[0] for s in SEEDS]
    print(f"X150 trained ({time.time() - t0:.0f}s)", flush=True)

    # ---------------------------------------------------------------- encode the mutants
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    enc_model = EsmModel.from_pretrained(MODEL_ID).to(device).eval()
    embs = []
    B = 256
    for i in range(0, len(needed), B):
        e = tok(needed[i:i + B], return_tensors="pt", padding=True)
        e = {k: v.to(device) for k, v in e.items()}
        with torch.no_grad():
            h = enc_model(**e).last_hidden_state
        embs.append(h[:, 1:-1, :].cpu())
    mut_emb = torch.cat(embs).to(device)
    mut_emb = torch.nn.functional.layer_norm(mut_emb, (d_res,))
    seq_ix = {s: i for i, s in enumerate(needed)}
    del enc_model
    torch.cuda.empty_cache()
    print(f"encoded {len(needed):,} sequences ({time.time() - t0:.0f}s)\n", flush=True)

    pep_onehot = {s: features.onehot_sequences(pd.Series([s]), 9)[0] for s in needed}

    # ---------------------------------------------------------------- mutagenesis
    results, summary = {}, []
    for a in alleles:
        ai = all_index[a]
        hla_oh = features.onehot_sequences(
            pd.Series([tr[tr.allele == a].hla_pseudoseq.iloc[0]]), 34)[0]
        all_oh = features.onehot_alleles(pd.Series([a]), vocab)[0]

        def score_b1(seqs):
            X = np.stack([np.concatenate([pep_onehot[s], hla_oh, all_oh]) for s in seqs])
            return np.mean([mdl.predict(X) for mdl in b1_models], axis=0)

        def score_x150(seqs):
            idx = torch.tensor([seq_ix[s] for s in seqs], device=device)
            pr = mut_emb[idx]
            hr = banks["hla_res"][ai].unsqueeze(0).expand(len(seqs), -1, -1)
            with torch.no_grad():
                out = [mdl(pr.mean(1), hr.mean(1), pr, hr).cpu().numpy() for mdl in x150_models]
            return np.mean(out, axis=0)

        pwm = {"B1": np.zeros((9, 20)), "X150": np.zeros((9, 20))}
        for ref in refs[a]:
            muts = mutants(ref)
            seqs = [s for _, _, s in muts]
            base_b1, base_x = score_b1([ref])[0], score_x150([ref])[0]
            sb, sx = score_b1(seqs), score_x150(seqs)
            for (pos, aa, _), vb, vx in zip(muts, sb, sx):
                j = features.AA_INDEX[aa]
                pwm["B1"][pos, j] += (vb - base_b1) / len(refs[a])
                pwm["X150"][pos, j] += (vx - base_x) / len(refs[a])

        entry = {"n_refs": len(refs[a]), "pwm": {k: v.tolist() for k, v in pwm.items()}}
        for model in ("B1", "X150"):
            order = {p + 1: [AA[j] for j in np.argsort(-pwm[model][p])] for p in range(9)}
            entry[f"{model}_top3"] = {p: "".join(order[p][:3]) for p in (1, 2, 3, 9)}
            if a in KNOWN:
                ranks = {p: rank_of(s, order[p]) for p, s in KNOWN[a].items()}
                entry[f"{model}_anchor_rank"] = ranks
                summary.append({"allele": a, "model": model,
                                **{f"P{p}_rank": r for p, r in ranks.items()}})
        results[a] = entry
        print(f"  {a}")
        for model in ("B1", "X150"):
            t3 = entry[f"{model}_top3"]
            extra = ""
            if a in KNOWN:
                extra = "   anchor mean rank " + ", ".join(
                    f"P{p} {r:.1f}" for p, r in entry[f"{model}_anchor_rank"].items())
            print(f"    {model:<5} top3  P1 {t3[1]}  P2 {t3[2]}  P3 {t3[3]}  P9 {t3[9]}{extra}")

    # ---------------------------------------------------------------- verdict
    print("\nANCHOR RECOVERY against published motifs (mean rank of preferred residues, 1 = best)")
    print("  random expectation is 10.5 of 20\n")
    sdf = pd.DataFrame(summary)
    if len(sdf):
        print(sdf.to_string(index=False))
        for model in ("B1", "X150"):
            g = sdf[sdf.model == model]
            vals = g[[c for c in g.columns if c.endswith("_rank")]].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            print(f"\n  {model:<5} mean anchor rank {vals.mean():.2f} "
                  f"({'recovers the motif' if vals.mean() < 7 else 'no better than chance'})")

    (OUT / "motif_recovery.json").write_text(
        json.dumps({"alleles": alleles, "known": {k: {str(p): sorted(v) for p, v in d.items()}
                                                  for k, d in KNOWN.items()},
                    "n_sequences_scored": len(needed), "n_novel": len(novel),
                    "results": results, "summary": summary}, indent=2, default=float),
        encoding="utf-8")
    print(f"\nwrote {OUT / 'motif_recovery.json'}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
