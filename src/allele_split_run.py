"""Does the conclusion survive a different split? Generalising to alleles never seen in training.

Why this split asks a different question
-----------------------------------------
The primary split holds out *peptides*. Every allele in validation also appears in training, so a
model can memorise "HLA-B*15:01 complexes are stable" and carry that to new peptides. That is the
right test for "given a known allele, rank candidate peptides".

`splits/allele_split.csv` holds out whole **alleles**: 64 train / 5 val / 6 test. The clinical
version of this is a patient with an HLA type the model has never seen -- which, with 20,000+ known
class I alleles and a long tail, is the common case rather than the exotic one.

Why the result might reverse
----------------------------
**B1-prime encodes allele identity as a one-hot over the 75 training alleles.** For an allele it
has never seen, every one of those features is zero. It is *structurally incapable* of
distinguishing two unseen alleles -- it must predict the same value for both. B1 has the 34-residue
pseudosequence and ESM-2 has the full 182-aa domain, so both can in principle place a new allele
relative to familiar ones.

If the ordering flips here, the practical conclusion of the whole project becomes far more useful:
one-hot for known alleles, sequence representations for novel ones. If it does not flip, the
negative result is robust to the split criterion, which is the obvious objection to a
single-split finding.

Note this does NOT touch the frozen peptide split or its sha guard. It is a declared second
experiment on a second, pre-existing split file.

    python -m modal run modal_app.py::allele_split
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import hashlib
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
BLOCKS = ("peptide", "pseudoseq", "allele")
SEEDS = (42, 43, 44)


def load_allele_split():
    """Dataset joined to the ALLELE split, with peptide cluster ids carried along.

    The cluster ids come from the peptide split file and are kept only so the bootstrap has a
    resampling unit; they play no part in the allele assignment.
    """
    df = pd.read_csv(features.DATA)
    asp = pd.read_csv(ROOT / "splits" / "allele_split.csv")
    psp = pd.read_csv(features.SPLIT)[["peptide", "cluster_id"]]

    sha = hashlib.sha256((ROOT / "splits" / "allele_split.csv").read_bytes()).hexdigest()
    print(f"allele_split.csv sha256 {sha[:16]}  ({len(asp)} alleles)")

    m = df.merge(asp, on="allele", validate="many_to_one")
    m = m.merge(psp, on="peptide", validate="many_to_one")
    assert len(m) == len(df), "join changed the row count"
    assert set(m.split) == {"train", "val", "test"}, f"unexpected splits: {set(m.split)}"
    # the defining property of this split, asserted rather than assumed
    for a, b in (("train", "val"), ("train", "test")):
        shared = set(m[m.split == a].allele) & set(m[m.split == b].allele)
        assert not shared, f"{a} and {b} share alleles: {shared}"
    return m, sha


def main():
    import torch
    from torch import nn

    esm_heads.DEVICE = device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    OUT.mkdir(exist_ok=True)
    t0 = time.time()

    m, sha = load_allele_split()
    tr = m[m.split == "train"].reset_index(drop=True)
    va = m[m.split == "val"].reset_index(drop=True)
    vocab = sorted(tr.allele.unique())
    y_tr, y_va = features.target(tr), features.target(va)
    allele_va = va.allele.to_numpy()

    print(f"train {len(tr):,} rows / {tr.allele.nunique()} alleles")
    print(f"val   {len(va):,} rows / {va.allele.nunique()} alleles  "
          f"({sorted(va.allele.unique())})")
    print(f"unseen alleles in val: {va.allele.nunique() - len(set(va.allele) & set(tr.allele))}"
          f" of {va.allele.nunique()}")
    print("test  NOT READ\n")

    results = {"allele_split_sha256": sha,
               "n_train_alleles": int(tr.allele.nunique()),
               "n_val_alleles": int(va.allele.nunique())}

    # ---------------------------------------------------------------- B0b: per-allele median
    # On an allele split every validation allele is unseen, so there IS no training median for it.
    # The only honest fallback is the global training median -- which makes B0b a constant, and
    # its Spearman undefined. That is not a failure, it is the point: the no-peptide baseline
    # has nothing to say about an allele it has never met.
    b0 = np.full(len(va), np.median(y_tr))
    r = metrics.evaluate(y_va, b0, allele_va)
    results["B0b_global_median_fallback"] = r
    print(f"  {'B0b (forced to global median)':<34} "
          f"pooled {str(r['spearman_pooled']):>7}  within {str(r['spearman_within_allele']):>7}")

    # ---------------------------------------------------------------- one-hot rungs
    Xtr_full = features.build(tr, BLOCKS, vocab)
    Xva_full = features.build(va, BLOCKS, vocab)
    Xtr_prime = features.build(tr, ("peptide", "allele"), vocab)
    Xva_prime = features.build(va, ("peptide", "allele"), vocab)
    # sanity: every validation allele is unseen, so the allele block must be entirely zero
    allele_block = Xva_prime[:, -len(vocab):]
    print(f"  (B1' allele one-hot on val sums to {allele_block.sum():.0f} -- "
          f"zero by construction, every val allele is unseen)")

    for name, Xt, Xv in (("B1_peptide_pseudoseq_allele", Xtr_full, Xva_full),
                         ("B1prime_peptide_allele_only", Xtr_prime, Xva_prime)):
        preds = []
        for seed in SEEDS:
            mdl, _, _ = baselines.fit_mlp(Xt, y_tr, Xv, y_va, seed=seed)
            preds.append(mdl.predict(Xv))
        r = metrics.evaluate(y_va, np.mean(preds, axis=0), allele_va)
        results[name] = r
        print(f"  {name:<34} pooled {r['spearman_pooled']:>7.3f}  "
              f"within {(r['spearman_within_allele'] or float('nan')):>7.3f}")

    # ---------------------------------------------------------------- frozen ESM-2
    z, pep_index, all_index = esm_heads.load_cache()
    banks = {}
    for k, src in (("pep_res", "peptide_residue"), ("hla_res", "hla_residue")):
        t = torch.tensor(z[src].astype(np.float32)).to(device)
        banks[k] = torch.nn.functional.layer_norm(t, (t.shape[-1],))
    banks["pep_pool"] = banks["pep_res"].mean(1)
    banks["hla_pool"] = banks["hla_res"].mean(1)
    d_res = banks["pep_res"].shape[-1]

    tensors = {}
    for nm, sub in (("train", tr), ("val", va)):
        pi, ai = esm_heads.row_indices(sub, pep_index, all_index)
        tensors[nm] = {"pi": torch.from_numpy(pi).to(device),
                       "ai": torch.from_numpy(ai).to(device),
                       "y": torch.tensor(features.target(sub).astype(np.float32)).to(device),
                       "allele": sub.allele.to_numpy(), "meta": sub}

    models = esm_heads.build_models(torch, nn, d_in_pooled=2 * d_res, d_res=d_res)
    for name, Model in models.items():
        preds = []
        for seed in SEEDS:
            _, _, _, pv = esm_heads.train_one(torch, nn, Model, banks, tensors, seed)
            preds.append(pv)
        r = metrics.evaluate(y_va, np.mean(preds, axis=0), allele_va)
        results[name] = r
        print(f"  {name:<34} pooled {r['spearman_pooled']:>7.3f}  "
              f"within {(r['spearman_within_allele'] or float('nan')):>7.3f}")

    # ---------------------------------------------------------------- the comparison
    b1 = results["B1_peptide_pseudoseq_allele"]["spearman_pooled"]
    bp = results["B1prime_peptide_allele_only"]["spearman_pooled"]
    x = results["X150_interaction_head"]["spearman_pooled"]
    print("\nCOMPARISON WITH THE PEPTIDE SPLIT")
    print(f"  {'rung':<30}{'peptide split':>15}{'allele split':>15}{'change':>10}")
    for name, prev in (("B1_peptide_pseudoseq_allele", 0.780),
                       ("B1prime_peptide_allele_only", 0.748),
                       ("X150_interaction_head", 0.754),
                       ("F150_pooled_head", 0.593)):
        now = results[name]["spearman_pooled"]
        print(f"  {name:<30}{prev:>15.3f}{now:>15.3f}{now - prev:>+10.3f}")

    print(f"\n  B1 {b1:.3f} vs X150 {x:.3f}  ->  "
          f"{'one-hot still ahead' if b1 > x else 'ESM-2 AHEAD -- the ordering FLIPPED'}")
    print(f"  B1' {bp:.3f} vs B1 {b1:.3f}  ->  the allele one-hot is uninformative here "
          f"by construction, so any gap is the pseudosequence doing real work")

    (OUT / "allele_split_metrics.json").write_text(json.dumps(results, indent=2, default=float),
                                                   encoding="utf-8")
    print(f"\nwrote {OUT / 'allele_split_metrics.json'}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
