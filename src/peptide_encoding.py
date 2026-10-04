"""Is one-hot the right peptide encoding? BLOSUM62 and a compact derived alternative.

The gap this closes
-------------------
The HLA side of our best model has been validated three independent ways (Boltz recovers its
positions; its Hamming distance predicts cross-allele agreement monotonically; compression beats
coverage). The **peptide** side has been validated zero ways. It is a positional one-hot, which
treats every substitution as equally foreign: L to M and L to D are the same distance, though one
is conservative and the other swaps a hydrophobic for a charged residue.

BLOSUM62 encodes exactly that chemistry, is what NetMHCpan has always used, and is a 20x20 table
of integers rather than a learned thing - so this is a cheap, falsifiable test of whether the
peptide representation is leaving anything on the table.

Three encodings, and why the third exists
------------------------------------------
    one-hot        9 x 20 = 180   position-pure, no chemistry
    BLOSUM62 rows  9 x 20 = 180   same width, chemistry instead of orthogonality
    BLOSUM PCA-5   9 x  5 =  45   the first 5 principal components of BLOSUM62

The PCA variant is included because Run 17 found that *compression* beats coverage on the HLA
side - 31 positions beat 97 beat 182. If the same holds for the peptide, a 5-dimensional residue
code should lose little and might gain. Its components are derived from the matrix fetched here,
not recalled, so no constant in this file comes from memory.

The HLA block and the allele block are held fixed throughout, so any difference is attributable to
the peptide encoding alone. A fourth and fifth configuration vary the HLA block instead, as a
check that the result is about peptides and not about BLOSUM in general.

    python src/peptide_encoding.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import baselines
import features
import metrics

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
SEEDS = baselines.SEEDS
AA = features.AMINO_ACIDS

# BLOSUM62, fetched 2026-10-04 from the Biopython distribution of the NCBI matrix.
# Row order is the NCBI order; only the 20 standard amino acids are retained.
_ORDER = "ARNDCQEGHILKMFPSTWYV"
_RAW = """
 4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
-1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
-2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
-2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
 0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
-1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
-1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
 0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
-2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
-1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
-1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
-1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
-1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
-2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
 1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
 0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
-2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
 0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""


def blosum62():
    """The matrix, reindexed to our alphabetical amino-acid order, with sanity checks."""
    rows = [r.split() for r in _RAW.strip().splitlines()]
    M = np.array(rows, dtype=np.float32)
    assert M.shape == (20, 20), M.shape
    assert np.allclose(M, M.T), "BLOSUM62 must be symmetric"
    idx = {a: i for i, a in enumerate(_ORDER)}
    # known values, so a transcription error cannot pass silently
    assert M[idx["L"], idx["L"]] == 4, "L/L should be 4"
    assert M[idx["L"], idx["I"]] == 2, "L/I should be 2"
    assert M[idx["L"], idx["D"]] == -4, "L/D should be -4"
    assert M[idx["W"], idx["W"]] == 11, "W/W should be 11"
    order = [idx[a] for a in AA]
    return M[np.ix_(order, order)]


def blosum_pca(M, k=5):
    """First k principal components of BLOSUM62: a compact residue code derived, not recalled."""
    X = M - M.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    comps = U[:, :k] * S[:k]
    var = (S ** 2 / (S ** 2).sum()).cumsum()[k - 1]
    return comps.astype(np.float32), float(var)


def encode(seqs, length, table):
    """Positional encoding with an arbitrary per-residue vector table (n, length * dim)."""
    dim = table.shape[1]
    out = np.zeros((len(seqs), length * dim), dtype=np.float32)
    for row, s in enumerate(seqs.to_numpy()):
        for pos, aa in enumerate(s):
            j = features.AA_INDEX.get(aa)
            if j is not None:
                out[row, pos * dim:(pos + 1) * dim] = table[j]
    return out


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    B = blosum62()
    P5, var5 = blosum_pca(B, 5)
    eye = np.eye(20, dtype=np.float32)
    print(f"BLOSUM62 loaded and checked. First 5 PCs retain {100 * var5:.0f}% of its variance.\n")

    m = features.load()
    tr = m[m.split == "train"].reset_index(drop=True)
    va = m[m.split == "val"].reset_index(drop=True)
    vocab = sorted(tr.allele.unique())
    y_tr, y_va = features.target(tr), features.target(va)
    allele_va = va.allele.to_numpy()
    a_tr = features.onehot_alleles(tr.allele, vocab)
    a_va = features.onehot_alleles(va.allele, vocab)

    configs = {
        "B1 one-hot pep + one-hot HLA (reference)": (eye, eye),
        "BLOSUM pep + one-hot HLA": (B, eye),
        "BLOSUM-PCA5 pep + one-hot HLA": (P5, eye),
        "one-hot pep + BLOSUM HLA": (eye, B),
        "BLOSUM pep + BLOSUM HLA": (B, B),
        "BLOSUM-PCA5 pep + BLOSUM-PCA5 HLA": (P5, P5),
    }

    results = {"pca5_variance_retained": var5}
    dest = OUT / "peptide_encoding.json"
    print(f"{'configuration':<42}{'feats':>7}{'pooled':>9}{'within':>9}{'vs B1':>9}{'time':>8}")
    ref = None
    for label, (pt, ht) in configs.items():
        t = time.time()
        Xt = np.hstack([encode(tr.peptide, 9, pt), encode(tr.hla_pseudoseq, 34, ht), a_tr])
        Xv = np.hstack([encode(va.peptide, 9, pt), encode(va.hla_pseudoseq, 34, ht), a_va])
        ps = [baselines.fit_mlp(Xt, y_tr, Xv, y_va, seed=s)[0].predict(Xv) for s in SEEDS]
        r = metrics.evaluate(y_va, np.mean(ps, axis=0), allele_va)
        if ref is None:
            ref = r["spearman_pooled"]
        w = r["spearman_within_allele"]
        results[label] = {"n_features": int(Xt.shape[1]), "pooled": r["spearman_pooled"],
                          "within": w, "seconds": round(time.time() - t, 1)}
        dest.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
        print(f"{label:<42}{Xt.shape[1]:>7}{r['spearman_pooled']:>9.3f}"
              f"{(w if w is not None else float('nan')):>9.3f}"
              f"{r['spearman_pooled'] - ref:>+9.3f}{time.time() - t:>7.0f}s", flush=True)

    best = max((k for k in configs), key=lambda k: results[k]["pooled"])
    gain = results[best]["pooled"] - ref
    print(f"\nVERDICT")
    print(f"  best: {best}  ({results[best]['pooled']:.3f} pooled, "
          f"{results[best]['within']:.3f} within)")
    print(f"  gain over one-hot B1: {gain:+.3f} pooled")
    print(f"  typical seed spread elsewhere in this project is about 0.015-0.03, so a gain")
    print(f"  under that is not a result. -> "
          f"{'worth adopting' if gain > 0.03 else 'WITHIN the usual seed noise; one-hot is fine'}")
    print(f"\nwrote {dest}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
