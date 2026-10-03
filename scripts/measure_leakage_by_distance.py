"""Measure how much a test peptide's label is predictable from its nearest TRAIN peptide,
as a function of edit distance. This is the evidence behind CLUSTER_MAX_DIST in make_splits.py.

Why this exists
---------------
The published convention for this dataset is 80% sequence identity, which for 9-mers is exactly
Hamming distance <= 1. Adopting a convention is not the same as checking it holds, so this script
checks it: if peptides at distance 2 still have strongly predictable labels, a threshold of 1 leaves
near-duplicates straddling the split.

Confound to avoid
-----------------
A first version of this measurement pooled pairs across alleles and reported a "null" of 0.335 for
unrelated pairs, which should have been ~0. Alleles differ systematically in stability, so both
members of every pair carried their allele's mean and correlated for that reason alone. Labels are
therefore converted to within-allele z-scores on log1p, comparing only deviation from the allele's
own norm. The corrected null comes out at 0.027, which is what tells us the probe now works.

    python scripts/measure_leakage_by_distance.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MAX_NEIGHBOURS = 3  # per test peptide per distance, to bound cost

df = pd.read_csv(ROOT / "context" / "dataset.csv")
sp = pd.read_csv(ROOT / "splits" / "peptide_split.csv")
m = df.merge(sp, on="peptide")

df = df.assign(z=np.log1p(df.thalf_hours))
df["z"] = df.groupby("allele").z.transform(lambda s: (s - s.mean()) / (s.std() or 1))
z = {(p, a): v for p, a, v in zip(df.peptide, df.allele, df.z)}

train, test = m[m.split == "train"], m[m.split == "test"]
trP, teP = sorted(train.peptide.unique()), sorted(test.peptide.unique())


def encode(ps):
    return np.frombuffer("".join(ps).encode(), np.uint8).reshape(len(ps), -1)


A, B = encode(trP), encode(teP)
D = np.zeros((len(B), len(A)), np.int8)
for i in range(0, len(B), 128):
    D[i : i + 128] = (B[i : i + 128][:, None, :] != A[None, :, :]).sum(2)

tr_al = train.groupby("peptide").allele.apply(set).to_dict()
te_al = test.groupby("peptide").allele.apply(set).to_dict()

rows = []
for i, p in enumerate(teP):
    for dist in range(1, 6):
        for j in np.where(D[i] == dist)[0][:MAX_NEIGHBOURS]:
            q = trP[j]
            for a in tr_al.get(q, set()) & te_al.get(p, set()):
                rows.append((dist, z[(p, a)], z[(q, a)]))
r = pd.DataFrame(rows, columns=["dist", "z_test", "z_neighbour"])

rng = np.random.default_rng(0)
trp_by_al = train.groupby("allele").peptide.apply(list).to_dict()
null = [
    (z[(p, a)], z[(rng.choice(trp_by_al[a]), a)])
    for p in teP
    for a in te_al.get(p, set())
    if a in trp_by_al
]
n = pd.DataFrame(null, columns=["x", "y"])

print("Within-allele label agreement between a test peptide and its nearest TRAIN peptide")
print("(both measured against the same allele; labels as within-allele z of log1p)\n")
print(f"{'edit dist':>10} {'identity':>9} {'n pairs':>8} {'Spearman':>9}")
for d, g in r.groupby("dist"):
    rho = g.z_test.corr(g.z_neighbour, method="spearman")
    print(f"{d:>10} {(9-d)/9:>8.3f} {len(g):>8} {rho:>9.3f}")
null_rho = n.x.corr(n.y, method="spearman")
print(f"{'random':>10} {'--':>9} {len(n):>8} {null_rho:>9.3f}   <- null; must be ~0 or the probe is confounded")

if abs(null_rho) > 0.08:
    raise SystemExit("null is not ~0: the measurement is confounded, do not trust the table above")
print("\nnull is ~0, so the distances above are measuring what they claim to.")
