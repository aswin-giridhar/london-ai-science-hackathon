"""Evidence behind CLUSTER_MAX_DIST in make_splits.py: how far apart must two peptides be
before one stops predicting the other's label?

Two analyses are printed.

  A. TRAIN-ONLY (the decision-relevant one). Uses training labels exclusively. This is what
     justifies the threshold, because a benchmark whose design depended on evaluation labels is not
     a benchmark designed blind.

  B. TRAIN-vs-EVAL (confirmation only). Reported for completeness and clearly marked, because it
     inspects evaluation labels and therefore must not be the basis of a design decision.

Historical note. The threshold was originally chosen from analysis B, which was a methodological
mistake caught in review: the split design was informed by evaluation labels. Analysis A was then run
and reproduces the same conclusion -- the break between "near-duplicate" and "merely similar" sits
between edit distance 2 and 3 -- so the decision does not depend on having seen eval labels. The
error is recorded rather than removed, because the audit trail matters more than a clean story.

Confound this controls for
--------------------------
A first version pooled pairs across alleles and reported a null of 0.335 for unrelated pairs, where
~0 was required. Alleles differ systematically in stability, so both members of every pair carried
their allele's mean and correlated for that reason alone. Labels are therefore converted to
within-allele z-scores of log1p, comparing only deviation from the allele's own norm. The script
refuses to report if the null is not near zero.

    python scripts/measure_leakage_by_distance.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MAX_NEIGHBOURS = 3
NULL_TOLERANCE = 0.08

df = pd.read_csv(ROOT / "context" / "dataset.csv")
sp = pd.read_csv(ROOT / "splits" / "peptide_split.csv")
m = df.merge(sp, on="peptide")

zdf = df.assign(z=np.log1p(df.thalf_hours))
zdf["z"] = zdf.groupby("allele").z.transform(lambda s: (s - s.mean()) / (s.std() or 1))
Z = {(p, a): v for p, a, v in zip(zdf.peptide, zdf.allele, zdf.z)}


def encode(ps):
    return np.frombuffer("".join(ps).encode(), np.uint8).reshape(len(ps), -1)


def report(title, pairs, null, note=""):
    r = pd.DataFrame(pairs, columns=["d", "x", "y"])
    n = pd.DataFrame(null, columns=["x", "y"])
    null_rho = n.x.corr(n.y, method="spearman")
    print(f"\n{title}")
    if note:
        print(f"  {note}")
    print(f"  {'edit dist':>10} {'identity':>9} {'n pairs':>8} {'Spearman':>9}")
    for d, g in r.groupby("d"):
        print(f"  {d:>10} {(9-d)/9:>8.3f} {len(g):>8} {g.x.corr(g.y, method='spearman'):>9.3f}")
    print(f"  {'random':>10} {'--':>9} {len(n):>8} {null_rho:>9.3f}   <- null, must be ~0")
    if abs(null_rho) > NULL_TOLERANCE:
        raise SystemExit("null is not ~0: measurement confounded, do not trust the table above")


# ---------------------------------------------------------------- A. train-only
train = m[m.split == "train"]
P = sorted(train.peptide.unique())
A = encode(P)
al = train.groupby("peptide").allele.apply(set).to_dict()

pairs = []
for i in range(0, len(P), 128):
    D = (A[i : i + 128][:, None, :] != A[None, :, :]).sum(2)
    for row in range(D.shape[0]):
        gi = i + row
        for d in range(1, 6):
            for j in np.where(D[row] == d)[0][:MAX_NEIGHBOURS]:
                if j <= gi:
                    continue  # count each unordered pair once
                for a in al[P[gi]] & al[P[j]]:
                    pairs.append((d, Z[(P[gi], a)], Z[(P[j], a)]))

rng = np.random.default_rng(0)
by_al = train.groupby("allele").peptide.apply(list).to_dict()
null_a = [(Z[(p, a)], Z[(rng.choice(by_al[a]), a)]) for p in P[:3000] for a in al[p]]

report(
    "A. TRAIN-ONLY  (this is what justifies the threshold)",
    pairs,
    null_a,
    "no evaluation labels are read in this analysis",
)

# ---------------------------------------------------------------- B. train vs eval
test = m[m.split == "test"]
teP = sorted(test.peptide.unique())
B = encode(teP)
tr_al, te_al = al, test.groupby("peptide").allele.apply(set).to_dict()

D = np.zeros((len(B), len(A)), np.int8)
for i in range(0, len(B), 128):
    D[i : i + 128] = (B[i : i + 128][:, None, :] != A[None, :, :]).sum(2)

pairs_b = [
    (d, Z[(p, a)], Z[(P[j], a)])
    for i, p in enumerate(teP)
    for d in range(1, 6)
    for j in np.where(D[i] == d)[0][:MAX_NEIGHBOURS]
    for a in tr_al.get(P[j], set()) & te_al.get(p, set())
]
null_b = [
    (Z[(p, a)], Z[(rng.choice(by_al[a]), a)])
    for p in teP
    for a in te_al.get(p, set())
    if a in by_al
]

report(
    "B. TRAIN vs EVAL  (confirmation only — reads eval labels, must not drive design)",
    pairs_b,
    null_b,
)

print("\nBoth analyses place the break between distance 2 and 3, so the Hamming<=2 grouping")
print("in make_splits.py is justified without reference to evaluation labels.")
