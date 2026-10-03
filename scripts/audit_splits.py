"""Independent audit of the splits. Assumes nothing from make_splits.py.

make_splits.py verifies itself using the same masked-key index it used to build the clusters, so a
bug in that index would hide from both. Everything here is re-derived from the CSVs by brute force.

    python scripts/audit_splits.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(0)

df = pd.read_csv(ROOT / "context" / "dataset.csv")
pep = pd.read_csv(ROOT / "splits" / "peptide_split.csv")
m = df.merge(pep, on="peptide", validate="many_to_one")

ok = True


def check(label, passed, detail=""):
    global ok
    ok = ok and passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))


def encode(peptides):
    return np.frombuffer("".join(peptides).encode(), dtype=np.uint8).reshape(len(peptides), -1)


print("1. Brute-force neighbour search (no masked-key index)")
groups = {s: sorted(pep[pep.split == s].peptide) for s in ("train", "val", "test")}
A = encode(groups["train"])
worst = {}
for s in ("val", "test"):
    B = encode(groups[s])
    best = np.full(len(B), 99, dtype=np.int16)
    for i in range(0, len(B), 128):  # chunk to bound memory
        chunk = B[i : i + 128]
        d = (chunk[:, None, :] != A[None, :, :]).sum(axis=2)  # exact Hamming, every pair
        best[i : i + 128] = d.min(axis=1)
    worst[s] = int(best.min())
    n_le1 = int((best <= 1).sum())
    check(
        f"no {s} peptide within Hamming<=1 of ANY train peptide",
        n_le1 == 0,
        f"closest is {best.min()} edits away; {int((best<=2).sum())} peptides sit at <=2",
    )

print("\n2. Leakage paths other than train<->eval")
B, C = encode(groups["val"]), encode(groups["test"])
best_vt = 99
for i in range(0, len(B), 128):
    d = (B[i : i + 128][:, None, :] != C[None, :, :]).sum(axis=2)
    best_vt = min(best_vt, int(d.min()))
check("val and test are not near-duplicates of each other", best_vt > 1, f"closest {best_vt} edits")

sets = {s: set(v) for s, v in groups.items()}
check("no peptide in two splits", not (sets["train"] & sets["val"] | sets["train"] & sets["test"] | sets["val"] & sets["test"]))
check("all 5,633 peptides assigned exactly once", sum(len(v) for v in sets.values()) == df.peptide.nunique())
spans = pep.groupby("cluster_id").split.nunique()
check("no cluster split across sides", int(spans.max()) == 1, f"max splits per cluster {spans.max()}")
check("every row got a split", m.split.notna().all() and len(m) == len(df))
check("no duplicate (peptide, allele) rows", not df.duplicated(["peptide", "allele"]).any())

print("\n3. The leak, quantified: a predictor that ONLY memorises peptide identity")
print("   For each eval row, look up the same peptide's labels in train and predict their mean.")
print("   This model has no features and cannot generalise. On an honest split it must fail.")


def peptide_lookup_score(train_df, eval_df):
    lut = train_df.groupby("peptide").thalf_hours.mean()
    hit = eval_df.peptide.isin(lut.index)
    if hit.sum() < 10:
        return 0.0, float(hit.mean())
    pred = eval_df.loc[hit, "peptide"].map(lut)
    rho = pd.Series(pred.values).corr(
        pd.Series(eval_df.loc[hit, "thalf_hours"].values), method="spearman"
    )
    return float(rho), float(hit.mean())


# (a) the naive split everyone would reach for first
shuf = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
cut = int(0.9 * len(shuf))
rho_rand, cov_rand = peptide_lookup_score(shuf.iloc[:cut], shuf.iloc[cut:])
print(f"   random row split : peptide found in train for {cov_rand:6.1%} of eval rows, Spearman {rho_rand:.3f}")

# (b) ours
rho_ours, cov_ours = peptide_lookup_score(m[m.split == "train"], m[m.split == "test"])
print(f"   our peptide split: peptide found in train for {cov_ours:6.1%} of eval rows, Spearman {rho_ours:.3f}")
check(
    "memorisation-only model gets no traction on our split",
    cov_ours == 0.0,
    f"vs {cov_rand:.0%} coverage on a random split, where it scores {rho_rand:.3f}",
)

print("\n4. Representativeness")
for col, fn in [
    ("rows", lambda s: len(s)),
    ("median t-half", lambda s: s.thalf_hours.median()),
    ("zero fraction", lambda s: (s.thalf_hours == 0).mean()),
    ("alleles/peptide", lambda s: s.groupby("peptide").allele.nunique().mean()),
    ("distinct alleles", lambda s: s.allele.nunique()),
]:
    vals = {s: fn(m[m.split == s]) for s in ("train", "val", "test")}
    fmt = lambda v: f"{v:,.0f}" if col == "rows" else f"{v:.3f}"
    print(f"   {col:<16} train {fmt(vals['train']):>8}  val {fmt(vals['val']):>8}  test {fmt(vals['test']):>8}")

tr = m[m.split == "train"].thalf_hours
for s in ("val", "test"):
    ev = m[m.split == s].thalf_hours
    # Kolmogorov-Smirnov without scipy: max gap between the two empirical CDFs
    grid = np.unique(np.concatenate([tr.values, ev.values]))
    ks = float(np.max(np.abs(np.searchsorted(np.sort(tr.values), grid, "right") / len(tr)
                             - np.searchsorted(np.sort(ev.values), grid, "right") / len(ev))))
    check(f"{s} label distribution close to train (KS)", ks < 0.10, f"KS = {ks:.4f}")

print("\n5. Known limitation, measured not assumed")
for s in ("val", "test"):
    B = encode(groups[s])
    best = np.full(len(B), 99, dtype=np.int16)
    for i in range(0, len(B), 128):
        d = (B[i : i + 128][:, None, :] != A[None, :, :]).sum(axis=2)
        best[i : i + 128] = d.min(axis=1)
    print(f"   {s}: {int((best<=2).sum()):>4} / {len(B)} peptides ({(best<=2).mean():5.1%}) are within 2 edits "
          f"of a train peptide — allowed by the 80% identity convention, which is Hamming<=1 for 9-mers")

print("\n" + ("AUDIT PASSED" if ok else "AUDIT FAILED"))
raise SystemExit(0 if ok else 1)
