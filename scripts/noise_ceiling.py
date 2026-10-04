"""How much headroom is left? An empirical ceiling from near-identical allele pairs.

The problem
-----------
There are **zero** replicate (peptide, allele) measurements in 28,166 rows, so the assay's
reproducibility cannot be estimated directly. Without it, "B1 scores 0.780" has no scale: it could
be close to the measurement limit or nowhere near it, and those imply completely different answers
to "would a better model help?".

The way in
----------
94% of peptides are measured against more than one allele, and the 34-residue pseudosequence tells
us how similar two grooves are. Two alleles differing at one or two of those 34 contact positions
are, biophysically, almost the same pocket. The same peptide measured against both should give
almost the same half-life.

So for each pair of near-identical alleles, take the peptides measured against both and correlate
their half-lives. That correlation is an **empirical upper bound on what any model could achieve**
when ranking peptides, because it is what the assay itself achieves when asked the same question
twice through two nearly identical pockets.

What this is and is not
-----------------------
It is **not** a pure replicate estimate. The scatter contains:

    assay noise                    what we want
    genuine micro-allele effects   real biology that a perfect model COULD capture
    peptide-set composition        controlled by using only shared peptides

So it is a **lower bound on the ceiling**: the true ceiling is at least this high, because some of
the observed scatter is real signal a model could in principle learn. Reporting it as "the
ceiling" would overstate it; reporting it as "the ceiling is at least this" is sound and is still
decision-relevant.

The distance-0 case, if it exists, is the cleanest: two alleles with *identical* pseudosequences
have identical contact residues, so a difference in half-life is either assay noise or an effect
of groove positions outside the pseudosequence.

    python scripts/noise_ceiling.py
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import features  # noqa: E402

RESULTS = ROOT / "results"
MIN_SHARED = 30          # peptides a pair must share before its correlation means anything
MAX_PSEUDO_DIST = 4      # report by distance; pairs beyond this are not "near-identical"


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / d) if d else np.nan


def main():
    m = features.load()
    info = m.drop_duplicates("allele")[["allele", "hla_pseudoseq"]].set_index("allele")
    alleles = sorted(info.index)
    print(f"{len(alleles)} alleles, {len(m):,} rows")

    # pseudosequence Hamming distance between every allele pair
    pseudo = {a: info.loc[a, "hla_pseudoseq"] for a in alleles}
    by_dist = {}
    for a, b in combinations(alleles, 2):
        d = sum(x != y for x, y in zip(pseudo[a], pseudo[b]))
        by_dist.setdefault(d, []).append((a, b))
    print("\npseudosequence Hamming distance between allele pairs (34 positions):")
    for d in sorted(by_dist)[:8]:
        print(f"  distance {d}: {len(by_dist[d]):>4} pairs")

    # peptide -> {allele: thalf}
    piv = m.pivot_table(index="peptide", columns="allele", values="thalf_hours")

    rows = []
    for d in sorted(by_dist):
        if d > MAX_PSEUDO_DIST:
            break
        for a, b in by_dist[d]:
            if a not in piv.columns or b not in piv.columns:
                continue
            both = piv[[a, b]].dropna()
            if len(both) < MIN_SHARED:
                continue
            x, y = np.log1p(both[a].to_numpy()), np.log1p(both[b].to_numpy())
            if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
                continue
            rows.append({"dist": d, "a": a, "b": b, "n": len(both),
                         "spearman": spearman(x, y),
                         "rmse_log": float(np.sqrt(np.mean((x - y) ** 2)))})

    if not rows:
        print(f"\nNo allele pair shares at least {MIN_SHARED} peptides within pseudosequence "
              f"distance {MAX_PSEUDO_DIST}. No empirical ceiling can be derived this way.")
        return

    df = pd.DataFrame(rows).sort_values(["dist", "spearman"], ascending=[True, False])
    print(f"\n{len(df)} allele pairs with >= {MIN_SHARED} shared peptides and "
          f"pseudosequence distance <= {MAX_PSEUDO_DIST}\n")
    print(f"  {'dist':>5}{'pairs':>7}{'median rho':>13}{'max rho':>10}"
          f"{'median n':>11}{'median RMSE':>13}")
    summary = {}
    for d, g in df.groupby("dist"):
        print(f"  {d:>5}{len(g):>7}{g.spearman.median():>13.3f}{g.spearman.max():>10.3f}"
              f"{g.n.median():>11.0f}{g.rmse_log.median():>13.3f}")
        summary[int(d)] = {"pairs": int(len(g)), "median_spearman": float(g.spearman.median()),
                           "max_spearman": float(g.spearman.max()),
                           "median_n": float(g.n.median()),
                           "median_rmse_log": float(g.rmse_log.median())}

    print("\n  closest pairs individually:")
    for _, r in df.head(10).iterrows():
        print(f"    d={r.dist}  {r.a:<14} vs {r.b:<14} n={r.n:>4}  "
              f"rho {r.spearman:.3f}  rmse_log {r.rmse_log:.3f}")

    best = df[df.dist <= 2]
    ceiling = float(best.spearman.median()) if len(best) else float(df.spearman.median())
    print("\nWHAT THIS MEANS")
    print(f"  Among allele pairs differing at <= 2 of 34 contact positions, the same peptide set")
    print(f"  measured through both grooves correlates at a median rho of **{ceiling:.3f}**.")
    print(f"  Our best model, B1, reaches 0.780 pooled / 0.633 within-allele on validation.")
    print()
    if ceiling < 0.78:
        print("  The assay's own agreement across near-identical grooves is BELOW our model's")
        print("  score. Read carefully: this does not mean the model beats the assay. It means")
        print("  the two numbers measure different things -- pooled Spearman is inflated by")
        print("  between-allele variance (B0b alone scores 0.563 with no peptide information),")
        print("  while this comparison is within a matched peptide set. The honest comparison is")
        print("  against WITHIN-allele Spearman, 0.633.")
    print()
    print("  Treat this as a LOWER bound on the ceiling: the scatter includes genuine")
    print("  micro-allele effects a perfect model could capture, not only assay noise.")

    out = {"min_shared_peptides": MIN_SHARED, "by_distance": summary,
           "n_pairs_total": int(len(df)),
           "ceiling_estimate_dist_le_2": ceiling,
           "pairs": df.to_dict("records")}
    dest = RESULTS / "noise_ceiling.json"
    dest.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
