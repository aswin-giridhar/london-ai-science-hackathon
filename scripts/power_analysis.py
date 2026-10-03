"""How small a difference can our test split actually resolve?

An ablation table is only interpretable if its rows differ by more than the noise. This measures the
minimum detectable effect on the frozen test split, under a paired bootstrap that resamples *peptide
clusters* rather than rows (rows within a cluster are not independent).

Run before the ablation, not after, so the write-up can state what the evaluation can and cannot
distinguish.

Method and its limits
---------------------
Predictions are simulated by adding Gaussian noise to the log target, with the noise level tuned by
bisection to hit a target Spearman. Two caveats, both of which matter for how far these numbers
generalise:

  1. Real model errors are structured -- heteroscedastic, allele-dependent, and worse near the
     censored floor -- while this noise is homoscedastic. Treat the numbers as indicative of
     resolution, not exact.
  2. The simulated improvement is spread uniformly across alleles. If a real improvement is
     concentrated in particular alleles (a foundation model helping most on rare ones, say), the
     within-allele metric could be more sensitive to it than this shows.

    python scripts/power_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPS = 200  # bootstrap resamples per realisation
REALISATIONS = 5  # independent noise draws per (metric, gap) cell
MIN_ROWS_PER_ALLELE = 12

rng = np.random.default_rng(0)
df = pd.read_csv(ROOT / "context" / "dataset.csv")
sp = pd.read_csv(ROOT / "splits" / "peptide_split.csv")
te = df.merge(sp, on="peptide").query("split == 'test'")

y = np.log1p(te.thalf_hours.values)
cl, al = te.cluster_id.values, te.allele.values
clusters = np.unique(cl)
idx_by_cluster = {c: np.where(cl == c)[0] for c in clusters}


def spearman(a, b):
    return np.corrcoef(pd.Series(a).rank().values, pd.Series(b).rank().values)[0, 1]


def pearson(a, b):
    return np.corrcoef(a, b)[0, 1]


def neg_log_rmse(a, b):
    return -np.sqrt(np.mean((a - b) ** 2))  # negated so that higher is better everywhere


def within_allele_spearman(a, b, alleles):
    scores = [
        spearman(a[alleles == A], b[alleles == A])
        for A in np.unique(alleles)
        if (alleles == A).sum() >= MIN_ROWS_PER_ALLELE
    ]
    return float(np.nanmean(scores))


def simulate(y, target_rho, rng):
    """Noise level tuned by bisection so the prediction lands at target_rho."""
    lo, hi = 0.01, 20.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if spearman(y, y + rng.normal(0, mid, len(y))) > target_rho:
            lo = mid
        else:
            hi = mid
    return y + rng.normal(0, (lo + hi) / 2, len(y))


def resample():
    picked = rng.choice(clusters, len(clusters), replace=True)
    return np.concatenate([idx_by_cluster[c] for c in picked])


print(f"test split: {len(te):,} rows across {len(clusters)} peptide clusters")
print(f"bootstrap: {REPS} reps, resampling clusters\n")

print("1. CONFIDENCE INTERVAL on a single Spearman")
for target in (0.45, 0.60, 0.75):
    pred = simulate(y, target, rng)
    vals = [spearman(y[i], pred[i]) for i in (resample() for _ in range(REPS))]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    print(f"   rho ~{target:.2f}   95% CI [{lo:.3f}, {hi:.3f}]   width {hi - lo:.3f}")

print("\n2. MINIMUM DETECTABLE PAIRED DIFFERENCE, by metric")
print("   Each cell: how many of the independent noise realisations gave a 95% CI on the")
print("   difference that excludes zero. A single realisation is a coin toss near the")
print(f"   boundary, so each cell runs {REALISATIONS} of them.\n")

metrics = [
    ("Spearman, pooled", lambda a, b, alleles: spearman(a, b)),
    ("Pearson on log", lambda a, b, alleles: pearson(a, b)),
    ("log-RMSE", lambda a, b, alleles: neg_log_rmse(a, b)),
    ("Spearman, within-allele", within_allele_spearman),
]
gaps = (0.01, 0.02, 0.03, 0.05)

print(f"   {'metric':<26}" + "".join(f"{g:>+9.2f}" for g in gaps))
for name, fn in metrics:
    row = f"   {name:<26}"
    for gap in gaps:
        hits = 0
        for _ in range(REALISATIONS):
            base = simulate(y, 0.60, rng)
            better = simulate(y, 0.60 + gap, rng)
            diffs = [
                fn(y[i], better[i], al[i]) - fn(y[i], base[i], al[i])
                for i in (resample() for _ in range(REPS))
            ]
            hits += np.percentile(diffs, 2.5) > 0
        row += f"{f'{hits}/{REALISATIONS}':>9}"
    print(row)

print(f"\n   Read 5/5 as 'reliably resolvable', 0/5 as 'reliably not', and anything between as")
print("   'on the boundary' -- which is itself the finding: do not report such a gap as a win.")
print("\nStable across realisations: the ORDERING. log-RMSE is the most sensitive, within-allele")
print("Spearman the least (it removes the between-allele confound but each per-allele estimate")
print("rests on ~38 rows). The exact threshold per metric is not stable; the ordering is.")
print("Use all three, with different jobs: see PLAN.md section 6b.")
