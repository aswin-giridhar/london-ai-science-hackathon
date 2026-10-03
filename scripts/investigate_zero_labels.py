"""What does `thalf_hours == 0.0` actually mean? Settled from the data, not assumed.

Background
----------
5,679 of 28,166 rows (20.2%) carry a half-life of exactly 0.0. Working notes in this repo first
asserted this was "an instrument detection floor". That was an unsupported inference, and the
official Serova brief, the background primer and the SPEARMINT preprint are all silent on it --
the preprint simply feeds log(1+0)=0 into MSE as a real value.

Two tests settle it without needing the assay's methods section.

Test 1: is there a gap above zero?
    A hard censoring threshold at some value V shows up as a point mass at 0 and then NOTHING until
    V. There is no such gap here -- 0.05 and 0.1 both occur -- so a simple "below the instrument's
    threshold" story does not fit on its own.

Test 2: what is the reporting precision?
    98.4% of non-zero values are exact multiples of 0.1 h, and exactly one row in the whole dataset
    sits below 0.1. So the reporting scale is 0.1 h, and any true value below 0.05 h rounds to 0.0.

Conclusion
----------
`0.0` means "measured half-life below 0.05 h" -- left-censored at 3 minutes -- not "a complex with
zero lifetime". A plain regression on log1p treats a censoring boundary as a point observation.

The zeros are also strongly allele-dependent, which matters more than the censoring for anyone
tempted to drop them: seven alleles are more than half zeros, including all three engineered
(C67S) constructs at 75-92%. Dropping zero rows removes 20% of the data and effectively deletes
those alleles.

    python scripts/investigate_zero_labels.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "context" / "dataset.csv")
t = df.thalf_hours
pos = t[t > 0]

print(f"rows {len(df):,} | zeros {int((t == 0).sum()):,} ({(t == 0).mean():.1%})\n")

print("TEST 1 — is there a gap above zero? (a censoring threshold would leave one)")
vals = np.sort(t.unique())
print(f"  smallest distinct values: {vals[:8]}")
print(f"  gap 0 -> next: {vals[1] - vals[0]:.3f} h; following gaps: {np.diff(vals[1:7]).round(3)}")
print("  -> no gap, so 'below the instrument threshold' does not fit on its own\n")

print("TEST 2 — what is the reporting precision?")
is_tenth = np.isclose((pos * 10) % 1, 0, atol=1e-9)
print(f"  non-zero values that are exact multiples of 0.1 h: {is_tenth.sum():,} ({is_tenth.mean():.1%})")
print(f"  non-zero values finer than 0.1 h:                  {(~is_tenth).sum():,} ({(~is_tenth).mean():.1%})")
print(f"  rows below 0.1 h anywhere in the dataset:          {int((pos < 0.1).sum())}")
print("  -> the reporting scale is 0.1 h, so any true value below 0.05 h rounds to 0.0\n")

print("CONCLUSION: 0.0 == 'half-life < 0.05 h', i.e. LEFT-CENSORED at 0.05, not a true zero.")
print("A regression on log1p treats that boundary as a point observation.\n")

print("Allele dependence — the practical consequence")
z = df.groupby("allele").thalf_hours.agg(n="size", zero_frac=lambda s: (s == 0).mean())
print(f"  alleles with any zeros: {int((z.zero_frac > 0).sum())} of {len(z)}")
print(f"  zero fraction: min {z.zero_frac.min():.3f}  median {z.zero_frac.median():.3f}  max {z.zero_frac.max():.3f}")
heavy = z[z.zero_frac > 0.5].sort_values("zero_frac", ascending=False)
print(f"\n  alleles that are more than half zeros ({len(heavy)}):")
for a, r in heavy.iterrows():
    print(f"    {a:<20} {int(r.n):>4} rows  {r.zero_frac:.1%} zero")
print(f"\n  dropping zero rows would remove {int((t == 0).sum()):,} rows ({(t == 0).mean():.1%})")
print("  and effectively delete the alleles above. Do not drop them silently.")

print("\nReconciliation with the published corpus")
eng = sorted(a for a in df.allele.unique() if "(" in a)
print(f"  {df.allele.nunique()} alleles - {len(eng)} engineered = {df.allele.nunique() - len(eng)}")
print(f"  the preprint and primer both report 72 alleles, so the engineered constructs")
print(f"  {eng} are additional to the published corpus.")
