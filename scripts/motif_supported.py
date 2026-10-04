"""Re-score the motif probe using only residues the training data actually constrains.

Why the raw probe needed fixing
--------------------------------
In-silico saturation mutagenesis asks the model about all 20 amino acids at all 9 positions. But
a training set only constrains the residues it contains. At HLA-A*02:01 P9, **eleven of the
twenty** amino acids appear in under 1% of the 821 training rows - arginine in **3 of 821 (0.4%)**
- so for those residues the model is extrapolating, and ranking an extrapolation against a
well-supported residue compares a guess to a measurement.

That is exactly what the raw probe did, and it produced a result that contradicted both the
literature and basic pocket chemistry: both models ranked **R first at A*02:01 P9**, where the
F pocket is hydrophobic and the published motif is V/L/I/A. The training data itself says
V 38%, L 34%, I 11%, A 6% - the data agrees with the literature, and only the extrapolation does
not.

So the ranking is recomputed over the **supported** residues only. The support threshold is
reported, and the result is shown at several thresholds so the conclusion cannot rest on one
arbitrary cut.

    python scripts/motif_supported.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import features  # noqa: E402

R = ROOT / "results"
AA = features.AMINO_ACIDS
THRESHOLDS = (0.0, 0.005, 0.01, 0.02, 0.05)


def main():
    d = json.loads((R / "motif_recovery.json").read_text(encoding="utf-8"))
    known = {a: {int(p): set(v) for p, v in dd.items()} for a, dd in d["known"].items()}
    m = features.load()
    tr = m[m.split == "train"]

    print("Support in the training data at the anchor positions\n")
    support = {}
    for a in known:
        sub = tr[tr.allele == a]
        for p in known[a]:
            counts = pd.Series([pep[p - 1] for pep in sub.peptide]).value_counts()
            frac = {aa: counts.get(aa, 0) / len(sub) for aa in AA}
            support[(a, p)] = frac
            top = "  ".join(f"{aa}:{100 * frac[aa]:.0f}%" for aa in
                            sorted(AA, key=lambda x: -frac[x])[:5])
            n_rare = sum(1 for aa in AA if frac[aa] < 0.01)
            print(f"  {a} P{p}  n={len(sub)}   {top}")
            print(f"      {n_rare} of 20 residues appear in under 1% of rows; "
                  f"known anchors {''.join(sorted(known[a][p]))}")

    print(f"\nAnchor mean rank at several support thresholds (1 = best; lower is better)\n")
    print(f"  {'threshold':<12}{'residues kept':>15}{'B1':>10}{'X150':>10}")
    out = {}
    for thr in THRESHOLDS:
        ranks = {"B1": [], "X150": []}
        kept_total = []
        for a in known:
            pwm = {k: np.array(v) for k, v in d["results"][a]["pwm"].items()}
            for p, preferred in known[a].items():
                keep = [aa for aa in AA if support[(a, p)][aa] >= thr]
                kept_total.append(len(keep))
                # the known anchors must themselves be supported, or the question is meaningless
                target = [aa for aa in preferred if aa in keep]
                if not target:
                    continue
                for model in ("B1", "X150"):
                    vals = {aa: pwm[model][p - 1, features.AA_INDEX[aa]] for aa in keep}
                    order = sorted(keep, key=lambda aa: -vals[aa])
                    pos = {aa: i + 1 for i, aa in enumerate(order)}
                    ranks[model].append(float(np.mean([pos[aa] for aa in target])))
        b1, x1 = float(np.mean(ranks["B1"])), float(np.mean(ranks["X150"]))
        kept = float(np.mean(kept_total))
        chance = (kept + 1) / 2
        print(f"  >= {thr:<9.1%}{kept:>15.1f}{b1:>10.2f}{x1:>10.2f}"
              f"   (chance {chance:.1f})")
        out[f"{thr}"] = {"mean_residues_kept": kept, "B1": b1, "X150": x1, "chance": chance}

    print("\n  'residues kept' is the mean number of amino acids scored per anchor position;")
    print("  chance is the mean rank a random ordering would give.\n")

    one = out["0.01"]
    print("VERDICT at the 1% support threshold")
    print(f"  B1   mean anchor rank {one['B1']:.2f} of {one['mean_residues_kept']:.0f} "
          f"(chance {one['chance']:.1f})")
    print(f"  X150 mean anchor rank {one['X150']:.2f} of {one['mean_residues_kept']:.0f} "
          f"(chance {one['chance']:.1f})")
    delta = one["X150"] - one["B1"]
    print(f"  difference {delta:+.2f} rank positions, over {len(known)} alleles and "
          f"{sum(len(v) for v in known.values())} anchor positions")
    print(f"  -> {'ESM-2 recovers the motif better' if delta < -1 else ('one-hot recovers it better' if delta > 1 else 'NO MEANINGFUL DIFFERENCE between the two')}")

    (R / "motif_supported.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {R / 'motif_supported.json'}")


if __name__ == "__main__":
    main()
