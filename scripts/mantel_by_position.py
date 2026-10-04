"""Anchor-restricted Mantel: does geometry at P2/P9 track stability where the average did not?

Run 5 measured which peptide positions the trained model relies on: **P9 >> P2 > P1 > P3**, with
P4-P8 contributing essentially nothing. Run 9 then tested whether predicted geometry tracks
half-life by averaging |dD| over **all nine** peptide rows -- spending five ninths of the signal
budget on positions our own ablation says are irrelevant. A real effect concentrated in the anchor
pockets could be diluted to nothing by that average.

This restricts the Mantel test to the rows that matter, and keeps P4-P8 as a negative control.

Two outcomes, both worth having:

    anchors survive, bulge does not   the Run 9 null was an averaging artefact, and anchor-pocket
                                      geometry carries stability signal -- mechanistically coherent
                                      with Run 5 and a genuine structural result

    nothing survives                  the structural null holds at position resolution too, which
                                      is a strictly stronger negative than Run 9 on its own

No new folds: the distance matrices are already in results/boltz_variance.json.

    python scripts/mantel_by_position.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import mantel as M  # noqa: E402

RESULTS = ROOT / "results"
PERMS = 10000
SEED = 2026


def main():
    src = RESULTS / "boltz_variance.json"
    if not src.exists():
        raise SystemExit(f"missing {src} -- run modal_boltz.py::variance_main first")
    d = json.loads(src.read_text(encoding="utf-8"))
    if "D" not in d:
        raise SystemExit("variance file has no D matrices; rerun with return_matrices=True")

    peps, thalf = d["peptides"], d["thalf"]
    mask = np.array(d["contact_mask"], dtype=bool)                  # (9, 182)
    D = {p: np.array(d["D"][p][0], dtype=np.float32) for p in peps}
    # sample 1 vs sample 0 of the same peptide is the model's own noise, per cell
    noise = {p: np.abs(np.array(d["D"][p][0], dtype=np.float32)
                       - np.array(d["D"][p][1], dtype=np.float32)) for p in peps}
    n = len(peps)

    print(f"{n} peptides on {d['allele']}")
    print("contacting cells per peptide position:")
    print("  " + "  ".join(f"P{i + 1}:{int(mask[i].sum())}" for i in range(9)))

    iu = np.triu_indices(n, 1)
    T = np.zeros((n, n))
    for i, a in enumerate(peps):
        for j, b in enumerate(peps):
            if i < j:
                T[i, j] = T[j, i] = abs(np.log1p(thalf[a]) - np.log1p(thalf[b]))

    def run(rows, label):
        sub = np.zeros_like(mask)
        sub[rows, :] = True
        sub &= mask
        if sub.sum() == 0:
            print(f"  {label:<30} no contacting cells in these rows")
            return None
        G = np.zeros((n, n))
        for i, a in enumerate(peps):
            for j, b in enumerate(peps):
                if i < j:
                    G[i, j] = G[j, i] = np.abs(D[a] - D[b])[sub].mean()
        r, p, _ = M.mantel(G, T, perms=PERMS, seed=SEED)
        nz = float(np.mean([noise[q][sub].mean() for q in peps]))
        sig = float(G[iu].mean())
        print(f"  {label:<30} r = {r:+.3f}  p = {p:.4f}   "
              f"signal {sig:.3f} A / noise {nz:.3f} A = {sig / nz:.2f}x   "
              f"{'SURVIVES' if p < 0.05 else 'no evidence'}")
        return {"rows": [int(x) + 1 for x in rows], "cells": int(sub.sum()),
                "mantel_r": r, "p": p, "signal_A": sig, "noise_A": nz,
                "signal_to_noise": sig / nz}

    print(f"\nMantel test restricted by peptide position, {PERMS:,} label permutations\n")
    out = {
        "anchors_P2_P9": run([1, 8], "anchors P2 + P9"),
        "P9_only": run([8], "P9 alone (largest ablation drop)"),
        "P2_only": run([1], "P2 alone (most stable drop)"),
        "ablation_top4": run([0, 1, 2, 8], "P1+P2+P3+P9 (ablation top 4)"),
        "bulge_P4_P8": run([3, 4, 5, 6, 7], "P4-P8 (negative control)"),
        "all_positions": run(list(range(9)), "all positions (= Run 9)"),
    }

    anchors, bulge = out["anchors_P2_P9"], out["bulge_P4_P8"]
    print("\nVERDICT")
    if anchors and anchors["p"] < 0.05 and (not bulge or bulge["p"] >= 0.05):
        print("  Anchor geometry tracks stability where the bulge does not, and where the")
        print("  all-position average did not. Run 9's null was an averaging artefact.")
    elif anchors and anchors["p"] < 0.05:
        print("  Anchors survive, but so does the control -- suspect a global confound,")
        print("  not an anchor-specific effect.")
    else:
        print("  No position subset survives. The structural null holds at position resolution,")
        print("  which is a stronger negative than Run 9 alone: it is not that we averaged the")
        print("  signal away, it is that there is no signal at the positions the model uses.")
    print("\n  Caveat: n = {} peptides on one allele. See results/mantel.json for the power".format(n))
    print("  curve -- at n=24 the all-position test had 92% power at a true r of 0.2.")

    dest = RESULTS / "mantel_by_position.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
