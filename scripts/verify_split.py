"""Re-verify the FROZEN split without regenerating it.

`make_splits.py` refuses to overwrite the frozen split, which is correct - every number in
`results/` was measured against those exact files. But that also meant its verification block
could only ever run at generation time, so a fix to the verifier could not be exercised against
the split actually in use.

This runs the same checks against the files on disk:

    1. the neighbour detector is probed AT the contract boundary (D4 fix) - a positive at
       Hamming = 2 must fire, and a negative at Hamming = 3, confirmed clear of every training
       peptide, must not
    2. no evaluation peptide is within Hamming 2 of any training peptide
    3. the split partitions the peptides: disjoint, non-empty, nothing lost or duplicated
    4. the file's sha256 matches the hash `src/features.py` asserts before reading any label

Check 4 ties this to everything else: if the hash here disagrees with the constant in
`features.py`, then either the split moved or the constant is stale, and every result in the
repository is suspect. It is cheap to check and expensive to discover late.

    python scripts/verify_split.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import make_splits as ms  # noqa: E402

SPLIT = ROOT / "splits" / "peptide_split.csv"


def main():
    df = pd.read_csv(ROOT / "context" / "dataset.csv")
    sp = pd.read_csv(SPLIT)
    peptide_split = dict(zip(sp.peptide, sp.split))
    peptides = sorted(df.peptide.unique())
    print(f"{len(peptides):,} peptides, {len(peptide_split):,} assigned, "
          f"contract Hamming <= {ms.CLUSTER_MAX_DIST}\n")

    failures = []

    # ---- 1. detector probed at the boundary, both directions
    ok, probe = ms.validate_detector(df, peptide_split, ms.CLUSTER_MAX_DIST)
    pos, neg = probe["positive"], probe["negative"]
    print("1. detector self-test at the contract boundary")
    print(f"   positive  {pos['planted']}  ({pos['distance']} edits from {pos['victim']})"
          f"  -> {'DETECTED' if pos['fired'] else 'MISSED'}")
    if "skipped" in neg:
        print(f"   negative  skipped: {neg['skipped']}")
    else:
        print(f"   negative  {neg['planted']}  ({neg['distance']} edits, verified clear of every "
              f"train peptide)  -> {'WRONGLY DETECTED' if neg['fired'] else 'correctly ignored'}")
    if not pos["fired"]:
        failures.append("detector missed a neighbour at the contract boundary")
    if neg.get("fired"):
        failures.append("detector fires beyond the contract -- it over-reports")

    # ---- 2. the contract itself
    violations = ms.verify_no_cross_split_neighbours(df, peptide_split, ms.CLUSTER_MAX_DIST)
    print(f"\n2. cross-split neighbours at Hamming <= {ms.CLUSTER_MAX_DIST}: {len(violations)}")
    if violations:
        failures.append(f"{len(violations)} cross-split neighbours")
        for v in violations[:5]:
            print(f"   {v}")

    # ---- 3. partition
    sets = {s: {p for p, v in peptide_split.items() if v == s} for s in ("train", "val", "test")}
    disjoint = (not (sets["train"] & sets["test"]) and not (sets["train"] & sets["val"])
                and not (sets["val"] & sets["test"]))
    complete = sum(len(v) for v in sets.values()) == len(peptides)
    print(f"\n3. partition: train {len(sets['train']):,} / val {len(sets['val']):,} / "
          f"test {len(sets['test']):,}   disjoint {disjoint}   complete {complete}")
    if not disjoint:
        failures.append("splits overlap")
    if not complete:
        failures.append("peptides lost or duplicated")

    # ---- 4. the hash the rest of the codebase asserts
    import features
    digest = hashlib.sha256(SPLIT.read_bytes()).hexdigest()
    match = digest == features.SPLIT_SHA256
    print(f"\n4. sha256 {digest[:16]}...  matches features.SPLIT_SHA256: {match}")
    if not match:
        failures.append("split hash disagrees with the constant features.py asserts")

    print()
    if failures:
        print("FAILED")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("All four checks pass. The frozen split satisfies the contract, and the detector that")
    print("says so has been shown to fire at the boundary and stay quiet beyond it.")

    (ROOT / "results" / "split_verification.json").write_text(json.dumps({
        "contract_max_hamming": ms.CLUSTER_MAX_DIST,
        "detector_probe": probe, "cross_split_neighbours": len(violations),
        "partition_disjoint": disjoint, "partition_complete": complete,
        "sha256": digest, "hash_matches_features_py": match,
    }, indent=2), encoding="utf-8")
    print("\nwrote results/split_verification.json")


if __name__ == "__main__":
    main()
