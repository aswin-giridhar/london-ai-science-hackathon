# scripts

One-off analysis and the split tooling. Model code does **not** go here — when the GeoStab-FT
pipeline is written it belongs in `src/`, so that reproducible artifacts stay separable from
exploratory analysis.

All scripts read `context/dataset.csv` by that exact path and run from the repo root.

**Nothing here is removable.** Every file is cited as provenance somewhere, and the generator in
particular is what makes the frozen split verifiable rather than an unexplained artifact. Rarely-run
is not the same as unneeded — it is guarded instead.

| Script | What it does | Run it when |
|---|---|---|
| `make_splits.py` | **Generates the frozen splits**, and is the *provenance* of `splits/` — a split with no generator is an unexplained CSV. Clusters peptides at Hamming ≤ 2, assigns whole clusters | **Almost never.** It now **refuses to overwrite** an existing split and exits 1; `--force` is required to replace one deliberately |
| `audit_splits.py` | Independent brute-force audit — re-derives every distance from the CSVs rather than reusing the generator's index | After any change that could touch the split. Exits non-zero on failure |
| `measure_leakage_by_distance.py` | Label agreement between peptides as a function of edit distance. The evidence behind `CLUSTER_MAX_DIST = 2` | To re-examine the threshold |
| `investigate_zero_labels.py` | What `thalf_hours == 0.0` actually is: reporting grid, point-mass size, allele dependence | To re-check the zero-label reading |
| `power_analysis.py` | CI width on a single metric, plus `paired_bootstrap()` — **the function to call on two real models' predictions** | Now, for the CI. `paired_bootstrap` once models exist |

## Known defects — flagged P0 by the plan of record

Two are fixed, two are not. Do not present a result that depends on an unfixed one without saying so.

| ID | Status | Detail |
|---|---|---|
| **A1** | ✅ fixed | `audit_splits.py` asserted Hamming ≤ 1 while the contract is ≤ 2, so a distance-2 violation would have passed. Now enforces `CONTRACT_MAX_DIST = 2`, and the split passes it — the split was right, the guard was loose |
| **M1** | ✅ fixed | `peptide_lookup_score` returned `0.0` on low coverage, which we reported as "Spearman 0.000". That was a sentinel, not a measurement. Now returns `None` and prints "undefined (no overlap)". **The 0% coverage is the real evidence** |
| **D1** | ✅ fixed 2026-10-04 | `measure_leakage_by_distance.py` used to fit its per-allele z-normalisation over **all** rows, so the analysis that claims "no evaluation labels are read" was measuring train pairs on a scale set partly by eval labels. Statistics are now fitted on **training rows only** and applied everywhere. The break still falls between distance 2 and 3 (0.634 then 0.406), so the Hamming <= 2 threshold stands and is now justified without touching eval |
| **D4** | ✅ fixed 2026-10-04 | `validate_detector` planted a **1-edit** known-positive while the contract boundary is **2 edits**, so a detector blind to 2-edit neighbours would have passed. It now plants a 2-edit positive (must fire) **and** a 3-edit negative verified clear of every training peptide (must not). Re-runnable against the frozen split with `scripts/verify_split.py` |

All four P0 defects (A1, M1, D1, D4) are now closed. `scripts/verify_split.py` re-checks the frozen split end to end and writes `results/split_verification.json`: detector fires at Hamming 2 and stays quiet at 3, 0 cross-split neighbours, partition disjoint and complete, and the file's sha256 matches the constant `src/features.py` asserts.
The D1 fix was the one that could have changed a decision, and it did not: distances 1 and 2 still
behave alike (0.616, 0.634) and 3 still drops sharply (0.406). That was predicted before the fix
and is now shown rather than assumed.

## Conventions

- Deterministic given their seeds; `make_splits.py` uses `SEED = 42`.
- They assert their own invariants and fail loudly rather than emitting a quietly wrong artifact.
- `investigate_zero_labels.py` and `measure_leakage_by_distance.py` read evaluation labels for
  diagnostics. Keep that in mind before quoting them as blind analyses.
