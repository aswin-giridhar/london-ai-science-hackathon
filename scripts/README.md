# scripts

One-off analysis and the split tooling. Model code does **not** go here — when the GeoStab-FT
pipeline is written it belongs in `src/`, so that reproducible artifacts stay separable from
exploratory analysis.

All scripts read `context/dataset.csv` by that exact path and run from the repo root.

| Script | What it does | Run it when |
|---|---|---|
| `make_splits.py` | **Generates the frozen splits.** Clusters peptides at Hamming ≤ 2, assigns whole clusters, writes `splits/*.csv` and `split_report.json` | **Almost never.** Running it overwrites the frozen split. It is not an onboarding step |
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
| **D1** | ❌ open | `measure_leakage_by_distance.py` fits its per-allele z-normalisation over **all** rows (line 46), then uses it in the analysis that claims "no evaluation labels are read". That claim is currently false. Fix: fit on training rows only, and assert that perturbing held-out labels cannot change analysis A's output |
| **D4** | ❌ open | `validate_detector` in `make_splits.py` plants a **1-edit** known-positive, but the contract boundary is **2 edits** — the self-test never exercises the actual cutoff. Fix: plant a 2-edit positive and a 3-edit negative |

D1 matters most: it undercuts the train-only re-derivation of the threshold. The conclusion may well
survive the fix — distances 1 and 2 behaved alike and 3 dropped sharply — but that has to be shown,
not assumed.

## Conventions

- Deterministic given their seeds; `make_splits.py` uses `SEED = 42`.
- They assert their own invariants and fail loudly rather than emitting a quietly wrong artifact.
- `investigate_zero_labels.py` and `measure_leakage_by_distance.py` read evaluation labels for
  diagnostics. Keep that in mind before quoting them as blind analyses.
