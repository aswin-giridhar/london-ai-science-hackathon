# Stability Lens: master execution plan

**Project:** ARE · London AI × Science Hackathon · Serova Protein Engineering Track. **Working branch:** `sh-arka22/are`. **Status:** current research and implementation roadmap; split tooling exists, but the model and structural pipelines are not implemented. This document governs the proposed work; [Science Skills](SCIENCE_SKILLS_PLAN.md) supplies the detailed evidence/structure companion and the [README](README.md) provides the project overview.

**Reviewed source boundary:** upstream `main` at `de2532f0a0d97067828dfd0077d0b852a845b920`, compared with personal-branch revision `1273a9735ba0d69d8debc5e53a631c611904a8ab`. The documentation revision analyzes upstream code without importing it, modifying scripts, or rerunning scientific measurements. Source behavior, saved reports, earlier reported research, synthetic checks, and future work are distinguished below.

## 0. Executive decision and falsifiable question

**Question:** On the existing peptide-component-disjoint benchmark, do sequence-only protein language-model features, predicted peptide–HLA interface geometry, or inverse-folding compatibility add useful measured half-life information beyond a fair supervised sequence baseline, and at what cost?[5][24]

The target is `thalf_hours`, not equilibrium affinity, pose accuracy, model confidence, or sequence likelihood. A bound structure is a static snapshot; dissociation is a kinetic process. None of the proposed protein models is automatically a validated half-life predictor.[1][5][27][41]

The priority order is fixed:

1. **Trustworthy evaluation:** reuse the frozen inputs and assignments, close the documented validation gaps, and freeze the model-selection protocol.
2. **Delivery floor:** B0/B1/F0 on the full sponsor dataset with a strong non-foundation baseline, allele-identity controls, within-allele reporting, and actual cost evidence.
3. **Main scientific extension:** a bounded S0 cohort and paired S1 test of geometry beyond sequence, availability, and confidence.
4. **Optional extension:** I1 peptide-only ProteinMPNN scoring on the same structures with a fixed-template control.
5. **Follow-ups:** interaction features, learning curves, larger encoders, calibrated uncertainty, and additional model families only after the core result is viable.

A positive, negative, or inconclusive matched comparison can be a successful scientific deliverable. An unrun branch must remain labeled untested. An inconclusive interval does not prove equivalence; a result from one checkpoint does not settle the value of an entire model family. This is a laboratory research-prioritization prototype, not a clinical or vaccine-selection system.[5]

## 1. Repository and evidence baseline

### What actually exists

| Asset | Current state | Consequence |
|---|---|---|
| `context/dataset.csv` and `.xlsx` | Checked-in challenge data | Use the CSV as the canonical computational input; do not create an untracked competing export |
| `splits/peptide_split.csv`, `allele_split.csv`, `split_report.json` | Frozen assignments and saved generation report | Consume and fingerprint them; do not regenerate to improve a score |
| `scripts/make_splits.py` | Existing generator | Running it overwrites the split artifacts |
| `scripts/audit_splits.py` | Existing independent distance computation and diagnostics | Its current pass/fail gates do not fully enforce the current split contract |
| `scripts/measure_leakage_by_distance.py` | Local earlier diagnostic; newer upstream version inspected | Distinguish local behavior from upstream analysis A/B |
| `experiments/` | Proposals, template, and result placeholders | A proposal is not a trained model or a measured result |
| This plan, Science Skills companion, README | Current planning/documentation layer | Define future requirements, not completed scientific work |
| Training code, model tests, evidence registries, structures, app | Not implemented in this checkout | Do not advertise runnable training/deployment commands or invented results |

This existing repository is the project workspace; do not initialize another Git root under `submission/`. The [background primer](Peptide-HLA%20Stability%20Primer.md) was written before the event and is retained as attributed reference material, not newly built hackathon implementation. Event eligibility and source redistribution conditions remain organizer/source questions, not assumptions inferred from public GitHub availability.[3][19]

### Frozen dataset and split findings

The following are **saved-report or previously reported audit findings**, not newly recomputed model results.[6][46]

| Finding | Recorded value / interpretation |
|---|---|
| Dataset | 28,166 measurements; 5,633 distinct 9-mer peptides; 75 alleles |
| HLA inputs | 182-aa α1/α2 domains, not full heavy chains; 34-position pseudosequences |
| Prior audit | 75 distinct domains, 74 pseudosequences, 5,679 zero-hour labels; no blank fields or duplicate `(allele, peptide)` pairs reported |
| Current grouping | Hamming distance ≤2 connected components across all alleles; seed 42 |
| Current components | 5,410 components; largest contains eight distinct peptides |
| Historical one-edit sensitivity | 5,494 components; largest five; not the current primary grouping |
| Primary train | 22,532 rows; 4,514 peptides; 75 alleles |
| Primary validation | 2,817 rows; 563 peptides; 73 alleles |
| Primary test | 2,817 rows; 556 peptides; 75 alleles |
| Secondary allele split | 64/5/6 train/validation/test alleles and 22,039/3,068/3,059 rows |

The engineered `HLA-B*14:01(C67S)` and `HLA-B*14:02(C67S)` share a pseudosequence; the earlier audit reported 368 repeated `(peptide, pseudosequence)` keys, not duplicate peptide–allele observations. Preserve allele identity or domain information rather than merge their different measurements.[6]

The exact meaning of zero-hour values is not established by the CSV. A detection floor is a hypothesis, not a known censoring threshold. Do not invent a censoring likelihood or call a zero-versus-positive classifier a validated binding classifier without assay evidence.[5][6]

### Input identity

| Artifact | SHA-256 recorded for this revision |
|---|---|
| `context/dataset.csv` | `384e0accd35c589f02bd3f7d702d31b00a27f8f6f728cacb52b4d8a6c83af44a` |
| `splits/peptide_split.csv` | `0d92a33bb64eb35d922f698283ca08f6556019e55c7f87c7ee5d31ab6b4d7988` |
| `splits/allele_split.csv` | `6632491f72c3af890b82c4e5753863866089ae43cc0dd3b5d0af1c5bd521bdd1` |
| `splits/split_report.json` | `b9893123ef40bacc28367b20e2eabec7bb8b7d9bd1bb3d0a41d87d160e28b539` |

Their Git blobs match the inspected upstream snapshot. Matching counts between the earlier exported-sheet audit and the current CSV do not establish byte identity of an unavailable earlier export. Bind future experiments to these current input identities; retain earlier findings with attribution.

### Prior research findings retained with their boundaries

These findings were reported by the earlier plan and its cited sources. They were **not recomputed during this script/documentation review**:

- The inspected SPEARMINT consensus reference maps 72/75 sponsor allele names; the missing names are the three C67S constructs. Separately, `HLA-B*08:03`, `HLA-A*02:50`, and `HLA-A*24:19` have domain discrepancies against that reference. Those two categories are not the same three alleles. Reconcile against a pinned official sequence source rather than assuming the sponsor input is wrong.[6][42]
- Earlier pilot mapping found full-chain lengths 365 for `HLA-A*02:01` and 362 for `HLA-B*15:01`, with exact supplied-domain matches starting at zero-based offset 24. These are prior reference observations, not permission to use a precursor as the mature structural construct.[6][42]
- SPEARMINT describes 27,034 curated measurements across 72 alleles, unlike the sponsor's 28,166 rows and 75 alleles. Its released train/validation/test counts are 21,626/2,705/2,700, totaling 27,031; the three-row discrepancy remains a limitation on claims of exact raw-to-split reproduction.[1][4]
- The preprint reports MINT transfer Spearman 0.791 versus ESM-2 transfer 0.745. Earlier reanalysis of 2,700 released Stage-2 test predictions reported MINT 0.7906 and NetMHCstabpan 0.8763; the latter is not a clean held-out comparator. Hours-scale CCC was 0.6735/0.4997, versus log-hours CCC 0.7879/0.7860. Always state the scale.[1][4]
- The earlier overlap audit reported all 2,700 Stage-2 test pairs in the sponsor CSV and 568 also in Stage-3 training. Of 1,133 Stage-3 test records, 979 had a reference identifier seen in training, leaving 154 new-reference rows. Peptide-disjoint testing is not automatically new-study testing.[1][4]
- Earlier reanalysis of released `MINT_stage3_film_v2` predictions reported SPA Spearman 0.3591 (217 rows), purified fluorescence 0.5666 (758), and cellular fluorescence 0.2204 (158). These are prediction-file reanalyses, not fresh inference; other published variants are different methods. MINT's interaction pretraining and architecture also prevent attributing its difference from ESM-2 solely to cross-chain attention.[1][4][22]

The referenced SPEARMINT code/prediction snapshot is `d548dee9b04dd0e3a9d7a50a593d2c030e31d2a5`. An attributed reproduction can be useful, but released stability-trained checkpoints are not the primary blind baselines on this sheet. A retrospective uncertainty analysis on those predictions would be a separate research task, not a replacement for B1/F0 here.[1][4][7]

## 2. Extracted scripts and validation gaps

### Local versus inspected upstream

| Script | Local Git blob | Upstream Git blob at `de2532f` | Status |
|---|---|---|---|
| `make_splits.py` | `21e96d3bb92d8ed910a73c492b6855e601f61533` | Same | Identical |
| `audit_splits.py` | `7b967dcbde8d7d97c094c0f60a1236ad7dc2c2da` | Same | Identical |
| `measure_leakage_by_distance.py` | `3dc48a1615af803ca8fe1d0d9bde49eff855a630` | `2f3f39cebd88aebd0d246aa7b9d19200b623c6d4` | Upstream adds A; local remains the earlier version |

The upstream [reconciliation](https://github.com/aswin-giridhar/london-ai-science-hackathon/blob/de2532f0a0d97067828dfd0077d0b852a845b920/RECONCILIATION.md) and [brainstorm](https://github.com/aswin-giridhar/london-ai-science-hackathon/blob/de2532f0a0d97067828dfd0077d0b852a845b920/experiments/BRAINSTORM.md) are reviewed source material, not files imported by this revision. Their assertions must be checked against the code rather than adopted wholesale.[47][48]

### Generator: `scripts/make_splits.py`

**Input:** canonical CSV. **Dependencies:** pandas and standard library. **Outputs overwritten when run:** the primary split CSV, secondary split CSV, and JSON report.[43]

- `UnionFind` and `cluster_peptides` (69–112) build exact global connected components using masked-position keys. At the active cutoff of two, groups include two-substitution neighbors as well as closer pairs.
- `greedy_assign` (146–172) assigns whole groups, largest first, according to relative remaining row quota with seeded tie-breaking. This avoids the old absolute-deficit imbalance; the primary allocation is row-weighted, not label-bin stratified.
- `greedy_assign_stratified` (128–143) applies that allocation inside secondary allele strata. `main` creates those strata from full-dataset per-allele median labels (290–297).
- `verify_no_cross_split_neighbours` (178–210) compares every nontraining peptide against training peptides. It does not separately check validation-versus-test distances.
- `validate_detector` (213–226) plants a one-substitution positive, not a two-substitution boundary case.
- `main` (257–394) reports cutoff sensitivity, validates several invariants, and writes fixed output paths. It asserts uniform peptide length, not specifically nine. The `if __name__ == "__main__"` guard prevents regeneration on import.

The module's one-edit convention description and historical threshold rationale are not the authoritative current constant: `CLUSTER_MAX_DIST = 2`. Generation is not an onboarding step or a harmless verification command.

### Independent audit: `scripts/audit_splits.py`

**Inputs:** CSV and saved primary assignment. **Dependencies:** NumPy/pandas, with SciPy used by pandas Spearman calculations. **Outputs:** stdout and exit status; no scientific artifact writes.[44]

The audit joins on peptide with `validate="many_to_one"`, computes Hamming distances by chunked brute force, and checks train–validation, train–test, validation–test, several coverage invariants, duplicate pairs, memorization coverage, and distribution summaries. It reads evaluation labels for diagnostic calculations and executes at import time.

Its three distance pass/fail checks still reject only distances ≤1 (48–61). Printing the number of neighbors at distance ≤2 does not enforce the two-edit contract. `peptide_lookup_score` returns `0.0` when fewer than ten matches exist (76–85); this is a sentinel, not a measured zero correlation. The current checks also do not replace full schema validation, exact assignment-key equality, or a complete secondary-allele audit.

### Distance diagnostic: inspected upstream `scripts/measure_leakage_by_distance.py`

**Inputs:** CSV and saved primary assignment. **Dependencies:** NumPy/pandas/SciPy. **Outputs:** printed descriptive statistics and a null-tolerance check. Both analyses execute at import/run time; no train-only CLI switch exists.[45]

The upstream file builds `Z[(peptide, allele)]` from `log1p` labels standardized per allele over the **entire dataset** (45–47). Analysis A later selects training peptide pairs (69–96) but uses that table; B compares test with training pairs (98–126).

**Therefore A is not strictly train-only.** An isolated synthetic check during this review held training labels fixed and changed only held-out labels; training normalized values and pooled ranks changed. This proves a dependency, not its numerical magnitude on the real dataset. Within a single finite, nondegenerate allele group, affine standardization preserves ranks; pooling multiple groups is different.

Other limits:

- Pair sampling takes at most three sorted-index neighbors per distance. A applies this cap before removing duplicate pair orientations, so eligible pairs can be missed (81–83).
- The A null uses the first 3,000 sorted training peptides, can sample self-matches, and couples RNG draws to set iteration. Determinism and representativeness need explicit policies.
- `report` prints a null correlation and tests `abs(null_rho) > 0.08` (54–66). An isolated empty-null fixture returned NaN without failing this guard.
- The final statement that the boundary lies between distances two and three is unconditional text (128–129), not a conclusion established by an implemented decision rule.

The source review and synthetic checks did not run the real dataset diagnostic, regenerate assignments, or reproduce its reported correlations. The newer A/B script remains upstream-only in this personal branch.

### Required future hardening, not completed fixes

| ID / priority | Gap | Required acceptance before relying on the result |
|---|---|---|
| D1 / P0 | All-row normalization enters nominally train-only A | Fit preprocessing on training rows only; perturbing/removing held-out labels cannot change A's artifacts, samples, or outputs; separate confirmation-only B |
| A1 / P0 | Audit gates enforce cutoff one instead of two | One declared cutoff applied to all three partition pairs; reject distance 0/1/2 fixtures and accept an isolated distance-3 case |
| M1 / P0 | Undefined null passes; low-coverage correlation is replaced by zero | Require finite statistics and adequate support; report undefined values, reasons, and coverage rather than fabricated zeros |
| D2 / P1 | Order-dependent capped pairs, null policy, unconditional conclusion | Filter unique eligible pairs before capping; specify seeded ordering, self-match policy, support, uncertainty, and a conclusion rule or descriptive-only reporting |
| V1 / P1 | Incomplete schema/assignment guarantees | Enforce exact 9-mer/alphabet/label contracts, unique and complete keys, allowed split names, no extra assignments, valid allele mapping, and proper secondary scope |
| P1 / P1 | Mutable outputs, import-time execution, incomplete run provenance | Separate loading/checking/generation; use isolated fixtures and pure helper seams in future tests; record versions, hashes, seeds, command, and status |

These findings do **not** demonstrate that the saved split contains a distance-two violation. The stored report records zero evaluation-to-training neighbors at the configured cutoff; upstream prose reports a closest cross-partition distance of three.[46][49] Keep those as recorded findings, not a fresh exhaustive audit. Do not reroll the split to repair the narrative.

## 3. Data and evaluation contract

### Canonical inputs and frozen assignments

One measurement is `(allele, peptide, thalf_hours, hla_seq, hla_pseudoseq)`. Preserve exact allele strings, engineered suffixes, and stable pair identity. Load `context/dataset.csv` and join the saved primary assignments many-to-one on peptide; require unique assignment keys, exact peptide-key coverage, allowed partition names, and nonmissing `cluster_id`/`split`.

Keep the current **80/10/10 train/validation/test, seed-42, Hamming≤2 component split**. Do not introduce the earlier proposed 70/10/10/10 replacement. An 80% identity convention for aligned 9-mers corresponds to Hamming≤1; the current grouping deliberately keeps two-substitution neighbors together as well. This is not a general variable-length sequence-alignment rule.

The secondary allele split holds out whole alleles but not necessarily peptides. Its median-label stratification is label-informed; it is not prospective external-assay validation. Record an unknown-allele policy for identity features and keep these metrics separate from the primary benchmark.

### Label and feature validation

Reject nonstandard amino acids, non-9-mers, unequal-length Hamming inputs, missing fields, negative/nonfinite labels, duplicate/conflicting observations, and inconsistent allele-to-sequence mappings. Log exclusions; do not silently rewrite sponsor constructs or merge pseudosequence collisions.

```text
y = log1p(thalf_hours)          after rejecting invalid labels
predicted_hours = expm1(y_hat)
```

Any clipping of predictions for presentation is a separate recorded choice. Never turn negative labels into valid zeros. Fit scalers, imputers, target normalization, learned feature selection, and other learned preprocessing on training rows only. A fixed pretrained encoder may cache label-independent embeddings, but any learned downstream transform remains training-only.

### Selection and exposure firewall

- Select architectures, feature definitions, heads, hyperparameters, seeds/protocols, and early stopping with training/validation information. Lock them before the scheduled test evaluation.
- Freeze S0 membership, reference/template selection rules, pose QC, retry caps, and fallback behavior before cohort inference. Do not replace failed test cases or select models according to their test result.
- Preserve the historical disclosure: evaluation-label diagnostics informed the threshold choice. Later supporting analysis cannot make that original choice prospectively blind; current upstream A also has the normalization dependency described above.
- Keep supervised-label exposure, own-partition peptide overlap, structural-template exposure, and design-time label exposure separate. PDB/Foldseek searches cannot certify an undisclosed pretraining corpus as clean.
- Released NetMHCstabpan, TLStab, and stability-trained MINT/SPEARMINT checkpoints are not independently held-out baselines on the source corpus. Do not use their predictions as primary training features.[1][4][5]
- External-assay claims require compatible metadata and independent exposure checks. A good score on this sponsor dataset does not establish assay transfer.[1]

### Metrics and uncertainty

Primary: held-out Spearman and paired change in Spearman on identical test pairs. Secondary: MAE/RMSE on `log1p(hours)`, supported within-allele performance, support counts, cost, latency, failures, and fallbacks. If CCC or hours-scale error is reported, name the scale explicitly.

Resample **peptide components**, retaining all rows and both models' predictions for each selected component. This paired bootstrap quantifies conditional test-sampling uncertainty, not retraining variability; report independently run seed variation separately. Pooled correlation can include between-allele effects, so show supported within-allele results rather than claiming every pooled association is either pure biology or entirely an artifact.

Empty, constant, insufficient, or nonfinite inputs yield undefined statistics with reasons and counts. A zero-coverage lookup is not evidence of a measured zero correlation. A confidence interval spanning no gain is inconclusive, not equivalence. Small structural cohorts remain pilots.

U1 is optional. The current split has no dedicated calibration partition. Any later group-aware calibration design must preserve the frozen test set and separate calibration from model selection. Standard row-level conformal guarantees do not automatically hold under peptide dependence or assay shift; ensemble spread and conformal output are not automatic out-of-distribution detectors.[8][17]

## 4. Models and controlled comparisons

### Experimental ladder

| ID | Proposed inputs / method | Required comparison and boundary |
|---|---|---|
| B0 | Training-only global/allele median log-half-life; global fallback for unsupported alleles | Sanity floor; constant predictions have undefined rank correlation, not zero by definition |
| B1 | B1a: positional peptide + pseudosequence features and explicit allele identity; B1b: positional peptide + 182-aa domain features, strongly regularized | Select using training/validation only; avoid a pseudosequence-collision straw man and report both variants |
| F0 | Frozen `facebook/esm2_t12_35M_UR50D`; separately encode peptide/domain, pool valid amino-acid tokens, concatenate, and fit a regularized Ridge/MLP head | Full-corpus comparison with B1, identical split and selection discipline; include a matched replacement of the HLA embedding with one-hot allele identity |
| F1 | Same encoder/cache with controlled pair interactions or explicit peptide-position features | Optional; match tuning/capacity policy and select before test |
| S0 | Label-blind, bounded structural cohort within the saved partitions | A cohort, not a model or a resplit; evaluate matched-size B1/F0 on these same pairs |
| S1 | Verified complex coordinates → QC, separate confidence, compact contact geometry → regularized heads | Geometry increment over the matched sequence/availability/confidence control, with all-input fallbacks |
| I1 | ProteinMPNN peptide-only normalized compatibility on S1 backbones, with fixed receptor context | Optional increment over matched no-I1 comparison plus fixed-template sensitivity |
| F2 / other encoders | Larger ESM-2 or alternative sequence representations | Follow-ups after the core result; no assumed gain or unmeasured runtime promise |
| U1 | Separately designed group-aware calibration/empirical coverage | Optional; no invented calibration partition or unsupported coverage guarantee |

Cache each unique peptide and domain once for independently encoded F0, with checkpoint/tokenizer/version and input hashes. Retain per-position peptide embeddings if F1 will need them. Do not include padding/special tokens in residue pooling. A joint-context/cross-attention encoder changes the caching and comparison contract.

### Model menu: one primary route, not an integration sweep

| Family | Primary route | Alternatives / limits |
|---|---|---|
| Sequence language model | ESM-2 35M frozen baseline | 150M/650M, ESMC, and ProtT5 are optional matched follow-ups, not drop-in equivalents.[13][14][38][39] |
| Complex structure | Boltz-2 coordinates | Chai-1 or Protenix only as an already-feasible fallback with pinned input/version/runtime. Boltz small-molecule affinity is excluded.[23][34][35] |
| Monomer/other folding | Not a substitute for a validated peptide complex | ESMFold or an AlphaFold Database monomer does not supply the peptide interface. Any newer complex-capable route needs its own pHLA smoke test and access check.[32][38] |
| Inverse folding | ProteinMPNN peptide score | LigandMPNN/ESM-IF require verified scoring/chain-context semantics; they do not produce measured half-lives or automatically validate redesigned peptides.[25][26][32] |
| Structure-aware language | Deferred SaProt | Correct structural-alphabet inputs and validated structures are required; amino-acid-only frozen features are not an equivalent structural comparison.[36] |
| External structural controls | Optional PANDORA/SwiftMHC/TFold context | Separate template, installation, allele, and target limits; structure or affinity performance is not half-life performance.[28][29][30][31][33][37] |

### Research synthesis and ranked backlog

The sponsor's three model classes motivate tests, not novelty guarantees. Prior structural classifiers used binder proxies rather than a multi-allele quantitative half-life endpoint; the earlier bounded search did not establish the absence of all related work. MINT's reported improvement combines supervision/architecture differences, not one isolated mechanism.[1][9][22][27]

| Candidate from earlier/upstream plans | Decision |
|---|---|
| Allele-identity controls, supported within-allele reporting, measured cost | Core |
| Learning curves using whole training components; explicit per-position features | Optional after the core sequence/structure comparison is viable |
| Positional occlusion and allele-specific anchor profiles | Descriptive follow-up/demo analysis, not causal proof of mechanism |
| ESM-2 model-size sweep | Optional cost-benefit experiment; experiment 001's 650M proposal does not override F0=35M |
| Zero-shot likelihood/context diagnostics | Optional; score/variant/prompt selection and checkpoint exposure still require evaluation discipline |
| Zero-label censoring | Deferred until assay semantics support an observation model; no unsupported novelty claim |
| Fine-tuning, EL/multitask transfer, model-family sweeps, molecular dynamics, sequence redesign | Outside the MVP |
| Conformal intervals and seed/mask spread | Optional, with empirical coverage and dependence/shift caveats |

Do not claim that mean pooling erases every trace of position from contextual embeddings, that a matched one-hot control proves protein knowledge is absent, or that zero-shot scoring removes all leakage surfaces. These are hypotheses about incremental utility under a protocol, not conclusions about a model's internal knowledge. Upstream brainstorm IDs reuse names such as B1/I1; reference them as `BRAINSTORM R4` or by descriptive name rather than changing this plan's model IDs.[48]

## 5. Evidence-grounded structural experiment

### Construct registry

Use the [companion's mapping contract](SCIENCE_SKILLS_PLAN.md#3-sequence-and-construct-registry). Preserve sponsor allele, mutation suffix, domain, pseudosequence, and pair ID. Resolve names against a pinned official IPD-IMGT/HLA release, require an exact supplied-domain match, and record accession, numbering, hashes, boundaries, and retrieval provenance.[54][55]

The primary proposed input is **mature extracellular HLA heavy chain + mature β2m + peptide**. Do not blindly feed a signal/transmembrane-containing precursor. A domain-only fallback is a separately named protocol. Multiple compatible extensions remain ambiguous; C67S constructs and the separately named domain discrepancies remain quarantined from the structural arm until resolved, while valid measured rows stay in sequence experiments.

Mapping statuses: `verified`, `ambiguous_extension`, `domain_mismatch`, `engineered_unverified`, `unmapped`. Extract/align sequences in code rather than manually rewriting amino-acid strings.

### References, QC, and exposure

PDB sequence matches are candidates, not proof of allele/construct identity. Inspect assembly, entity and author/label chain IDs, residue numbering, peptide identity/length, β2m, modifications, missing atoms, method, resolution, and citation. A qualifying reference may be absent.

Keep reference/QC use, fixed-template inverse-folding control, and structural-exposure audit distinct. A crystal complex is not an independent half-life example without compatible measured stability and assay metadata. Align on mapped HLA groove residues; call peptide RMSD a pose-error measurement only for an appropriate matched peptide/construct reference. Otherwise label it a reference overlay.

Freeze QC and label-blind selection rules after training-only probes. Predictor confidence is a separate channel, read from documented fields/scales; experimental B-factors are not pLDDT.

### Compact contact-feature contract

For every peptide position P1–P9, count mapped 182-domain HLA residues with at least one peptide/HLA heavy-atom pair at distance **≤4.5 Å**, then divide by the recorded mapped-residue denominator. Count unique receptor residues rather than atom pairs. Keep the nine-vector, raw counts, missing-coordinate masks, deterministic alternate-conformer policy, and feature-schema version.

Only peptide–HLA-domain contacts belong in this vector, not intrachain or β2m contacts. The cutoff is an engineering definition, not a universal physical threshold. P2/P9 are useful visual summaries, not universal exclusive anchors; allele-specific exceptions exist. More contacts or improved anchors need not produce longer measured half-life.[52][53]

Hydrogen-bond geometry, salt bridges, buried surface area, and pocket depth remain stretch features until their algorithms, atom requirements, units, and fixtures are specified. A generic distance contact or PyMOL display is not a verified hydrogen bond.

### Matched training eligibility and failure handling

1. Run three training-only probes after budget approval to assess mapping, pose validity, memory, runtime, and cost.
2. Freeze S0 IDs within the saved partitions before folding the cohort; predeclare seeds, attempts, QC/confidence-based pose selection, concurrency, retry reserve, and spend stop conditions.
3. Fit the availability/QC diagnostic on all S0 training rows. Fit the confidence and confidence-plus-geometry heads on **identical valid-structure training rows**; their paired test difference is the primary geometry contrast.
4. Both structural comparators use the same predeclared matched F0 fallback for invalid test structures. Every preselected test pair receives an operational prediction with `model_used` and `fallback_reason`.
5. Report a secondary valid-structure comparison on identical valid test pairs, including F0 trained on the same eligible training rows. Do not attribute different training eligibility to feature quality.
6. Report selected, attempted, valid, fallback, and inverse-score counts and costs. Never replace a failed test pair or report only successful folds as all-input performance.
7. I1 scores only the peptide with receptor context fixed and normalization explicit. Include an allele-compatible fixed-template sensitivity control selected without the evaluated peptide's label, excluding evaluated test-component peptides where possible. The control is not a new all-atom predicted complex.
8. If I1 shrinks training eligibility, refit the no-I1 comparator on the same eligible rows and fall back to the already-frozen no-I1 pipeline when a score is unavailable.

The main comparison sequence is matched-size B1/F0 → F0+availability/QC → plus confidence → plus geometry → optional I1, with structure-only prediction as a secondary diagnostic. A self-generated pose can favor its conditioning peptide's inverse-folding score; this circularity must stay visible.[25]

## 6. Workspace, interfaces, and artifact contracts

```text
Sources / literature --------------------------------> claim ledger
context/dataset.csv + saved splits + hashes
    |
    +--> future validation / hardened checks --> B0 / B1
    |
    +--> cached frozen ESM-2 embeddings --> F0
    |
    +--> verified construct registry --> locked S0
              +--> PDB references --> chain / pose / exposure checks
              +--> Boltz-2 --> QC + confidence + contact vector
                                   +--> optional peptide-only ProteinMPNN
                                                   |
                         matched heads + explicit operational fallbacks
                                                   |
                         paired metrics + support + failures + cost
                                                   |
                         real-output prediction / evidence / pose demo
```

The existing `scripts/` and frozen `splits/` are reused. Future `src/` modules will validate/load those assignments rather than quietly replace them. No nested `submission/` repository or duplicate raw CSV is required.

| Proposed future artifact | Contract / purpose |
|---|---|
| Input/run manifest | Dataset/split hashes, code revision, model/tokenizer revision, feature schema, versions, seeds, command, hardware, timing, and cost |
| `data/evidence_manifest.jsonl` | Claim, source/version/passage, endpoint, assay, license, status, retrieval date and hashes |
| `data/hla_mapping.jsonl` | Exact sponsor/reference/construct identity, boundaries, mutations, β2m, mapping status |
| `data/reference_manifest.jsonl` | PDB assembly/chain/residue maps, experimental quality, coordinates hash, role, exposure status |
| `data/structures_manifest.jsonl` | Pair/cohort IDs, construct/model versions, attempts, QC/confidence scale, reference IDs, failures, coordinates hash, runtime/cost |
| `models/selection.md` | Development-selected models and input contracts, versions/licenses, feasibility decisions |
| Prediction/result records | Pair/component/split/cohort, target scale, prediction, model used, fallback reason, paired metrics and denominators |
| `experiments/` write-ups | Hypothesis fixed before execution, owner, status, exact protocol, budget, result and limitations |

These manifests and model/result records are **not implemented**. Proposed modules cover input validation, embeddings, structural inputs, folding, pose QC, features, inverse scoring, training, evaluation, optional uncertainty, and Streamlit. Proposed tests are listed in §8; their paths are specifications, not present files.

Future interface requirements:

```text
fold_one(pair_id, peptide, hla_construct_sequence, construct_type,
         beta2m_sequence, checkpoint_revision)
  -> structure_uri_or_null, chain_map, confidence_fields, seconds, status, failure_reason

featurize(structure_uri, chain_map, mapped_domain, feature_schema_version)
  -> contact_vector, raw_counts, denominator, missing_masks, qc_status

score_inverse(structure_uri, peptide_chain_id, fixed_receptor_chains)
  -> peptide_normalized_score_or_null, scoring_protocol, failure_reason

predict(peptide, allele)
  -> predicted_hours, model_id, interval_or_null, structure_uri_or_null,
     structure_status, provenance, warnings, fallback_reason
```

Names above express proposed contracts rather than existing callable APIs. Save enough metadata to prevent mixing versions or reordering paired predictions. Unsupported input must be rejected or explicitly handled, not silently coerced. The sponsor CSV has no assay column; do not expose an assay switch unless a genuinely assay-conditioned model is implemented. Local Python calls suffice for a demo; a remote endpoint is optional.

## 7. Execution gates, resources, and ownership

### Dependency gates rather than expired start times

| Gate | Required evidence / output | Current status and stop rule |
|---|---|---|
| A0 — trustworthy inputs/checks | Frozen identities; complete two-edit and schema/coverage validation; isolated diagnostics and defined-statistic policies | Assets exist, hardening open. Do not call an audit PASS a complete validation certificate |
| A1 — sequence floor | B0/B1/F0, allele control, cached embeddings, development-selected heads and traceable predictions | Not implemented; complete the baseline before expanding model scope |
| B — structural feasibility | Verified constructs/references, approved spending boundary, three training-only probes, observed runtime/memory/cost, frozen QC | Planned; Modal access confirmed, paid-job approval absent. One already-ready substitute or explicit no-go |
| C — locked structural cohort | Frozen S0 IDs/counts, input versions, concurrency/attempt caps, manifests including failures | Planned; stop admitting jobs beyond remaining budget and result-freeze capacity |
| D — scientific freeze | Frozen choices, scheduled test comparison, paired uncertainty/support, failures/cost, optional I1 | Planned; no test-selected architectures or hidden exclusions |
| E — submission | Exercised real-output demo, recording, attribution, repository link and description | Planned; use local/recorded real output if hosting fails |

Each implementation task needs input artifacts, dependencies, output, acceptance tests, a responsible role, and a stop/fallback condition. Suggested roles are data/evaluation, structure/modeling, and evidence/demo; do not invent assigned people. Solo work follows A0/A1 before structural expansion.

### Modal and spending

**The user confirms Modal GPU access.** Account/billing state, GPU model/VRAM, image compatibility, concurrent capacity, available credits, and a numerical budget have not been independently verified. **Paid jobs require separate approval.** This documentation revision launches none.

Before a paid probe or batch, record: approved amount/currency, account route, GPU/image/checkpoint, bounded wall-clock and attempts, maximum concurrency, measured per-pair cost including overhead, retry reserve, cache strategy, admission/stop rule, and preserved result/demo freeze window. Use measured probe cost to size S0; nominal sponsor offers are not verified balances. Durable approved storage must retain manifests and real outputs.[11][12]

### Other platforms and science infrastructure

- Science Skills supplies evidence retrieval, sequence annotation, references, and visualization—not a half-life model. Its source is pinned to `68832757cbbf941c620b71df5756cf6e5cc287b0`; use the [companion matrix](SCIENCE_SKILLS_PLAN.md#2-science-skills-workflow-matrix) for exact deliverables and limits.[50][51]
- Hugging Face can supply pinned public weights; target-supervised SPEARMINT weights remain outside the primary comparison.[7][13]
- Cognition/Devin, Google research-agent tools, Anthropic, Amass, and other offers are optional engineering/evidence support. Access, billing routes, balances, expiry, and prize criteria must be checked at use, not inferred from logos or stale authentication output.[3][15][16][18]
- Earlier event materials gave conflicting Cognition challenge wording: reproduction-plus-extension versus at least five parallel sessions ending in branches/PRs. Confirm with organizers rather than changing the scientific plan to claim eligibility.[3][18]
- Do not perform automatic account logout/login, install a bundle-wide skill collection, or add unrelated integrations. Credentials belong in approved secret stores, not source, manifests, screenshots, or demos.
- Review applicable software, material, and database terms. Keep required notices; do not distribute modified HLA sequence mirrors without reviewing the relevant conditions.[51][55]

### Deadline and demo

The earlier plan records **Sunday 4 October 2026, 14:45 BST** as the detailed-rules cutoff, with rounded or conflicting timings elsewhere. Preserve that earlier recorded cutoff and ask organizers which presentation timing controls; it was not freshly reverified in this code review. Prepare 90-second and two-minute demonstrations, recording, GitHub link, and short description with a submission buffer.[2][3][18]

The demo must show a supported peptide/allele, a real model prediction and identity, the matched comparison, support/failure/cost information, and an evidence drawer. A displayed structure is labeled `experimental reference`, `predicted complex`, or `no valid structure`. Experimental labels appear only in clearly marked evaluation examples; offline fallback uses real saved outputs. Do not invent patients, improved clinical outcomes, intervals, or geometry where no valid structure exists.

Rollback: preserve sequence results if folding fails; preserve S1 without I1 if inverse scoring fails; preserve CPU baseline evidence if GPU work cannot proceed. A failed or negative branch remains in the scientific record even if its live feature is not shown. No deadline or sponsor rule justifies fabricating a result.

## 8. Verification and future acceptance tests

### This documentation revision

Only `HACKATHON_PLAN.md`, `SCIENCE_SKILLS_PLAN.md`, and `README.md` change. Verify full diffs, Markdown/fences/tables/anchors/local links, pinned references, example syntax, cross-document contracts, and unchanged protected files. Use saved-report values rather than recalculating them. No full script, dataset audit, training test suite, paid compute, or deployment is run for this update.

The synthetic diagnostic checks described in §2 were limited in-memory probes of reviewed code paths, not a full scientific test run or an implemented regression suite.

### Later code-hardening and model tests — not yet implemented

| Proposed area / files | Required fixtures and assertions |
|---|---|
| Split checks: `tests/test_split.py` | Across all three partition pairs, reject distance 0/1/2 and accept an isolated distance-3 case; transitive components stay intact; one- and two-edit planted positives both detected |
| Input contracts: `tests/test_labels.py` | Exact schemas/keys, allowed partitions, no missing/extra assignments, duplicate/conflicting pairs, invalid alphabet/length, negative/nonfinite labels; valid zero remains finite under `log1p` |
| Diagnostic isolation: `tests/test_diagnostic_isolation.py` | Fixed training rows with changed/permuted/removed evaluation labels produce identical A normalization, samples, and output; confirmation analysis is separate |
| Diagnostic sampling/metrics: `tests/test_diagnostics.py` | Unique eligible pairs before caps; deterministic seeded samples under input/process-order changes; explicit self-pair/null policy; constant/empty/insufficient/nonfinite inputs do not silently pass |
| Mapping: `tests/test_chain_mapping.py` | Exact supplied domain, separately tested mismatches/engineered variants, ambiguous extension, mature boundaries, β2m identity, author/label numbering and insertion codes |
| References: `tests/test_reference_manifest.py` | Wrong peptide/allele, incomplete groove, assembly selection, experimental/predicted distinction, missing-reference fallback |
| Geometry: `tests/test_interface_features.py` | Fixtures on both sides of 4.5 Å; unique receptor-residue counting; rigid rotation/translation invariance; no β2m/intrachain contamination; missing/alternate atom policy |
| Inverse scoring: `tests/test_inverse_score.py` | Peptide-only mask/normalization, fixed receptor context, alignment failure, template provenance, matched no-I1 eligibility |
| Metrics/fallbacks: `tests/test_metrics.py` | Pair-order checks, component resampling, same training eligibility, undefined correlations, every selected input predicted, correct valid/full-cohort denominators |
| Inference/demo: `tests/test_inference.py` and smoke checks | Save/load consistency, unsupported inputs, real output/status labels, no B-factor/pLDDT confusion, offline artifact fallback |

Future environments must pin dependencies and isolate pure helper seams before importing scripts with top-level I/O. Run only narrow tests relevant to the implemented change; do not run the generator on frozen benchmark files as a test. No packaging or `pytest` setup exists here yet.

## 9. Definition of done and open decisions

### Documentation completion

- [ ] Master/companion/README agree on current assets, pinned upstream analysis, and proposed work.
- [ ] Frozen split details, label rules, comparison/fallback contracts, and resource gates agree.
- [ ] Identified script gaps remain visibly open, with exact future acceptance cases.
- [ ] Existing scripts, dataset, splits/report, and background primer are unchanged.
- [ ] Focused documentation checks pass; the reviewed update is committed and verified on `sh-arka22/are` with the user's GitHub identity.

### Scientific completion — separate and not achieved by this revision

- [ ] A0 hardening validated without regenerating the benchmark.
- [ ] B0/B1/F0 return traceable, same-split held-out results or a documented blocker.
- [ ] Construct/reference/probe gate produces a measured structural go/no-go.
- [ ] If viable, S0/S1 and optional I1 comparisons include matched controls, failures, uncertainty, support, and actual cost.
- [ ] Any uncertainty output has an appropriate separate design and empirical evidence.
- [ ] Demo/recording uses real outputs and is exercised before submission.

Open external decisions: approved spend cap and GPU specification; source/redistribution and event eligibility; mentor priorities for assay, allele, absolute hours versus ranking, and engineered constructs; actual team ownership and resource availability. These are explicit preflight requirements, not invented facts.

The documentation-only update does not authorize model implementation, script synchronization, fixes, spending, or deployment. Subsequent work stays on the personal branch unless the user directs otherwise. The [Science Skills companion](SCIENCE_SKILLS_PLAN.md) retains the detailed evidence, construct, reference, visualization, licensing, and acceptance contracts supporting this roadmap.

## Sources

[1] https://www.biorxiv.org/content/10.64898/2026.06.28.735023v1.full.pdf
[2] https://iterate.inc/london-ai-science?welcome=true&tab=home
[3] https://iterate.inc/london-ai-science?welcome=true&tab=resources
[4] https://github.com/pirl-unc/spearmint
[5] https://drive.google.com/file/d/11i-LSRZlgLS7T2IVM-12xuddwuryDmnu/view?usp=sharing
[6] https://docs.google.com/spreadsheets/d/1NtZNvcF3u0KFn-1bbuA50CF3IXvs1l4HfbaR3KRLvso/edit?usp=sharing
[7] https://huggingface.co/dkarthikeyan1/spearmint
[8] https://arxiv.org/abs/1905.03222
[9] https://www.nature.com/articles/s41467-025-67971-3
[10] https://pmc.ncbi.nlm.nih.gov/articles/PMC4976001
[11] https://modal.com/docs/guide/batch-processing
[12] https://modal.com/docs/guide/gpu
[13] https://huggingface.co/facebook/esm2_t12_35M_UR50D
[14] https://huggingface.co/facebook/esm2_t30_150M_UR50D
[15] https://amass.tech
[16] https://antigravity.google/use-cases/science
[17] https://arxiv.org/abs/1904.06019
[18] https://docs.google.com/presentation/d/1zpERXrvfL2akHcEySw7gJxSWedW-3K3uk2kobw4jtDY/edit?usp=sharing
[19] https://luma.com/3iipivod
[20] https://www.biorxiv.org/content/biorxiv/early/2026/06/29/2026.06.28.735023/DC1/embed/media-1.pdf?download=true
[21] https://doi.org/10.1016/j.immuno.2023.100030
[22] https://github.com/VarunUllanat/mint
[23] https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md
[24] https://github.com/aswin-giridhar/london-ai-science-hackathon/blob/main/context/serova_primer.pdf
[25] https://github.com/dauparas/ProteinMPNN
[26] https://github.com/dauparas/LigandMPNN
[27] https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2020.01583/full
[28] https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2022.878762/full
[29] https://pmc.ncbi.nlm.nih.gov/articles/PMC13106973
[30] https://github.com/X-lab-3D/swiftmhc-inference
[31] https://www.sciencedirect.com/science/article/pii/S0969212623004136
[32] https://github.com/facebookresearch/esm
[33] https://zenodo.org/records/15495898
[34] https://github.com/chaidiscovery/chai-lab
[35] https://github.com/bytedance/Protenix
[36] https://github.com/westlake-repl/SaProt
[37] https://github.com/X-lab-3D/PANDORA
[38] https://github.com/Biohub/esm
[39] https://github.com/agemagician/ProtTrans
[40] https://github.com/aswin-giridhar/london-ai-science-hackathon/blob/main/context/NOTES.md
[41] https://www.kavrakilab.org/publications/abella2020-pnas.pdf
[42] https://raw.githubusercontent.com/pirl-unc/spearmint/d548dee9b04dd0e3a9d7a50a593d2c030e31d2a5/refs/2field_hla_consensus_seqs.csv
[43] https://github.com/aswin-giridhar/london-ai-science-hackathon/blob/de2532f0a0d97067828dfd0077d0b852a845b920/scripts/make_splits.py
[44] https://github.com/aswin-giridhar/london-ai-science-hackathon/blob/de2532f0a0d97067828dfd0077d0b852a845b920/scripts/audit_splits.py
[45] https://github.com/aswin-giridhar/london-ai-science-hackathon/blob/de2532f0a0d97067828dfd0077d0b852a845b920/scripts/measure_leakage_by_distance.py
[46] https://github.com/aswin-giridhar/london-ai-science-hackathon/blob/de2532f0a0d97067828dfd0077d0b852a845b920/splits/split_report.json
[47] https://github.com/aswin-giridhar/london-ai-science-hackathon/blob/de2532f0a0d97067828dfd0077d0b852a845b920/RECONCILIATION.md
[48] https://github.com/aswin-giridhar/london-ai-science-hackathon/blob/de2532f0a0d97067828dfd0077d0b852a845b920/experiments/BRAINSTORM.md
[49] https://github.com/aswin-giridhar/london-ai-science-hackathon/blob/de2532f0a0d97067828dfd0077d0b852a845b920/splits/README.md
[50] https://github.com/google-deepmind/science-skills/tree/68832757cbbf941c620b71df5756cf6e5cc287b0
[51] https://github.com/google-deepmind/science-skills/blob/68832757cbbf941c620b71df5756cf6e5cc287b0/SKILL_LICENSES.md
[52] https://haematologica.org/article/view/5692
[53] https://pmc.ncbi.nlm.nih.gov/articles/PMC3032881
[54] https://www.ebi.ac.uk/ipd/imgt/hla
[55] https://github.com/ANHIG/IMGTHLA/blob/Latest/LICENCE.md
