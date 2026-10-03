---
agent: devin-local
session: heady-bubble
created: 2026-10-03T14:04:32Z
---
# Stability Lens: Science Skills and structural-evidence companion

Specify the evidence retrieval, verified constructs, reference structures, geometric features, and failure-aware comparisons supporting the [master execution plan](HACKATHON_PLAN.md).

## Summary

This is the technical companion to the current master plan, not a second implementation status report or a directive to edit only one document. The original front matter records the earlier revision proposal's provenance. The current documentation update reconciles this companion, `HACKATHON_PLAN.md`, and `README.md`; it does not install skills, change scripts, regenerate splits, train models, fold complexes, or deploy a service.

The sequence → structure → optional inverse-folding direction remains: B0/B1/F0 provide the full-dataset delivery floor, while matched S0/S1 comparisons test the incremental value of geometry. Science Skills supplies evidence, sequences, references, and visualization—not measured half-life predictions. The pre-event `Peptide-HLA Stability Primer.md` remains unchanged background.

The existing 80/10/10, seed-42, Hamming≤2 component split is authoritative. Modal GPU access is confirmed by the user; hardware/runtime compatibility and a numerical budget remain preflight items, and paid jobs require separate approval.

## Research provenance and current evidence boundary

- **Earlier research:** the initial document review preceded Git integration and the current script inventory. The repository now contains tracked plans, context data, split tooling, frozen assignments/report, and experiment proposals; model training, structural modules, registries, tests, and the demo remain unimplemented.
- Read the supplied ARQ and Tavily Dynamic Search guidance. Inspected Graft/Graphify guidance; there is no usable graph artifact here, and Graft MCP exposed no tools. No index was built.
- **Earlier research:** read the relevant Google DeepMind Science Skills instructions at **`68832757cbbf941c620b71df5756cf6e5cc287b0`**, the repository’s v1.2.1 revision. Inspected PDB, bioRxiv, and Foldseek helper code plus PyMOL recipes.
- **Earlier research:** used bounded Tavily discovery/extraction to verify the official IPD-IMGT/HLA source and two relevant biological caveats. No raw research output was deliberately saved outside this plan.
- **Earlier research:** the inspected official HLA repository reported **IPD-IMGT/HLA 3.65.0**, dated 2026-07-14. This identifies the inspected release, not a promise that it remains the latest during implementation.
- Retain the current plan’s previously reported dataset audit and published-prediction reanalysis **as attributed prior findings**. Do not recompute them merely to rewrite the document, label them as new measurements, or silently treat unavailable audit artifacts as reproduced.
- No Science Skills database wrappers, GPU jobs, training, inference, installation, or deployment ran during this revision research.

Historical pre-revision fingerprints from the original proposal (not current hashes of revised documents):

| Document | SHA-256 |
|---|---|
| `HACKATHON_PLAN.md` | `3c2f8b1c8f8ff0a59d5a2c9e6c9c7ea665e520a170f97ef3f348015d3bee83f3` |
| `Peptide-HLA Stability Primer.md` | `11af78a17fb12d3a271404efd995bb124fe1acc3ed0f995d99d7a5b81fb05020` |

### Upstream script review and unresolved gates

The [master plan's script extraction](HACKATHON_PLAN.md#2-extracted-scripts-and-validation-gaps) is based on upstream `de2532f0a0d97067828dfd0077d0b852a845b920`, compared with personal-branch revision `1273a9735ba0d69d8debc5e53a631c611904a8ab`. Local generation/audit scripts match that snapshot; the local distance diagnostic is still the earlier version. This update does not synchronize it.

The upstream diagnostic's new analysis A forms training pairs but normalizes labels using all dataset rows before selecting them. A synthetic held-out-label perturbation changed training normalized values and pooled ranks. Its null guard also accepted an undefined correlation in an isolated fixture. These findings do not remeasure the sponsor data or quantify its metric bias. The independent audit still enforces only a one-edit cutoff in its pass/fail checks, weaker than the saved two-edit contract.

Treat diagnostic isolation, cutoff-boundary checks, explicit undefined statistics, deterministic pair sampling, and schema/provenance hardening as open implementation gates. Do not claim a clean train-only justification, complete audit certification, or newly reproduced model performance. Keep the historical disclosure that evaluation labels influenced the original split design; neither a later analysis nor this document erases that history.

The current report records 5,410 components (largest eight) and primary train/val/test counts of 22,532/2,817/2,817 rows, 4,514/563/556 peptides, and 75/73/75 alleles. Read these as saved-report facts, not freshly recalculated findings. One-edit counts remain historical sensitivity evidence only.

## Technical workflow specifications

### 1. Scope tiers and scientific target

The master plan distinguishes four scope tiers:

1. **Delivery floor:** B0/B1/F0, measured on the full sponsor dataset under one locked split.
2. **Main scientific differentiator:** paired F0 versus F0 + geometric features on one predeclared, bounded structural cohort.
3. **Exploratory extension:** ProteinMPNN peptide-only scoring on those same structures, with a fixed-template control.
4. **Research infrastructure:** Science Skills supplies evidence, sequences, references, and visualization—not half-life predictions.

Keep the existing 28,166-row versus paper-curated-data distinction, 182-aa domain versus full-chain distinction, target-supervised checkpoint exclusions, source attribution, and nonclinical scope. Keep Boltz-2 coordinates as the primary co-fold route and exclude its small-molecule affinity output.

Do not add AlphaGenome, ChEMBL, clinical-trial mining, large MSA campaigns, molecular dynamics, sequence redesign, or another foundation-model sweep to the MVP. Do not describe “no paper found in our bounded search” as proof of novelty.

### 2. Science Skills workflow matrix

Each workflow below supports a concrete artifact and the master plan's execution gates; it is not a list of sponsor integrations to install.

| Priority / upstream directory | Input and inspected helper | Concrete contribution | Limits / fallback |
|---|---|---|---|
| Core: `literature_search_europepmc` | DOI/topic → `scripts/europepmc_api.py search`, then eligible full text | Claim ledger with DOI, version/date, exact supporting passage, endpoint, assay, and limitations | Wrapper forces open-access search; no match is not evidence of absence from the field |
| Core: `literature_search_biorxiv` | Known SPEARMINT DOI → `scripts/search_by_doi.py` | Verify preprint metadata and the version being cited | Helper takes the first returned record; explicitly inspect returned version/date instead of claiming it found the latest; do not browse months of preprints for one DOI |
| Core: `uniprot_database` | Bounded accession/gene lookup → `scripts/uniprot_tools.py` | Verified protein annotations, processing boundaries, β2-microglobulin metadata, and cross-references | Not an HLA allele resolver; never replace a two-field allele with a generic canonical gene sequence |
| Core: `pdb_database` | Verified heavy-chain sequence → search, metadata, coordinate helpers | Small experimental pHLA reference panel, chain/assembly maps, template/exposure ledger | A PDB entry can contain several complexes, non-9-mer peptides, TCRs, engineered chains, or missing atoms |
| Core when structures exist: `pymol` | Validated local coordinate file plus explicit chain map | Human-auditable groove overlay, contact view, PNG and editable `.pse` session | Visualization is not kinetic validation; test headless rendering on the actual host |
| Optional: `alphafold_database_fetch_and_analyze` | Verified UniProt accession → fetch/pLDDT/PAE helpers | Monomer confidence/domain context only if it resolves a construct question | Does not run a custom co-fold or provide peptide-interface features; skip unless exact sequence identity is established |
| Optional: `foldseek_structural_search` | Existing coordinate file → `scripts/search.py --databases pdb100` | Candidate structural relatives for the reference/exposure ledger | Requires coordinates and uploads them; a global HLA fold match does not validate the peptide pose or prove absence of training overlap |
| Exception path: `protein_sequence_msa`, `protein_sequence_similarity_search` | Heavy-chain sequences → Clustal Omega or MMseqs2/BLAST helpers | Investigate unresolved HLA mapping with explicit identity/coverage denominators | Not the 9-mer leakage checker; do not replace the exact Hamming≤2 component contract with heuristic homology search |

Pin source revision, command, retrieval date, input/output hash, and tool version for every executed workflow. Use Tavily for discovery and cross-checking; use the specialized, rate-limited wrappers for production database retrieval.

**Licensing and installation:** the source repository distinguishes Apache-2.0 software, CC-BY materials, and separate database terms. Review terms and record required notices before actual database use. The HLA database’s `LICENCE.md` specifically requires permission before distributing modified data; publish retrieval instructions and hashes rather than an unreviewed modified sequence mirror. Do not blindly run the repository’s bundle-wide `npx` install. Any later approved project-local skill configuration belongs in `.devin/skills/`; it is outside this document-edit approval.

### 3. Sequence and construct registry

The planned `src/structure_inputs.py` responsibility is:

- Preserve the sponsor allele string, engineered-variant suffix, 182-aa sequence, pseudosequence, and pair ID.
- Resolve named alleles against a **pinned official IPD-IMGT/HLA release**. The existing SPEARMINT consensus reference remains useful prior evidence, but is not permission to overwrite a mismatched sponsor sequence.
- Require an exact match across the supplied 182-aa domain before calling a mapping verified. Record positions using an explicit numbering convention.
- If several full sequences remain compatible with the supplied fragment, record the ambiguity; do not silently choose an extension and call it uniquely identified.
- For the primary pilot, propose a **verified mature extracellular heavy-chain construct plus mature β2-microglobulin and the peptide**. Record the exact sequences and boundaries. Do not feed the entire signal/transmembrane-containing precursor by default.
- A domain-only construct, if used as fallback, is a separately named protocol and cannot silently replace the primary construct within one comparison.
- Keep the prior domain mismatches (`HLA-B*08:03`, `HLA-A*02:50`, `HLA-A*24:19`) separate from the three engineered C67S constructs; quarantine unresolved inputs from the structural arm while retaining their valid measured rows in sequence experiments.
- Perform sequence extraction/alignment in code, not by manually rewriting amino-acid strings.

Proposed `data/hla_mapping.jsonl` fields:

`allele_id, sponsor_domain_sha256, reference_database, reference_release, reference_accession, reference_sequence_sha256, domain_mapping, construct_type, construct_boundaries, construct_sequence_sha256, beta2m_accession, beta2m_sequence_sha256, mutations, mapping_status, source_url, retrieved_at`.

Mapping statuses must distinguish `verified`, `ambiguous_extension`, `domain_mismatch`, `engineered_unverified`, and `unmapped`. A record is not eligible for the primary structural cohort merely because its allele name resembles a database annotation.

### 4. Reference-first structural triage

Before the three training-only co-fold probes:

1. Use PDB sequence search on the verified HLA domain/heavy chain to discover candidate entities. A high-identity search is discovery, not proof of exact allele identity.
2. Fetch schema before constructing attribute/GraphQL metadata queries. Explain that the query discovers heavy-chain matches, then separately checks whether the associated biological assembly contains an appropriate peptide complex.
3. Inspect experimental method/resolution, assembly, entity and chain IDs, peptide sequence/length, β2m, modifications, engineered mutations, missing residues, alternate conformers, and primary citation.
4. Map both `label_asym_id`/`auth_asym_id` and label/author residue numbering. Do not assume the peptide is chain C or that the biological assembly equals the asymmetric unit.
5. Select a small reference set by exact construct compatibility, usable groove/peptide coordinates, and experimental quality, without using test half-lives. Record why candidates were rejected. No qualifying reference is an allowed outcome.
6. Keep three uses separate:
   - **Reference/QC:** experimentally grounded groove orientation and chain checks.
   - **Fixed-template I1 control:** one allele-compatible backbone selected without the evaluated peptide’s label; exclude evaluated test-component peptides from template selection where possible.
   - **Exposure audit:** exact/near peptide–HLA matches found in public structures, with release dates and known predictor cutoffs.

Do not label an arbitrary matched crystal complex as a new independent half-life test set. Measured stability must have an explicit source and compatible construct/assay before any such claim.

Do not introduce PANDORA/MODELLER setup as a required detour. PDB retrieval does not thread a new peptide, relax side chains, or generate its bound coordinates. Fixed-template ProteinMPNN scoring is a backbone-compatibility control, not a new predicted all-atom complex.

For overlays, superpose on mapped HLA groove residues first. Only call peptide RMSD a pose-error measurement when the reference matches the peptide and relevant construct. Otherwise label the image a reference overlay. Do not align the peptide onto itself and present the minimized number as docking accuracy.

Proposed `data/reference_manifest.jsonl` fields:

`reference_id, pdb_id, assembly_id, entity_ids, label_chain_map, auth_chain_map, residue_map, peptide_sequence, hla_sequence_hash, beta2m_sequence_hash, experimental_method, resolution_or_null, modifications, missing_residues, release_date, primary_citation, coordinate_sha256, eligibility, use_role, exposure_status`.

### 5. Structure-feature contract

The master plan adopts this proposed feature contract:

**MVP geometry:** for each peptide position P1–P9, compute the fraction of mapped 182-domain HLA residues with at least one peptide/HLA heavy-atom pair at distance **≤4.5 Å**. This is a prespecified engineering definition, not a universal physical threshold. Use one deterministic atom/alternate-conformer policy and record the mapped-residue denominator.

Store the 9-position vector, raw counts, missing-coordinate masks, and feature-schema version. This avoids conflating a larger side chain’s atom count with the number of contacted receptor residues. Use the full position vector; show P2/P9 for readability without asserting that all alleles share those as their only anchors. Literature describes P3-dominant exceptions.

**Separate channels:** pose-validity flags and predictor confidence remain distinct from contact geometry. Read confidence using the selected predictor’s documented output and scale; never interpret a crystallographic B-factor as pLDDT.

**Stretch only:** buried surface area, chemically typed hydrogen-bond geometry, salt bridges, and pocket depth. Promote them only after their algorithms, atom requirements, units, and fixture tests are fixed. A generic distance cutoff or PyMOL polar-contact display alone must not be described as a verified hydrogen-bond count.

**Scientific hypothesis:** contacts may predict half-life, but more contacts or “better” anchors are not guaranteed to increase persistence. Published experimental work found that anchor improvements did not always improve measured stability. Use this as a falsification caveat, not as a reason to abandon the paired test.

### 6. Selection and failure-aware evaluation

Use the saved 80/10/10 Hamming≤2 component assignments, training-only learned transforms, and matched S0 rows. These are required implementation contracts, not claims that the current scripts enforce every safeguard:

- Freeze model/head families and hyperparameter selection on training/validation data. Test results report the predeclared comparison; they do not select the winning model for that same test or decide which negative findings to retain.
- Freeze structural cohort membership before folding. Do not replace failed or low-confidence test pairs with easier examples.
- Predeclare bounded attempts per pair, selection by label-blind QC/confidence, seeds, cost cap, and stopping rules.
- Keep supervised-label contamination, exact peptide-component leakage, and structural-template exposure as separate audit axes. Public PDB/Foldseek search cannot prove the contents of an undisclosed pretraining corpus.

On S0, run the following using the same training/development partitions, head-selection protocol, and paired test rows:

| Comparison | Purpose |
|---|---|
| Matched-size B1 and F0 | Preserve the non-foundation and sequence-only controls |
| F0 + availability/QC indicators | Detect signal from which structures can be produced or pass QC |
| F0 + availability/QC + predictor confidence | Test confidence separately from physical contacts |
| F0 + availability/QC + confidence + contact geometry | Primary incremental geometry test |
| Structure-only head | Secondary diagnostic; not a replacement for the paired control |
| Above + peptide-only ProteinMPNN score | Optional incremental inverse-folding comparison with fixed-template sensitivity control |

Train the availability/QC-only diagnostic on all S0 training rows, since availability itself is defined for every attempted pair. Train the confidence and confidence-plus-geometry heads on exactly the same valid-structure training rows. Their paired difference is the primary geometry contrast; both use the same F0 fallback for invalid test structures. If I1 has a smaller eligible training set, refit its no-I1 comparator on that same set and use the already-frozen no-I1 pipeline as the fallback for missing inverse scores. Do not compare extra-feature models with different training eligibility and attribute the whole difference to the added feature.

For valid/invalid handling:

1. **Primary full-cohort operational result:** every preselected test pair receives a prediction. If a structural branch is unavailable/invalid, use its predeclared matched F0 fallback and record `model_used`/`fallback_reason`. Fit hybrid heads only using the explicitly eligible training rows; never use failed test outcomes to decide eligibility.
2. **Secondary valid-structure analysis:** compare F0 and the hybrid on the identical valid subset; include an F0 control refit on the same eligible training rows to distinguish feature effects from training-set selection. Label this analysis conditional on structure availability.
3. Report selected count, attempted count, valid count, fallback count, inverse-score availability, and costs. A complete-case gain alone is not the all-input operational gain.

Primary metric remains held-out Spearman with **paired component bootstrap**, plus log-scale MAE/RMSE and per-allele support. State that resampling test components quantifies conditional test-sampling uncertainty, not retraining uncertainty. Report undefined correlations/intervals for insufficient or constant data rather than coercing them to zero. Pooled correlation can reflect between-allele differences, so always show supported within-allele results.

If test support is small, call the result a pilot and report interval width; do not promise significance from an arbitrary subset size. A confidence interval spanning no gain means inconclusive evidence, not proof of equivalence.

**Target contract:** reject negative/nonfinite labels, then use `y = log1p(thalf_hours)`. Do not clip invalid labels into valid zeros. Presentation clipping of predictions is a separate recorded choice.

**U1 remains optional:** pooled row-level conformal guarantees do not automatically follow when observations share peptide components or undergo assay shift. Use an appropriate group-aware design before claiming coverage guarantees; otherwise report empirical coverage only. Neither conformal intervals nor ensemble spread automatically detect out-of-distribution inputs. The current split has no separate calibration partition; any future design must preserve the test set and separate calibration from model selection rather than silently introduce a four-way resplit.

### 7. Evidence artifacts and demo

The proposed evidence-to-prediction flow is:

```text
Paper/source lookup → claim ledger ──────────────────────────────┐
Existing CSV + saved split → hashes + hardened checks            │
       ├─ B0/B1 + frozen ESM-2 → F0                              │
       └─ HLA sequence registry → eligible locked S0              │
              ├─ PDB references → chain/pose/exposure checks      │
              └─ verified construct + peptide → Boltz-2          │
                    → explicit QC/availability/confidence        │
                    → compact interface features                 │
                    → optional peptide-only ProteinMPNN          │
                                  ↓                              │
             paired heads + fallback + one locked evaluation     │
                                  ↓                              │
             metrics / uncertainty / coverage / measured cost    │
                                  └────────── provenance ────────┘
                                                ↓
                     Streamlit prediction + evidence/pose drawer
```

The following artifact specifications remain unimplemented; no new service is required:

- `data/evidence_manifest.jsonl`: claim ID, DOI/URL, paper/version/type, exact support location, endpoint/assay, license, status (`supported`, `hypothesis`, `unverified`), and retrieval details.
- `data/hla_mapping.jsonl` and `data/reference_manifest.jsonl`: registries described above.
- Planned `data/structures_manifest.jsonl`: include construct/coordinate hashes, chain-map version, reference IDs, QC reason, confidence field/scale, attempts, failure status, feature version, and per-pair runtime/cost.
- Planned result records: include full-cohort and valid-subset identifiers, denominator counts, fallback fraction, reference/exposure flags, and actual model used.
- Keep raw data/coordinates local or in approved storage until redistribution is cleared; never put credentials into artifacts.

**Demo layout:** one real peptide/allele, actual predicted half-life and model ID, measured label only in clearly marked evaluation examples, sequence-vs-structure ablation, cost/failure rate, and an evidence drawer. Mark a displayed object `experimental reference`, `predicted complex`, or `no valid structure`; never imply an unrelated reference is the selected peptide’s pose.

Use explicit peptide polymer-chain selections in PyMOL. The skill’s small-molecule `organic` recipe is unsuitable for selecting a protein peptide chain. Save PNG plus `.pse`, check nonzero atoms, and use static real-output images as a nonblocking fallback if headless rendering or interactive 3D fails.

### 8. Resource-gated delivery

Retain the earlier recorded **Sunday 4 October 2026, 14:45 BST** submission cutoff and timing-conflict warning as prior event-source information, not freshly reverified scheduling. Follow dependency gates instead of expired start times.

| Gate | Required artifact / decision | Current state / fallback |
|---|---|---|
| A0 — trustworthy inputs/checks | Frozen input identities, complete two-edit/schema checks, isolated diagnostics, defined-statistic policies | Artifacts exist; hardening is open. An existing audit PASS is not full certification |
| A1 — reliable baseline | B0/B1/F0, fair allele controls, validation-selected heads, cached embeddings and traceable predictions | Not implemented; precedes GPU-heavy expansion |
| B — structural feasibility | Verified constructs/references, approved budget, three training-only probes, actual runtime/memory/cost, frozen QC | Planned; one already-ready alternative or explicit untested structure |
| C — locked cohort | S0 IDs/counts, worker/attempt limits, spend stop rule, complete manifests | Planned; stop admitting work beyond budget or result-freeze capacity |
| D — scientific freeze | Frozen choices, paired scheduled evaluation, uncertainty/support/failures, optional I1 | Planned; positive, negative, inconclusive, and unrun outcomes stay explicit |
| E — submission | Exercised real-output demo, recording, attribution, description, repository link | Planned; preserve submission buffer and offline real-output fallback |

Modal GPU access is confirmed by the user, not independently runtime-tested. GPU model/VRAM, image compatibility, account route, available credits, concurrency, and a numerical cap remain preflight items. **Paid jobs require separate approval.** Estimate cohort cost from a later approved probe including overhead and retry reserve; record a spend cap, maximum attempts/concurrency, and admission/stop rule. This revision launches no jobs.

Parallel team: data/evaluation owner; structural-input/model owner; evidence/demo owner. Solo: A0/A1 first, then B/C; skip optional searches, larger encoders, and U1 before compromising the core comparison. The evidence workflow must support delivery, not consume the weekend.

Science Skills is research infrastructure, not a half-life model. Do not infer access to Antigravity, other paid APIs, or prize eligibility from sponsor offers. Verify account/billing routes before use; retain no automatic logout/login instructions.

## Document roles and approved scope

| Document | Responsibility |
|---|---|
| [Master plan](HACKATHON_PLAN.md) | Authoritative current execution/evaluation contract, extracted script behavior, open hardening gates, model ladder, resources, and acceptance criteria |
| This companion | Detailed source workflows, construct/reference registries, feature definitions, failure controls, visualization, and source terms |
| [README](README.md) | Accessible project overview, current implementation status, navigation, and safe onboarding |
| Background primer | Unchanged pre-event reference; not a current execution contract |

The approved revision updates only the master plan, this companion, and README on `sh-arka22/are`, then verifies and publishes those documents with the user's GitHub identity. Scripts, input data, split files/report, experiments, and primer stay unchanged. Future manifests, modules, tests, images, and `.devin/skills` paths are specifications, not files created by this update. No installation, full scientific script run, training, folding, paid compute, or deployment is included.

## Verification

### Documentation verification

- [ ] Review all three document diffs and their shared split/model/feature/fallback contracts.
- [ ] Confirm the primer, scripts, canonical input, and frozen split/report fingerprints are unchanged.
- [ ] Preserve saved counts and prior findings with their evidence status; do not re-derive them for a rewrite.
- [ ] Verify Markdown structure, fences, tables, internal anchors, local links, source references, and example syntax.
- [ ] Distinguish the local diagnostic from the newer pinned upstream analysis and keep all identified hardening gaps open.
- [ ] Check no unsupported novelty, clinical, confidence-to-half-life, clean-pretraining, or prospective-blind-design claim appears.
- [ ] Check no test-driven model/subset/template selection or silent fold removal remains.
- [ ] Label resources, uploads, licenses, future commands, and unimplemented artifacts accurately.
- [ ] State that no full scientific script or model test suite ran for this documentation update; the isolated synthetic diagnostic checks are not a completed regression suite.

The repository now supports `git diff --check`, full diff review, and unchanged-file comparisons. Use a read-only standard-library Markdown/link scan rather than inventing a nonexistent build. Proposed implementation tests below require separate code work before they can run.

### Future implementation acceptance tests

- `tests/test_chain_mapping.py`: exact 182-aa mapping, mismatched allele, ambiguous extension, C67S quarantine, mature-chain boundaries, β2m identity, label/auth renumbering and insertion codes.
- `tests/test_reference_manifest.py`: wrong peptide, wrong allele, incomplete groove, experimental versus predicted type, assembly selection, and missing-reference fallback.
- `tests/test_interface_features.py`: known synthetic contact distances on both sides of 4.5 Å; unique receptor-residue counting; rigid rotation/translation invariance; no intra-chain/β2m contacts; alternate conformers and missing atoms handled deterministically.
- `tests/test_metrics.py`: identical pair order, paired component resampling, constant/insufficient targets, complete-case versus full-cohort counts, F0 fallback, and no model selection on test outcomes.
- `tests/test_inverse_score.py`: only the peptide is scored, receptor context stays fixed, score normalization is explicit, template provenance is attached.
- `tests/test_split.py` and `tests/test_labels.py`: all three partition pairs reject distance 0/1/2 and accept an isolated distance-3 fixture; transitive components, exact assignment coverage, allowed split names, exact 9-mer/alphabet rules, negative/nonfinite-label rejection, zero-hour finiteness, and save/load/unsupported-input handling.
- `tests/test_diagnostic_isolation.py`: held-out-only label perturbation/permutation/removal leaves training normalization, samples, and analysis-A outputs unchanged; confirmation-only evaluation is separate.
- `tests/test_diagnostics.py`: undefined/constant/empty/nonfinite statistics cannot silently pass or become zero; sampling filters unique eligible pairs before capping and has deterministic ordering, a declared null/self-match policy, and support counts.
- Demo smoke check: real prediction/artifact, correct structure-status label, experimental B-factor not shown as pLDDT, offline real-output fallback, working recording.

Proposed narrow commands **only after those files and dependencies exist**:

```text
uv run pytest -q tests/test_chain_mapping.py tests/test_reference_manifest.py tests/test_interface_features.py
uv run pytest -q tests/test_split.py tests/test_labels.py tests/test_metrics.py tests/test_inverse_score.py tests/test_inference.py
```

## Risks / Considerations

1. **Skills are workflow assistance, not scientific guarantees.** Their scripts still require input/schema review, numerical checks, and dependency/version pinning.
2. **PDB does not solve the label problem.** Exact matched sequence/construct structures may be absent, and crystal availability is not a measured stability outcome.
3. **Construct mismatch can dominate the experiment.** Keep verified-domain identity and full-construct assumptions visible; do not “repair” the sponsor assay molecule into a different one.
4. **Static geometry may add no kinetic signal.** A negative or inconclusive paired result is valid; do not expand scope merely to force a positive metric.
5. **Small selected subsets and fold failures change the estimand.** Report both operational all-pair performance and conditional valid-pose analysis, with matching controls.
6. **Online search cannot prove full pretraining decontamination.** Record what was checked and what remains unknown.
7. **Wrappers have concrete quirks.** PDB returns all hits unless pagination is set; its implemented count flag is `--count-only`, despite an inconsistent prose example. bioRxiv DOI retrieval does not explicitly choose the newest version. Foldseek needs an external timeout and narrow database selection rather than an unbounded polling dependency.
8. **Tavily compatibility was observed.** The installed CLI advertises `--chunks-per-source`, but the extraction backend rejected it. Plain extraction plus bounded in-memory filtering succeeded; do not prescribe an update or configuration change just to work around this.
9. **License/output side effects remain gated.** Running skills can create downloads, caches, notices, or external uploads. Those executions are outside this documentation-only update.
10. **Access is not an approved execution budget.** Modal access is user-confirmed; hardware/runtime compatibility, spending cap, and paid-job approval remain outstanding, as do rendering support, mentor decisions, and submission-rule ambiguities.
11. **Existing tooling is not complete certification.** The audit threshold mismatch, all-row diagnostic normalization, undefined-statistic behavior, and sampling/conclusion limits remain executable-code gaps. This companion specifies acceptance criteria but does not repair them.

## Verified Sources for the Revision

The master plan retains earlier sources [1]–[42] with their original evidence boundaries and adds the pinned script review. The following workflow sources were inspected during the earlier Science Skills research, not newly executed by this revision:

- Google DeepMind Science Skills, pinned source and license overview: https://github.com/google-deepmind/science-skills/tree/68832757cbbf941c620b71df5756cf6e5cc287b0
- Skill-specific terms: https://github.com/google-deepmind/science-skills/blob/68832757cbbf941c620b71df5756cf6e5cc287b0/SKILL_LICENSES.md
- PDB skill and scripts: https://github.com/google-deepmind/science-skills/tree/68832757cbbf941c620b71df5756cf6e5cc287b0/skills/pdb_database
- UniProt skill: https://github.com/google-deepmind/science-skills/tree/68832757cbbf941c620b71df5756cf6e5cc287b0/skills/uniprot_database
- Europe PMC skill: https://github.com/google-deepmind/science-skills/tree/68832757cbbf941c620b71df5756cf6e5cc287b0/skills/literature_search_europepmc
- bioRxiv skill and DOI helper: https://github.com/google-deepmind/science-skills/tree/68832757cbbf941c620b71df5756cf6e5cc287b0/skills/literature_search_biorxiv
- PyMOL instructions and recipes: https://github.com/google-deepmind/science-skills/tree/68832757cbbf941c620b71df5756cf6e5cc287b0/skills/pymol
- AFDB skill: https://github.com/google-deepmind/science-skills/tree/68832757cbbf941c620b71df5756cf6e5cc287b0/skills/alphafold_database_fetch_and_analyze
- Foldseek skill and wrapper: https://github.com/google-deepmind/science-skills/tree/68832757cbbf941c620b71df5756cf6e5cc287b0/skills/foldseek_structural_search
- MSA skill: https://github.com/google-deepmind/science-skills/tree/68832757cbbf941c620b71df5756cf6e5cc287b0/skills/protein_sequence_msa
- Sequence-search skill: https://github.com/google-deepmind/science-skills/tree/68832757cbbf941c620b71df5756cf6e5cc287b0/skills/protein_sequence_similarity_search
- Official IPD-IMGT/HLA database: https://www.ebi.ac.uk/ipd/imgt/hla
- Official sequence repository: https://github.com/ANHIG/IMGTHLA
- HLA data terms: https://github.com/ANHIG/IMGTHLA/blob/Latest/LICENCE.md
- Anchor-position exceptions: “The nature of peptides presented by an HLA class I low expression allele”: https://haematologica.org/article/view/5692
- Anchor improvements need not improve stability: “Real time detection of peptide–MHC dissociation reveals that improvement of primary MHC-binding residues can have a minimal, or no, effect on stability”: https://pmc.ncbi.nlm.nih.gov/articles/PMC3032881
- Reviewed scripts and supporting snapshot: https://github.com/aswin-giridhar/london-ai-science-hackathon/tree/de2532f0a0d97067828dfd0077d0b852a845b920/scripts
- Script contracts, evidence qualifications, and future hardening: [master plan §2](HACKATHON_PLAN.md#2-extracted-scripts-and-validation-gaps)
