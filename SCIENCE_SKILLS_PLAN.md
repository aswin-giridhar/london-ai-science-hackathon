---
agent: devin-local
session: heady-bubble
created: 2026-10-03T14:04:32Z
---
# Stability Lens: Science-Skills-grounded revision

Improve the Serova hackathon plan by grounding structural modeling in verified allele sequences and experimental references, then testing incremental half-life prediction with explicit provenance, failure handling, and compute limits.

## Summary

Revise **only `HACKATHON_PLAN.md`** after approval. Preserve its sequence → structure → inverse-folding research direction, peptide-component-disjoint evaluation, and honest negative-result fallback. Add a small, concrete Google DeepMind Science Skills workflow instead of another collection of models.

**Recommended project framing:** “Stability Lens tests whether predicted peptide–HLA geometry adds useful half-life information beyond sequence, with every construct, pose, result, and failure traceable to its evidence.”

The immediate deliverable is an improved planning document—not an installed skill bundle, trained model, new code repository, or deployment. Keep `Peptide-HLA Stability Primer.md` unchanged as background. Put relevant corrections and caveats into the plan so the implementation does not inherit the primer’s overclaims.

## Research completed and evidence boundary

- Read both supplied documents in full. The workspace contains those documents, `.claude` settings, and a `graft` cache; no implemented pipeline, package manifest, tests, or structural files are present.
- Read the supplied ARQ and Tavily Dynamic Search guidance. Inspected Graft/Graphify guidance; there is no usable graph artifact here, and Graft MCP exposed no tools. No index was built.
- Read the relevant Google DeepMind Science Skills instructions at **`68832757cbbf941c620b71df5756cf6e5cc287b0`**, the repository’s v1.2.1 revision. Inspected PDB, bioRxiv, and Foldseek helper code plus PyMOL recipes.
- Used bounded Tavily discovery/extraction to verify the official IPD-IMGT/HLA source and two relevant biological caveats. No raw research output was deliberately saved outside this plan.
- Confirmed that the official HLA repository reports **IPD-IMGT/HLA 3.65.0**, dated 2026-07-14. This identifies the inspected release, not a promise that it remains the latest during implementation.
- Retain the current plan’s previously reported dataset audit and published-prediction reanalysis **as attributed prior findings**. Do not recompute them merely to rewrite the document, label them as new measurements, or silently treat unavailable audit artifacts as reproduced.
- No Science Skills database wrappers, GPU jobs, training, inference, installation, or deployment ran during this revision research.

Source-document fingerprints before any approved edit:

| Document | SHA-256 |
|---|---|
| `HACKATHON_PLAN.md` | `3c2f8b1c8f8ff0a59d5a2c9e6c9c7ea665e520a170f97ef3f348015d3bee83f3` |
| `Peptide-HLA Stability Primer.md` | `11af78a17fb12d3a271404efd995bb124fe1acc3ed0f995d99d7a5b81fb05020` |

## Implementation Steps

### 1. Tighten the executive decision without changing the target

Revise existing §§0–2 to distinguish:

1. **Delivery floor:** B0/B1/F0, measured on the full sponsor dataset under one locked split.
2. **Main scientific differentiator:** paired F0 versus F0 + geometric features on one predeclared, bounded structural cohort.
3. **Exploratory extension:** ProteinMPNN peptide-only scoring on those same structures, with a fixed-template control.
4. **Research infrastructure:** Science Skills supplies evidence, sequences, references, and visualization—not half-life predictions.

Keep the existing 28,166-row versus paper-curated-data distinction, 182-aa domain versus full-chain distinction, target-supervised checkpoint exclusions, source attribution, and nonclinical scope. Keep Boltz-2 coordinates as the primary co-fold route and exclude its small-molecule affinity output.

Do not add AlphaGenome, ChEMBL, clinical-trial mining, large MSA campaigns, molecular dynamics, sequence redesign, or another foundation-model sweep to the MVP. Do not describe “no paper found in our bounded search” as proof of novelty.

### 2. Add a compact Science Skills → deliverable matrix

Place the matrix near the evidence baseline and connect it to the execution gates, not just the sponsor table.

| Priority / upstream directory | Input and inspected helper | Concrete contribution | Limits / fallback |
|---|---|---|---|
| Core: `literature_search_europepmc` | DOI/topic → `scripts/europepmc_api.py search`, then eligible full text | Claim ledger with DOI, version/date, exact supporting passage, endpoint, assay, and limitations | Wrapper forces open-access search; no match is not evidence of absence from the field |
| Core: `literature_search_biorxiv` | Known SPEARMINT DOI → `scripts/search_by_doi.py` | Verify preprint metadata and the version being cited | Helper takes the first returned record; explicitly inspect returned version/date instead of claiming it found the latest; do not browse months of preprints for one DOI |
| Core: `uniprot_database` | Bounded accession/gene lookup → `scripts/uniprot_tools.py` | Verified protein annotations, processing boundaries, β2-microglobulin metadata, and cross-references | Not an HLA allele resolver; never replace a two-field allele with a generic canonical gene sequence |
| Core: `pdb_database` | Verified heavy-chain sequence → search, metadata, coordinate helpers | Small experimental pHLA reference panel, chain/assembly maps, template/exposure ledger | A PDB entry can contain several complexes, non-9-mer peptides, TCRs, engineered chains, or missing atoms |
| Core when structures exist: `pymol` | Validated local coordinate file plus explicit chain map | Human-auditable groove overlay, contact view, PNG and editable `.pse` session | Visualization is not kinetic validation; test headless rendering on the actual host |
| Optional: `alphafold_database_fetch_and_analyze` | Verified UniProt accession → fetch/pLDDT/PAE helpers | Monomer confidence/domain context only if it resolves a construct question | Does not run a custom co-fold or provide peptide-interface features; skip unless exact sequence identity is established |
| Optional: `foldseek_structural_search` | Existing coordinate file → `scripts/search.py --databases pdb100` | Candidate structural relatives for the reference/exposure ledger | Requires coordinates and uploads them; a global HLA fold match does not validate the peptide pose or prove absence of training overlap |
| Exception path: `protein_sequence_msa`, `protein_sequence_similarity_search` | Heavy-chain sequences → Clustal Omega or MMseqs2/BLAST helpers | Investigate unresolved HLA mapping with explicit identity/coverage denominators | Not the 9-mer leakage checker; do not replace the exact one-edit peptide graph with heuristic homology search |

Pin source revision, command, retrieval date, input/output hash, and tool version for every executed workflow. Use Tavily for discovery and cross-checking; use the specialized, rate-limited wrappers for production database retrieval.

**Licensing and installation:** the source repository distinguishes Apache-2.0 software, CC-BY materials, and separate database terms. Review terms and record required notices before actual database use. The HLA database’s `LICENCE.md` specifically requires permission before distributing modified data; publish retrieval instructions and hashes rather than an unreviewed modified sequence mirror. Do not blindly run the repository’s bundle-wide `npx` install. Any later approved project-local skill configuration belongs in `.devin/skills/`; it is outside this document-edit approval.

### 3. Replace the generic mapping gate with a sequence/construct registry

Extend existing §3 and the planned `src/structure_inputs.py` responsibility:

- Preserve the sponsor allele string, engineered-variant suffix, 182-aa sequence, pseudosequence, and pair ID.
- Resolve named alleles against a **pinned official IPD-IMGT/HLA release**. The existing SPEARMINT consensus reference remains useful prior evidence, but is not permission to overwrite a mismatched sponsor sequence.
- Require an exact match across the supplied 182-aa domain before calling a mapping verified. Record positions using an explicit numbering convention.
- If several full sequences remain compatible with the supplied fragment, record the ambiguity; do not silently choose an extension and call it uniquely identified.
- For the primary pilot, propose a **verified mature extracellular heavy-chain construct plus mature β2-microglobulin and the peptide**. Record the exact sequences and boundaries. Do not feed the entire signal/transmembrane-containing precursor by default.
- A domain-only construct, if used as fallback, is a separately named protocol and cannot silently replace the primary construct within one comparison.
- Keep the current three domain mismatches and C67S constructs quarantined from the structural arm until reconciled. Keep their valid measured rows in the sequence experiment.
- Perform sequence extraction/alignment in code, not by manually rewriting amino-acid strings.

Proposed `data/hla_mapping.jsonl` fields:

`allele_id, sponsor_domain_sha256, reference_database, reference_release, reference_accession, reference_sequence_sha256, domain_mapping, construct_type, construct_boundaries, construct_sequence_sha256, beta2m_accession, beta2m_sequence_sha256, mutations, mapping_status, source_url, retrieved_at`.

Mapping statuses must distinguish `verified`, `ambiguous_extension`, `domain_mismatch`, `engineered_unverified`, and `unmapped`. A record is not eligible for the primary structural cohort merely because its allele name resembles a database annotation.

### 4. Add reference-first structural triage, not a mandatory new modeling branch

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

### 5. Specify a smaller, more defensible structure-feature experiment

Update §§3–4 and the planned feature contract:

**MVP geometry:** for each peptide position P1–P9, compute the fraction of mapped 182-domain HLA residues with at least one peptide/HLA heavy-atom pair at distance **≤4.5 Å**. This is a prespecified engineering definition, not a universal physical threshold. Use one deterministic atom/alternate-conformer policy and record the mapped-residue denominator.

Store the 9-position vector, raw counts, missing-coordinate masks, and feature-schema version. This avoids conflating a larger side chain’s atom count with the number of contacted receptor residues. Use the full position vector; show P2/P9 for readability without asserting that all alleles share those as their only anchors. Literature describes P3-dominant exceptions.

**Separate channels:** pose-validity flags and predictor confidence remain distinct from contact geometry. Read confidence using the selected predictor’s documented output and scale; never interpret a crystallographic B-factor as pLDDT.

**Stretch only:** buried surface area, chemically typed hydrogen-bond geometry, salt bridges, and pocket depth. Promote them only after their algorithms, atom requirements, units, and fixture tests are fixed. A generic distance cutoff or PyMOL polar-contact display alone must not be described as a verified hydrogen-bond count.

**Scientific hypothesis:** contacts may predict half-life, but more contacts or “better” anchors are not guaranteed to increase persistence. Published experimental work found that anchor improvements did not always improve measured stability. Use this as a falsification caveat, not as a reason to abandon the paired test.

### 6. Close the selection, failure, and test-set loopholes

Preserve the current peptide-component split, train-only transforms, and matched S0 rows; make the execution contract unambiguous:

- Freeze model/head families and hyperparameter selection on training/development data. **Remove wording that retains a branch only if its locked-test result improves.** Test results report the predeclared comparison; they do not select the winning model for that same test.
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

**Target fix:** after rejecting negative/nonfinite labels, use `y = log1p(t_half_hours)`. Remove `max(t_half_hours, 0)` from the target contract because it silently turns bad data into valid zero-hour labels. Presentation clipping of predictions is a separate recorded choice.

**U1 remains optional:** pooled row-level conformal guarantees do not automatically follow when observations share peptide components or undergo assay shift. Use an appropriate group-aware design before claiming coverage guarantees; otherwise report empirical coverage only. Neither conformal intervals nor ensemble spread automatically detect out-of-distribution inputs.

### 7. Connect evidence artifacts to the pipeline and demo

Replace the existing architecture diagram with:

```text
Paper/source lookup → claim ledger ──────────────────────────────┐
Sponsor CSV → audit/hash → locked peptide-component split        │
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

Extend the planned artifact inventory instead of creating an elaborate new service:

- `data/evidence_manifest.jsonl`: claim ID, DOI/URL, paper/version/type, exact support location, endpoint/assay, license, status (`supported`, `hypothesis`, `unverified`), and retrieval details.
- `data/hla_mapping.jsonl` and `data/reference_manifest.jsonl`: registries described above.
- Existing `data/structures_manifest.jsonl`: add construct/coordinate hashes, chain-map version, reference IDs, QC reason, confidence field/scale, attempts, failure status, feature version, and per-pair runtime/cost.
- Existing result records: add full-cohort and valid-subset identifiers, denominator counts, fallback fraction, reference/exposure flags, and actual model used.
- Keep raw data/coordinates local or in approved storage until redistribution is cleared; never put credentials into artifacts.

**Demo layout:** one real peptide/allele, actual predicted half-life and model ID, measured label only in clearly marked evaluation examples, sequence-vs-structure ablation, cost/failure rate, and an evidence drawer. Mark a displayed object `experimental reference`, `predicted complex`, or `no valid structure`; never imply an unrelated reference is the selected peptide’s pose.

Use explicit peptide polymer-chain selections in PyMOL. The skill’s small-molecule `organic` recipe is unsuitable for selecting a protein peptide chain. Save PNG plus `.pse`, check nonzero atoms, and use static real-output images as a nonblocking fallback if headless rendering or interactive 3D fails.

### 8. Replace expired starts with resource-gated delivery

Retain the current plan’s earlier **Sunday 4 October 2026, 14:45 BST** submission cutoff and timing-conflict warning as sourced event requirements from the prior plan, not freshly reverified event information. Replace the Saturday 13:00 start with dependency gates and status fields.

| Gate | Required artifact / decision | Stop or fallback |
|---|---|---|
| A — reliable baseline | Audited inputs, locked split, B0/B1 training/development results, embedding cache/F0 | No GPU-heavy expansion before the split tests and baseline work |
| B — structural feasibility | Verified pilot constructs, reference triage, three training-only probes, observed runtime/memory/cost, frozen QC policy | If mapping, pose quality, or cost fails, use one already-ready alternative or label structure untested |
| C — locked cohort | S0 IDs/counts, actual available budget, worker cap, bounded retries, all-pair manifests | Stop admitting jobs that cannot fit the budget and result-freeze window; do not choose test rows by label/performance |
| D — scientific freeze | Frozen development-selected models, paired evaluation, uncertainty/counts/failures, optional I1 | No new architecture after evaluation; report positive, negative, inconclusive, or unrun honestly |
| E — submission | Verified local/live demo, recording, attribution, description, repository link | Preserve the existing Sunday demo/submission buffer; submit before the earlier cutoff |

Estimate cohort cost from measured pilot cost **including overhead and retry reserve**, not nominal credit offers. Record an explicit spend cap and stop condition; no new paid jobs are authorized by this revision.

Parallel team: data/evaluation owner; structural-input/model owner; evidence/demo owner. Solo: A first, then B/C; skip optional searches, larger encoders, and U1 before compromising the core comparison. The evidence workflow must support delivery, not consume the weekend.

Update the Google DeepMind sponsor row to name this actual use of Science Skills. Do not imply access to Antigravity, paid model APIs, or sponsor-prize eligibility has been verified. Replace stale authentication-state assertions with “verify current account/billing route before use”; retain no automatic logout/login commands.

## Exact Document Edit Map

| Existing location | Revision |
|---|---|
| Title, executive decision, §0 | Keep Stability Lens and falsifiable half-life question; introduce the evidence-grounded structural spine and explicit scope tiers |
| §§1–2 | Preserve prior audits/source distinctions; add pinned Science Skills matrix, evidence hierarchy, and biological caveats |
| §3 | Add sequence/reference registries, failure-aware evaluation, target-validation fix, split/exposure distinctions |
| §4 | Keep baseline/model menu; make contact MVP precise, add confidence/availability controls, freeze model selection before test |
| §5 | Replace flow and extend planned manifests/interfaces; distinguish reference structures from predicted target structures |
| §6 | Explain concrete Google DeepMind skill use; remove stale live-account assertions without assuming new access |
| §7 | Replace elapsed starts with dependency gates, measured-budget decisions, and the existing submission buffer |
| §§8–9 | Add acceptance tests, explicit model/pose/coverage statuses, and documentation-only completion boundary |
| Sources | Preserve existing citations; append pinned skill and verified new source links, repair numbering/anchors |

Keep the main plan readable: consolidate repeated cautions into contracts/decision tables rather than appending another long, competing plan. Preserve useful existing findings and model alternatives, but keep optional tools visibly off the critical path.

## Files to Modify

- **Modify after approval:** `/Users/arkajyotisaha/Desktop/Hackathon/ARE/HACKATHON_PLAN.md`.
- **Keep unchanged:** `/Users/arkajyotisaha/Desktop/Hackathon/ARE/Peptide-HLA Stability Primer.md`.
- **Planning artifact only:** `/Users/arkajyotisaha/.devin/plans/plan-1e9f464293078ecb.md`.

The listed future manifests, modules, tests, images, and `.devin/skills` locations are specifications to include in the hackathon document, **not files to create in this revision**. No Git initialization, installation, memory write, paid compute, deployment, commit, or push is included.

## Verification

### For the approved documentation revision

- [ ] Review the complete document diff against the edit map; check that the primer fingerprint is unchanged.
- [ ] Preserve audit counts as previously reported, without re-deriving or relabeling them as fresh verification.
- [ ] Trace every added tool capability to the pinned instructions/helper code listed below.
- [ ] Check Markdown heading hierarchy, tables, fences, citation numbering, source links, and internal anchors.
- [ ] Check names and contracts agree across the flow, ladder, file inventory, acceptance tests, and definition of done.
- [ ] Check no unsupported novelty, clinical benefit, pLDDT-to-half-life, static-pose-to-kinetics, or clean-pretraining claim appears.
- [ ] Check no test-driven model/subset/template selection or silent invalid-fold removal remains.
- [ ] Check optional tools, unconfirmed resources, external uploads, data rights, and future commands are labeled clearly.
- [ ] State that no training/test suite exists or ran; do not invoke a nonexistent build.

Read-only document checks can use `shasum -a 256` on both documents and a Python standard-library Markdown/link/fence scan. Since ARE is not a Git root, compare against an in-memory pre-edit snapshot rather than claiming `git diff` is available.

### Acceptance tests to add to the future implementation plan

- `tests/test_chain_mapping.py`: exact 182-aa mapping, mismatched allele, ambiguous extension, C67S quarantine, mature-chain boundaries, β2m identity, label/auth renumbering and insertion codes.
- `tests/test_reference_manifest.py`: wrong peptide, wrong allele, incomplete groove, experimental versus predicted type, assembly selection, and missing-reference fallback.
- `tests/test_interface_features.py`: known synthetic contact distances on both sides of 4.5 Å; unique receptor-residue counting; rigid rotation/translation invariance; no intra-chain/β2m contacts; alternate conformers and missing atoms handled deterministically.
- `tests/test_metrics.py`: identical pair order, paired component resampling, constant/insufficient targets, complete-case versus full-cohort counts, F0 fallback, and no model selection on test outcomes.
- `tests/test_inverse_score.py`: only the peptide is scored, receptor context stays fixed, score normalization is explicit, template provenance is attached.
- Existing split/label/inference tests: exact one-edit global peptide groups, negative-label rejection, zero-hour finiteness, train-only preprocessing, save/load equivalence, and unsupported-input handling.
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
9. **License/output side effects are deferred.** Executing several skills creates output files, caches, or notice receipts. That is outside this Plan-mode/document-only approval.
10. **Operational resources remain unverified.** Actual GPU availability, budget, rendering platform support, mentor decisions, and submission-rule ambiguities stay explicit gates—not assumptions.

## Verified Sources for the Revision

The existing plan’s sources [1]–[42] remain attached to their existing claims. Add the following sources for new workflow claims, using a consistent numbering scheme in the final document:

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
