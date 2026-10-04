# Stability Lens

**Testing whether protein foundation models and predicted peptide–HLA geometry improve measured complex half-life prediction.**

London AI × Science Hackathon · Serova Protein Engineering Track · 3–4 October 2026.

Stability Lens asks a practical scientific question: given a nine-residue peptide and an HLA class I variant, can we predict how long their complex remains intact, and do expensive protein models add useful information beyond a small supervised sequence model? The planned comparison progresses from sequence features to predicted interface geometry and, optionally, inverse-folding compatibility. Every comparison uses measured half-life labels, controlled data partitions, and explicit records of cost and failure.

**Current status:** the sequence ladder is **complete and measured**, from baselines through LoRA fine-tuning, with cluster-bootstrap confidence intervals on every claim. Eight logged runs; every number, and what it does not establish, is in [`results/README.md`](results/README.md). The structural arm and the demo are not built. A well-supported negative or inconclusive result is a valid outcome - and the headline result so far is of that kind.

## Measured results

Validation rows, frozen split, three seeds per rung, seed-ensemble mean reported. The test split has
not been read by anything. `rho` is Spearman; *within-allele* is the mean over alleles with at least
12 rows, which is the clinically relevant question - rank one patient's candidate peptides inside
their own allele.

| Rung | rho pooled (95% CI) | rho within-allele (95% CI) | Protein LM? |
|---|---|---|---|
| B0b per-allele median, **no peptide information** | 0.563 [0.522, 0.598] | undefined | no |
| F150 frozen ESM-2 150M, mean-pooled | 0.593 [0.556, 0.625] | 0.278 [0.222, 0.320] | **yes** |
| B1' one-hot peptide + allele identity | 0.748 [0.719, 0.775] | 0.557 [0.508, 0.585] | no |
| X150 frozen ESM-2 150M, residue cross-attention | 0.754 [0.727, 0.777] | 0.558 [0.508, 0.585] | **yes** |
| L150 **LoRA-adapted** ESM-2 150M, same head | 0.757 [0.732, 0.780] | 0.572 [0.524, 0.599] | **yes** |
| **B1 one-hot peptide + HLA pseudosequence + allele** | **0.780 [0.756, 0.802]** | **0.633 [0.588, 0.654]** | no |

Intervals are 2,000 paired draws resampling the 540 **peptide clusters**, seed 2026. Rows inside a
cluster are within 2 substitutions, so resampling rows would claim precision we have not earned.

Three findings, in the order we would present them:

**1. Pooled Spearman has a floor of 0.563, and it contains no peptide information at all.** B0b
predicts the training median of whichever allele a row belongs to. Its within-allele correlation is
*undefined by construction* - it predicts a constant inside each allele - and it still scores 0.563
pooled. Pooled figures on this endpoint are largely measuring which groove, not which peptide, so we
report within-allele as primary.

**2. The head design matters more than the foundation model.** F150 and X150 share the same frozen
embeddings and differ only in how they read them. Replacing mean-pooling with residue-level
cross-attention is worth **+0.161 pooled and +0.280 within-allele**, against seed spreads of 0.025
and 0.039. Mean-pooling a 9-mer averages away position, and position is where the P2/P9 anchor
residues live.

**3. No ESM-2 configuration beats one-hot, and none of the three obvious fixes changes that.**
X150 and B1' are statistically indistinguishable (pooled difference -0.005, interval
[-0.026, +0.014] - a *bounded* null, ruling out any PLM advantage above ~0.026). B1 beats the
LoRA-adapted model outright: **+0.023 [+0.007, +0.039] pooled, +0.060 [+0.027, +0.094]
within-allele**, both excluding zero. And each knob one would reach for lands inside noise:

| knob tried | result | against |
|---|---|---|
| **fine-tuning** - LoRA r=8 on K/V, 614,400 params | +0.003 pooled | CI [-0.012, +0.018], **spans zero** |
| **depth** - five ESM-2 layers, fairly normalised | 0.108 range | worst seed spread 0.157, **within noise** |
| **scale** - ESM-2 8M to 650M, 80x parameters | **0.007 range** | worst seed spread 0.036, **within noise** |

**4. It is not redundant - it is nearly worthless, which is a number not a verdict.** Residual
correlation between B1 and X150 is +0.751, so the two do not make identical errors. Ensembling
them gives **+0.009 pooled, interval [+0.001, +0.017]** - real and reproducible - and **+0.007
within-allele with an interval spanning zero**. So a frozen ESM-2 contributes about one hundredth
of a Spearman point on top of a one-hot encoding of the same sequences, and nothing measurable on
the metric that matters clinically. A control confirms the method: averaging B1 with B1', which
share almost all their features, adds exactly +0.000.

**5. The data-efficiency defence fails, and runs backwards.** A pretrained encoder is supposed to
win when labels are scarce. Trained on 5 / 10 / 25 / 50 / 100% of the training clusters, one-hot
leads at **every** budget, and the gap is **widest at 25%** (-0.118 pooled, -0.240 within-allele)
and narrowest at 100% (-0.019). Scarcity makes the foundation model more disadvantaged, not less -
plausibly because the interaction head must learn how to *read* the embedding, which carries its
own sample complexity.

**6. Where one-hot is structurally blind, the gap closes.** On a second split holding out whole
**alleles** (64 train / 5 val / 6 test), B1' - which encodes allele identity as a one-hot over
training alleles and therefore cannot represent an unseen one at all - falls furthest, **-0.273**.
B1 and X150 converge to **0.642 vs 0.641** pooled, from a 0.026 gap on the peptide split. That is
the first evidence here that a learned sequence representation buys something one-hot cannot,
though B1 still leads within-allele (0.539 vs 0.380), so it is a direction rather than a reversal.

**7. We also know why the embeddings underperform.** Ablating each peptide position in the trained model gives a sharply peaked
profile - P9 > P2 > P1 > P3, with P4-P8 contributing nothing - which is the textbook
B-pocket/F-pocket anchor picture recovered from data alone, and it holds in every seed
independently. ESM-2's own masked-position likelihood profile over the same peptides is **flat**,
and the two rank-correlate at **-0.03**. Its per-peptide likelihood predicts stability at
**rho +0.002**. The model's notion of which residues matter is orthogonal to the one this endpoint
needs, which is why a supervised head with per-residue access is the only thing that recovers the
signal - and why a mean-pool destroys it.

We do not compare these to published figures on other splits. A number measured on a different
partition, with a different HLA input and a different model, is context, not a scoreboard.

## Contents

- [Repository status and navigation](#repository-status-and-navigation)
- [Biology and prediction target](#biology-and-prediction-target)
- [Dataset and input contracts](#dataset-and-input-contracts)
- [Evaluation and leakage controls](#evaluation-and-leakage-controls)
- [Model ladder](#model-ladder)
- [Structural experiment](#structural-experiment)
- [Science Skills and evidence workflow](#science-skills-and-evidence-workflow)
- [Planned architecture and artifacts](#planned-architecture-and-artifacts)
- [Getting started with the existing repository](#getting-started-with-the-existing-repository)
- [Delivery plan and demo](#delivery-plan-and-demo)
- [Limitations and interpretation](#limitations-and-interpretation)
- [Sources and attribution](#sources-and-attribution)

## Repository status and navigation

| Available now | Purpose |
|---|---|
| [Official Serova brief](context/serova_primer.pdf) | Challenge question, dataset definition, model families, and evaluation expectations |
| [Challenge CSV](context/dataset.csv) | Measured peptide–HLA half-lives and sequence inputs. **The CSV is the only authoritative copy** — the `.xlsx` export was removed because Excel date-parsed 11,907 of 28,166 `thalf_hours` values (42%), e.g. `1.8` became 1 August 2026 |
| [Split guide](splits/README.md) and [saved report](splits/split_report.json) | Frozen assignments, reported counts, and generation-time checks |
| [Split generator](scripts/make_splits.py) and [independent audit](scripts/audit_splits.py) | Existing code for constructing assignments and checking distances, coverage, and representativeness |
| [Distance diagnostic](scripts/measure_leakage_by_distance.py) | Exploratory label-agreement analysis used when revising the grouping threshold; not a half-life model benchmark |
| [Experiment index](experiments/README.md) and [template](experiments/TEMPLATE.md) | Hypotheses, ownership, time boxes, and eventual result write-ups |
| **[Can we answer the question?](archive/superseded-decisions/POWER_ANALYSIS.md)** | **Minimum detectable effect on the frozen test split**, and why the metric choice decides whether the ablation table is interpretable |
| **[Plan of record — GeoStab-FT](docs/GeoStab-FT-build-plan.pdf)** | **The governing build plan.** ESM-2 150M + LoRA, Boltz-2 geometry as a learned attention bias |
| **[Architecture](ARCHITECTURE.md)** | **What we are building** — the pipeline, the two tracks, the caching, the two heads, and the ablation table, with diagrams |
| **[Merged plan and decision](archive/superseded-decisions/PLAN.md)** | **Start here.** The three proposals evaluated against the brief, the best idea identified, conflicts resolved, and one ordered path with kill criteria |
| [Reconciliation notes](archive/superseded-decisions/RECONCILIATION.md) | Per-plan review with the measured evidence behind each resolution |
| [Track 3 Build Plan](archive/plans/Track%203%20Build%20Plan%20%E2%80%94%20Structure-Aware%20pHLA%20Stability%20Prediction.pdf) | Threading-based structure ensemble, variance-as-kinetic-signal hypothesis, kill gates |
| [**Archived source material**](archive/README.md) | The three original proposals and background reading, with what was taken from each and why |
| [Science Skills revision plan](archive/plans/SCIENCE_SKILLS_PLAN.md) | Detailed proposed construct verification, reference retrieval, structural controls, and delivery gates |
| [Earlier hackathon plan](archive/plans/HACKATHON_PLAN.md) | Detailed research rationale and implementation proposals; historical decisions may be superseded by this README and the saved split files |
| [Peptide–HLA background primer](archive/background/Peptide-HLA%20Stability%20Primer.md) | Background written on 2 October 2026, before the event; retained as reference material |
| [Working notes](context/NOTES.md), [SPEARMINT summary](context/SPEARMINT_SUMMARY.md), and [paper PDF](archive/background/spearmint.pdf) | Background and earlier analysis; these are not new experimental results |

The Science Skills document proposes revisions to the earlier hackathon plan; both are planning records rather than execution logs. They retain historical descriptions from before this directory was connected to GitHub. The current README and saved split files describe the current project state. The background primer predates the event and is included as reference material, not as work claimed to have been built during the hackathon.

Earlier experiment proposals have not all been synchronized with the newer plan. In particular, experiment 001 names ESM-2 650M, whereas the proposed delivery baseline below starts with ESM-2 35M. The current **80/10/10 train/validation/test split with Hamming distance ≤2 grouping remains authoritative**; older one-edit grouping descriptions and a planning example involving a fourth calibration partition do not override the saved assignments. Optional experiments must record their final checkpoint and protocol before running. Treat claims in working notes about zero-label censoring, novelty, or expected scores as hypotheses unless supported by source evidence.

## Biology and prediction target

### Peptides, HLA, and the binding groove

Cells present fragments of proteins on their surface for inspection by T cells. In humans, HLA class I complexes contain an HLA heavy chain, the supporting protein **β2-microglobulin (β2m)**, and a bound peptide. The heavy chain's **α1/α2 domains** form the peptide-binding groove. This dataset contains only peptides of length nine, commonly called **9-mers**.

An **allele** is a particular genetic variant. In `HLA-A*02:01`, `HLA-A` identifies the gene and the fields identify an allele group and protein variant. Different HLA sequences change the chemistry and shape of the groove, so the same peptide can behave differently with different alleles. An engineered suffix such as `C67S` records a cysteine-to-serine substitution and must be preserved in the input identity.

Peptide positions are numbered **P1–P9**. Side chains at certain positions fit into pockets and act as **anchors**. P2 and the final residue are common examples, but anchor preferences vary by allele, including P3-dominant cases. The structural experiment therefore retains features for all nine positions.

### Stability is a kinetic property

The target is the measured **dissociation half-life**, `thalf_hours`: the time required for half of an initially assembled population of complexes to dissociate under the assay conditions.

For an ideal single-exponential dissociation process without rebinding:

```text
fraction remaining at time t = exp(-k_off × t)
t_half = ln(2) / k_off
```

If time is measured in hours, `k_off` has units of inverse hours. These equations explain the concept; they do not establish that every assay follows a single kinetic mechanism.

**Binding affinity** describes equilibrium binding strength. In a simple two-state binding model, `K_d = k_off / k_on`. Two interactions can have the same affinity but different off-rates and half-lives. A model of affinity, a plausible 3D structure, or a high sequence likelihood is therefore not automatically a model of stability.

Stability may help prioritize laboratory experiments, but it is only one factor in antigen presentation and immune response. This project does not establish T-cell recognition, immunogenicity, vaccine efficacy, or clinical benefit.

### Why sequence, structure, and inverse folding are different tests

| Representation | What it supplies | What still has to be learned or tested |
|---|---|---|
| Positional sequence features | Which amino acids occur at which peptide/HLA positions | Their relationship to measured half-life |
| Protein language-model embeddings | Numerical representations learned from protein sequences | Whether frozen representations improve a supervised half-life predictor |
| Predicted complex coordinates | A proposed bound pose and its geometric contacts | Whether geometry adds information beyond sequence and predictor confidence |
| Inverse-folding score | Compatibility of a sequence with a specified backbone | Whether that compatibility predicts kinetics rather than merely the modeled pose |

A static bound structure does not describe every unbinding pathway or its energy barriers. Structural features are a falsifiable hypothesis about half-life, not a physical guarantee.

## Dataset and input contracts

The [saved split report](splits/split_report.json) records **28,166 measurements, 5,633 distinct peptides, and 75 HLA alleles**. The [existing dataset notes](context/NOTES.md) report 36 HLA-A and 39 HLA-B alleles, with no HLA-C examples. These are existing dataset/report findings, not newly computed model results.

| CSV column | Meaning | Modeling consequence |
|---|---|---|
| `allele` | HLA identifier, including engineered-variant suffixes | Preserve exact identity and pair it with the peptide |
| `peptide` | Nine amino acids in one-letter notation | Validate length and alphabet; group related peptides across alleles |
| `thalf_hours` | Measured half-life in hours | Supervised regression target, not an affinity label |
| `hla_seq` | 182-residue α1/α2 groove domain | Suitable supplied sequence input; **not a full heavy chain** |
| `hla_pseudoseq` | 34 selected peptide-contacting positions | Compact representation; not a contiguous protein construct for folding |

A **pseudosequence** collects selected positions from the HLA sequence. It is useful for sequence models but does not uniquely distinguish every construct here: the existing notes report 74 distinct pseudosequences for 75 alleles, with a collision between the two engineered HLA-B*14 variants. The baseline must retain allele identity or the supplied domain sequence rather than merge their labels.

The earlier audit reports **5,679 zero-hour labels**. Their assay meaning is not established by the CSV alone; do not assume a known detection limit or censoring threshold. The planned target is:

```text
y = log1p(thalf_hours)
predicted_hours = expm1(predicted_y)
```

`log1p` accommodates zero and reduces the influence of large hour values. Reject negative and nonfinite labels before transforming; silently clipping invalid labels to zero would fabricate valid observations. If predictions are clipped for display, record that separately from evaluation.

Before model training, the planned validation gate also checks sequence alphabets and lengths, missing fields, duplicate `(allele, peptide)` observations, and conflicting allele-to-sequence mappings. Record input hashes, source, units, and any exclusions. These model-input checks are not all implemented by the existing split script.

## Evaluation and leakage controls

### The frozen primary split

A peptide can appear in multiple rows because it was measured against several alleles. Randomly splitting rows can put the same peptide in both training and test data. The current split instead groups unique peptides globally, across all alleles, before assigning rows.

For two aligned 9-mers, **Hamming distance** counts differing positions. At most one substitution gives at least `8/9 ≈ 88.9%` identity; two give `7/9 ≈ 77.8%`. An 80% identity convention at this fixed length would therefore group identical and one-substitution neighbors. **The current repository deliberately groups peptides at Hamming distance ≤2**, keeping more similar peptides together than that convention. Connected components of the neighbor graph move together, including transitively linked peptides; this is a Hamming rule, not general alignment identity for variable-length sequences.

The [split guide](splits/README.md) records the threshold change in commit `36ae9d4`, motivated by exploratory label-agreement diagnostics. Those diagnostics inspect evaluation labels, so disclose that the split design was informed by exploratory label analysis; it is not a benchmark designed without viewing any evaluation labels. From this point, freeze the assignments and model-selection protocol. The diagnostics do not prove that all remaining biological similarity or data exposure has been eliminated.

The stored assignments use seed **42** and **5,410 peptide components**. The largest component contains eight distinct peptides. Counts below come directly from the saved report at split revision `36ae9d4`:

| Partition | Measurement rows | Distinct peptides | HLA alleles |
|---|---:|---:|---:|
| Train | 22,532 | 4,514 | 75 |
| Validation (`val`) | 2,817 | 563 | 73 |
| Test | 2,817 | 556 | 75 |

The report records a passing planted-neighbor detector check and zero validation/test neighbors of training peptides at Hamming distance ≤2. The generator's detector compares evaluation peptides against training peptides. The separate audit also computes validation-to-test distances, but its current neighbor pass/fail checks reject only distances ≤1; an audit `PASS` alone does not enforce the full new ≤2 contract. The split guide reports a closest cross-partition distance of three. These are recorded findings and implementation boundaries, not a fresh audit run for this README or proof of clean model pretraining.

Use these saved files for every primary experiment. **Do not regenerate splits to obtain a better score.** Freeze feature definitions, model selection, hyperparameters, and stopping rules using training/validation data before the final test evaluation. Fit scalers, imputers, and learned feature selection on training data only.

### The secondary allele split asks a different question

[Allele assignments](splits/allele_split.csv) hold out whole alleles: the report records 64/5/6 train/validation/test alleles and 22,039/3,068/3,059 measurement rows. This tests generalization to unseen HLA variants. It does not also hold out peptides, so it is not evidence of simultaneous generalization to unseen peptides and unseen alleles.

The generator stratifies this secondary assignment using allele-level median-label bins. Document that design; it is not a prospective external assay validation. Never mix its metrics with the primary peptide-split comparison. Any allele-identity feature needs an explicit unknown-allele policy for this analysis.

### Which metrics answer the question?

| Planned measure | Interpretation |
|---|---|
| Held-out Spearman correlation, `ρ` | Whether higher predicted half-lives correspond to higher measured half-lives; rank agreement does not establish calibration in hours |
| Paired change in Spearman, `Δρ` | Incremental benefit on the exact same test pairs |
| MAE and RMSE on `log1p(hours)` | Prediction error on the declared transformed scale |
| Supported within-allele results | Whether pooled performance hides between-allele effects or weak subgroups |
| Validity, fallback rate, runtime, memory, and cost | Whether the added feature source is usable across the selected inputs |

Uncertainty for paired metric differences should resample **peptide components**, retaining all their rows and the paired predictions from both models. Row-wise bootstrap treats related observations as independent and can understate uncertainty. This component bootstrap measures conditional test-sampling uncertainty; it does not include retraining variability. Report seed variation separately if models are retrained.

Undefined correlations for constant or insufficient data remain undefined. A wide interval spanning no improvement is inconclusive, not proof of equivalence. A small structural test cohort must be described as a pilot. If concordance correlation (CCC) is reported, state whether it was computed on hours or log-hours.

### Three different kinds of exposure

1. **Supervised-label exposure:** released NetMHCstabpan and stability-trained MINT/SPEARMINT checkpoints have source-corpus exposure and are not independently held-out baselines on this challenge sheet. Any reproduction belongs in a separately labeled literature comparison.
2. **Peptide overlap:** the current frozen component split limits exact, one-substitution, and two-substitution peptide overlap between our own partitions.
3. **Structure/template exposure:** a structure predictor may have seen a related PDB complex. Record public matches, dates, and known training cutoffs; absence of a match cannot certify an undisclosed training corpus as clean.

SPEARMINT's curated dataset, full-chain inputs, split, and published performance differ from this repository's experiment. Its reported scores are context, not our baseline or expected result.

## Model ladder

The delivery floor is a completed comparison of **B0/B1/F0 on the full dataset**. The main extension is a bounded, paired structure experiment.

> **Note.** The table below is the *original* plan and uses the earlier naming (F0 = ESM-2 35M). What was actually built is the ladder in [ARCHITECTURE.md](ARCHITECTURE.md) section 3 - B0/B1/B1' then F150/X150 on ESM-2 **150M** - with results in [results/README.md](results/README.md). The rows below that are still unbuilt are S0/S1/I1/U1.

| ID | Inputs and approach | Question answered |
|---|---|---|
| **B0 — sanity baseline** | Training-only median log-half-life by allele, with a global fallback | Can a learned model outperform a simple allele-level prior? |
| **B1 — supervised sequence control** | Regularized small model over positional peptide features plus HLA pseudosequence and allele identity; compare a supplied-domain encoding variant using training/validation only | What can the available sequence data support without a pretrained protein model? |
| **F0 — frozen sequence encoder** | `facebook/esm2_t12_35M_UR50D`; separately encode peptide and 182-aa HLA domain, pool residue embeddings, concatenate, and fit a small regularized regression head | Do general protein representations add useful signal over B1? |
| **F1 — optional interaction features** | Same F0 embeddings with controlled pair interactions and peptide-position features | Do explicit interactions help without changing encoder or evaluation data? |
| **S0 — cohort, not a model** | Predeclared subset of the existing train/val/test assignments with verified structural inputs and an affordable cost | Which identical pairs will support the structural comparison? |
| **S1 — structural features** | Boltz-2 complex coordinates, pose checks, separate confidence fields, and compact peptide–HLA contact features | Does geometry add predictive signal beyond sequence, availability, and confidence? |
| **I1 — optional inverse folding** | ProteinMPNN peptide-only compatibility scoring on the same backbones, with a fixed-template control | Does sequence–backbone compatibility add information beyond S1? |
| **U1 — optional uncertainty** | Separately designed group-aware calibration and empirical coverage analysis | How informative are prediction intervals under the tested conditions? |

**Frozen** means ESM-2 weights are not updated using half-life labels; the regression head is still supervised. Cache unique sequence embeddings because F0 encodes chains independently. Pool amino-acid token representations rather than padding or special tokens. A future joint-context encoder would require a different cache contract.

Larger encoders, fine-tuning, SaProt, or alternate complex predictors are follow-ups, not simultaneous MVP requirements. Chai-1 or Protenix is a fallback only if it is already feasible within the measured budget. An AlphaFold Database monomer does not supply a custom bound-peptide pose. Boltz-2's small-molecule affinity output is not a peptide half-life prediction.

[Experiment 002](experiments/002-likelihood-scoring.md) proposes language-model likelihood diagnostics. These remain optional: concatenating sequences does not establish biological conditioning, masking/seed spread is not automatically calibrated uncertainty, and choosing scoring variants still requires validation discipline even without fitting a regression head. The entire dataset is not an untouched test set once labels guide such decisions.

U1 is outside the delivery floor. The current split has no dedicated calibration partition. Any future calibration design must preserve the frozen test set, keep peptide groups together, and separate calibration from model selection. Do not claim conformal coverage under group dependence or assay shift merely because a standard implementation returns intervals.

## Structural experiment

### 1. Verify the molecule before modeling it

An allele name alone is insufficient to identify a valid structural input. Resolve it against a pinned **IPD-IMGT/HLA** release and require an exact match across the supplied 182-residue domain. Record numbering, sequence hashes, reference accession, and construct boundaries. Multiple compatible extensions must remain marked as ambiguous.

The proposed primary construct contains the **mature extracellular HLA heavy chain, mature β2m, and the peptide**. Signal peptides and transmembrane regions must not be fed in by default. A domain-only fallback is a distinct protocol and cannot silently replace that construct within a comparison.

The earlier planning audit flagged domain discrepancies for `HLA-B*08:03`, `HLA-A*02:50`, and `HLA-A*24:19` against its inspected consensus reference. Treat these as prior mapping warnings to reconcile against the official sequence source, not proof that the sponsor data is incorrect. Those cases and unverified C67S constructs stay out of the structural arm until resolved; valid measured rows remain eligible for sequence experiments.

Mapping records should distinguish `verified`, `ambiguous_extension`, `domain_mismatch`, `engineered_unverified`, and `unmapped`.

### 2. Find experimental references and define pose checks

Use verified sequences to discover candidate PDB structures, then inspect the biological assembly, peptide identity and length, HLA construct, β2m, experimental quality, modifications, and missing atoms. Preserve author and label chain/residue identifiers; the peptide is not necessarily chain C.

Experimental references have three separate roles: checking groove geometry, providing a fixed backbone for the inverse-folding control, and documenting possible structural exposure. A crystal complex does not become a half-life test example without compatible measured stability and assay metadata.

Superpose on mapped HLA groove residues for visualization. Call peptide root-mean-square deviation (RMSD) a pose-error measurement only when the reference matches the peptide and relevant construct; otherwise it is a reference overlay. Quality-control rules must be frozen from reference inspection and training-only probes before test inference.

### 3. Measure a compact, interpretable feature

For each peptide position P1–P9, the proposed MVP feature is the **fraction of mapped HLA-domain residues with at least one peptide/HLA heavy-atom distance ≤4.5 Å**:

```text
contact_fraction(position i) = contacted mapped HLA residues / mapped HLA residues
```

The 4.5 Å cutoff is a prespecified engineering definition, not a universal law of binding. Count unique receptor residues, not atom pairs. Restrict contacts to the peptide and mapped 182-residue HLA domain; β2m and within-chain contacts do not belong in this vector. Record raw counts, the denominator, missing-coordinate masks, and a deterministic alternate-conformer policy.

Keep three channels separate: **pose validity**, **predictor confidence**, and **contact geometry**. Read confidence according to the predictor's documented field and scale. Where provided, pLDDT estimates local structural confidence rather than kinetic stability; experimental B-factors describe atomic displacement/disorder and are not pLDDT. Hydrogen bonds, salt bridges, buried surface area, and pocket depth are stretch features requiring explicit algorithms and validation. A generic distance contact is not a verified hydrogen bond.

### 4. Compare identical cohorts and retain failures

Run three training-only complexes first to measure feasibility, memory, runtime, and cost. Freeze S0 membership within the existing partitions before cohort inference, including seeds, bounded attempts, selection rules, and a spend cap. Do not replace failed test complexes with easier examples.

The planned controlled ladder on S0 is:

| Comparison | Purpose |
|---|---|
| Matched-size B1 and F0 | Separate structural benefit from a smaller training cohort |
| F0 + availability/QC | Detect information in which inputs produce usable structures |
| F0 + availability/QC + confidence | Establish the predictor-confidence control |
| F0 + availability/QC + confidence + contact geometry | Primary test of incremental geometric information |
| Structure-only head | Secondary diagnostic of structural features |
| Above + peptide-only ProteinMPNN score | Optional inverse-folding increment with a matched no-I1 control |

Train the availability/QC diagnostic on all S0 training rows. Train the confidence and confidence-plus-geometry heads on exactly the same valid-structure training rows; their paired difference is the primary geometry contrast. Both use the same predeclared matched F0 fallback for invalid test structures. Also report the comparison against F0 itself.

The **primary operational result** includes every preselected test pair, including those receiving fallback predictions. A **secondary valid-structure analysis** uses identical valid test pairs and includes an F0 control trained on the same eligible training rows. Report selected, attempted, valid, fallback, and inverse-score counts with costs. A gain after discarding failed structures is not an all-input performance gain.

### 5. Control the inverse-folding circularity

ProteinMPNN scores sequence compatibility conditional on a backbone. Score only the peptide, keep receptor context fixed, and normalize by the number of scored peptide residues. A structure generated using that peptide can make the peptide appear compatible partly by construction.

Repeat scoring on an allele-compatible fixed template selected without the evaluated peptide's label, excluding evaluated test-component peptides from template selection where possible. This sensitivity control does not create a new all-atom prediction or eliminate every exposure concern. If I1 has fewer eligible training rows, refit its no-I1 comparator on those same rows and use the already-frozen no-I1 pipeline when a score is unavailable.

## Science Skills and evidence workflow

[Google DeepMind Science Skills](https://github.com/google-deepmind/science-skills/tree/68832757cbbf941c620b71df5756cf6e5cc287b0) supplies proposed research workflows for evidence retrieval and inspection. It does not provide a ready-made half-life predictor. The revision plan inspected revision `68832757cbbf941c620b71df5756cf6e5cc287b0`; no skill bundle is installed by this README.

| Workflow | Planned deliverable | Boundary |
|---|---|---|
| Europe PMC literature search | Claim ledger with source passage, DOI, endpoint, assay, and limitation | The inspected wrapper searches open-access material; missing hits do not prove novelty |
| bioRxiv DOI lookup | SPEARMINT metadata and cited version/date | Inspect returned versions; the helper does not guarantee the latest record |
| UniProt | Processing boundaries, annotations, β2m identity, and cross-references | Generic protein annotations do not resolve a specific HLA allele |
| Official IPD-IMGT/HLA sequence retrieval | Pinned allele/construct registry | Preserve sponsor sequence identity and unresolved mismatches |
| PDB search and retrieval | Experimental reference and exposure manifest | A sequence hit still needs assembly, peptide, and construct checks |
| PyMOL | Groove overlays, contact views, PNGs, and editable sessions | Select peptide polymer chains explicitly; a visual contact is not kinetic validation |
| Optional AlphaFold Database / Foldseek | Monomer context or candidate structural relatives | AFDB is not custom co-folding; Foldseek needs coordinates and uploads them |
| Exception-only MSA / sequence search | Resolve difficult heavy-chain mappings | Does not replace the exact 9-mer neighbor check |

Record input/output hashes, retrieval date, source revision, tool version, and command for each executed workflow. Bound database queries and retries. Science Skills software, teaching materials, and upstream databases have separate terms; preserve applicable notices and review sequence redistribution conditions. The plan favors retrieval instructions and hashes over an unreviewed modified HLA sequence mirror.

## Planned architecture and artifacts

```text
Sources and literature ------------------------------> claim ledger
Challenge CSV + frozen split
    |
    +--> input validation --> B0 / B1
    |
    +--> cached frozen ESM-2 embeddings --> F0
    |
    +--> verified construct registry --> locked S0 cohort
              |
              +--> PDB references --> chain / pose / exposure checks
              |
              +--> Boltz-2 --> validity + confidence + contact features
                                   |
                                   +--> optional ProteinMPNN peptide scores
                                                   |
                         paired regression heads + explicit fallbacks
                                                   |
                         held-out metrics + uncertainty + cost / failures
                                                   |
                         planned demo: prediction + evidence / pose view
```

The following artifacts are specifications, not existing files:

| Planned artifact | Contents |
|---|---|
| `data/evidence_manifest.jsonl` | Claim, supporting source/passage, endpoint, assay, license, and evidence status |
| `data/hla_mapping.jsonl` | Sponsor/reference hashes, domain mapping, construct boundaries, β2m, mutations, and mapping status |
| `data/reference_manifest.jsonl` | PDB/assembly/chain/residue maps, experimental metadata, coordinate hash, role, and exposure status |
| `data/structures_manifest.jsonl` | Pair IDs, exact constructs, model version, attempts, confidence scale, QC/failure reasons, coordinate hash, and cost |
| `models/selection.md` | Checkpoints, licenses, input contracts, feasibility probes, and decisions |
| Prediction and result records | Frozen split/cohort identity, target scale, predictions, `model_used`, fallback reason, paired metrics, and support counts |

Future modules cover validation, embeddings, structural inputs, folding, pose checks, contact features, inverse scoring, training, evaluation, and a Streamlit interface. Future acceptance tests must exercise mapping mismatches, synthetic contact fixtures, split invariants, invalid labels, paired metric alignment, fallback behavior, and save/load inference consistency. No `src/` pipeline, packaged training command, or model test suite is supplied yet.

## Getting started with the existing repository

Clone and inspect the shared materials:

```bash
git clone https://github.com/aswin-giridhar/london-ai-science-hackathon.git
cd london-ai-science-hackathon
```

For the data-loading example, a local Python environment with pandas is sufficient:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pandas
```

This is minimal inspection setup, not a pinned model-training environment. Load the **saved** primary assignment with a checked many-to-one join:

```python
import pandas as pd

pairs = pd.read_csv("context/dataset.csv")
assignments = pd.read_csv("splits/peptide_split.csv")

assert assignments["peptide"].is_unique
assert assignments["split"].isin(["train", "val", "test"]).all()
assert set(pairs["peptide"]) == set(assignments["peptide"])

pairs = pairs.merge(
    assignments,
    on="peptide",
    how="left",
    validate="many_to_one",
    indicator=True,
)
assert pairs["_merge"].eq("both").all()
pairs = pairs.drop(columns="_merge")

train = pairs.loc[pairs["split"].eq("train")].copy()
val = pairs.loc[pairs["split"].eq("val")].copy()
test = pairs.loc[pairs["split"].eq("test")].copy()
```

Keep `cluster_id` attached for grouped analysis. Loading a partition does not authorize using its labels for model selection. This snippet does not perform the full model-input validation gate.

`scripts/make_splits.py` documents how the existing assignments were generated. Running it writes the split files and report; it is not a required onboarding or validation step. Audit any proposed split change separately rather than overwriting the team's frozen benchmark.

The audit and distance-diagnostic scripts are available for separate split investigations. They read evaluation labels and require NumPy/pandas plus SciPy for the pandas Spearman calculations. They are not part of the minimal data-loading setup above, and rerunning them does not create a new independent test set. Preserve the distinction between diagnostic measurements and final model evaluation.

To contribute an experiment, copy the template, choose an unused number, and update the index with an owner, hypothesis, status, compute budget, and comparison. For example, if 003 is still unused:

```bash
cp experiments/TEMPLATE.md experiments/003-structure-contact-ablation.md
```

Record checkpoint versions, target scale, training eligibility, exact split/cohort IDs, selection rules, failures, and costs in the write-up. Results stay marked `proposed`, `running`, `done`, `abandoned`, or `inconclusive` according to what actually happened. Commit useful negative findings with their evidence.

## Delivery plan and demo

| Gate | Required evidence | Fallback |
|---|---|---|
| **A — reliable sequence baseline** | Validated inputs, frozen split identity, B0/B1 and cached F0, development-selected heads | Resolve this before expanding GPU work |
| **B — structural feasibility** | Verified constructs, reference triage, three training-only probes, observed runtime/memory/cost, frozen QC policy | One already-ready alternative or explicitly untested structure |
| **C — locked structural cohort** | S0 pair IDs/counts, worker limit, spend cap, bounded retries, and manifests | Stop admitting jobs that cannot fit the budget and result-freeze window |
| **D — scientific freeze** | Paired held-out comparisons, support, uncertainty, failures, and optional I1 | Report positive, negative, inconclusive, or unrun outcomes honestly |
| **E — submission** | Exercised local/live demo, recording, citations, repository link, and description | Local real-output demo and recording if hosting fails |

Estimate cost from measured probes including overhead and retry reserve. Sponsor offers do not establish actual access, billing route, or usable budget. No paid compute is launched by following this documentation update.

The planned Streamlit demo shows one supported peptide/allele, a real prediction and model ID, sequence-versus-structure comparisons, coverage/failure rate, and an evidence drawer. Every displayed structure is labeled **experimental reference**, **predicted complex**, or **no valid structure**. Measured labels appear only in clearly identified evaluation examples. Offline fallbacks must use real saved outputs, not fabricated predictions.

The earlier event plan records **Sunday 4 October 2026, 14:45 BST** as the submission cutoff from the detailed rules, with conflicting rounded times in other materials. Confirm against the [organizer resources](https://iterate.inc/london-ai-science?welcome=true&tab=resources), preserve recording/submission time, and prepare both 90-second and two-minute demonstrations. This README does not claim the schedule or sponsor entitlements were freshly reverified.

Completion means a reproducible same-split result with its limitations and costs, plus a working demonstration of actual outputs. It does not require every optional model to succeed.

## Limitations and interpretation

- **No measured improvement yet.** A plan, pretrained checkpoint, or rendered complex is not an experiment result.
- **Restricted input domain.** The current dataset contains 9-mers from HLA-A/B; other lengths, HLA-C, and unseen assay conditions require separate validation.
- **Assay dependence.** Half-lives measured by different assays need not be interchangeable; good performance here does not establish external-assay transfer.
- **Partial molecular information.** The supplied domain is not a complete mature structural construct. Reference ambiguity and engineered variants need explicit resolution.
- **Selection and availability.** A small structurally feasible cohort can differ from the full dataset. Report conditional and all-input results with matched controls.
- **Uncertain physical relationship.** More contacts, better anchors, higher confidence, or higher inverse-folding likelihood need not imply longer half-life.
- **Incomplete exposure knowledge.** Public searches cannot prove the contents of a model's pretraining data.
- **Data and code terms.** Existing public files retain their source terms; availability in this repository is not a blanket redistribution license. No project-wide license is currently supplied.

## Sources and attribution

The challenge dataset originates from Rasmussen and colleagues. Following the track brief, work using it should cite:

> Rasmussen M, Fenoy E, Harndahl M, Kristensen AB, Nielsen IK, Nielsen M, Buus S. **Pan-Specific Prediction of Peptide–MHC Class I Complex Stability, a Correlate of T Cell Immunogenicity.** *The Journal of Immunology.* 2016;197(4):1517–1524. [doi:10.4049/jimmunol.1600582](https://doi.org/10.4049/jimmunol.1600582).

The pseudosequence convention follows the NetMHCpan family:

> Reynisson B, Alvarez B, Paul S, Peters B, Nielsen M. **NetMHCpan-4.1 and NetMHCIIpan-4.0.** *Nucleic Acids Research.* 2020;48(W1):W449–W454. [doi:10.1093/nar/gkaa379](https://doi.org/10.1093/nar/gkaa379).

Further sources and planned tools:

- [Serova challenge brief](context/serova_primer.pdf): authoritative local challenge description.
- [SPEARMINT preprint](https://doi.org/10.64898/2026.06.28.735023) and [released repository](https://github.com/pirl-unc/spearmint): related stability research, with different curation, inputs, and evaluation. The cited version is a preprint, not peer-reviewed evidence of this project's performance.
- [ESM-2 35M checkpoint](https://huggingface.co/facebook/esm2_t12_35M_UR50D): proposed frozen sequence encoder.
- [Boltz prediction documentation](https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md) and [ProteinMPNN](https://github.com/dauparas/ProteinMPNN): proposed complex coordinates and conditional sequence scoring.
- [Pinned Science Skills source](https://github.com/google-deepmind/science-skills/tree/68832757cbbf941c620b71df5756cf6e5cc287b0) and [skill-specific terms](https://github.com/google-deepmind/science-skills/blob/68832757cbbf941c620b71df5756cf6e5cc287b0/SKILL_LICENSES.md): proposed evidence workflows.
- [IPD-IMGT/HLA](https://www.ebi.ac.uk/ipd/imgt/hla), [official sequence repository](https://github.com/ANHIG/IMGTHLA), and [data terms](https://github.com/ANHIG/IMGTHLA/blob/Latest/LICENCE.md): allele nomenclature and sequence provenance. The earlier plan inspected release 3.65.0; record the actual pinned release when retrieving inputs.
- [Allele-specific anchor preferences](https://haematologica.org/article/view/5692) and [experimental limits of anchor optimization](https://pmc.ncbi.nlm.nih.gov/articles/PMC3032881): biological reasons to test rather than assume contact–stability relationships.

See the [revision plan](archive/plans/SCIENCE_SKILLS_PLAN.md) for the detailed evidence boundaries and proposed acceptance tests. Attribute every dataset, pretrained model, structural reference, and external workflow actually used in future experiments.
