# The plan — three proposals evaluated, one path chosen

**Written 2026-10-03 16:35 BST. 22h 10m to the 14:45 Sunday cutoff. Nothing has been trained yet.**

Three plans now exist in this repo, written independently. This document evaluates them against what
the brief actually scores, picks the best idea, and states one ordered path. It supersedes nothing —
each source plan remains the authority on its own detail — but where they conflict, the resolution
is here.

---

## 1. What the brief rewards (the scoring function we are optimising)

From the Serova brief, §5–6, verbatim where it matters:

- *"We are **not** looking for an approach that tops an arbitrary leaderboard."*
- *"NetMHCstabpan … is trained on the entirety of this dataset, hence is **unfair to use as a direct comparator**."*
- *"**Negative results are just as good as positive results** where well supported."*
- Judged on: *"how you tackle this question, your approach to the data, chosen evaluation metrics,
  your conclusions, and the evidence you provide to support them"*, with *"ML best practices,
  engineering & compute requirements during the timeframe, and **incorporation of the biological
  background**"*.

Plus the event-wide rubric, 20 points each: technicality, creativity, usefulness, **demo**, track fit.

**Consequence:** the deliverable is an *argument*, and the ablation table is the evidence. A higher
Spearman with a weaker argument scores worse than a lower one with a clean one.

§10 maps every clause of §5 and §6 to where this repo addresses it, including the clauses it does
**not** yet address.

### The hypothesis our split embodies

The brief asks for splits established early *"with a sensible hypothesis"*. Ours, stated plainly:

> **H₀ (primary, `peptide_split`):** a model that has never seen a peptide — nor any peptide within
> two substitutions of it — must predict its half-life from sequence and HLA context alone. This
> measures generalisation to **new fragments**, which is the clinical use case: a neoantigen from a
> patient's tumour has no measured neighbours.
>
> **H₁ (secondary, `allele_split`):** a model that has never seen an *allele* must still predict for
> it. A one-hot allele encoding cannot represent an unseen allele at all, so this is the sharpest
> available test of whether a protein foundation model contributes anything beyond a lookup table.

Both splits are frozen, audited, and committed before any model was trained.

## 2. The three plans

| | **Stability Lens** (`HACKATHON_PLAN.md`, `SCIENCE_SKILLS_PLAN.md`) | **Track 3 Build Plan** (`Track 3 Build Plan — ….pdf`) | **Repo infrastructure** (`splits/`, `scripts/`, `experiments/`) |
|---|---|---|---|
| Core move | Controlled sequence→structure→inverse-folding ladder with verified constructs and an evidence ledger | Threading gives a pose *ensemble*; feature **variance** across it is the kinetic signal | Build and audit the measurement apparatus first |
| Greatest strength | Evaluation rigour. *"Test results report the predeclared comparison; they do not select the winning model"* | One falsifiable original claim, plus hard kill gates at 16:30 / 20:00 / 02:00 / 08:00 | Artifacts that exist and are verified, not proposed |
| Greatest weakness | Its stated deliverable is *"an improved planning document"*; full scope is a week | Rebuilds the split; conformal calibration is methodologically unsound (§5) | No modelling at all |
| Compute realism | Thorough but heavy (β2m constructs, PDB triage, Boltz-2) | Deliberately avoids PyRosetta/MODELLER/APE-Gen; threading not docking | n/a |
| Already done? | Proposal | Proposal | **Done and audited** |

All three independently agree on the things the brief scores: not a leaderboard, negative results
valid, one frozen split, model selection frozen before test, cluster-bootstrap by peptide component.
That agreement is itself worth a sentence in the pitch.

## 3. The best idea of the three — and the risk that could kill it

**Edouard's ensemble-variance hypothesis is the best scientific idea in this repo.**

> *"Structure-based pHLA work has always predicted the wrong quantity. pHLA-RF and 3pHLA-score both
> produce a score for a single static pose, which approximates binding free energy. Half-life is
> kinetic. … We use the variance of the features across that ensemble as a proxy for local
> conformational entropy. Variance is the kinetic signal."*

Why it wins on merit:

1. **It is the only genuinely novel claim among the three.** The sequence ladder reproduces published
   work; the evidence ledger is good practice. This is a hypothesis nobody has tested.
2. **It explains a failure rather than just adding a feature.** It says *why* prior structural work
   underperformed on stability — wrong thermodynamic quantity — which is a scientific argument, not
   an engineering one.
3. **It is falsifiable in a single table row**: mean-only features vs mean-and-std, same head, same
   split. One number either way, and a null is reportable.
4. **It is cheap relative to co-folding.** Threading plus repacking, not docking, and explicitly no
   PyRosetta.
5. It is **track-aligned**: it uses inverse-folding models (ESM-IF / ProteinMPNN) reading geometry,
   which is the brief's third model family and the one nobody else will touch.

### The risk that has to be tested before the overnight run

**Repacking variance may be measuring the solver, not the physics.** Running FASPR three times with
different seeds samples the *stochasticity of the side-chain packing algorithm*. That is not a
Boltzmann ensemble, and its spread is not thermal conformational entropy. If the variance is
algorithmic noise, the headline feature is noise with a physical-sounding name.

This is testable in under an hour, before committing the 28k × 3 overnight job:

- **Reproducibility test.** Take ~200 pairs spanning the half-life range. Generate ensemble A with
  seeds {1,2,3} and ensemble B with seeds {4,5,6}. Correlate the per-pair std from A against B. If
  the correlation is weak, the std is solver noise and the branch should die at this gate rather
  than at 02:00.
- **Signal test.** On those same 200 pairs, check the std correlates with `log1p(t½)` *at all*,
  within allele. A flat result here is not fatal (the head may still find it useful in combination)
  but a *non-reproducible* std is.

The build plan already has a 200-pair featuriser gate. Add these two checks to it. Thirty minutes
buys the difference between "our headline feature is reproducible" and discovering at 02:00 that it
is not.

**Runner-up idea**, and the one to lead with if structure dies: the **held-out-allele axis**. A
protein foundation model should beat a one-hot allele encoding precisely where the allele was never
seen, because one-hot *cannot represent it at all*. That is the sharpest possible test of the brief's
literal question, it needs no structure, and the split already exists.

## 4. What is already built — do not rebuild any of it

| Artifact | State | Why it matters |
|---|---|---|
| `splits/peptide_split.csv` | Frozen, audited | 22,532 / 2,817 / 2,817 rows, clustered at Hamming ≤ 2 |
| `splits/allele_split.csv` | Frozen, stratified | The held-out-allele axis both other plans ask for |
| `scripts/audit_splits.py` | Passing | Brute-force O(n²) check, independent of the generator |
| `scripts/measure_leakage_by_distance.py` | Run | Train-only derivation of the threshold |
| `scripts/investigate_zero_labels.py` | Run | Settles what `t½ = 0` means |
| `experiments/BRAINSTORM.md` | Written | ~40 candidate directions with costs and failure modes |

**The leak, quantified.** A predictor that does nothing but memorise peptide identity scores
Spearman **0.337 on a random row split** (finding its peptide for 93.6% of eval rows) and **0.000 on
ours** (0.0% coverage). That gap is what the split is buying.

## 5. Conflicts between the plans, resolved

| # | Conflict | Resolution | Evidence |
|---|---|---|---|
| 1 | Split threshold: 80% identity (Hamming ≤ 1) vs Hamming ≤ 2 | **Hamming ≤ 2.** Already frozen and audited | Train-only neighbour agreement: 0.612 at d=1, 0.636 at d=2, **0.406 at d=3**, null 0.000. Break is between 2 and 3 |
| 2 | Conformal calibrated on the validation split | **Carve calibration rows out of train.** Never val or test | Val is simultaneously doing model selection — the plan retrains "in under a minute so we can ablate aggressively". Selecting on the rows you calibrate on breaks exchangeability and voids the guarantee |
| 3 | `t½ = 0` is "an assay detection floor" | **It is a rounding floor at 0.05 h.** Reporting scale is 0.1 h (98.4% of non-zero values are exact multiples; exactly one row below 0.1) | So `0.0` means "< 0.05 h", left-censored at a *known* threshold. Two-head classify-then-regress handles it; a Tobit with a known cut is also now available |
| 4 | NetMHCstabpan as "the baseline to beat"; Table 2 as "our scoreboard" | **Reference only, never a target.** It trained on the entire corpus, so our test peptides are in its training data *however* we split | The brief rules it out in its own words |
| 5 | ESM-2 650M vs 35M | **Start at 35M.** Settle it with a size ladder if there is slack — that *is* the brief's cost-benefit question | `experiments/001` names 650M; Stability Lens starts at 35M on compute grounds |
| 6 | Boltz-2 affinity head (`training pipeline.pdf`) | **Cannot be used.** Boltz docs: the affinity binder *"must be a ligand chain (not a protein, DNA or RNA)"* | A 9-mer is a protein chain. `HACKATHON_PLAN.md` already rejects this; the diagram appears superseded |
| 7 | 72 alleles (papers) vs 75 (our file) | Both correct: 75 − 3 engineered `(C67S)` = 72 | Row count reconciles to within 3 rows |

## 6. The path — ordered, with what kills each step

Merged from all three. Times are wall-clock targets, not estimates of effort.

### Floor — must exist tonight, guarantees a submission
Nothing here needs structure, GPUs beyond one small card, or any unresolved decision.

| Step | What | Kill / fallback |
|---|---|---|
| F1 | **Trivial baselines**: global median, per-allele median. 30 min | — |
| F2 | **Supervised sequence baseline** (`experiments/000`): peptide one-hot/BLOSUM + pseudosequence **+ explicit allele identity**, LightGBM and/or small MLP, `log1p` target | — |
| F3 | **The one-hot allele control.** Replace the HLA representation with a one-hot over 75 alleles | If this matches the embeddings, the model is memorising allele identity — a *finding*, and the headline |
| F4 | **Frozen ESM-2 35M embeddings + head** (`experiments/001`). Cache 5,633 peptide and 75 HLA vectors once, plus the 9 per-residue peptide vectors | Expect ≈ 0.57. Above 0.85 on the main split means stop and audit |
| F5 | **Report within-allele Spearman alongside pooled, from the first number onward** | Pooled correlations on this data are partly a between-allele artefact: unrelated pairs score 0.335 pooled and 0.027 within-allele |

### Differentiators — in value order, each independently killable
| Step | What | Kill criterion |
|---|---|---|
| D1 | **Affinity pre-training transfer** (NetMHCpan 4.1 corpus, ~170k rows). Decontaminate it against the stability test clusters | Largest single published jump: 0.574 → 0.745 |
| D2 | **Conformal intervals + coverage**, calibrated on train-carved rows | Cheap, and no published stability model reports uncertainty |
| D3 | **Ensemble-variance structure branch** — but run the reproducibility test in §3 *first* | If std is not reproducible across seed sets, kill at the 200-pair gate, not at 02:00 |
| D4 | **Inverse-folding zero-shot score** (ESM-IF / ProteinMPNN), peptide given pocket | Zero-shot, no training, runs early, uses the brief's third model family |
| D5 | **Held-out-allele evaluation** on `splits/allele_split.csv` | Where a foundation model should beat one-hot. Only 5–6 alleles held out, so report intervals and call it directional |

### Under the hood — the brief asks for these by name
*"These models … produce embeddings, log-likelihoods/perplexities, and both internal and external
confidence metrics. They can be run with different inputs masked and different seeds, fine-tuned or
have prediction heads built on top of their embeddings. We encourage you to explore under the hood!"*

That is a list of six routes. Using only embeddings answers one sixth of it. Each below is cheap and
needs no structure branch:

| Route | What | Cost |
|---|---|---|
| Embeddings → head | F4 above | the standard route |
| **Log-likelihood / perplexity** | Pseudo-log-likelihood of the peptide in HLA context; also *context minus no-context*, which isolates how much the groove explains the peptide. **No training at all**, so no leakage surface and the whole dataset is an evaluation set | ~3h (`experiments/002`) |
| **Internal confidence** | The model's own per-residue confidence as a feature *and* as an uncertainty estimate. Distinct from conformal, which is ours not the model's | ~1h |
| **Masked inputs** | Score with each peptide position masked in turn — doubles as the anchor-rediscovery result below | ~2h |
| **Different seeds** | 5-member deep ensemble. Mean is the prediction, spread is epistemic uncertainty. Also a free accuracy gain | ~1h on cached features |
| Fine-tuning | **Deliberately not done** — and we say so, with the compute reason. "We chose not to, here is the budget" is a legitimate answer to a cost-benefit question | vetoed |

**Compute discipline the brief asks for explicitly** (*"data subsetting or careful choice of which
model(s) to evaluate"*): we choose ESM-2 35M over 650M, cache all 5,633 peptide and 75 HLA
embeddings once rather than re-encoding per row, and subset the structure branch rather than folding
28k pairs. Record measured wall-clock and cost per route — that record is itself a deliverable.

### Free wins — hours, not days, and each is a slide
- **Position occlusion**: mask each of P1–P9, measure the prediction change. Does the model
  rediscover the anchors from sequence alone? Cheap, visual, and it is §6's biology criterion.
  Note the primer gives only P2/PΩ with no exceptions — real alleles include P3-dominant cases, so
  finding the exceptions is the *better* result.
- **Learning curves** at 5/10/25/50/100% of training data. If foundation models help most where
  labels are scarce, that is a precise, transferable answer to the brief's actual question.
- **Per-allele Spearman vs training support.** Exposes whether we are only fitting the data-rich alleles.

### Always
Write up every result including the failures; a clean negative on a leakage-free split is a
legitimate finding and the brief says so explicitly. Build the demo from whatever exists at the time
rather than waiting for the best model.

## 6b. Chosen metrics, and why each one

§6 judges *"chosen evaluation metrics"* — so the choice needs a reason, not a default.

| Metric | Why this one |
|---|---|
| **Spearman ρ — within-allele *and* pooled** | Primary. Assays disagree on absolute hours but rank consistently. **Pooled alone is misleading here**: unrelated pairs score 0.335 pooled against 0.027 within-allele, because alleles differ systematically in stability. Reporting only the pooled number inflates it with a between-allele effect |
| **Pearson r on `log1p`** | Catches the NetMHCstabpan failure mode: it ranks at 0.876 while its Pearson is 0.532, so its absolute hours are badly calibrated. Rank alone would hide that |
| **RMSE / MAE in log space** | Absolute accuracy, in the space the model is trained in |
| **Precision@10 within a supported allele** | The clinical use is re-ranking a shortlist of candidate neoantigens, so top-k matches the application. Within-allele, because a per-patient shortlist is single-allele |
| **Interval coverage at 90%** | No published stability model reports uncertainty. If we claim 90% and observe 89%, that is a calibration result; if we observe 60%, we say so |
| **Per-allele ρ against training support** | Exposes whether we are only fitting the data-rich alleles |
| **Classifier AUROC for the zero head** | The `t½ < 0.05 h` class is 20% of rows; "will this bind at all" is a usable output in its own right |
| **Paired bootstrap over peptide *clusters*** | Resample clusters, not rows, or the interval is too narrow. A CI spanning zero is inconclusive, not equivalence |

## 6c. Biological background, and where it changes the modelling

§6 lists *"incorporation of the biological background"* as a scoring criterion. Each item below
changed a concrete decision rather than decorating the write-up.

- **Stability is kinetic; affinity is thermodynamic.** The brief: affinity depends on both the
  association and dissociation rates, while stability *"is largely a function of the dissociation
  rate"*. → Affinity is excellent *pre-training* (0.574 → 0.745 published) but not a substitute, and
  it is why a static predicted pose may carry no kinetic signal at all.
- **Anchor positions are allele-specific.** Usually P2 and PΩ, but there are P3-dominant alleles.
  → Keep features for all nine positions; do **not** hard-code P2/P9. The primer gives only P2/PΩ
  with no exceptions, so anchor features built from it alone would bake in a simplification across
  all 75 alleles. Finding the exceptions is the better result.
- **The groove is conserved; the trench residues are not.** Two α-helices over a β-sheet floor, with
  34 contact positions. → This is why a 34-residue pseudosequence characterises an allele — and why
  it still fails for two engineered mutants that share one (§5 conflict table).
- **Each person carries a different allele set.** → Generalisation to unseen alleles is the clinically
  meaningful axis, not a curiosity. Hence `allele_split`.
- **A complex that dissociates before reaching the surface is never inspected.** → The `t½ < 0.05 h`
  class is not a nuisance to be dropped; it is the biologically meaningful "never presented" label.
- **The three `(C67S)` constructs are engineered, not natural**, and are 75–92% zeros. → Their
  measurements may reflect the construct rather than the biology; quarantine them from the structural
  arm and report them separately.
- **Why structure might legitimately fail.** A co-folded pose is a single *bound* snapshot, while
  dissociation is an escape process with its own barrier and possibly several pathways. → Stated up
  front as the falsification condition for the structure branch, not as an excuse afterwards.

## 7. Vetoes — agreed across the plans

1. **No PyRosetta, MODELLER or APE-Gen installs.** Each is a plausible four-hour hole.
2. **No fine-tuning a 650M model.** Most expensive, least interpretable, highest chance of consuming
   the night for one number.
3. **No rebuilding the split.** It is frozen and audited; a second split makes every prior ablation
   row incomparable.
4. **Test set is touched once.** Selecting on test is leakage reintroduced by hand.
5. **Any metric above ≈ 0.85 on the main split is leakage, not a result.** Stop and audit.

## 8. Open decisions for a human

1. **The pre-event primer is committed, and `HACKATHON_PLAN.md` says it should not be.** Its own
   words: *"Do not commit or present the pre-event primer … as a newly built submission"* and
   *"because the primer predates kickoff, do not include it or copy its text into the judged artifact
   without organizer approval."* The submission rules say *"Build entirely during the event. No prior
   commits to the repo."* The README currently discloses it as reference material not claimed as
   hackathon work, which is honest — but the safest reading is to remove it from the judged repo and
   keep it locally, or ask an organiser. **This needs deciding before submission, not at 14:00 Sunday.**
2. **Calibration partition**: carve it from train now, or drop the conformal *guarantee* and report
   empirical coverage only?
3. **Who owns what.** The build plan's workstream split (data/splits · sequence model · structure ·
   featurisation · demo) is sensible; it needs names against it.

## 9. Coverage of the brief, clause by clause

Checked against the brief text, not from memory. **Gaps are marked as gaps.**

### §5 — Notes on the challenge

| Clause (brief) | Addressed? | Where |
|---|---|---|
| "not looking for an approach that tops an arbitrary leaderboard" | ✅ | §1; NetMHCstabpan demoted to reference in §5 #4; veto #5 |
| "NetMHCstabpan … unfair to use as a direct comparator" | ✅ | §5 #4 — and noted that it trained on the whole corpus, so our test peptides are in its training data *however* we split |
| "establishing splits early on **with a sensible hypothesis**" | ✅ | §1 "The hypothesis our split embodies"; built, audited and frozen before any training |
| "a **simple supervised neural network** trained on peptide and HLA pairs … to contextualise" | ✅ | F2. **Note:** the brief says *neural network*, so the MLP is required, not optional — LightGBM is an addition, not a substitute |
| "data subsetting or careful choice of which model(s) to evaluate" | ✅ | §6 "Compute discipline"; 35M over 650M; structure branch subset |
| "embeddings" | ✅ | F4 |
| "**log-likelihoods / perplexities**" | ✅ | §6 Under the hood; `experiments/002` |
| "**internal and external confidence metrics**" | ⚠️ *partial* | Internal confidence listed in §6 Under the hood but not yet owned by anyone. External = our conformal (D2) |
| "run with **different inputs masked**" | ✅ | §6 Under the hood; doubles as anchor occlusion |
| "**different seeds**" | ✅ | §6 Under the hood — 5-member deep ensemble |
| "fine-tuned" | ✅ *by decision* | Vetoed with the compute reason stated. A justified "no" answers a cost-benefit question |
| "prediction heads built on top of their embeddings" | ✅ | F4, and the two-head zero formulation |
| "negative results are just as good … **Are they useful**, not **prove they are useful**" | ✅ | §1, §6 "Always", D3 kill criterion, §5 #4 |

### §6 — Evaluation criteria

| Clause | Addressed? | Where |
|---|---|---|
| "how you tackle this question" | ✅ | This document |
| "your approach to the data" | ✅ | Split + audit; zero-label investigation; the four traps |
| "**chosen evaluation metrics**" | ✅ | §6b, with a reason per metric |
| "your conclusions" | ⏳ *pending* | Nothing trained yet |
| "the **evidence** you provide to support them" | ✅ *apparatus ready* | `scripts/` are reproducible and self-asserting; the ablation table is the deliverable |
| "watertight evaluation, ML best practices" | ✅ | Frozen split, test touched once, selection frozen before test, cluster bootstrap, vetoes |
| "**engineering & compute requirements during the timeframe**" | ✅ | §6 compute discipline; kill gates; measured cost recorded per route |
| "**incorporation of the biological background**" | ✅ | §6c — each item tied to a decision it changed |

### Honest gaps

1. **Nothing has been trained.** Every ✅ above is apparatus, not result. This is the only gap that
   matters, and it closes by running F1–F4.
2. **Internal model confidence has no owner.** It is one of the six routes the brief names and is
   roughly an hour of work.
3. **The affinity corpus is not downloaded**, so D1 — the largest published single gain — has not
   started.
4. **No external/independent evaluation set.** Published work shows accuracy collapses across assays.
   We cannot fix that in the time; it belongs in Limitations, stated plainly rather than omitted.

## 10. Where everything lives

| Path | What |
|---|---|
| `README.md` | Project overview and status |
| **`PLAN.md`** | **This file — the merged decision** |
| `RECONCILIATION.md` | Detailed review of each plan, with evidence |
| `HACKATHON_PLAN.md`, `SCIENCE_SKILLS_PLAN.md` | Stability Lens plans |
| `Track 3 Build Plan — ….pdf` | The build plan |
| `Peptide-HLA Stability Primer.md` | Pre-event background — see §8.1 |
| `context/` | The brief, the dataset, working notes, the SPEARMINT summary |
| `splits/` | Frozen splits and what they guarantee |
| `scripts/` | Split generation, independent audit, two data diagnostics |
| `experiments/` | Proposals, template, and the ~40-idea catalogue |
