# Results log

Every run that produced a number, what it showed, and what it does **not** establish. Append to this
rather than overwriting — superseded numbers stay, with the reason they were superseded.

Regenerate any row with the command given. The test split has **not been read by anything yet**.

---

## Run 1 — baselines, single seed · 2026-10-03 21:36

`python src/baselines.py` with `SEEDS = (42,)`. **Superseded by Run 2** — kept because the
difference between them is itself a finding.

| Rung | ρ pooled | ρ within-allele |
|---|---|---|
| B1 peptide + pseudoseq + allele | 0.751 | 0.580 |
| B1′ peptide + allele only | 0.688 | 0.465 |
| **delta (B1 − B1′)** | **+0.063** | **+0.115** |

## Run 2 — baselines, three seeds · 2026-10-03 21:42

`python src/baselines.py` (seeds 42, 43, 44; reported model is the seed-ensemble mean).

| Rung | n | ρ pooled | ρ within-allele | r (log) | RMSE log |
|---|---|---|---|---|---|
| B0a global median | 2,817 | **undefined** | undefined | undefined | 1.143 |
| B0b per-allele median | 2,817 | **0.563** | **undefined** | 0.499 | 0.977 |
| B1 peptide + pseudoseq + allele | 2,817 | **0.780** | **0.633** | 0.791 | 0.666 |
| B1′ peptide + allele only | 2,817 | 0.748 | 0.557 | 0.758 | 0.712 |

Per-seed pooled ρ — B1: 0.751 / 0.760 / 0.755 · B1′: 0.688 / 0.692 / 0.702.

### What these runs establish

**1. Pooled Spearman must be read against 0.563, not against zero.**
B0b predicts the training median of whichever allele a row belongs to. It contains **no peptide
information whatsoever**, and its within-allele correlation is *undefined by construction* because it
predicts a constant inside each allele. It still scores **0.563 pooled**.

That is the between-allele confound made concrete. B1's 0.780 is a gain of **+0.217 over knowing
nothing about the peptide**, which is the honest framing. Any rung reported later without this
context overstates itself.

**2. The HLA sequence adds over the allele label — but half as much as one seed suggested.**

| | Run 1 (1 seed) | Run 2 (3 seeds) | seed spread |
|---|---|---|---|
| delta pooled | +0.063 | **+0.032** | 0.014 |
| delta within-allele | +0.115 | **+0.076** | 0.010 |

Both Run 2 deltas exceed the seed spread, so the effect is real. But **the single-seed estimate was
roughly double the three-seed one.** One run gives a value; several tell you whether it means
anything. Quote Run 2.

**3. Seed ensembling is a free gain.** Individual B1 seeds score 0.751–0.760 pooled; their mean
scores **0.780**. About +0.025 for no extra training, and it is why the ensemble is the reported model.

**4. Early stopping is unstable for B1′.** Best epoch varied 8 / 25 / 15 across seeds, against a
tight 8 / 9 / 10 for B1. The weaker feature set has a flatter validation curve, so its epoch choice
is noisier — a reason to prefer the ensemble over any single run.

### What these runs do NOT establish

- **Nothing about test performance.** These are validation numbers and validation drove early
  stopping, so they are mildly optimistic.
- **No confidence intervals.** Seed spread is not a CI. Use `paired_bootstrap()` from
  `scripts/power_analysis.py` on saved predictions for that; it resamples peptide clusters.
- **No mechanism for finding 2.** The pseudosequence is *constant within an allele*, so it cannot
  directly rank peptides inside one. The plausible explanation is parameter sharing — the model can
  learn a peptide-residue/groove-residue interaction and transfer it across similar alleles — but
  that is a **hypothesis**, untested. A direct check would compare per-allele gains against allele
  similarity.
- **Nothing about foundation models.** No ESM-2 rung has been run.

### Protocol actually used

- Trained on the 22,532 training rows; epoch selected on the 2,817 validation rows.
- sklearn's own `early_stopping` carves a *random* slice out of training and would ignore the
  peptide-cluster structure, so the epoch loop is stepped manually against the frozen split.
- Target `log1p(thalf_hours)`; features are fixed one-hot encodings with nothing fitted.
- Allele vocabulary derived from **training rows only**; an unseen allele encodes as all-zero.
  (0 alleles appear in val but not train, so this path was not exercised.)
- MLP 256→64, ReLU, alpha 1e-4, lr 1e-3, batch 256, max 60 epochs, patience 8.

## Run 3 — frozen ESM-2 150M embedding cache · 2026-10-03 22:15

`python src/embed.py`. Not a model result — the input every ESM rung reads. Logged because two
of its properties constrain what the later numbers can mean.

| | |
|---|---|
| Model | `facebook/esm2_t30_150M_UR50D`, 148M params, hidden 640 |
| Encodes | **5,708** (5,633 distinct peptides + 75 distinct HLA domains) instead of 28,166 — 4.9x fewer |
| Output | `peptide_residue (5633, 9, 640)`, `hla_residue (75, 182, 640)`, float16 |
| Cost | **188 s on CPU**, 4 threads. 83 MB on disk |

### What this run established

**1. The cache is only valid while the encoder is frozen and the chains are encoded separately.**
Recorded as `VALID_ONLY_IF` inside `cache/esm2_150m.manifest.json`. The moment LoRA trains (L150) or
the two chains attend to each other, an HLA embedding starts depending on which peptide it is paired
with, and every number computed from this file becomes wrong. L150 must re-encode.

**2. `pooler.dense` is randomly initialised.** `EsmModel.from_pretrained` reports it MISSING from the
checkpoint and newly initialises it. `embed.py` uses `last_hidden_state` and never touches
`pooler_output`, so this is harmless **here** — but anything that reaches for the pooler would be
mean-pooling an untrained projection. Noted so a later rung does not pick it up by reflex.

### Two defects this run exposed, both fixed

| | |
|---|---|
| **Unreadable cache, exit code 0** | `hla.allele.to_numpy()` returns dtype `object`; `np.load(allow_pickle=False)` refuses object arrays. The first encode wrote an 83 MB file that `esm_heads.py` could not open, and the script exited **0**. Fixed by forcing `dtype="U"`, and `embed.py` now reads its own output back with `allow_pickle=False` and asserts shapes and keys before claiming success. An exit code is not a check |
| **10.5 GB of duplicated tensors** | `esm_heads.py` originally gathered a per-row `(182, 640)` HLA tensor — 10.5 GB for the training split. It now holds the unique-sequence banks (**179 MB measured**) and indexes them per batch. Same 4.9x redundancy `embed.py` exploits, one layer down |

### Environment note

Anaconda's MKL and pip's torch each load `libiomp5md.dll`, aborting the process. `esm_heads.py` sets
`KMP_DUPLICATE_LIB_OK=TRUE` with `OMP_NUM_THREADS=MKL_NUM_THREADS=4` before importing numpy. The
flag's own documentation warns it "may silently produce incorrect results", so it was **checked, not
trusted**: torch matmul vs numpy max abs diff **0.0**, a seeded X150 forward repeats bit-identically,
and 4-thread vs 1-thread differs by **8.2e-08** (float32 reduction order). Recorded so the result is
reproducible and the risk is visible rather than buried in a shell variable.

## Run 4 - F150 / X150, frozen ESM-2 150M, two heads - 2026-10-03 22:53

`python -m modal run modal_app.py` on an **NVIDIA A10**, seeds 42/43/44, reported model is the
seed-ensemble mean. The encoder is frozen throughout; only the heads train.

| Rung | n | rho pooled | rho within-allele | r (log) | RMSE log |
|---|---|---|---|---|---|
| F150 mean-pooled head | 2,817 | **0.593** | **0.278** | 0.593 | 0.875 |
| X150 residue interaction head | 2,817 | **0.754** | **0.558** | 0.765 | 0.704 |

Per-seed pooled - F150: 0.588 / 0.574 / 0.581 - X150: 0.715 / 0.710 / 0.690.

### H2 answered: the head design dominates

| | X150 | F150 | delta | seed spread | verdict |
|---|---|---|---|---|---|
| pooled | 0.754 | 0.593 | **+0.161** | 0.025 | larger than seed noise |
| within-allele | 0.558 | 0.278 | **+0.280** | 0.039 | larger than seed noise |

Nothing changed but the head. Same frozen embeddings, same rows, same split, same budget. Swapping
mean-pooling for residue-level cross-attention **doubles** within-allele Spearman.

Mean-pooling a 9-mer averages away position, and position is where anchor residues live - P2 and P9
dock into the B and F pockets. F150's within-allele 0.278 is what survives that averaging. This is
the cheapest result on the board and it indicts a standard practice: "ESM-2 embeddings + MLP" is
usually implemented as mean-pool, and on this endpoint that choice costs more than the model size.

### The ladder as it now stands (validation, same split, all internal)

| Rung | rho pooled | rho within | Uses a protein LM? |
|---|---|---|---|
| B0a global median | undefined | undefined | no |
| B0b per-allele median | 0.563 | undefined | no |
| F150 frozen ESM-2, mean-pooled | 0.593 | 0.278 | **yes** |
| X150 frozen ESM-2, cross-attention | 0.754 | 0.558 | **yes** |
| B1' one-hot peptide + allele ID | 0.748 | 0.557 | no |
| B1 one-hot peptide + pseudoseq + allele | **0.780** | **0.633** | no |

**The uncomfortable reading, and the one we should report.** X150 (0.754 / 0.558) and B1' (0.748 /
0.557) are the same number to within seed noise. A frozen 150M protein language model with a
residue-level interaction head performs **no better than a one-hot peptide encoding plus an allele
lookup**. And B1 - which adds the 34-residue HLA pseudosequence and still uses no protein language
model at all - beats both.

Note also that F150, the configuration closest to a conventional "ESM-2 embeddings" baseline,
clears the **no-peptide-information** floor of 0.563 by only +0.030 pooled.

### Cross-check: the GPU result is not a new system

Run 4 was produced remotely. The local CPU run was left going until the remote one returned, so the
two could be compared on a known-good reference before the CPU job was killed:

| | CPU | A10 | diff |
|---|---|---|---|
| F150 seed 42 | 0.581 | 0.588 | 0.007 |
| F150 seed 43 | 0.571 | 0.574 | 0.003 |
| F150 seed 44 | 0.571 | 0.581 | 0.010 |
| X150 seed 42 | 0.722 | 0.715 | 0.007 |

All inside the 0.025 seed spread. The residual difference is expected: CUDA reductions run in a
different order, so early stopping selects a different epoch (CPU 30, GPU 26 for X150 seed 42).

### Cost

| | |
|---|---|
| Remote compute | **64 s** total for both rungs, 3 seeds each |
| Wall clock | 77 s including image upload and container start |
| Same work on 4 CPU threads | ~65 min (X150 measured at 1,242 s for one seed) |
| Speed-up | **~65x** |
| Per optimiser step | 6.6 ms - the model is 561K parameters, so this is launch-bound, not FLOP-bound. A faster GPU would not help this rung |

### What Run 4 does NOT establish

- **Nothing about fine-tuning.** Every ESM rung here is frozen. L150 (LoRA r=8 on K/V) is the
  brief's actual question and has not been run. Adaptation could reorder this table entirely.
- **Nothing about test performance.** Validation numbers, and validation drove early stopping.
- **No confidence intervals.** Seed spread is not a CI. Use `paired_bootstrap()` from
  `scripts/power_analysis.py` on `esm_heads_val_predictions.json`.
- **Not a claim about ESM-2 in general.** It is a claim about ESM-2 **150M, frozen**, with these two
  heads, on this endpoint and this split. The 650M model is untested here.
- **No mechanism for why one-hot competes.** A plausible explanation is that ESM-2 embeddings are
  contextual - the vector at P2 already mixes the whole 9-mer - while one-hot is position-pure, and
  position-purity may suit an anchor-driven endpoint with 22,532 training rows. That is a
  **hypothesis**, untested.

## Run 5 - anchor probes: where the signal lives - 2026-10-03 23:33

`python -m modal run modal_app.py::anchors` on an NVIDIA A10, **91.5 s**. Two probes that measure
deliberately different things and are reported separately.

Run 4 explained X150 beating F150 by appealing to anchors: mean-pooling destroys position, and
P2/P9 dock into the B and F pockets. That was a story that fit the numbers. Run 5 tests it. The
prediction - "P2 and P9 should drop most" - was written into `src/anchors.py` **before** the probe
ran, as an assertion the script checks and prints a verdict on.

### Probe A - position ablation on the trained X150

Zero the peptide residue embedding at position k, re-score validation, measure the loss of skill.
Three seeds; intact X150 scores 0.715 / 0.710 / 0.690 pooled.

| position | drop in pooled rho | drop in within-allele rho |
|---|---|---|
| P1 | +0.063 +- 0.033 | +0.082 +- 0.008 |
| **P2** | **+0.084 +- 0.004** | **+0.077 +- 0.022** |
| P3 | +0.025 +- 0.007 | +0.041 +- 0.028 |
| P4 | +0.004 +- 0.010 | +0.015 +- 0.009 |
| P5 | +0.002 +- 0.005 | +0.004 +- 0.014 |
| P6 | +0.006 +- 0.013 | +0.020 +- 0.016 |
| P7 | +0.006 +- 0.007 | +0.011 +- 0.013 |
| P8 | +0.007 +- 0.009 | +0.004 +- 0.012 |
| **P9** | **+0.195 +- 0.136** | **+0.135 +- 0.051** |

**The pre-registered prediction was correct, and it holds seed by seed rather than only on
average** - which matters, because an average can be carried by one lucky run:

| seed | P2 | P9 | best of P4-P8 |
|---|---|---|---|
| 42 | +0.083 | +0.271 | +0.013 |
| 43 | +0.087 | +0.179 | +0.010 |
| 44 | +0.083 | +0.135 | +0.013 |

In all three, both anchors clear the entire P4-P8 core by roughly an order of magnitude. The
script also checks whether the across-position range exceeds the worst seed spread, and prints a
warning if the profile cannot separate positions at all. It did not fire.

The profile is **P9 > P2 > P1 > P3 >> P4-P8 ~ 0**, which is the textbook pMHC class I picture
arrived at from data alone: P2 and P9 are the canonical B- and F-pocket anchors, P1 contacts the A
pocket, and P4-P8 bulge into solvent. The model learned where the pockets are without being told.

One honest caveat: **P9's magnitude is unstable** (0.135 to 0.271 across seeds) even though its
rank never is. Quote the rank, not the number. And zeroing an embedding puts the input off the
manifold the encoder was trained on, so this measures *reliance*, not strict causal necessity.

### Probe B - ESM-2 masked-position pseudo-log-likelihood (no training at all)

Mask position k, ask ESM-2 for the log-probability of the residue actually there. No labels, no
fitting, so no leakage surface anywhere.

| | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 |
|---|---|---|---|---|---|---|---|---|---|
| mean log P | -3.98 | -2.91 | -2.97 | -2.97 | -2.93 | -2.91 | -2.94 | -2.92 | -2.93 |

**Essentially flat.** Every position except P1 sits within 0.06 nats of every other, and P1 is
merely the *least* predictable, not the most important.

### The finding that connects Runs 4 and 5

Rank correlation between the two profiles: **-0.03**. Zero.

**What ESM-2 finds predictable has no relationship to what predicts half-life.** The ablation
profile is strongly peaked at the anchors; the likelihood profile is flat. They are measuring
different things, and the project's headline follows directly: a frozen protein language model
underperforms one-hot encoding here because its internal notion of which residue matters is
*orthogonal* to the one this endpoint needs. X150 recovers the anchors only because the head is
given per-residue access and supervised labels to learn from - not because the embeddings
foreground them.

That is a mechanism for the Run 4 result, not a restatement of it.

**Caveat on Probe B.** This scores a bare 9-mer with no HLA context, so ESM-2 has very little to
condition on - which may itself be why the profile is flat. Concatenating peptide and HLA was not
done, because concatenation does not establish biological conditioning (the same caution the repo
README already carries). A genuine conditional test needs a model that takes two chains.

### What Run 5 does NOT establish

- **Not a causal claim.** Ablation measures what the trained model leans on, not what physically
  governs off-rate.
- **Nothing about L150.** This ablates the frozen-encoder model. An adapted encoder could relocate
  the signal.
- **No confidence intervals on the drops.** Seed spread is reported; a cluster bootstrap on the
  per-position drops is not run.
- **Validation rows only.** The test split remains unread.

## Run 6 - hurdle model for the zero class: a declared test that came back negative - 2026-10-04 00:23

`python src/hurdle.py`, 287 s on CPU, seeds 42/43/44, B1 feature set throughout.

20.6% of validation rows (579 of 2,817) sit on t-half = 0. ARCHITECTURE.md section 6 required this
to be tested rather than assumed: the CSV never establishes a detection limit, so adding a censored
head by default would be correcting for a mechanism nobody has shown exists.

| scored on | model | rho pooled | rho within | RMSE log |
|---|---|---|---|---|
| all 2,817 rows | PLAIN single regressor | **0.780** | **0.633** | 0.666 |
| all 2,817 rows | HURDLE classifier x regressor | 0.779 | 0.621 | 0.667 |
| 2,238 non-zero rows | PLAIN single regressor | **0.730** | **0.618** | 0.692 |
| 2,238 non-zero rows | HURDLE regressor, fitted on non-zero only | 0.729 | 0.600 | 0.693 |

Zero vs non-zero classifier: **AUC 0.886** (0.889 / 0.885 / 0.882 per seed).

### What this establishes

**The hurdle structure buys nothing: -0.001 pooled on both framings.** Not a small gain, not a
small loss - no difference, on either the full set or the measurable subset, and the within-allele
figures are slightly *worse*.

The informative part is that this sits next to **AUC 0.886**. The features separate the point mass
from the measurable rows very well, so the zeros are far from unpredictable. Yet modelling them as
a separate regime adds nothing. The reading is that the single regressor has already absorbed
whatever the classifier knows, and the two-part model adds machinery without adding information.

That is a stronger conclusion than "the hurdle did not help": it is evidence the zeros are **not a
separate regime the plain model was failing to represent**. Which in turn means the plan was right
not to assume censoring - and we can now say so from a measurement rather than from caution.

**Decision: do not add a censored or hurdle head.** Reported because the negative is the result.

### What it does NOT establish

- **Nothing about the mechanism.** Whether those rows are an assay floor or genuine non-binders is
  still undetermined. This tests whether modelling them separately helps *prediction*, not what
  they are.
- **No confidence interval on the delta.** -0.001 is well inside any plausible interval, so the
  conclusion is safe, but `bootstrap_ci.py` has not been pointed at these predictions.
- **Only the B1 feature set.** A hurdle built on X150 embeddings was not tried.
- Validation rows; the test split remains unread.

## Run 7 - confidence intervals on every claim - 2026-10-04 00:28

`python scripts/bootstrap_ci.py`. **2,000 paired draws, seed 2026, percentile bounds 2.5/97.5,
resampling the 540 peptide clusters** - not rows. Rows inside a cluster are within 2 substitutions
of each other, so resampling rows would treat near-duplicates as fresh information and give an
interval narrower than the evidence supports.

Every comparison is **paired**: each draw scores both models on the same resampled rows, so "this
draw happened to contain easy peptides" affects both equally and cancels.

### Per-rung 95% CI

| rung | pooled rho | within-allele rho |
|---|---|---|
| B0b per-allele median | 0.563 [0.522, 0.598] | undefined |
| F150 mean-pooled | 0.593 [0.556, 0.625] | 0.278 [0.222, 0.320] |
| X150 cross-attention | 0.754 [0.727, 0.777] | 0.558 [0.508, 0.585] |
| B1' one-hot + allele id | 0.748 [0.719, 0.775] | 0.557 [0.508, 0.585] |
| B1 one-hot + pseudoseq + allele | **0.780 [0.756, 0.802]** | **0.633 [0.588, 0.654]** |

X150 and B1' have within-allele intervals that are **identical to three decimal places**.

### Paired comparisons, 95% CI on (B - A)

| question | comparison | pooled | within-allele | verdict |
|---|---|---|---|---|
| Does a one-hot allele lookup match the frozen PLM? | B1' - X150 | -0.005 [-0.026, +0.014] | -0.001 [-0.045, +0.041] | **spans zero** |
| Does the one-hot baseline beat the frozen PLM? | B1 - X150 | +0.026 [+0.011, +0.042] | +0.074 [+0.042, +0.106] | **survives** |
| H2: does the residue head beat mean pooling? | X150 - F150 | +0.161 [+0.137, +0.188] | +0.276 [+0.222, +0.330] | **survives** |
| Does the HLA sequence add over the allele label? | B1 - B1' | +0.032 [+0.016, +0.048] | +0.074 [+0.042, +0.107] | **survives** |
| Does peptide information add over an allele median? | B1 - B0b | +0.217 [+0.185, +0.251] | unavailable | **survives** |

### What the intervals change

**1. "X150 ties B1-prime" is now a bounded null, not an eyeball.** The interval
[-0.026, +0.014] rules out any advantage to the frozen protein language model larger than about
0.026 Spearman **in either direction**. That is different from, and much stronger than, "we could
not tell them apart" - we are not short of resolution, the effect is simply smaller than 0.03.

**2. "B1 beats X150" survives, and by more than the pooled number suggests.** +0.026 pooled looks
marginal; +0.074 within-allele with an interval of [+0.042, +0.106] does not. On the metric that
matters clinically, a one-hot encoding with no protein language model beats frozen ESM-2 by a
margin that excludes zero comfortably.

**3. H2 is the largest effect measured anywhere in this project.** +0.276 within-allele,
[+0.222, +0.330]. The head design moves the result an order of magnitude more than the choice to
use a foundation model at all.

**4. One interval is reported unavailable rather than invented.** B1 - B0b within-allele: B0b
predicts a constant inside each allele, so most draws produce no defined statistic. Fewer than 95%
of draws were valid, and ARCHITECTURE.md section 7 requires reporting the interval unavailable
rather than silently resampling until enough succeed. It does.

### A correctness note on the implementation

The first version of this script could not finish 2,000 draws inside 30 minutes - it rebuilt a
pandas DataFrame and grouped on an object-dtype string column on every draw, at 156 ms each. The
vectorised replacement groups integer allele codes in numpy.

**The substitution was verified before it was trusted.** 2,466 of the 2,817 validation rows are
tied, overwhelmingly on the t-half = 0 point mass, so a naive argsort rank would silently compute a
*different statistic* under the same name. `scipy.stats.rankdata` was checked against
`power_analysis.spearman` to 2.2e-16 across 200 random subsets, and the vectorised within-allele
metric against `metrics.evaluate` - the function that produced every published number - to 1.1e-16.

## Run 8 - L150: LoRA adaptation, and the completed ladder - 2026-10-04 00:38

`python -m modal run modal_app.py::l150`. Three seeds, **one A100 each in parallel**, 4,491 s wall.
LoRA r=8, alpha=16, dropout 0.05 on the key and value projections of all 30 layers: **60 modules,
614,400 trainable parameters against 148,138,841 frozen (0.41%)**. B is zero-initialised, so at
step 0 the model is exactly frozen X150.

L150 reuses the **X150 head class itself** - same architecture, same initialisation, same optimiser,
same seeds, same split, same 40-epoch / patience-6 budget. The only difference is that gradients
reach the encoder, so `L150 - X150` is attributable to adaptation and nothing else.

### H1: does adapting the encoder help?

| | L150 | X150 | delta | 95% CI | verdict |
|---|---|---|---|---|---|
| pooled | 0.757 | 0.754 | +0.003 | [-0.012, +0.018] | **spans zero** |
| within-allele | 0.572 | 0.558 | +0.014 | [-0.020, +0.045] | **spans zero** |

Per-seed pooled 0.721 / 0.721 / 0.706, spread 0.015. Best epochs 16 / 21 / 15.

**No.** And the interval bounds it: adaptation cannot be worth more than **+0.018 pooled or +0.045
within-allele**. Training 614,400 parameters for 75 minutes of A100 time moved the result less than
changing the random seed does.

### The completed ladder, with intervals

| Rung | rho pooled | rho within-allele | Protein LM? |
|---|---|---|---|
| B0b per-allele median | 0.563 [0.522, 0.598] | undefined | no |
| F150 frozen, mean-pooled | 0.593 [0.556, 0.625] | 0.278 [0.222, 0.320] | yes |
| B1' one-hot peptide + allele id | 0.748 [0.719, 0.775] | 0.557 [0.508, 0.585] | no |
| X150 frozen, cross-attention | 0.754 [0.727, 0.777] | 0.558 [0.508, 0.585] | yes |
| L150 **LoRA-adapted**, cross-attention | 0.757 [0.732, 0.780] | 0.572 [0.524, 0.599] | yes |
| **B1 one-hot + pseudoseq + allele** | **0.780 [0.756, 0.802]** | **0.633 [0.588, 0.654]** | **no** |

**A one-hot MLP with no protein language model anywhere beats every ESM-2 rung, including the
fine-tuned one, and the margin excludes zero:**

| comparison | pooled | within-allele |
|---|---|---|
| B1 - L150 | +0.023 [+0.007, +0.039] | **+0.060 [+0.027, +0.094]** |
| B1 - X150 | +0.026 [+0.011, +0.042] | +0.074 [+0.042, +0.106] |

### Why the epoch budget was the decisive implementation detail

A first version of this rung used `MAX_EPOCHS=12, PATIENCE=3`. L150's selected epochs were
**16 / 21 / 15** - all three seeds would have been cut off mid-climb, and at epoch 6 L150 was
scoring 0.63-0.67. That run would have reported "adaptation actively hurts", which is false and
would have been an artefact of a budget I chose, not a property of the model.

X150 selected epochs 26 / 20 / 20 out of 40. Matching the budget is what makes the comparison a
comparison. **A matched residual requires a matched selection budget**, and a shorter one for the
new arm is not a conservative choice - it is a wrong one.

### What Run 8 does NOT establish

- **Not a claim about fine-tuning in general.** It is LoRA r=8 on K/V of ESM-2 **150M** for 40
  epochs on 22,532 rows. Higher rank, more modules, a larger encoder, or longer training are all
  untested. 650M in particular is untested and is where the published 0.574 figure came from.
- **Nothing about test performance.** Validation rows, and validation drove early stopping.
- **No claim that foundation models are useless for this endpoint.** The measured claim is narrower
  and more useful: on this dataset, at this scale, the extracted signal does not exceed a one-hot
  encoding of the same sequences, and adaptation does not close the gap.
- Why is in Run 5: ESM-2's salience is orthogonal to the endpoint's.

## Run 9 - Boltz-2 structures: they vary, but not with stability - 2026-10-04 01:00

`modal_boltz.py::variance_main`, then `scripts/mantel.py` and `scripts/mantel_by_position.py`.
**boltz 2.2.1** on an A100. 24 peptides on HLA-A*02:01, 2 diffusion samples each = **48 folds in
783 s**. Cost including the install probe: **~$0.6**.

**Scope: B1 only.** Structures are generated and characterised. The geometry residual (B2) is not
attempted - underpowered by roughly an order of magnitude at any cohort foldable tonight.

**Construct - a recorded deviation from the plan of record.** The plan assumed a verified full
heavy chain plus beta-2 microglobulin. The dataset carries only the **182-aa alpha1/alpha2
domain**. That domain is the complete peptide-binding groove (two helices over a beta-sheet floor),
so every peptide contact is present; b2m packs under **alpha3**, which we also lack, so including
it would leave it nothing to dock against. Groove + peptide is both what the data supports and an
exact match for the D[9,182] spec.

### Does the pose depend on the peptide at all?

Two diffusion samples of the *same* peptide give the model's own noise floor for free.

| | all cells | contacting cells (<8 A) |
|---|---|---|
| within-peptide, same sequence different seed | 0.145 A | 0.196 A |
| across-peptide, different sequences | 0.735 A | 0.808 A |
| **ratio** | 5.1x | **4.1x** |

**Yes.** The pessimistic hypothesis - that Boltz returns one canonical pose regardless of peptide,
having been trained on MHC complexes with no concept of an off-rate - is **false**.

### Does that variation track half-life? No, and the test was powered.

Mantel test: correlate the peptide-by-peptide geometry distance matrix against the half-life
distance matrix, with significance from permuting **peptide labels** - because the 276 pairs come
from only 24 independent peptides, and an ordinary p-value would be badly anti-conservative.

| | Mantel r | p |
|---|---|---|
| Spearman | **-0.111** | 0.116 |
| Pearson | -0.120 | 0.134 |

**Power at n=24: 92% to detect a true r of 0.2, 100% at r >= 0.3.** Not an underpowered shrug.

### Position-resolved: the null is not an averaging artefact

Run 5 found the model relies on P9 >> P2 > P1 > P3, with P4-P8 contributing nothing. The test above
averaged over all nine peptide rows, spending five ninths of its budget on positions our own
ablation calls irrelevant. Repeated per position:

| positions | Mantel r | p | signal / noise |
|---|---|---|---|
| anchors P2 + P9 | -0.071 | 0.283 | **6.6x** |
| P9 alone | -0.061 | 0.302 | 5.8x |
| **P2 alone** | **+0.002** | 0.984 | **8.2x** |
| P1+P2+P3+P9 | -0.105 | 0.107 | 6.4x |
| P4-P8 (negative control) | -0.095 | 0.169 | 3.6x |
| all positions | -0.111 | 0.116 | 4.1x |

**No subset survives**, and the anchor rows are the *most cleanly resolved* of all - P2's geometry
sits 8.2x above the diffusion noise floor while its correlation with half-life is +0.002. We can
see the anchor pocket very precisely, and it carries no stability information.

Stronger than the averaged test alone: we did not dilute a signal, there is none at the positions
the model actually uses.

### What Run 9 does NOT establish

- **One allele, one folding model, 24 peptides.** HLA-A*02:01 and Boltz-2 only.
- **Raw distances only.** Anchor burial depth, buried surface area, hydrogen-bond counts, backbone
  bulge height and interface pLDDT are untested, and are where a signal would most plausibly hide.
- **Nothing about B2.** The residual head was never fitted.

## Run 10 - negative control, and what it revealed about our protocol - 2026-10-04 04:09

`python src/shuffle_control.py`, 287 s CPU, plus a follow-up discriminating test.

Destroy the relationship between training features and training labels, keep validation labels
intact, retrain. A clean pipeline must score ~0.

| | pooled | within-allele |
|---|---|---|
| intact labels (reproduces B1) | +0.780 | +0.633 |
| labels shuffled **within each allele** | +0.557 | **+0.046** |
| labels shuffled **globally** | **+0.104** | +0.057 |

The within-allele shuffle behaves exactly as designed: pooled lands on the 0.563 allele-median floor
(allele identity surviving by construction, not leakage) and within-allele collapses to +0.046.
**No peptide-level leakage.**

The global shuffle gave +0.104, not 0. Rather than accept the FAIL flag, the causes were separated:

| shuffled labels | pooled |
|---|---|
| fixed 9 epochs, **no selection** | +0.028 |
| fixed 20 epochs, **no selection** | +0.020 |
| **with epoch selection** | **+0.104** |

Selection chose epochs **0, 4 and 6** - it grabs the luckiest early epoch before anything is learned.

### What this establishes, which is worth more than a passed control

**Not leakage - epoch-selection optimism, now quantified at roughly +0.08 pooled.** Every validation
number in this file is selected on validation, so every one carries that bias. Three consequences:

1. **Absolute validation figures are optimistic.** B1's 0.780 is not an estimate of held-out
   performance. Only the test read settles that.
2. **Paired differences between rungs remain valid**, because every rung pays the same bias under
   the same protocol. A direct vindication of reporting paired bootstrap intervals rather than
   comparing two independent numbers.
3. It explains the shape of the whole table: rungs differing by less than ~0.08 should never have
   been separated on validation alone - which is exactly what the Run 7 intervals already say.

## Run 11 - layer sweep: the attack the negative result had to survive - 2026-10-04 04:28

`modal_app.py::layers`, A10, 207 s. Every ESM-2 result in Runs 3-8 read `last_hidden_state` - the
**final** layer. The obvious attack is "you read the wrong layer": a masked LM's last layer is
specialised for the masked-token objective, and intermediate layers are widely found to transfer
better downstream.

| layer | raw pooled | raw within | **LN pooled** | **LN within** |
|---|---|---|---|---|
| 6 | 0.570 | 0.201 | **0.726** | **0.526** |
| 12 | 0.553 | -0.040 | 0.682 | 0.458 |
| 18 | 0.528 | 0.054 | 0.638 | 0.354 |
| 24 | 0.551 | undefined | 0.627 | 0.283 |
| 30 | 0.757 | 0.565 | 0.734 | 0.524 |

### The raw columns are a preprocessing artefact that nearly got reported as a finding

The first run produced only the raw columns. They say layer 30 wins by a mile while the middle
layers sit at the no-peptide floor, one of them predicting a **constant**. That is the signature of
a training failure, not a property of depth: ESM-2's residual stream grows in norm with depth and
the model applies a **final LayerNorm** before `last_hidden_state`, so layer 30 arrives normalised
and the others arrive raw at very different scales.

Applying the same parameter-free `F.layer_norm` to **every** layer moves layer 6 from 0.570 to
**0.726**. It also slightly *hurts* layer 30 (0.757 -> 0.734), which is the expected sign - layer 30
already carries ESM-2's own LayerNorm, so a second one is redundant.

### Verdict

- **Range across depth 0.108 vs worst seed spread 0.157 -> depth is WITHIN seed noise.** No layer
  can be claimed better than another on this evidence.
- **No layer beats B1.** Best normalised: layer 30 at 0.734 pooled, layer 6 at 0.526 within-allele,
  against B1's **0.780 / 0.633**.

The headline survives the strongest attack available to it, and "we checked five depths and
normalised them fairly" is a materially better sentence than "we used the default".

## Files

| File | What |
|---|---|
| `baselines_val_metrics.json` | All metrics per rung, including per-seed values and spread |
| `baselines_val_predictions.json` | Validation predictions per rung, aligned with peptide / allele / cluster_id — the input `paired_bootstrap()` needs |
| `esm_heads_val_metrics.json` | F150 / X150 metrics, per-seed values and spread |
| `esm_heads_val_predictions.json` | F150 / X150 validation predictions, same alignment |
| `anchors.json` | Per-position ablation drops (per seed) and ESM-2 masked-position log-probabilities |
| `hurdle_val_metrics.json` | Hurdle vs plain, scored on all rows and on non-zero rows, plus classifier AUC |
| `hurdle_val_predictions.json` | Predictions for both arms, with an `is_nonzero` flag |
| `bootstrap_ci.json` | Per-rung and paired 95% intervals, with `spans_zero` flags |
| `l150_val_metrics.json` | L150 metrics, per-seed values, best epochs, trainable parameter count |
| `l150_val_predictions.json` | L150 ensemble and per-seed validation predictions |
| `l150_seed{42,43,44}.json` | Raw per-seed payloads, written as each container returned |
| `bootstrap_l150_vs_b1.json` | The B1 - L150 interval |
| `boltz_probe.json` | Boltz-2 install probe: version, CLI, first fold |
| `boltz_variance.json` | 24 peptides x 2 samples, full D matrices and the contact mask |
| `mantel.json` | Mantel r, permutation p, and the power curve at n=24 |
| `mantel_by_position.json` | Mantel restricted to each peptide-position subset |
| `shuffle_control.json` | Negative control, three label conditions |
| `layer_sweep.json` | Five ESM-2 depths, raw and LayerNormed |
| `scale_sweep.json` | ESM-2 8M to 650M with the identical head |

## How to reproduce

| Run | Command |
|---|---|
| 1, 2 | `python src/baselines.py` |
| 3 | `python src/embed.py` (writes `cache/esm2_150m.npz`; needs `SSL_CERT_FILE` set to the local bundle) |
| 4 | `python -m modal run modal_app.py` (GPU), or `python src/esm_heads.py` (CPU, ~65 min) |
| 5 | `python -m modal run modal_app.py::anchors` |
| 6 | `python src/hurdle.py` |
| 7 | `python scripts/bootstrap_ci.py` (~17 min, CPU) |
| 8 | `python -m modal run modal_app.py::l150` (3x A100, ~75 min) |
| 9 | `python -m modal run modal_boltz.py::variance_main`, then `python scripts/mantel.py` and `python scripts/mantel_by_position.py` |
| 10 | `python src/shuffle_control.py` |
| 11 | `python -m modal run modal_app.py::layers` |
| 12 | `python -m modal run modal_app.py::scale` |

`src/features.py` asserts the frozen split's sha256 before reading any label, so a run against a
changed split fails immediately rather than reporting a number against different data.
