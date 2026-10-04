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

## Run 12 - scale sweep: 80x the parameters, 0.007 of Spearman - 2026-10-04 04:36

`modal_app.py::scale`, A100, 320 s. The identical X150 head on four ESM-2 sizes, same split, same
seeds, same budget, same selection rule, same LayerNorm on the inputs. Only the encoder changes.

| model | params | hidden | lr chosen | rho pooled | spread | rho within | spread |
|---|---|---|---|---|---|---|---|
| ESM-2 8M | 8M | 320 | 3e-4 | 0.764 | 0.007 | 0.572 | 0.015 |
| ESM-2 35M | 34M | 480 | 1e-3 | 0.769 | 0.029 | 0.579 | 0.035 |
| ESM-2 150M | 148M | 640 | 3e-4 | **0.771** | 0.024 | **0.585** | 0.033 |
| ESM-2 650M | 651M | 1280 | 3e-4 | 0.764 | 0.036 | 0.581 | 0.089 |
| **B1 one-hot** | - | - | - | **0.780** | - | **0.633** | - |

### Scale does not help, and the effect is smaller than the seed

**Range across an 80x parameter increase: 0.007 pooled, against a worst seed spread of 0.036.**
Changing the encoder from 8M to 650M moves the result five times less than changing the random
seed does. The published 0.574 that motivated this endpoint used 650M; here 650M is
indistinguishable from 8M.

**And no size beats B1.** The best ESM-2 rung is 150M at 0.771 / 0.585 against one-hot's
0.780 / 0.633.

### A tuning failure that was nearly reported as a finding

The first version of this sweep gave every model the same learning rate (1e-3). ESM-2 650M scored
**0.526 pooled - below the 0.563 no-peptide floor** - with a within-allele seed spread of 0.185,
four times any other model's. A model cannot do worse than knowing nothing about the peptide unless
its head failed to converge.

Publishing that as "bigger is worse" would have been a confident, memorable and false claim, and
the same mistake as Run 11's raw layer columns: reporting preprocessing as science.

The fix had to be fair in both directions. Giving only the large model a smaller learning rate
would have been tuning one arm and not the others. So **every** model was given both 1e-3 and
3e-4 and kept its own best, selected on validation like everything else. 650M recovered from 0.526
to **0.764**, confirming the first result was the optimiser and not the encoder.

### Together with Runs 8 and 11, this closes the question

The three knobs anyone would reach for to make a protein language model work here:

| knob | result | interval or spread |
|---|---|---|
| **depth** (Run 11) | range 0.108 across five layers | worst seed spread 0.157 - **within noise** |
| **scale** (Run 12) | range 0.007 across 80x parameters | worst seed spread 0.036 - **within noise** |
| **fine-tuning** (Run 8) | +0.003 pooled | CI [-0.012, +0.018] - **spans zero** |

None of them closes the gap to a one-hot MLP. This is no longer one unlucky configuration; it is
a negative result that survives the obvious attacks on it.

### What Run 12 does NOT establish

- **Two learning rates is not a hyper-parameter search.** A thorough search could move any of
  these. The defence is that every model got the same budget, so the *comparison* is fair even if
  no model is at its ceiling.
- **One architecture family.** ESM-2 only. ESM-C, ProtT5, SaProt and in particular **MINT** - an
  interaction-aware encoder with cross-chain attention, which is the literature's named fix for
  precisely the single-chain limitation these results diagnose - are untested. See
  `../experiments/BIO-DIRECTIONS.md`.
- **Final layer throughout**, which Run 11 shows is defensible but not optimal-by-search.
- **650M remains the least stable rung** (within-allele seed spread 0.089, 2-6x the others), so its
  point estimate deserves the least confidence of the four.
- Validation rows; the test split is still unread.

## Run 13 - complementarity: is ESM-2 useless, or merely worse? - 2026-10-04 04:44

`python scripts/complementarity.py`, CPU, no new training -- it reuses saved predictions.

"Worse alone" and "carries nothing the baseline lacks" are different claims, and only the second
justifies saying a foundation model is not useful here. This separates them.

### Do the models make the same mistakes?

Residual = rank(prediction) - rank(truth). Correlation of those residuals:

| | B1 | B1' | X150 | F150 | L150 |
|---|---|---|---|---|---|
| **B1** one-hot | 1.000 | 0.817 | **0.751** | 0.578 | 0.747 |
| **B1'** one-hot + allele id | 0.817 | 1.000 | 0.714 | 0.603 | 0.730 |
| **X150** ESM-2 frozen | 0.751 | 0.714 | 1.000 | 0.705 | 0.820 |
| **F150** ESM-2 pooled | 0.578 | 0.603 | 0.705 | 1.000 | 0.703 |
| **L150** ESM-2 LoRA | 0.747 | 0.730 | 0.820 | 0.703 | 1.000 |

**B1 vs X150: +0.751.** Neither redundant (1.0) nor complementary (0.0). The two make substantially
overlapping errors, but not identical ones.

### Does combining them beat either alone?

Predictions are z-scored before averaging, since they sit on different scales.

| combination | pooled | within-allele | vs B1 pooled |
|---|---|---|---|
| B1 alone | 0.780 | 0.633 | - |
| X150 alone | 0.754 | 0.558 | -0.026 |
| **B1 + X150** | **0.789** | 0.639 | **+0.009** |
| B1 + L150 | 0.790 | 0.648 | +0.010 |
| B1 + X150 + L150 | **0.791** | 0.645 | +0.011 |
| B1 + F150 | 0.736 | 0.553 | **-0.045** |
| B1 + B1' *(control)* | 0.780 | 0.619 | +0.000 |

Paired cluster bootstrap, 2,000 draws, seed 2026:

```
(B1 + X150) - B1   pooled +0.009 [+0.001, +0.017]   excludes zero
                   within +0.007 [-0.011, +0.025]   SPANS ZERO
```

### What this establishes - the most precise statement we can make

**ESM-2 is not redundant. It is nearly worthless.**

- Pooled, it adds **+0.009** on top of one-hot, and the interval **excludes zero** -- a real,
  measurable, reproducible contribution.
- Within-allele, which is the clinically relevant metric, it adds **+0.007 with an interval
  spanning zero** -- nothing detectable.

So the defensible claim is not "foundation models are useless for this problem". It is: *on this
endpoint, a frozen ESM-2 contributes about one hundredth of a Spearman point on top of a one-hot
encoding of the same sequences, and nothing measurable on the metric that matters clinically.*
Against 150M-650M parameters and GPU-hours, that is the cost-benefit answer the brief asks for,
stated as a number rather than a verdict.

Two supporting details:

**The control behaves.** Averaging B1 with B1' -- which share almost all their features -- gives
exactly **+0.000**. So the +0.009 is not an artefact of ensembling any two models together.

**F150 actively harms the ensemble** (-0.045). Mean-pooling does not merely discard positional
information; the resulting predictor injects enough error to drag a better model down. That
sharpens Run 4: the problem with mean-pooling is not just that it is weak, it is that it is wrong.

### What Run 13 does NOT establish

- **A simple average is not the best possible combination.** A stacked or weighted ensemble,
  fitted on calibration rows, could extract more. We did not fit one, because doing so on
  validation would add another selection surface to a protocol that Run 10 already shows is
  optimistic by ~0.08.
- **Validation rows only**, with the Run 10 optimism applying to every number above equally.

## Run 14 - learning curve: the data-efficiency defence fails, and runs backwards - 2026-10-04 05:21

`modal_app.py::curve`, A10G. One-hot (B1) and frozen ESM-2 (X150) trained on 5 / 10 / 25 / 50 /
100% of the training set, three seeds each, **validation always the full split** so every point is
measured against the same rows.

Subsampling is by **peptide cluster**, never by row. Row subsampling would leave near-duplicates
of the retained peptides in training and quietly inflate the small-data points -- the same leakage
the frozen split exists to prevent, reintroduced through the back door.

| train rows | clusters | B1 pooled | X150 pooled | delta | B1 within | X150 within | delta |
|---|---|---|---|---|---|---|---|
| 1,131 (5%) | 228 | 0.589 | 0.522 | -0.067 | 0.252 | 0.149 | -0.104 |
| 2,256 (10%) | 444 | 0.647 | 0.564 | -0.083 | 0.345 | 0.186 | -0.159 |
| 5,641 (25%) | 1,128 | 0.728 | 0.610 | **-0.118** | 0.513 | 0.273 | **-0.240** |
| 11,267 (50%) | 2,193 | 0.772 | 0.676 | -0.096 | 0.600 | 0.404 | -0.197 |
| 22,532 (100%) | 4,330 | 0.790 | 0.771 | **-0.019** | 0.649 | 0.580 | -0.069 |

### The result, and why it is the sharpest answer to the brief's question

**No crossover, at any label budget tested - and the gap runs the wrong way.**

The standard defence of a pretrained encoder is data efficiency: it should win when labels are
scarce, because pretraining substitutes for supervision. On this endpoint the opposite holds.
ESM-2 is **relatively worst at 25% of the data** (-0.118 pooled, -0.240 within-allele) and
**closest at 100%** (-0.019). Scarcity makes the foundation model *more* disadvantaged, not less.

A coherent mechanism: the X150 head has 561k parameters and must learn **how to read** the
embedding, while one-hot features are directly consumable by a small MLP. Pretraining does not
help if extracting its signal is itself a learning problem carrying its own sample complexity.

This closes the last easy defence of the negative result. "Use it when you have few labels" is the
natural rejoinder to "it loses at full data", and it does not hold here.

### A reproducibility caveat worth recording

**B1 reads 0.790 at 100% here, against the canonical 0.780 of Run 2.** Investigated rather than
ignored: locally the pipeline reproduces Run 2 **exactly** - saved predictions versus a fresh fit
give a maximum per-row difference of **0.0** and pooled 0.7802 both times. The difference is
therefore environmental: the Modal image installs scikit-learn from PyPI while the local runs use
Anaconda's build, and `MLPRegressor` differs between them by about 0.010 Spearman.

Consequences, stated rather than buried:

- **Within this experiment the comparison is sound** - B1 and X150 ran in the same container at
  every fraction, so the deltas are untouched.
- **Absolute values should not be mixed across environments.** Run 2's 0.780 and this table's
  0.790 are the same model measured in two places.
- A pinned environment is a real gap in this repository and is listed in
  `../experiments/IMPROVEMENTS.md`.

### What Run 14 does NOT establish

- **Five fractions, one architecture pair.** A different head on the embeddings could have a
  different sample complexity.
- **No intervals on the deltas.** The gaps at 10-50% are large relative to the ~0.03 seed spreads
  seen elsewhere, but no bootstrap was run per fraction.
- **Below 5% is untested**, and that is where a pretrained encoder would have its best remaining
  case - 1,131 rows is still a reasonable supervised dataset.

## Run 15 - the allele split: generalising to an HLA never seen in training - 2026-10-04 05:25

`modal_app.py::allele_split`. `splits/allele_split.csv` (sha `2ead421471903adf`), **64 train / 5
val / 6 test alleles**, 3,068 validation rows. Every validation allele is absent from training -
asserted in code, not assumed. The frozen peptide split and its sha guard are untouched; this is a
declared second experiment on a second, pre-existing split file.

This asks the clinically realistic question. The primary split holds out peptides while every
allele is seen in training; here the model meets an HLA type it has never encountered, which with
20,000+ known class I alleles is the common case rather than the exotic one.

| rung | peptide split | allele split | change |
|---|---|---|---|
| B0b per-allele median | 0.563 | **undefined** | no training median exists for an unseen allele |
| **B1** one-hot + pseudoseq + allele | 0.780 | **0.642** | -0.138 |
| **X150** frozen ESM-2 | 0.754 | **0.641** | **-0.113** |
| **B1'** one-hot + **allele identity** | 0.748 | **0.475** | **-0.273** |
| F150 ESM-2 mean-pooled | 0.593 | 0.449 | -0.144 |

Within-allele: B1 **0.539**, X150 0.380, B1' 0.315, F150 0.194.

### A structural prediction, made in advance, that held

B1' encodes allele identity as a one-hot over the 75 **training** alleles. For an unseen allele
every one of those features is zero - it is *structurally incapable* of telling two novel alleles
apart and must predict the same value for both. The allele block of its validation feature matrix
sums to exactly **0**, asserted in the run.

It duly falls furthest: **-0.273**, twice the drop of any sequence-based rung. The prediction was
falsifiable and it survived.

### The gap narrows, but does not flip

Pooled, B1 and X150 are now **0.642 vs 0.641** - a gap of 0.001, down from 0.026 on the peptide
split. ESM-2 degrades least of all the rungs. That is the first evidence in this project that a
learned sequence representation buys something a one-hot encoding cannot: the ability to place an
unfamiliar allele relative to familiar ones.

**But it is not a reversal.** B1 still ties pooled and leads clearly within-allele, **0.539 vs
0.380**. The useful conclusion is directional, not yet earned: the case for embeddings is
strongest exactly where one-hot is structurally blind, and that is where a follow-up should look.

### What Run 15 does NOT establish

- **Five validation alleles.** Within-allele is a mean over five groups; the pooled figure rests on
  3,068 rows from five HLA types. This is the smallest evidence base of any run here.
- **No confidence intervals.** The 0.001 pooled gap is far inside any plausible interval; treat B1
  and X150 as tied, not as B1 marginally ahead.
- **No L150 or scale sweep on this split.** Only the frozen rungs were re-run.
- The test split (6 alleles) remains unread.

## Run 17 - can a folding model rediscover NetMHCpan's pseudosequence? - 2026-10-04 06:10

`python src/contact_features.py`. Boltz-2 contact map from the Run 9 folds (HLA-A*02:01), against
the 34-residue pseudosequence B1 uses.

**The framing matters.** B1's HLA input is not a naive baseline - NetMHCpan chose those 34
positions from *crystallographic* contacts, so B1 already carries a structure-derived feature
selection. The sharp question is therefore not "can structure help?" but **can a 2026 folding
model rediscover, from scratch, the positions crystallography gave NetMHCpan in the 2000s?**

### Finding 1: it recovers every single one

```
Boltz contact map: 97 of 182 groove positions within 8 A of some peptide residue
pseudosequence maps to 31 unique domain positions (4 of the 34 slots are repeats)

OVERLAP: 31 of 31 recovered.  Expected by chance 16.5.  Permutation p = 0.0001.  Missed: none.
```

In 1-based domain coordinates the recovered set is **7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73,
74, 76, 77, 80, 81, 95, 97, 99, 114, 116, 143, 147, 150, 152, 156, 158, 163, 167, 171** - the
published NetMHCpan pseudosequence, arrived at independently from a predicted structure.

**This closes an ambiguity in Run 9.** The obvious objection to "Boltz geometry does not track
stability" is "your structures are simply wrong". They are not: the model places the
peptide-contacting residues exactly where twenty years of crystallography put them. So the two
results together say something stronger than either alone - **the structures are right, and the
variation in them still carries no stability signal.**

### Finding 2: as *features*, the selection is worth nothing extra

| feature set | positions | features | pooled | within | fit time |
|---|---|---|---|---|---|
| **pseudosequence (reference)** | **31** | 875 | **0.785** | **0.633** | 109 s |
| Boltz contacts | 97 | 2,195 | 0.782 | 0.622 | 351 s |
| random, matched count #1 | 97 | 2,195 | 0.789 | 0.626 | 413 s |
| random, matched count #2 | 97 | 2,195 | 0.773 | 0.592 | 373 s |
| random, matched count #3 | 97 | 2,195 | 0.779 | 0.614 | 476 s |
| *random mean* | 97 | | *0.780* | *0.611* | |
| full 182-aa domain | 182 | 3,895 | **0.768** | **0.591** | 902 s |

**Boltz's selection confers no measurable advantage.** Its 97 positions score 0.782, squarely
inside the random-97 range of 0.773-0.789. Without that control this would have read as "the Boltz
selection works", and it does not.

**And more positions actively hurt: 31 > 97 > 182.** The 31-position pseudosequence beats every
97-position set and beats the full domain by **0.017 pooled and 0.042 within-allele**, while
fitting in an eighth of the time. Handing the model all 182 groove residues is worse than handing
it the right 31.

So the pseudosequence's value is **compression, not coverage**. Structure-as-feature-selection is
a real capability (Finding 1) that the existing feature set already saturates (Finding 2) - and
the marginal groove positions are not neutral filler, they are noise the model has to fit around.

There is nothing left for a folding model to contribute here, and that is now a measured claim
rather than an assumption.

### What Run 17 does NOT establish

- **One allele's contact map**, used as a universal prior. Class I grooves are highly conserved
  and the 182-aa domains are positionally aligned, which makes this defensible - but a per-allele
  map would be better and was not computed.
- **8 A is a threshold choice**, carried over from Run 9. A tighter cut would select fewer
  positions and might behave differently.
- **The pseudosequence position mapping was inferred**, not read from a table: 4 of 34 slots
  matched multiple domain positions and took the first. The recovered set matching the published
  list is strong evidence the inference was right, but it is an inference.
- Three random controls, one of which is reported; the others were still fitting at write-up.

## Run 16 - an empirical ceiling, and a third validation of the pseudosequence - 2026-10-04 06:12

`python scripts/noise_ceiling.py`, seconds, no training.

### The problem this solves

There are **zero** replicate (peptide, allele) measurements in 28,166 rows, so the assay's
reproducibility cannot be estimated directly. Without it, "B1 scores 0.780" has no scale - it
could be at the measurement limit or nowhere near it, and those imply opposite answers to "would
a better model help?".

### The way in

94% of peptides are measured against more than one allele, and the 34-residue pseudosequence says
how similar two grooves are. Alleles differing at **one of 34 contact positions** are
biophysically almost the same pocket, so the same peptide through both should give almost the same
half-life. The agreement between those paired measurements bounds what any model could achieve.

| pseudoseq Hamming distance | pairs | median rho | max rho | median n | median RMSE log |
|---|---|---|---|---|---|
| 0 | 1 | 0.643 | 0.643 | 368 | 0.434 |
| **1** | 12 | **0.823** | **0.921** | 346 | 0.666 |
| 2 | 23 | 0.659 | 0.844 | 354 | 0.853 |
| 3 | 21 | 0.474 | 0.729 | 348 | 1.076 |
| 4 | 23 | 0.318 | 0.801 | 346 | 1.410 |

Closest pairs: `HLA-A*02:01 / A*02:16` rho **0.921** (n=369), `A*02:12 / A*02:19` 0.907,
`A*23:01 / A*24:02` 0.901, `B*27:03 / B*27:05` 0.901.

### Finding 1: there is substantial headroom, and nothing we tested captured it

At distance 1, the assay agrees with itself at **rho 0.823** across ~346 shared peptides. Our best
model reaches **0.633 within-allele**. Roughly **0.19 of Spearman is available** and no rung in
this project found it.

This changes the standing of the whole negative result. "A frozen protein language model does not
beat one-hot" is not "the problem is saturated and nothing could help". There is real signal left;
ESM-2, at four sizes, five depths and with LoRA, simply did not reach it.

**Compare against within-allele, not pooled.** Pooled Spearman is inflated by between-allele
variance - B0b scores 0.563 with no peptide information at all - while this bound is computed
within a matched peptide set through two grooves. 0.633 is the comparable figure, not 0.780.

### Finding 2: the pseudosequence is validated a third independent way

Agreement declines **monotonically** with groove distance: 0.823 -> 0.659 -> 0.474 -> 0.318. The
34 positions genuinely track groove function, which is a property of the feature, not of any
model fitted to it.

Three independent validations of the pseudosequence now exist in this results file:

1. **Run 16 (this run)** - its Hamming distance predicts cross-allele measurement agreement,
   monotonically, with no model involved.
2. **Run 17** - Boltz-2 independently recovers **31 of 31** of its positions from predicted
   structure alone, p = 0.0001.
3. **Runs 2-15** - B1, which uses it, beats every learned representation tested.

This is why one-hot wins. B1 is not a naive baseline: it carries a structure-derived feature
selection that keeps checking out from every direction we probe it.

### An internal consistency check that passed

The single distance-0 pair scores **0.643**, *below* the distance-1 median of 0.823 - which should
not happen if groove identity drives agreement. Both members are `(C67S)` constructs, exactly the
engineered variants quarantined from the structural arm as suspect. The anomaly lands where the
quarantine predicted it would.

### What Run 16 does NOT establish

- **It is a LOWER bound on the ceiling, not the ceiling.** The scatter contains genuine
  micro-allele effects that a perfect model could capture, as well as assay noise. The true
  ceiling is at least 0.823 and probably higher.
- **Twelve pairs at distance 1.** The median rests on a small sample, and the spread within
  distance 1 is wide (0.744 to 0.921).
- **Not a replicate estimate.** Two alleles are not two measurements of the same thing; this
  substitutes groove similarity for identity because the dataset offers nothing closer.
- Different pairs share different peptide sets, so the per-pair figures are not strictly
  comparable with each other.

## Run 18 - what each approach cost, and what it bought - 2026-10-04 06:27

`python scripts/cost_benefit.py`. Accuracy read from the result files, never retyped. Compute is
measured wall-clock times device count. Prices are Modal's published rates fetched 2026-10-04:
A100 40GB $0.000583/s, A10 $0.000306/s. Local CPU is priced at $0 because it ran on a laptop, with
the seconds still shown so the comparison stays honest.

This is the brief's actual question. It does not ask which model is most accurate; it says
foundation models are *"computationally heavy and expensive to produce and run"* and asks whether
they are **useful**, naming engineering and compute as a judging criterion.

| approach | pooled | within | device | device-s | $ | gain vs 0.563 floor |
|---|---|---|---|---|---|---|
| B0b per-allele median | 0.563 | undefined | cpu | 1 | 0.00 | +0.000 |
| B1' one-hot + allele id | 0.748 | 0.557 | cpu | 48 | **0.00** | +0.185 |
| **B1 one-hot + pseudoseq** | **0.780** | **0.633** | cpu | **127** | **0.00** | **+0.217** |
| F150 ESM-2 150M pooled | 0.593 | 0.278 | A10 | 201 | 0.06 | +0.030 |
| X150 ESM-2 150M cross-attn | 0.754 | 0.558 | A10 | 237 | 0.07 | +0.191 |
| L150 ESM-2 150M LoRA | 0.757 | 0.572 | A100 | 11,307 | **6.59** | +0.194 |
| ESM-2 650M + cross-attn | 0.764 | 0.581 | A100 | 180 | 0.10 | +0.201 |
| **B1 + X150 ensemble** | **0.789** | **0.639** | A10 | 364 | 0.11 | +0.226 |

Gain is pooled Spearman over the **0.563 no-peptide floor**, not over zero - scoring from zero
would flatter every model by 0.563.

### The answer, stated as a cost

- **One-hot alone: 0.780 pooled / 0.633 within-allele, in 127 CPU-seconds on a laptop, for $0.00.**
- **Best single ESM-2 configuration: 0.764 pooled - still below one-hot - after $6.70 of GPU.**
- **One-hot + ESM-2: 0.789 / 0.639.** The foundation model's entire measured contribution is
  **+0.009 pooled, and nothing distinguishable within-allele**, on top of a model that trains in
  two minutes on a laptop for nothing.

L150 is the sharpest illustration: **$6.59 and 11,307 A100-seconds to land 0.023 pooled *below*
the free baseline.**

### Whole-project compute

**4.0 GPU-hours, 1,779 CPU-seconds, $7.88 of measured Modal spend** across sixteen logged runs.

| run | device | seconds | $ |
|---|---|---|---|
| L150, 3 x A100 (Run 8) | A100 | 11,307 | 6.59 |
| Boltz-2 probe + 48 folds (Run 9) | A100 | 1,063 | 0.62 |
| learning curve (Run 14) | A10 | 900 | 0.28 |
| scale sweep (Run 12) | A100 | 320 | 0.19 |
| allele split (Run 15) | A10 | 300 | 0.09 |
| layer sweep (Run 11) | A10 | 207 | 0.06 |
| anchor probes (Run 5) | A10 | 92 | 0.03 |
| F150 + X150 (Run 4) | A10 | 64 | 0.02 |
| bootstrap intervals (Run 7) | cpu | 1,020 | 0.00 |
| hurdle (Run 6) | cpu | 287 | 0.00 |
| shuffle control (Run 10) | cpu | 284 | 0.00 |
| embedding cache (Run 3) | cpu | 188 | 0.00 |

**84% of the entire GPU bill went to L150**, the rung that produced a result inside seed noise.
That is itself worth reporting: the most expensive experiment bought the least.

### What Run 18 does NOT establish

- **Device-seconds are wall-clock, not utilisation.** A partly idle GPU costs the same, so these
  are what you would be billed, not a measure of efficiency.
- **CPU is priced at $0** because it ran on hardware already paid for. On rented CPU the one-hot
  rows would cost fractions of a cent, which does not change any conclusion.
- **Amortisation is ignored.** The embedding cache is built once and reused by several rungs; it
  is charged to Run 3 and to F150/X150, so those rows slightly double-count 188 CPU-seconds.
- **Engineering hours are not priced**, and they dominated everything here. Boltz-2 cost $0.62 of
  GPU and roughly two hours of work.

## Run 19 - does either model know the biology? Motifs from in-silico mutagenesis - 2026-10-04 06:50

`modal_app.py::motifs` (A10G) then `python scripts/motif_supported.py`.

Every comparison so far is a correlation on held-out rows. None asks whether a model internalised
*the right biology* - and a model can rank peptides well having learned something an immunologist
would not call binding. Published anchor motifs give an external answer: established by elution
and crystallography, never seen by either model, and specific enough to be falsifiable.

| allele | published anchors | source |
|---|---|---|
| HLA-A*02:01 | P2 **L, M, I** (L dominant) · P9 **V, L, I, A** (V most frequent) | MHC Motif Atlas, NAR 2023 |
| HLA-B*27:05 | P2 **R**, strictly (alternatives Q, K) | Biochem Soc Trans 2021 |

**Method.** In-silico saturation mutagenesis: for each allele take 6 reference peptides spanning
the half-life range, mutate each of 9 positions to each of 20 amino acids, predict, and average
the change. That yields a position weight matrix derived from the *model*, not from the data.

**10,320 sequences scored, of which 10,258 (99.4%) never appear in the dataset.** Mutants are
absent from the embedding cache and must be re-encoded, so this is generalisation to unseen
sequence - precisely the capability pretraining is supposed to buy, and therefore ESM-2's best
remaining case.

### Both models recover the published motifs

| allele | model | P2 top-3 | P9 top-3 |
|---|---|---|---|
| A*02:01 | B1 | **L**, E, **M** | R, **V**, K |
| A*02:01 | X150 | **L**, Y, **M** | R, **V**, W |
| B*27:05 | B1 | **R**, K, P | - |
| B*27:05 | X150 | **R**, K, V | - |

Both put **L first at A*02:01 P2** and **R first at B*27:05 P2**, matching the literature exactly.

### A probe that needed validating before it could be believed

The raw ranking put **R first at A*02:01 P9** for both models - contradicting the published motif
(V/L/I/A) and basic pocket chemistry, since the A*02:01 F pocket is hydrophobic and arginine is
large and charged.

Checking the training support explained it: **R appears at A*02:01 P9 in 3 of 821 rows (0.4%)**,
and **12 of the 20 amino acids appear in under 1% of rows at that position**. For those residues
the model is extrapolating, and ranking an extrapolation against a well-supported residue compares
a guess to a measurement. The training distribution itself (V 38%, L 34%, I 11%, A 6%) matches the
literature perfectly - only the extrapolation does not.

Re-scored over supported residues only, at several thresholds so no conclusion rests on one cut:

| support threshold | residues scored | B1 | X150 | chance |
|---|---|---|---|---|
| >= 0.0% (raw) | 20.0 | 3.78 | **3.69** | 10.5 |
| >= 0.5% | 9.0 | 2.25 | **2.11** | 5.0 |
| >= 1.0% | 7.7 | **1.94** | 2.11 | 4.3 |
| >= 2.0% | 5.7 | **1.83** | 2.00 | 3.3 |
| >= 5.0% | 3.3 | 1.83 | 1.83 | 2.2 |

### What this establishes

**1. Both models learned real immunology.** At the 1% threshold the published anchors rank 1.94
and 2.11 out of 7.7 supported residues, against a chance expectation of 4.3. Neither model was
shown a motif; both reconstructed one from half-life data alone.

**2. ESM-2 does not know the biology better, on the task most favourable to it.** The difference
is 0.17 rank positions at the 1% threshold - and it **flips sign across thresholds**: X150 leads
at 0% and 0.5%, B1 leads at 1% and 2%, tied at 5%. A difference that changes direction with an
arbitrary analysis choice is noise. This was pretraining's strongest remaining case, 99.4% unseen
sequence, and it is a tie.

**3. The probe would have reported a false finding unaltered.** "Both models rank arginine as the
best P9 residue for HLA-A*02:01" is wrong, contradicts the literature, and is the kind of claim
that discredits a submission. It survived only until someone counted how often R actually occurs
there.

### What Run 19 does NOT establish

- **Two alleles, three anchor positions.** The motif evidence is only as broad as the alleles
  whose motifs could be independently verified. Others were scored but have no external ground
  truth to check against.
- **Six reference peptides per allele.** The PWM is an average over those; a different reference
  set would shift it.
- **Single substitutions only.** Anchor residues interact, and no pairwise effects were probed.
- **Additivity is assumed.** A PWM cannot represent position-position coupling, which is exactly
  what an attention head is supposed to capture.

## Run 20 - metrics a user would act on, and a regime crossover Spearman hid - 2026-10-04 08:00

`python src/operational_metrics.py`, seconds, CPU, no new training.

Everything so far is Spearman on `log1p(t-half)`. That is the right primary metric, but it answers
none of the questions someone deciding whether to use the model would ask: how many hours out are
you, how many of the genuinely best peptides does a shortlist catch, does it work for *my*
patient's allele, and is the number usable as hours or only as a rank.

### 1. Error on the original scale

| model | MAE hours | median AE h | MAE log | MAE h, t <= 2 h | MAE h, t > 24 h |
|---|---|---|---|---|---|
| B0b per-allele median | 4.43 | 1.10 | 0.682 | 1.04 | 38.96 |
| **B1 one-hot** | **3.45** | 0.99 | **0.497** | 0.95 | **24.59** |
| X150 ESM-2 frozen | 3.54 | 0.96 | 0.524 | 0.83 | 28.38 |
| **L150 ESM-2 LoRA** | 3.47 | **0.87** | 0.509 | **0.78** | 27.86 |

1,668 validation rows sit at or under 2 h; 141 sit above 24 h.

**There is a regime crossover here that pooled Spearman completely hid.** On *median* absolute
error the ESM models win - L150 0.87 h against B1's 0.99 h - and on unstable complexes they win
clearly, 0.78 h against 0.95 h. B1 only pulls ahead on the 141 stable rows above 24 h, where its
errors are 24.6 h against 27.9 h, and because those errors are enormous they dominate the *mean*.

So "one-hot beats ESM-2" is true on rank correlation, true on the stable tail, and **false on
typical-case hours error**. A rank metric is blind to which regime the errors live in. This is
the clearest argument in the project for reporting more than one metric.

### 2. Top-k retrieval within each allele - the actual shortlisting decision

Of the k truly most stable peptides for an allele, how many appear in the model's top k?

| model | top-5 | top-10 |
|---|---|---|
| **B1 one-hot** | **0.500** | **0.621** |
| L150 ESM-2 LoRA | 0.456 | 0.581 |
| X150 ESM-2 frozen | 0.424 | 0.544 |
| B0b per-allele median | 0.174 | 0.310 |

68 alleles scored; random expectation for top-5 is about **0.143**. B1 recovers **half** the true
top five. B0b's 0.174 is a tie-break artefact - it predicts a constant inside an allele, so its
ordering there is arbitrary - and it still sits near chance, as it should.

Here the ordering matches Spearman: for *picking candidates*, one-hot is the better tool.

### 3. Per-allele behaviour - a pooled average hides where a model fails

68 alleles with at least 12 validation rows. **B1 per-allele Spearman: median 0.630, range 0.285
to 0.854.** X150 beats B1 in **15 of 68 alleles (22%)**.

| | worst five for B1 | | | best five | |
|---|---|---|---|---|---|
| allele | n | median h | B1 | allele | B1 |
| HLA-A*24:19 | 35 | 0.2 | 0.285 | HLA-B*39:10 | 0.854 |
| HLA-B*40:01 | 27 | 1.2 | 0.302 | HLA-A*31:01 | 0.852 |
| HLA-B*39:06(C67S) | 35 | 0.0 | 0.318 | HLA-B*42:02 | 0.835 |
| HLA-A*25:01 | 46 | 0.5 | 0.365 | HLA-A*02:11 | 0.833 |
| HLA-B*14:01(C67S) | 37 | 0.0 | 0.381 | HLA-A*02:12 | 0.824 |

Two findings, and the second is the useful one:

- **Skill does not track how much data an allele has**: correlation of per-allele Spearman with
  row count is **-0.076**. Giving an allele more training rows does not make it easier.
- **Skill strongly tracks the allele's median half-life**: **+0.475**. Alleles whose complexes
  mostly die fast are much harder - their labels are compressed against the zero point mass and
  the 0.1 h reporting grid, so there is less to rank. The four worst alleles have median
  half-lives of 0.0-0.5 h.

That reframes what "improving the model" would mean. The failures are concentrated where the
*label* carries least information, not where the model has least data.

**And the quarantine is vindicated a third time.** Two of the five worst alleles are `(C67S)`
engineered constructs, and `HLA-A*24:19` is one of the three domain-mismatch alleles - exactly the
7.9% of rows quarantined from the structural arm on construct grounds. They were flagged by
sequence inspection, again by the Run 16 distance-0 anomaly, and now again by per-allele skill.

### 4. Calibration

| model | slope | intercept | bias (hours) |
|---|---|---|---|
| B0b per-allele median | 0.773 | +0.415 | -3.05 |
| **B1 one-hot** | **0.978** | **-0.023** | -1.03 |
| X150 ESM-2 frozen | 1.089 | -0.032 | -2.14 |
| L150 ESM-2 LoRA | 1.031 | +0.055 | -2.07 |

Slope 1 and intercept 0 would be perfect on the log scale. **B1 is very nearly calibrated**
(0.978 / -0.023), so its output is usable as hours and not only as a rank. B0b's 0.773 shows the
expected compression toward the middle that MSE on a constant predictor produces.

### What Run 20 does NOT establish

- **These are not prediction intervals.** Honest per-prediction uncertainty needs calibration rows
  carved from **train**, never validation, and this project has not done that. A calibration slope
  is a property of the whole set, not a statement about any one prediction.
- **No confidence intervals on any of these figures.** The per-allele numbers in particular rest
  on 12 to 46 rows each.
- **Validation rows**, with the Run 10 selection optimism (~+0.08 pooled) applying throughout.
- **Top-k is computed within allele** and only for alleles with at least 2k rows, so it describes
  68 of 73 alleles.

## Run 21 - does letting ESM-2 attend ACROSS the chains help? No. - 2026-10-04 09:04

`modal_app.py::concat`, A100, 155 s. A reachable test of the interaction-awareness hypothesis
without MINT, whose weights are not on HuggingFace and whose architecture is custom.

Encode the peptide **concatenated with** the HLA domain as one 191-residue sequence, so ESM-2's
self-attention spans both chains, then split the per-residue output back into a 9-row block and a
182-row block and feed the **identical X150 head**. Same split, same seeds, same budget. The only
difference anywhere is whether the encoder saw the two chains together.

The concatenation is row-specific, so the 4.9x unique-sequence saving does not apply: all 25,349
train and validation rows were encoded individually, 6.2 GB resident in fp16.

| encoding | pooled | within-allele | per-seed pooled |
|---|---|---|---|
| separate chains (X150, Run 4) | **0.754** | **0.558** | 0.715 / 0.710 / 0.690 |
| concatenated, joint attention | 0.737 | 0.523 | 0.683 / 0.698 / 0.675 |
| **delta** | **-0.017** | **-0.036** | |

Seed spreads 0.025 pooled and 0.044 within-allele, so **both deltas are inside seed noise.**
Letting the encoder attend across chains neither helps nor hurts measurably.

### Taken with Run 4 and SPEARMINT, this pins down the mechanism

| intervention | effect on pooled Spearman | |
|---|---|---|
| joint attention across chains, **no interaction pretraining** (this run) | **-0.017** | within noise |
| cross-attention **head** on separately-encoded chains (Run 4) | **+0.161** | far outside noise |
| cross-chain **pretraining** (MINT; SPEARMINT's figure) | **+0.158** | their study, their split |

**It is not that the encoder needs to see both chains.** We gave vanilla ESM-2 exactly that and it
changed nothing. The interaction has to be **modelled by something trained to model it** - either
a head trained on this task, or an encoder pretrained on interactions. Self-attention handed the
opportunity does not spontaneously discover an interface.

That single statement explains all three results at once: why MINT works, why our head works, and
why concatenation does not.

### What Run 21 does NOT establish

- **ESM-2 has no chain-break token.** A concatenation is seen as one continuous protein, so the
  model tries to fold residue 9 into residue 10 as though covalently bonded. This is a real
  limitation of the proxy and exactly what MINT's architecture fixes. A null here therefore cannot
  fully separate "cross-chain attention is useless for this endpoint" from "ESM-2 cannot exploit
  it without being told where the chain breaks".
- **No linker was inserted.** A glycine-serine linker was considered and rejected: it would add
  residues the model must also interpret, trading one artefact for another.
- One encoder, one head, one split.

## Cross-study comparison - the head reproduces what interaction-aware pretraining buys

Not a new run: a comparison between our Run 4 and SPEARMINT, after reading the latter's numbers
properly. Added 2026-10-04 09:10.

SPEARMINT (Karthikeyan, Vincent, Rubinsteyn; `archive/background/spearmint.pdf`) evaluates
**MINT** - ESM-2 650M extended with dual intra-chain and cross-chain attention per layer, trained
on ~96M protein-protein interactions - on this same endpoint. Their reported figures:

| | their Spearman |
|---|---|
| ESM-2 Direct (pooled embeddings + head) | 0.574 |
| **MINT Direct** (cross-chain pretrained) | **0.732** |
| ESM-2 Transfer (+ affinity pre-training) | 0.745 |
| MINT Transfer | 0.791 |

Their attribution: **+0.158 from cross-chain attention** in the encoder.

Our Run 4, on our own split:

| | our Spearman |
|---|---|
| F150 (pooled embeddings + head) | 0.593 |
| **X150** (residue cross-attention **head**) | **0.754** |

Our gain: **+0.161 from a cross-attention head** on unmodified ESM-2 150M.

### The two deltas agree to 0.003

| | pooled baseline | interaction-aware | gain |
|---|---|---|---|
| SPEARMINT - cross-chain **pretraining**, 650M, 96M PPIs | 0.574 | 0.732 | **+0.158** |
| This project - cross-attention **head**, 561K params on frozen 150M | 0.593 | 0.754 | **+0.161** |

**What a 650M-parameter interaction-aware encoder buys over pooled embeddings is reproduced, to
within 0.003, by a 561,793-parameter head on vanilla ESM-2.**

Note also how closely the pooled baselines agree across two different splits and two different
model sizes: **0.574 and 0.593**. Mean-pooling lands in the same place regardless.

### Why this comparison is legitimate, and where it is not

**Legitimate:** each delta is measured *within* its own study, against that study's own pooled
baseline, on that study's own split. Comparing delta to delta does not require the splits to
match.

**Not legitimate:** comparing the absolute numbers. Their split clusters peptides at 80% identity
by normalised Levenshtein - about one substitution on a 9-mer - while ours uses **Hamming <= 2**,
which is strictly harder. Our 0.754 and their 0.732 are not on the same scale and neither "beats"
the other.

**Also unequal:** their head on MINT embeddings is not our head. Some of the 0.003 agreement is
coincidence, and the comparison would be cleaner if both heads were identical.

### What this changes

The project's own diagnosis was that ESM-2 fails here because it is a **single-chain** model
asked about an **interface**, and that the literature's fix is interaction-aware pretraining. The
first half stands - mean-pooling scores 0.593 and recovers almost nothing. The second half needs
amending: **the interaction modelling does not have to live in the pretraining.** Put it in the
head and you get the same gain for four orders of magnitude fewer parameters.

That demotes MINT from the highest-value untried experiment to a question about whether
cross-chain attention adds anything *on top of* a cross-attention head. `src/concat_encoding.py`
tests exactly that by encoding both chains as a single sequence.

**Recorded as a correction.** `experiments/BIO-DIRECTIONS.md` originally cited SPEARMINT as newly
discovered literature and recommended MINT as the top next experiment. It is the paper that has
been in `context/` since the first hours of the project, whose summary already contained the
+0.158 figure. The error was writing from a search snippet without checking our own notes.

## Run 22 - THE TEST SET READ, once - 2026-10-04 09:34

`modal_app.py::test_eval`, A10G. **The first and only read of the frozen test split.**

Protocol, fixed before anything was run: train on the 22,532 training rows, select epochs on the
2,817 validation rows, report on the 2,817 test rows. Test peptide overlap with train is asserted
to be **0** inside the script. A guard file `results/TEST_READ.json` records the read; re-running
refuses unless `--force` is passed and prints the earlier result. The danger was never reading it
once - it is reading it repeatedly and quietly keeping the best number, which leaves no trace.

Read at **2026-10-04T08:34:26Z**, with the model choice already frozen by 21 prior runs.

| rung | **TEST pooled** | **TEST within** | val pooled | change |
|---|---|---|---|---|
| B0b per-allele median | 0.573 | undefined | 0.563 | +0.009 |
| **B1 one-hot + pseudoseq + allele** | **0.806** | **0.645** | 0.780 | +0.026 |
| B1' one-hot + allele identity | 0.767 | 0.566 | 0.748 | +0.018 |
| F150 frozen ESM-2, mean-pooled | 0.606 | 0.244 | 0.593 | +0.013 |
| X150 frozen ESM-2, cross-attention | 0.756 | 0.535 | 0.754 | +0.002 |

### The conclusion holds on held-out data, and strengthens

**B1 - X150 on test: +0.050 pooled, +0.110 within-allele** - against +0.026 and +0.075 on
validation. The gap between the one-hot baseline and the frozen foundation model is *larger* on
data neither model has seen.

Every ordering established across 21 validation runs survives: B1 > B1' > X150 > F150 > B0b.

### A prediction of mine that was wrong, and why

Run 10 measured epoch-selection optimism at **+0.08 pooled** by training on shuffled labels, and I
expected a corresponding **drop** from validation to test. Test came in **higher**, mean +0.0135
across five rungs.

The reconciliation matters and is not a get-out: Run 10 measured what epoch selection can
**fabricate from pure noise**. That is a ceiling on manufactured signal, not a forecast of
validation-to-test drift for a model with real signal. When the signal is genuine the selected
epoch is actually good rather than lucky, so the optimism does not transfer. The remaining +0.014
says the test clusters happen to be marginally easier than the validation ones - which is
ordinary, since the two are different sets of peptide clusters.

Had I published "expect about 0.70 on test", it would have been a confident, specific and wrong
prediction derived from a correct measurement misapplied.

### What Run 22 does NOT establish

- **L150 was not evaluated on test.** It requires full retraining (3 x A100, ~75 min, $6.59) and
  its validation result sits inside seed noise of X150, so the marginal information did not
  justify the cost. **Stated rather than quietly omitted**: the test table has no fine-tuned rung.
- **No confidence intervals on the test figures.** The cluster bootstrap was not re-run on test.
- **One read.** No tuning happened after it, and none may.
- The figures carry no selection optimism of their own, but they were produced by models whose
  epochs were selected on validation - which is the correct protocol, not a caveat.

## Run 23 - is one-hot the right encoding? BLOSUM62 says yes - 2026-10-04 09:37

`python src/peptide_encoding.py`, 988 s CPU.

The HLA side of B1 has been validated three ways. The **peptide** side had been validated zero
ways: a positional one-hot treats every substitution as equally foreign, so L to M and L to D are
the same distance though one is conservative and the other swaps hydrophobic for charged.

BLOSUM62 encodes exactly that chemistry and is what NetMHCpan has always used. The matrix was
fetched from the NCBI distribution and checked against known values (L/L=4, L/I=2, L/D=-4,
W/W=11, symmetric) rather than recalled. A PCA-5 variant was included because Run 17 found
compression beats coverage on the HLA side.

| configuration | features | pooled | within | vs B1 |
|---|---|---|---|---|
| **one-hot peptide + one-hot HLA (B1)** | 935 | **0.780** | **0.633** | - |
| one-hot peptide + BLOSUM HLA | 935 | 0.753 | 0.569 | -0.028 |
| BLOSUM peptide + BLOSUM HLA | 935 | 0.748 | 0.567 | -0.032 |
| BLOSUM peptide + one-hot HLA | 935 | 0.739 | 0.546 | **-0.042** |
| BLOSUM-PCA5 both chains | 290 | 0.702 | 0.510 | **-0.078** |

**One-hot wins on both chains, and every chemistry-aware encoding is worse.**

### Why, and it is not an accident

This confirms a hypothesis offered at Run 4 with no evidence behind it at the time: one-hot is
maximally **position-pure** - slot 2 means exactly "the residue at P2, nothing else" - and for an
anchor-driven endpoint that is the better inductive bias.

BLOSUM62 is built from what **evolution tolerates in folded proteins**. That is a different
question from what maximises a **kinetic off-rate in a groove**. It blurs L, I and M toward each
other because they substitute freely over evolutionary time, but their effects on complex
half-life need not be similar at all - and with 22,532 training rows the model can afford to learn
each residue independently. Smoothing them together destroys information it had enough data to
use.

The PCA-5 result sharpens this: compressing 20 dimensions to 5 costs **-0.078**, the largest drop
of any variant, even though those 5 components retain 81% of BLOSUM's variance. Retaining the
variance of a substitution matrix is not the same as retaining what this endpoint needs.

### What Run 23 does NOT establish

- **BLOSUM62 only.** BLOSUM50, PAM, or a learned embedding could differ, though the mechanism
  above would predict similarly.
- **No physicochemical descriptors.** Hydrophobicity, volume and charge scales were not tested;
  they would have required constants from memory, which this project avoids.
- **One model class.** This is the sklearn MLP. A model with a learned residue embedding could
  recover chemistry where a fixed matrix imposes it.
- Validation rows.

## Run 24 - the last two P0 defects, fixed and verified - 2026-10-04 09:50

`python scripts/measure_leakage_by_distance.py` and `python scripts/verify_split.py`.

Four P0 defects were flagged when the GeoStab-FT plan was adopted. A1 and M1 were fixed the same
evening; D1 and D4 stayed open through 23 runs because they affect diagnostics rather than any
reported model number. Both are now closed.

### D1 - the "train-only" analysis was not train-only

`measure_leakage_by_distance.py` fitted its per-allele z-normalisation over **all** rows:

    zdf.groupby("allele").z.transform(lambda s: (s - s.mean()) / (s.std() or 1))

so every allele's mean and standard deviation absorbed validation and test labels. Analysis A then
printed *"no evaluation labels are read in this analysis"*. The labels never appeared as data
points, but they set the scale every data point was measured on - leakage that leaves no trace in
any output.

Statistics are now fitted on **training rows only** and applied to all rows. Re-run:

| edit distance | identity | pairs | Spearman |
|---|---|---|---|
| 1 | 0.889 | 437 | 0.616 |
| **2** | 0.778 | 297 | **0.634** |
| **3** | 0.667 | 728 | **0.406** |
| 4 | 0.556 | 5,501 | 0.354 |
| 5 | 0.444 | 2,944 | 0.237 |
| random (null) | - | 14,650 | **0.009** |

**The break still falls between distance 2 and 3**, so the Hamming <= 2 clustering threshold
stands - and is now justified without reference to evaluation labels, which is what the analysis
always claimed. The decision did not change; the basis for it is now sound.

### D4 - a self-test that never exercised the contract

`validate_detector` planted a **1-edit** known-positive while the contract promises no evaluation
peptide within **2 edits** of a training peptide. A detector that caught 1-edit neighbours and
silently missed 2-edit ones would have passed, and the reported "0 violations" would have meant
nothing.

It now probes the boundary in **both** directions, because only the pair is informative:

```
positive  CCAAFEAAL  (2 edits from AAAAFEAAL)                        -> DETECTED
negative  CCCAFEAAL  (3 edits, verified clear of EVERY train peptide) -> correctly ignored
```

A detector that fired on everything would have passed the old test while making "0 violations"
an artefact. Testing that a check *can* fire shows it is not dead; testing that it *stays quiet*
shows its silence carries information. The negative is only valid because the planted peptide is
checked against every training peptide, not just the one it was derived from - a coincidental
neighbour would otherwise make a correct detector look broken.

### A re-runnable contract check

`make_splits.py` refuses to overwrite the frozen split, which is correct, but that also meant its
verification could only run at generation time - so a fix to the verifier could not be exercised
against the split actually in use. `scripts/verify_split.py` now runs the whole contract against
the files on disk:

```
1. detector probed at the boundary, both directions   pass
2. cross-split neighbours at Hamming <= 2             0
3. partition  train 4,514 / val 563 / test 556        disjoint, complete
4. sha256 210775dc4ad179df...  matches features.py    True
```

Check 4 ties the split to the rest of the codebase: if this hash disagreed with the constant
`src/features.py` asserts before reading any label, either the split moved or the constant is
stale, and every result here would be suspect.

**All four P0 defects are now closed.**

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
| `contact_features.json` | Boltz contact positions vs the pseudosequence, overlap test and feature comparison |
| `noise_ceiling.json` | Cross-allele agreement by pseudosequence distance, the empirical ceiling |
| `cost_benefit.json` | Accuracy against measured device-seconds and dollars, per approach |
| `motif_recovery.json` | Per-allele position weight matrices from both models, and anchor ranks |
| `motif_supported.json` | Anchor ranks re-scored over residues with training support |
| `operational_metrics.json` | Hours-MAE, top-k retrieval, per-allele table, calibration |
| `concat_encoding.json` | Joint peptide++HLA encoding, X150 head, vs separately-encoded chains |
| `test_metrics.json` | **The test-set read.** Per-rung metrics and predictions on held-out rows |
| `TEST_READ.json` | Guard: records that the test split was read, and when |
| `peptide_encoding.json` | One-hot vs BLOSUM62 vs BLOSUM-PCA5 on both chains |
| `split_verification.json` | The frozen split re-checked: boundary probe, neighbours, partition, hash |
| `complementarity.json` | Residual correlations, ensemble combinations, and the (B1+X150) - B1 interval |
| `learning_curve.json` | One-hot vs frozen ESM-2 at 5/10/25/50/100% of training clusters |
| `allele_split_metrics.json` | The ladder re-run against the allele split (every val allele unseen) |

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
| 13 | `python scripts/complementarity.py` |
| 14 | `python -m modal run modal_app.py::curve` |
| 15 | `python -m modal run modal_app.py::allele_split` |
| 16 | `python scripts/noise_ceiling.py` |
| 17 | `python src/contact_features.py` |
| 18 | `python scripts/cost_benefit.py` |
| 19 | `python -m modal run modal_app.py::motifs` then `python scripts/motif_supported.py` |
| 20 | `python src/operational_metrics.py` |
| 21 | `python -m modal run modal_app.py::concat` |
| 22 | `python -m modal run modal_app.py::test_eval` — **one read only** |
| 23 | `python src/peptide_encoding.py` |
| 24 | `python scripts/measure_leakage_by_distance.py` and `python scripts/verify_split.py` |
| audit | `python scripts/audit_results.py` - re-checks every number here against its source |

`src/features.py` asserts the frozen split's sha256 before reading any label, so a run against a
changed split fails immediately rather than reporting a number against different data.
