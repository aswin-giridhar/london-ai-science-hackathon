# Brainstorm — every direction we could take, and what each would prove

Written 2026-10-03. A catalogue, not a plan: most of this will not be built. Pick from it, claim a
file, and write the proposal. Anything here can be argued with.

**Costs are estimates, not measurements** — nothing in this document has been run. Treat every
"~2h" as a guess to be replaced by a measured number the moment someone runs it.

---

## 0. The question tree

The brief asks one question: *can existing protein foundation models improve prediction of
peptide-HLA complex stability?* That is too broad to answer in a day. It decomposes into
sub-questions that are each answerable, and choosing which ones we answer **is** the submission:

```
Are foundation models useful here?
├── Useful compared to WHAT?
│   ├── vs. nothing (global mean, per-allele mean)
│   ├── vs. hand-crafted sequence features (BLOSUM/one-hot + MLP)   ← the honest bar
│   └── vs. a one-hot lookup of the 75 alleles                      ← the sharpest null
├── Useful WHERE?
│   ├── on unseen peptides          (peptide_split — the default)
│   ├── on unseen alleles           (allele_split — where a pLM should win structurally)
│   ├── on rare alleles / thin data (do they help most where data is scarce?)
│   └── in the low-label regime     (learning curves)
├── Useful HOW?
│   ├── embeddings → trained head
│   ├── zero-shot likelihood (no training at all)
│   ├── model confidence → uncertainty
│   └── predicted structure → geometric features
└── Useful ENOUGH to justify the cost?
    └── does a 650M model beat an 8M one? does any of it beat a 90-second MLP?
```

**The last branch is the one the brief actually cares about** and the one most teams will skip. Its
own words: foundation models are *"computationally heavy and expensive to produce and run. We are
asking the question **Are they useful for this problem**."* A result showing a 35M model matches a
650M model is a better answer to that question than a 0.01 Spearman gain.

---

## 1. Baselines and controls

Nothing below is optional. Without these, every foundation-model number is uninterpretable.

| # | Idea | Answers | Est. cost | How it could fail |
|---|---|---|---|---|
| B1 | **Global mean / median predictor** | The floor. Spearman ≈ 0 by construction | minutes | — |
| B2 | **Per-allele mean** | How much is explained by allele identity alone, with zero peptide information | minutes | Will be higher than people expect — that is the point |
| B3 | **BLOSUM/one-hot + MLP** (`exp 000`) | The honest bar a foundation model must clear | ~2h | Tuning it too hard, or not enough — either makes the comparison dishonest |
| B4 | **One-hot allele + peptide features** | **The sharpest null.** There are only 75 distinct HLA sequences. If one-hot matches an HLA embedding, the model is memorising allele identity, not using protein knowledge | ~1h on top of B3 | None — this is the control that makes or breaks the headline claim |
| B5 | **Peptide-only and HLA-only ablations** | How much of the signal is interaction vs. marginal | ~1h | — |
| B6 | **Nearest-neighbour by BLOSUM similarity** | A non-parametric bar; also quantifies how much is "find a similar peptide" | ~1h | — |

**B2 and B4 are where the honest story lives.** If per-allele mean gets a large share of the
achievable Spearman, then most of what any model "knows" is which allele it is looking at, and we
should say so loudly rather than let a pooled number imply otherwise.

---

## 2. Protein language models — representation routes

| # | Idea | Answers | Est. cost | How it could fail |
|---|---|---|---|---|
| R1 | **Frozen embeddings + shallow head** (`exp 001`) | The standard approach; the thing everyone builds | ~4h | Lands at or below B3, which is itself a publishable answer |
| R2 | **Per-position vs mean-pooled peptide embedding** | **Mean-pooling a 9-mer destroys position**, and position is exactly what anchor residues are. Keeping all 9 × 1280 should beat pooling | ~1h on top of R1 | If it doesn't help, that is informative about what the embedding encodes |
| R3 | **Layer sweep** — which transformer layer to read | Middle layers often beat the last for structural/biophysical properties. The brief says "explore under the hood"; this is that | ~2h, inference only | Needs embeddings cached per layer — storage, not compute |
| R4 | **Model-size ladder**: ESM-2 8M → 35M → 150M → 650M | **Directly answers the cost-benefit question.** If 35M ≈ 650M, that is a headline finding | ~3h | Larger models may not fit the GPU budget; subsample if so |
| R5 | **Backbone comparison**: ESM-2 vs ESMC vs ProtT5 vs SaProt | Is the result a property of *protein LMs* or of *one* protein LM? | ~4h | SaProt needs structure tokens — see S4 |
| R6 | **Cross-attention over concatenation** | The published ablation says +0.158 ρ. Our data says the same peptide ranges 14.6h → 0h across holders, so the interaction is the signal | ~6h, needs training | Expensive; the cached-embedding trick stops working once chains attend |
| R7 | **Fine-tuning: last-k layers or LoRA** | Does adaptation beat frozen features? | ~6h+ | Most likely to blow the GPU budget for the least interpretable gain |

**R2 and R4 are the best value in this section.** Both are cheap, both produce a crisp finding, and
neither is what a rushed team would do first.

---

## 3. Protein language models — zero-shot routes (no training at all)

The brief names these explicitly and the published SPEARMINT work tries none of them.

| # | Idea | Answers | Est. cost | How it could fail |
|---|---|---|---|---|
| Z1 | **Pseudo-log-likelihood of the peptide in HLA context** (`exp 002`) | Does the model already "know" what a stable complex looks like? | ~3h | Plausibly ρ ≈ 0 — a clean, reportable null |
| Z2 | **Context minus no-context likelihood** | Isolates how much the groove *explains* the peptide, rather than how typical the peptide is on its own | +1h over Z1 | Arguably the most meaningful of the three scores |
| Z3 | **Masked-marginal scoring** (mutation-effect style) | The standard zero-shot protocol from the pLM literature, applied to a new endpoint | +2h | — |
| Z4 | **Seed and mask-pattern variation** | A free ensemble *and* a free uncertainty estimate | +1h | Spread may not correlate with error — worth knowing either way |

**Why this section matters out of proportion to its cost:** a zero-shot result needs no training, so
there is no train/test leakage surface at all, and the entire dataset becomes an evaluation set.
That is a methodologically clean claim and it is cheap.

---

## 4. Structure routes

The expensive family. The brief permits subsetting explicitly: *"either through data subsetting or
careful choice of which model(s) to evaluate."*

| # | Idea | Answers | Est. cost | How it could fail |
|---|---|---|---|---|
| S1 | **Boltz-2 co-fold a small cohort → interface contact features** | Does predicted geometry add anything beyond sequence? | ~8h+ | **The class I groove is highly conserved.** Predicted structures may barely differ between peptides, so the features carry little variance. State this as the falsification up front |
| S2 | **ESMFold / Chai-1 as a cheaper co-fold** | Same question, lower cost | ~5h | Monomer-oriented models may handle the complex poorly |
| S3 | **One template per allele + thread peptides** | Sidesteps per-pair folding entirely: fold once per allele (75), then score any peptide against it | ~4h | A fixed backbone ignores peptide-induced conformational change — which may be exactly the signal |
| S4 | **SaProt with Foldseek 3Di structure tokens** | A structure-aware LM without running a folding model per pair | ~4h | Still needs a structure per allele; depends on S3 |
| S5 | **Inverse folding score (ProteinMPNN / ESM-IF)** | "How well does this peptide fit this groove backbone?" — a natural stability proxy, and uses the brief's third model family | ~4h after S3 | Scores sequence-given-backbone, which is not the same as kinetic stability |
| S6 | **PDB reference panel** | Real experimental pHLA structures for QC and orientation | ~3h | Exact allele+peptide matches may not exist; it is a reference, not a label source |

**The honest framing for this whole section:** a negative result here is scientifically interesting.
"Predicted static geometry adds nothing beyond sequence for a kinetic property" is a real finding,
and the brief says so.

---

## 5. The label problem — the gap nobody in the literature has filled

| # | Idea | Answers | Est. cost | How it could fail |
|---|---|---|---|---|
| L1 | **Censored (Tobit-style) likelihood for `t½ = 0`** | **Now settled from the data** (`investigate_zero_labels.py`): the reporting scale is 0.1 h (98.4% of non-zero values are exact multiples of it, one row below 0.1 in the whole set), so `0.0` means **half-life < 0.05 h — left-censored**, not a true zero. SPEARMINT feeds `log(1+0)=0` into MSE regardless. Censoring threshold is known, so a Tobit likelihood is directly implementable | ~3h | May not move Spearman, since rank metrics are insensitive to the floor's exact value. Could still improve calibration and absolute error |
| L2 | **Two-stage: classify "zero vs positive", then regress** | A cleaner decomposition; also gives a usable "will this bind at all" classifier | ~3h | Two models to tune; error compounds |
| L3 | **Within-allele target normalisation** | We measured that pooled correlations are inflated by allele-level differences. Training on within-allele z-scores forces the model to learn peptide effects | ~2h | Loses the ability to predict absolute hours without adding the allele mean back |
| L4 | **Target transform sweep**: log1p vs Box-Cox vs rank | Is log1p actually the right choice, or just the inherited one? | ~1h | Likely a small effect; cheap enough to settle |
| L5 | **Sample weighting by allele support** | 7 alleles have <50 rows; should they count equally? | ~1h | — |

**L1 is the most novel thing on this list.** It is a real gap in the published work, it is cheap, and
it is defensible whichever way the result goes.

---

## 6. Evaluation and evidence — where the brief's points actually live

§6 judges *"your approach to the data, chosen evaluation metrics, your conclusions, and the evidence
you provide to support them."* These are not chores; they are the scored deliverable.

| # | Idea | Answers | Est. cost | How it could fail |
|---|---|---|---|---|
| E1 | **Within-allele Spearman alongside pooled** | We already measured the confound: unrelated pairs correlate at 0.335 pooled and 0.027 within-allele. **Any pooled headline is partly a between-allele artefact** | ~1h | None. This is nearly free and most teams will miss it |
| E2 | **Learning curves**: 5 / 10 / 25 / 50 / 100% of training data | **The single sharpest test of the brief's question.** If pLMs help most when labels are scarce, that is a precise, transferable answer | ~3h | Needs the same pipeline run 5× — cheap with cached embeddings |
| E3 | **Per-allele breakdown vs. training support** | Do foundation models rescue the rare alleles? | ~2h | Small per-allele n makes intervals wide; report them |
| E4 | **Paired bootstrap CIs over peptide clusters** | Is any gap real, or noise? A CI spanning zero is inconclusive, not equivalence | ~2h | Must resample *clusters*, not rows, or the CI is too narrow |
| E5 | **Rescue analysis**: which cases does the pLM fix that the baseline gets wrong? | Turns a scalar into a mechanism. Excellent demo material | ~2h | — |
| E6 | **Allele-split evaluation (unseen HLAs)** | Where one-hot *structurally cannot work*. The sharpest place for a pLM to win | ~2h | Only 5–6 alleles held out; wide intervals, label shift — treat as directional |
| E7 | **Calibration / empirical coverage of uncertainty** | Are the uncertainty estimates honest? | ~2h | Conformal guarantees need group-aware design when peptides are clustered |

**E1 and E2 are the highest value-per-hour in this entire document.**

---

## 7. Interpretability and biology — the §6 criterion most teams will ignore

| # | Idea | Answers | Est. cost | How it could fail |
|---|---|---|---|---|
| I1 | **Position occlusion: mask each of P1–P9, measure prediction change** | **Does the model rediscover the anchor positions?** If importance concentrates at P2 and P9, it has learned the binding motif from sequence alone | ~2h | Some alleles are P3-dominant — a uniform P2/P9 story would be the oversimplification, and finding the exceptions is a *better* result |
| I2 | **Per-allele anchor profiles** | Different alleles have different anchors. Does the model learn *per-allele* motifs, or one average motif? | ~3h | — |
| I3 | **Embedding geometry: do alleles cluster by known supertype?** | Is the HLA representation biologically structured or arbitrary? | ~2h | Pretty, but weak evidence on its own |
| I4 | **Which residues drive the zero-label class?** | Interpretable failure modes for the 0.0 floor | ~2h | Depends on L2 |

**I1 is the best demo in this document.** It is cheap, it is visual, it directly serves the biology
criterion, and if it works it shows the model learned real immunology from sequence alone. If it
*doesn't* work, that is equally reportable — it would mean the model is fitting something other than
the known binding mechanism.

---

## 8. Uncertainty

| # | Idea | Answers | Est. cost | How it could fail |
|---|---|---|---|---|
| U1 | **Deep ensemble / seed variation** | Simplest honest uncertainty | ~2h | Underestimates epistemic uncertainty |
| U2 | **Model's own confidence metrics** | The brief names these; near-free | ~1h | May not correlate with error — test it |
| U3 | **Gaussian likelihood head** (predict μ and σ) | Uncertainty from one model, no ensemble | ~2h | Needs careful loss handling with the censored floor |
| U4 | **Group-aware conformal prediction** | Distribution-free intervals | ~3h | Standard conformal assumes exchangeability — clustered peptides break it |

---

## 9. Demo

Demo is 20% of the event-wide rubric and a regression table is not a demo.

- **D1** — Enter a peptide and allele, get a predicted half-life **with an interval**, plus the
  position-importance plot from I1. Shows prediction *and* reasoning.
- **D2** — Side-by-side: baseline vs foundation model on the same input, showing where they disagree.
- **D3** — The "shortlist" framing from the clinical use case: paste 20 candidate peptides, get them
  ranked with uncertainty, since that is the actual downstream task.
- **D4** — An evidence drawer: every number traceable to the script that produced it.

---

## 10. Anti-goals — attractive traps

Listed because each one will tempt someone at 3am.

1. **Reproducing SPEARMINT.** Not asked for, not feasible, and the brief rules its comparator out.
2. **Chasing ρ = 0.79.** There is no leaderboard. A clean 0.60 with honest evidence beats an
   unexplained 0.78.
3. **Full fine-tuning of a 650M model.** Most expensive, least interpretable, most likely to eat the
   night and produce one number.
4. **Boltz-2 across the whole dataset.** Subset or skip; the brief explicitly permits subsetting.
5. **Adding a fourth and fifth model instead of a control.** Breadth without B4 proves nothing.
6. **Reporting pooled Spearman alone.** We have measured that it is partly a between-allele artefact.
7. **Touching the test set more than once.** Selecting on test is leakage reintroduced by hand.
8. **Over-tuning the baseline.** A straw-man baseline makes the whole comparison dishonest; so does
   a baseline tuned harder than the model it is compared against.
9. **Letting "no result" mean "don't write it up".** The brief says a supported negative result is as
   good as a positive one. An unwritten experiment scores zero.

---

## 11. If I had to pick — a 23-hour skeleton

Not a decision, a proposal. Ordered by value per hour, with everything after the line optional.

**Gate A — the delivery floor (must exist by tonight)**
1. B1, B2 — trivial baselines (30 min)
2. B3 — supervised MLP baseline, `exp 000` (~2h)
3. R1 — frozen ESM-2 embeddings + head, `exp 001` (~4h)
4. **B4 — the one-hot allele control** (~1h) ← without this, nothing above means anything
5. E1 — within-allele reporting on all of the above (~1h)

**Gate B — the differentiators (what makes this a submission rather than a tutorial)**
6. E2 — learning curves (~3h) — the sharpest answer to the brief's question
7. I1 — position occlusion / anchor rediscovery (~2h) — the biology criterion, and the demo
8. Z1/Z2 — zero-shot likelihood (~3h) — novel vs the literature
9. E4 — bootstrap CIs over clusters (~2h) — makes every claim defensible

**Gate C — if time remains**
10. R4 — model-size ladder (cost-benefit evidence)
11. E6 — allele-split evaluation (unseen HLAs)
12. L1 — censored likelihood for the 0.0 floor
13. S3/S5 — one template per allele, inverse-folding score

**Always, in parallel:** write up every result including the failures, keep the evidence ledger, and
build the demo from whatever exists at the time rather than waiting for the best model.

---

## 12. Cross-cutting rules, learned the hard way in this repo

These came out of mistakes already made here. They apply to every experiment above.

1. **Validate a probe before believing it.** A leakage measurement here reported a null of 0.335
   where ~0 was required; it was confounded by pooling across alleles and the entire table was wrong.
   A detector that cannot fail is not measuring anything — plant a known-positive.
2. **Pooled correlations on this dataset are partly between-allele effects.** Always report
   within-allele too.
3. **A guard must separate.** The split's representativeness assertion scores 0.36 on the buggy
   version and 1.00 on the fixed one. A threshold nothing can fail is a bug wearing a safety costume.
4. **Measure the constant, don't inherit it.** The published 80%-identity convention turned out to be
   too loose here, and measuring it took twenty minutes.
5. **Reproduce a number before publishing it.** Two runs of the same configuration tell you whether
   the first one meant anything.
6. **Row counts can look perfect while the split is broken.** 80/10/10 was exact while the test set
   held no peptide measured against more than 6 alleles.
7. **Quote measured costs, not nominal ones.** Every "~2h" in this document is a guess until someone
   runs it and replaces it.
