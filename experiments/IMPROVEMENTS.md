# How to make this submission harder to fault

Written against the challenge's own evaluation criteria, after eight logged runs. Every cost is an
estimate unless marked **measured**. Items are ordered within each section by *value per hour*, and
the first section is the one that matters most: things that could make our headline **wrong**.

Current headline, for reference: *no ESM-2 rung beats a one-hot MLP on this endpoint, and LoRA
adaptation does not close the gap.* See `../results/README.md`.

---

## 0. Threats to the conclusion we have already drawn

A negative result is only as good as the effort spent trying to overturn it. The brief says
"negative results are just as good as positive results **where well supported**". These are the
attacks a knowledgeable judge would make, roughly in the order they would land.

### T1 — We used only the final layer of ESM-2. ⚠ highest risk, cheapest fix

`src/embed.py:60` and `src/l150.py:154` both read `last_hidden_state`. That is the single most
criticisable choice in the project. It is well established that the **last layer of a masked
language model is specialised for the masked-token objective**, and that intermediate
representations transfer better to downstream tasks. We never checked.

If a middle layer beats one-hot, our headline is **false** and we would have published it.

- **Test:** re-encode with `output_hidden_states=True`, keep layers {6, 12, 18, 24, 30}, train the
  X150 head on each. One encode pass, five cheap head trainings.
- **Cost:** ~40 min, maybe $1 of A10. The encode dominates.
- **Either outcome is a result.** A flat layer profile strengthens the negative considerably — "we
  checked every depth" is a much better sentence than "we used the default".

### T2 — One model, one size

We tested ESM-2 **150M** only. The claim we can support is "a 150M protein language model does not
help here"; the claim a reader will hear is "protein language models do not help here". The
published 0.574 that motivated this endpoint used **650M**.

- **Test:** the same X150 head over ESM-2 **8M / 35M / 150M / 650M**. Four points is enough to see
  a trend.
- **Cost:** ~1.5h. 650M encode is ~4x the 150M encode (188 s **measured**), so still minutes.
- **Why it is the strongest single addition:** a *flat* scaling curve turns a local negative into a
  general one — "we scaled the encoder 80x and the gap did not close" is a far more interesting
  claim than anything we currently have. A *rising* curve honestly bounds our conclusion instead,
  which is also worth knowing and worth saying.

### T3 — Every conclusion rests on one split

`splits/allele_split.csv` already exists (64 train / 5 val / 6 test alleles) and **has never been
used**. It asks a different and arguably more clinically meaningful question: does this generalise
to an allele never seen in training — a new patient with a rare HLA type?

- **Test:** rerun B0/B1/B1'/F150/X150 against the allele split. No new model code; only the split
  file changes.
- **Cost:** ~1h.
- **Why it matters:** if the ordering (one-hot ≥ ESM-2) holds under *both* splits, the conclusion
  is robust to the split criterion, which is the obvious objection to a single-split result. And
  there is a real chance it **flips**: B1' encodes allele identity as a one-hot over 75 alleles,
  which is **structurally incapable** of generalising to an unseen allele, while an embedding of
  the HLA sequence can. That would be a genuinely interesting reversal and a much richer story
  than either result alone.

### T4 — Only one model family, and we now have structures

The brief names ESMC, ProtT5 and SaProt. **SaProt is the pointed omission**: it is structure-aware,
and as of Run 9 we have Boltz-2 structures for this system. A structure-aware PLM on a task where
structure is the mechanism is the most on-thesis model we have not tried.

- **Cost:** ~2h (new tokeniser, foldseek 3Di alphabet).
- Lower priority than T1-T3 only because those are cheaper and attack the conclusion more directly.

### T5 — The baseline and the PLM heads were not tuned with equal effort

B1 is an MLP on 935 one-hot inputs; X150 is a 561k-parameter attention head. Neither got a
hyper-parameter search. We should **say so explicitly** rather than let a reader assume parity.
The honest framing: both were given the same budget and the same selection rule, and neither was
tuned; a tuned comparison is future work.

- **Cost:** 10 min of writing, or ~2h for a small matched random search over both.

### T6 — No negative control has been run

We have never checked that the pipeline returns ~0 on shuffled labels. If it returns 0.3, something
leaks and every number is suspect.

- **Test:** shuffle `thalf_hours` within the training split, retrain B1, confirm validation
  Spearman collapses to ~0.
- **Cost:** ~20 min. **Do this regardless of everything else** — it is the cheapest single
  credibility item on the list, and its absence is conspicuous.

---

## 1. Approach to the data

Facts established by measurement, and what is still missing.

| Measured | Value | Consequence |
|---|---|---|
| Replicate (peptide, allele) pairs | **0 of 28,166** | **No empirical noise ceiling is obtainable.** We cannot say "the model is near the measurement limit" — there is nothing in this file to estimate that from. Say so; do not invent one |
| Peptide lengths | all **9** | Real class I binds 8-11mers. Our scope is 9-mers; state it as a scope limit, not a silent assumption |
| Alleles | 75; median **368** rows, min **7**, max **1070** | 7 alleles have <100 rows. Per-allele metrics on those are noise |
| Distinct label values | **944** over 28,166 rows | Heavy discretisation, on top of the 20% zero point mass |

**Worth adding:**

1. **A per-allele results table** (ρ, n, and CI per allele), not just the mean. It shows *where* the
   model works, exposes the 7 low-n alleles honestly, and is the natural place to discuss supertypes.
   ~30 min.
2. **Supertype grouping.** HLA class I alleles cluster into supertypes (A2, A3, A24, B7, B27, B44,
   B58, B62) by pocket chemistry. Asking whether errors cluster by supertype connects the ML result
   to the biology and tests the "parameter sharing across similar alleles" hypothesis that
   `results/README.md` currently flags as untested. ~1h.
3. **Quantify the discretisation ceiling.** With a 0.1 h reporting grid, how much Spearman is
   destroyed by rounding alone? Simulate: take predictions, round to the grid, re-score. Gives an
   upper bound that is *derived* rather than guessed. ~30 min.
4. **Document the 7.9% construct quarantine** (3x C67S, 3 domain-mismatch alleles) in the results
   rather than only in the architecture doc, since it bears on the structural arm's coverage.

---

## 2. Evaluation metrics

What we have: Spearman pooled and within-allele, Pearson on log, RMSE on log, cluster-bootstrap CIs
on every claim, seed spreads everywhere.

**Gaps, in order of how much a judge would care:**

1. **The test split has never been read.** This is correct discipline and it is also a hole: there
   are currently **no held-out numbers at all**. Needs one single read after the model choice is
   frozen, reporting validation and test side by side. The gap between them is itself evidence
   about how much the validation-driven early stopping cost us. ~20 min, one-shot, needs a decision.
2. **Operational error in hours.** Everything is on `log1p`. A clinician asks "how many hours out
   are you?" `ARCHITECTURE.md` §7 already promises hours-MAE and we do not report it. ~15 min.
3. **Top-k precision.** The actual use is *pick the most stable candidates from a shortlist*. "Of
   the true top-10 peptides for this allele, how many are in our predicted top-10?" is closer to
   the decision than any correlation, and it is where a model can look good on ρ and be useless.
   ~30 min.
4. **Calibration and prediction intervals.** Requires calibration rows carved from **train**, never
   val (§2). Without it we can show a point prediction and no honest uncertainty. ~1h.
5. **A cost-per-result table.** The brief weighs "engineering & compute requirements" explicitly.
   We have the measurements already — put them in one table: GPU-seconds, dollars, and Spearman
   gained, per rung. It directly answers "are foundation models worth their cost here", which is
   the actual research question. ~15 min, **highest value-per-minute item in this section**.

---

## 3. Exploring the models "under the hood"

The brief lists what it wants explored. Our coverage:

| Route | Status |
|---|---|
| Embeddings → head | done (F150 pooled, X150 per-residue) |
| Fine-tuning | done (L150, LoRA r=8 on K/V) |
| Log-likelihood / perplexity | done, unconditioned only — **ρ +0.002** |
| Masked inputs | done — per-position ablation and masked-position scoring |
| Different seeds | done — three everywhere, spreads reported |
| **Internal confidence metrics** | **not done** — named explicitly in the brief |
| **External confidence metrics** | **not done** |
| **Layer choice** | **not done** — see T1 |
| **Model scale / family** | **not done** — see T2, T4 |

**Internal confidence is the named gap.** Concretely available and cheap:

- **Attention entropy** over the peptide positions — a confident, focused attention pattern vs a
  diffuse one, as a per-prediction uncertainty signal.
- **Embedding norm** and **per-residue pseudo-perplexity** as difficulty proxies.
- **Seed-ensemble disagreement** as epistemic uncertainty — we already train 3 seeds, so the
  spread is *already computed and thrown away*. Correlating per-row seed disagreement with absolute
  error is nearly free and would show whether the ensemble knows when it is wrong. **~20 min, do it.**

One more, directly on-thesis: **do ESM-2's own attention maps recover the groove contacts?** We now
have Boltz distance matrices to compare against. "The language model's attention does / does not
align with the physical contacts" is a strong, visual, biologically grounded result. ~1.5h.

---

## 4. Engineering and compute

Already good: every run is timed and priced, the embedding cache is justified by a measured 4.9x
redundancy, GPU and CPU paths are cross-checked, and the frozen split is asserted by sha256 before
any label is read.

**Worth adding:**

1. **The cost table** (see §2.5). We have the numbers; they are scattered across eight run entries.
2. **Inference latency per prediction** — needed for the demo and for any deployment claim. ~10 min.
3. **A `requirements.txt` or lockfile.** Reproducibility is an ML best practice and we have none.
   ~10 min.
4. **One end-to-end smoke test** that runs the whole pipeline on ~200 rows and asserts the outputs
   exist and are finite. Protects against the class of failure that bit us three times tonight —
   exit code 0 with no usable output. ~40 min.
5. **Run the documented commands verbatim from a clean shell** before submitting. Our README's
   reproduce table has never been executed as written.

---

## 5. Biological background

Strongest part of the submission already: the position-ablation profile recovers P2/P9 anchors from
data alone, seed by seed, against a pre-registered prediction.

**To deepen it:**

1. **Map the 182 HLA positions to pockets A-F** and check whether the model's attention concentrates
   on the B and F pockets specifically. The anchor result is currently peptide-side only; the
   groove-side mirror would make it much stronger. ~1h.
2. **Anchor-motif consistency.** HLA-A\*02:01 prefers L/M at P2 and V/L at P9. Does the model's
   predicted stability respect the known motif? A per-allele motif-vs-prediction agreement figure is
   directly interpretable to an immunologist. ~1h.
3. **Supertype generalisation** (see §1.2), which doubles as the mechanism test for the B1 vs B1'
   gap.
4. **Frame the within-allele metric biologically** in the write-up: ranking candidates *inside one
   patient's allele* is the clinical decision, which is why pooled Spearman flatters every model in
   this literature. We have the measurement (B0b = 0.563 with no peptide information); the framing
   should be explicit.

---

## 6. The demo / app

Not started, 20% of the rubric, and currently unowned.

**Minimum defensible demo (90 s):**
peptide + allele in -> predicted half-life **with a bootstrap interval** -> the P1-P9 ablation
profile with the anchors lit up -> the Boltz pose beside it. Three real outputs, one screen, all
backed by numbers in `results/`.

**What makes it a *good* demo rather than a wrapper:**

- Show the **baseline alongside the model**. A UI that displays "our model: 12 h; allele median:
  9 h" makes the project's whole argument visible in one line.
- Make the **negative result** the interface. Let the user toggle "use the foundation model" and
  watch the prediction barely move. That is memorable, honest, and nobody else will do it.
- Show the **interval**, not just the point. It is the difference between a demo and a toy.
- A **failure case** on screen, chosen in advance, not hidden.

---

## 7. If I had to pick: the ranked plan

| # | Item | Cost | Why |
|---|---|---|---|
| 1 | **Shuffled-label negative control** (T6) | 20 min | Cheapest credibility item that exists; its absence is conspicuous |
| 2 | **Layer sweep** (T1) | 40 min | Could make the headline wrong. Must be checked before we publish it |
| 3 | **Cost-per-Spearman table** (§2.5, §4.1) | 15 min | Directly answers the research question; data already measured |
| 4 | **Seed disagreement vs error** (§3) | 20 min | Already computed and discarded; buys "internal confidence", which the brief names |
| 5 | **Scale sweep 8M-650M** (T2) | 1.5h | Turns a local negative into a general one |
| 6 | **Allele-split replication** (T3) | 1h | Robustness to split criterion, and may flip the result |
| 7 | **Demo** | 2h | 20% of the rubric, unowned |
| 8 | **Test-set single read** | 20 min | Needs a decision, then the model choice freezes |
| 9 | Hours-MAE + top-k precision | 45 min | Operational metrics the brief's framing implies |
| 10 | Pocket mapping / motif agreement | 2h | Deepens the strongest existing result |

**1-4 total about 1.5 hours and would measurably harden the submission.** 5 and 6 are the two that
could change what we are claiming, which is why they are worth more than their cost suggests.

The thing I would protect at all costs: **do not let the layer sweep go unrun and then publish
"foundation models do not help here".** That is the one finding that a reviewer could overturn in
twenty minutes, and it would discredit everything else by association.
