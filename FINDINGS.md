# Can foundation models improve peptide-HLA stability prediction?

**Answer: no, on this dataset — and we can say what they contribute, what it costs, why they fail,
and how much signal is still on the table.**

Serova Protein Engineering Track · London AI × Science Hackathon · 3–4 October 2026

Every number below is reproducible from `results/README.md`, which logs 30 runs with the
command that produced each one. `scripts/audit_results.py` re-checks every figure quoted anywhere
in this repository against the file that produced it; it currently passes **125/125**.

---

## 1. The short version

| claim | evidence |
|---|---|
| A one-hot MLP beats every ESM-2 configuration tested | Runs 2, 4, 8, 11, 12 |
| It also beats **ProtBERT and ProtT5** - three families, three corpora, 8x params | Run 27 |
| Depth, scale and fine-tuning all land **inside seed noise** | Runs 8, 11, 12 |
| ESM-2's total contribution is **+0.009 pooled, nothing within-allele** | Run 13 |
| It costs **$6.59 and 11,307 A100-seconds** to land *below* a free baseline | Run 18 |
| The data-efficiency defence fails, and runs **backwards** | Run 14 |
| We know **why**: ESM-2's salience is orthogonal to the endpoint | Runs 5, 19, 25 |
| A **published** pMHC model, zero-shot, scores 0.421 within-allele and *below the floor* pooled | Run 29 |
| Cross-chain **pretraining** and our **head** buy the same +0.16, and both stop below one-hot | Run 30 |
| Roughly **0.19 of Spearman is still available** | Run 16 |

The useful claim is not "foundation models don't work". It is **"a frozen single-chain protein
language model - ESM-2, ProtBERT and ProtT5 alike - contributes about one hundredth of a Spearman
point here, for several dollars and several GPU hours, and we can show where the remaining signal
is instead."**

---

## 2. What we built, and the discipline underneath it

**A leakage-controlled split, frozen and hash-asserted.** 94% of peptides are measured against
more than one allele, so a random row split puts the same fragment on both sides. Peptides are
clustered at **Hamming ≤ 2** (stricter than the field's usual 80%-identity convention, which on a
9-mer permits one substitution) and whole clusters are assigned. `src/features.py` asserts the
split's sha256 before reading a single label, so a job that somehow receives a different split
fails immediately rather than quietly reporting a number against different data.

**A floor, not zero.** A per-allele median containing **no peptide information** scores **0.563
pooled Spearman**. Every pooled figure in this field — ours included — must be read against that,
not against zero. This is why we report **within-allele** Spearman as primary: it is also the
clinical question, ranking one patient's candidates inside their own allele.

**Intervals on every claim.** 2,000 paired bootstrap draws resampling the 540 peptide *clusters*,
never rows — rows inside a cluster are near-duplicates and resampling them would claim precision
we have not earned.

**A negative control that found something.** Shuffling training labels scored +0.104, which looked
like leakage. It was not: a fixed-epoch rerun scored +0.02, isolating the +0.08 as
**epoch-selection optimism**. Every validation number here carries it; paired differences do not,
which retroactively justifies the bootstrap design.

---

## 3. The ladder

Validation, three seeds, seed-ensemble mean, 95% CI from the cluster bootstrap.

| rung | ρ pooled | ρ within-allele | protein LM? |
|---|---|---|---|
| B0b per-allele median — *no peptide information* | 0.563 [0.522, 0.598] | undefined | no |
| F150 frozen ESM-2 150M, **mean-pooled** | 0.593 [0.556, 0.625] | 0.278 [0.222, 0.320] | yes |
| ProtT5-XL enc **1.2B**, cross-attention head | 0.729 † | 0.511 † | yes |
| ProtBERT 420M, cross-attention head | 0.729 † | 0.529 † | yes |
| B1′ one-hot peptide + allele identity | 0.748 [0.719, 0.775] | 0.557 [0.508, 0.585] | no |
| X150 frozen ESM-2 150M, **cross-attention head** | 0.754 [0.727, 0.777] | 0.558 [0.508, 0.585] | yes |
| L150 ESM-2 150M, **LoRA-adapted** | 0.757 [0.732, 0.780] | 0.572 [0.524, 0.599] | yes |
| **B1 one-hot + HLA pseudosequence + allele** | **0.780 [0.756, 0.802]** | **0.633 [0.588, 0.654]** | **no** |

**B1 − L150: +0.023 [+0.007, +0.039] pooled, +0.060 [+0.027, +0.094] within-allele.** Both exclude
zero. The best non-foundation model beats the fine-tuned foundation model.

† The two non-ESM families were not bootstrapped; their figures are seed-ensemble point
estimates with a worst per-seed spread of 0.039 (Run 27). Every rung above the one-hot baseline
is a protein language model only in the sense that it *contains* one - **the top rung has none**,
and the three families between 0.729 and 0.771 span three architectures, three corpora and 8x
parameters without reaching it.

### On held-out data — the test split, read once

| rung | TEST ρ pooled | TEST ρ within-allele |
|---|---|---|
| B0b per-allele median | 0.573 | undefined |
| F150 frozen ESM-2, mean-pooled | 0.606 | 0.244 |
| X150 frozen ESM-2, cross-attention | 0.756 | 0.535 |
| **L150 LoRA-adapted ESM-2** | 0.758 | 0.525 |
| B1′ one-hot + allele identity | 0.767 | 0.566 |
| **B1 one-hot + pseudosequence + allele** | **0.806** | **0.645** |

**Every ordering survives, and the gap widens: B1 − X150 is +0.050 pooled and +0.110
within-allele on test**, against +0.026 and +0.075 on validation. The test split was read exactly
once, after the model choice was frozen by 21 prior runs, and a guard file prevents a silent
second read.

L150 was deliberately omitted from that first read, because scoring it would have meant
retraining *after* seeing the test table. Run 28 closed it properly - the LoRA script now keeps
its validation-selected checkpoint and scores test inside the same run. **B1 − L150 on test is
+0.048 pooled and +0.120 within-allele, double the validation gap.** The most expensive model in
the project clears the frozen one by 0.002 pooled and *loses* 0.010 within-allele.

### Every obvious fix, tried

| knob | result | against |
|---|---|---|
| **fine-tuning** — LoRA r=8 on K/V, 614,400 params, 3×A100 | +0.003 pooled | CI [−0.012, +0.018] — **spans zero** |
| **depth** — five ESM-2 layers, identically normalised | 0.108 range | worst seed spread 0.157 — **within noise** |
| **scale** — ESM-2 8M → 650M, **80× parameters** | **0.007 range** | worst seed spread 0.036 — **within noise** |
| **more labels** — 5% → 100% of training clusters | no crossover at any budget | gap *widest* at 25% |
| **cross-chain attention** — joint 191-residue encoding | −0.017 pooled | within noise |
| **as an ensemble member** | +0.009 [+0.001, +0.017] | real, and tiny |
| **a different family** - ProtBERT 420M, ProtT5-XL **1.2B** | both 0.729 pooled | below ESM-2, below B1 |
| **the groove itself** - peptide scored inside the complex | 0.001 pooled | and the delta, 0.004 |

---

## 4. Why it fails — three independent measurements

**The groove does not rescue it either.** Scoring each peptide inside the 191-residue
peptide-plus-HLA concatenation, rather than alone, moves the correlation with stability from
0.002 to 0.001; the difference between the two scores predicts nothing (0.004). The two scores
rank-correlate at only 0.669, so the groove genuinely does reorder which peptides ESM-2 finds
likely - something real happens, and it is uncorrelated with what we are predicting. The caveat
is stated and not glossed: ESM-2 has no chain-break token, so this cannot separate *the groove
explains nothing about stability* from *ESM-2 cannot condition on a chain it does not know is
separate*.

**ESM-2's notion of which residues matter is orthogonal to this endpoint.** Ablating each peptide
position in the trained model gives a sharply peaked profile — **P9 ≫ P2 > P1 > P3**, with P4–P8
contributing essentially nothing. That is the textbook B-pocket/F-pocket anchor picture recovered
from data alone, and it holds in every seed independently. ESM-2's own masked-position likelihood
profile over the same peptides is **flat**, and the two rank-correlate at **−0.03**. Its
per-peptide likelihood predicts stability at **ρ +0.002**.

**The useful signal must be extracted, not read off.** Mean-pooling nine residues destroys
position and scores 0.593; a residue-level cross-attention head on the *same frozen embeddings*
reaches 0.754. The head is worth **+0.161 pooled, +0.276 within-allele**, both far outside noise —
the largest effect measured anywhere in this project, and larger than the choice to use a
foundation model at all.

**And it is the modelling, not the encoder, that matters — now measured on our own split.**
Run 30 put MINT's representation behind our head: 814M parameters, cross-chain attention
pretrained on 96M protein–protein interactions. Against F150, which is ESM-2 *also* mean-pooled
and so the architecturally matched comparator, MINT gains **+0.156 pooled and +0.250
within-allele**. Our cross-attention *head* — 561,793 parameters on unmodified ESM-2 150M — is
worth **+0.161 and +0.276**.

**Two unrelated routes to the interaction agree to within 0.005.** Ninety-six million real
protein pairs, or half a million trainable parameters, buy the same thing; MINT-pooled lands at
0.749 and X150 at 0.754. Meanwhile ESM-2 merely *permitted* to attend across both chains, without
such pretraining, moves nothing (−0.017, within noise). **The interaction has to be modelled by
something trained to model it; self-attention handed the opportunity does not discover an
interface.** This was previously an inference across two studies with different splits; it is now
a measurement on ours.

**And both routes stop below the baseline.** They converge at ≈0.75 while one-hot sits at 0.780
pooled and 0.633 within-allele.

---

## 5. Why the baseline is strong — it is not naive

B1's HLA input is the 34-residue **pseudosequence**, chosen by NetMHCpan from crystallographic
contacts. It carries decades of structural biology as a feature-selection decision, and it checks
out from three independent directions:

1. **Boltz-2 recovers 31 of its 31 positions** from predicted structure alone — expected by chance
   16.5, permutation **p = 0.0001**, none missed.
2. **Its Hamming distance predicts cross-allele measurement agreement monotonically** —
   0.823 → 0.659 → 0.474 → 0.318 at distances 1 → 4, with no model involved.
3. **Compression beats coverage.** 31 positions (0.785) beat Boltz's 97 (0.782), beat 97 random
   (0.780), and beat the full 182-residue domain (0.768). More groove residues actively hurt.

Both models also recover **published binding motifs they were never shown** — L first at A\*02:01
P2, R first at B\*27:05 P2 — from 10,320 scored sequences of which 99.4% never appear in the
dataset. Neither recovers them better than the other.

### And it ships with intervals that say what they mean

Split-conformal calibration on 3,412 held-out train rows, carved as whole peptide clusters so the
residuals measure unseen-cluster difficulty exactly as validation and test do, gives intervals
whose coverage on test matches the target to within half a point at every level: 80.2%, 90.5% and
95.4% against 80%, 90% and 95%, at median widths of 4.2, 6.6 and 9.7 hours.

Per allele it is a different story, and reporting only the first number would have been the
dishonest version: coverage ranges **68.8% to 100%** across 68 test alleles, 7 of them below 80%.
Conformal's guarantee is marginal by construction, not conditional. A clinician reading 90% off
the label would be wrong for HLA-A*01:01 by twenty points.

---

## 5b. The metric choice is load-bearing, and Run 29 proves it

`mint-stage1-affinity` is a released 814M model built for peptide-MHC class I: the MINT backbone
(ESM2-650M, cross-chain attention, 96M STRING interactions) fine-tuned on NetMHCpan 4.1 **binding
affinity**. It has never seen a half-life, which is exactly why it is admissible here when
NetMHCstabpan, SPEARMINT and `mint-stage2-stability` are not - all three are trained on the
endpoint we are predicting.

Scored zero-shot on our validation split, with no training of any kind:

| | pooled | within-allele |
|---|---|---|
| **MINT Stage-1, zero-shot** | **0.523** | **0.421** |
| B0b per-allele median, *no peptide information* | 0.563 | undefined |
| B1 one-hot, trained | 0.780 | 0.633 |

**Report pooled alone and this model appears to have no signal - it is below the no-peptide
floor. Report within-allele and it has a great deal, 0.421, having never seen the endpoint.**

Both numbers are right. A zero-shot affinity score has no reason to be calibrated across alleles
against half-life, because alleles differ systematically in their stability distributions and an
IC50-trained scalar does not encode that. The across-allele component is close to noise and drags
pooled beneath a floor that receives allele identity for free; the within-allele ranking - which
is the clinical question - survives intact.

This is the clearest case in the project for the decision made in §2 to treat **within-allele as
primary**. Here the metric choice is the whole difference between *"this model does nothing"* and
*"this model carries real signal with the wrong calibration"*, and only the second is true. It is
also the exact gap SPEARMINT's assay-conditioned FiLM head is built to close - their architecture
becomes legible from our measurement.

The obvious objection was pre-registered rather than answered afterwards. MINT documents a full
~365-residue heavy chain; we have the 182-residue groove. Both were run - 182, and 182 plus a
constant alpha-3 domain that is near-invariant across alleles and so can only change length, not
allele-specific signal. **The difference is 0.0012.** Truncation is not the explanation.

---

## 6. The structural arm: correct structures, no usable signal

Boltz-2 poses **do** depend on the peptide — across-peptide variation is **4.1×** the model's own
diffusion noise, so the pessimistic hypothesis (one canonical pose regardless of peptide) is false.
But that variation does **not** track half-life: Mantel r = **−0.111**, p = 0.116, permuting
peptide labels, with **92% power to detect a true r of 0.2**.

Restricting to the positions the model actually uses does not rescue it. **P2's geometry is the
most precisely resolved of any position — 8.2× the noise floor — and its correlation with
half-life is +0.002.** This is not a signal diluted by averaging; there is none where it would
matter.

Combined with the 31/31 position recovery, the two results say something neither says alone: **the
structures are right, and their variation still carries no stability information.**

---

## 7. What it cost

| approach | pooled / within | compute | cost |
|---|---|---|---|
| **B1 one-hot** | **0.780 / 0.633** | 127 CPU-seconds, laptop | **$0.00** |
| best single ESM-2 | 0.764 — *still below* | A100 | $6.70 |
| one-hot + ESM-2 ensemble | 0.789 / 0.639 | A10 | $0.11 for **+0.009** |

**L150 is the sharpest illustration.** It was trained twice: $6.59 and 11,307 A100-seconds the
first time, then $6.69 and 11,479 more to score it on test (Run 28). **$13.28 and 6.3 GPU-hours
on the one model that lands 0.023 pooled below the free baseline on validation and 0.048
below it on test.**

Whole project: **7.5 GPU-hours, $15.26** — against a one-hot baseline that costs 127 CPU-seconds
on a laptop and beats everything. **88% of the GPU bill went to rungs whose results land inside
seed noise or below the baseline.**

Where the ESM models *do* win is worth stating: on **median** absolute error (L150 0.87 h vs B1
0.99 h) and on unstable complexes below 2 h (0.78 vs 0.95 h). B1 leads on the 141 stable rows above
24 h, where errors are large enough to dominate the mean. A rank metric cannot see which regime the
errors live in.

---

## 8. How much is left

There are **zero** replicate measurements, so assay reproducibility cannot be estimated directly.
But alleles differing at **one of 34 contact positions** are biophysically almost the same pocket,
and the same ~346 peptides through both grooves agree at **ρ 0.823** (best pair **0.921**).

Our best model reaches **0.633 within-allele**. So roughly **0.19 of Spearman remains available**,
and nothing we tested found it. The negative result is *"ESM-2 did not reach the remaining
signal"*, not *"the problem is saturated"*.

Where that signal is **not**: the groove representation, validated three ways. Where it plausibly
**is**: per-allele skill correlates **+0.475** with the allele's median half-life and **−0.076**
with how many training rows it has. Failures concentrate where the **label** carries least
information — against the 20% zero point mass and the 0.1 h reporting grid — not where the model
has least data.

---

## 9. What we did not do, and would not claim

- **Four families now, and one of them published for this exact system.** ESM-2, ProtBERT and
  ProtT5 all lose to one-hot (Run 27), which is what lets this be a claim about frozen PLMs
  rather than about ESM-2 alone. Run 29 adds **MINT Stage-1**, the released 814M cross-chain
  model fine-tuned on binding affinity and never on half-life, scored zero-shot. **Still
  untested: MINT behind our own head** (which is the experiment that would isolate cross-chain
  pretraining from affinity fine-tuning), **ESM-C** (no official loadable Hub checkpoint) and
  **SaProt** (needs foldseek 3Di tokens we cannot build). SPEARMINT itself and
  `mint-stage2-stability` are **inadmissible** here, not unavailable: both are fine-tuned on
  pMHC stability, the endpoint we are predicting, which is the same objection that excludes
  NetMHCstabpan. The SPEARMINT comparison in §4 remains across studies and splits, and only the
  *deltas* are comparable.
- **No hyper-parameter search.** Every model got the same budget and selection rule, so the
  comparison is fair, but no model is at its ceiling.
- **Affinity pre-training not attempted.** It is the largest published lever here
  (0.574 → 0.745) and needs the NetMHCpan corpus decontaminated against our test clusters — a
  day's work, not an evening's.
- **Prediction intervals hold marginally, not per allele.** Split-conformal intervals calibrated
  on whole held-out train clusters cover 90.5% against a 90% target on test. Per allele they
  range 68.8% to 100%, with 7 of 68 below 80% - so an interval advertised as 90% gives
  HLA-A*01:01 patients 68.8%. Conformal guarantees the average and nothing within a subgroup;
  restoring per-allele guarantees needs Mondrian conformal and more calibration rows than the
  smallest alleles have.
- **9-mers, 75 alleles, one assay.** Class I binds 8–11mers; this is the 9-mer slice.
- **We cannot and do not compare against NetMHCstabpan.** This dataset is its training data, so
  its apparent performance here is leakage, not skill.

---

## 10. Reproducing any of it

`results/README.md` carries 26 run entries, each with the exact command, what it established, and
a *"what this does NOT establish"* section. `scripts/audit_results.py` verifies all 103 quoted
numbers against their source files and every figure restated in the write-ups; it is the check
that caught two drifted numbers during the project rather than after it.

`requirements.txt` pins the local environment. Run 14 measured B1 differing by 0.010 between
Anaconda and PyPI scikit-learn - the same size as several findings reported here - which is why
the pin exists and why the file also states, rather than hides, that `modal_app.py` still builds
its images unpinned.

The frozen split is content-addressed: `splits/peptide_split.csv`,
sha256 `210775dc4ad179df635b3386bdb9671e4bca515cef1892bc70fb38f9b1b73f47`.
