# 000 — Simple supervised baseline  ·  rungs B0 / B1 / B1′

> **B0** training-only median, **B1** positional one-hot MLP (180 peptide + 680 pseudosequence values)
> **+ allele identity**, **B1′** the same with HLA replaced by a one-hot over 75 alleles.
> Naming follows `../ARCHITECTURE.md` §3.

**Owner:** Claude · **Status:** **done** · **Time box:** 2h · **Compute:** CPU or one small GPU, ~free

> The brief asks for this by name: *"A simple supervised neural network trained on peptide and HLA
> pairs would be a helpful baseline to produce to contextualise the results on your splits."*
> Every other experiment is measured against this. It should be built first and by whoever is free
> earliest.

## Hypothesis
A small supervised network over one-hot / BLOSUM-encoded peptide and HLA pseudo-sequence features
reaches useful accuracy on our split, and establishes the number any foundation model must beat to
have earned its compute.

## What would prove me wrong
Nothing — this is a measurement, not a claim. But it fails as a baseline if it is tuned so hard that
it stops being "simple", or so carelessly that it is a straw man. Both make the comparison
dishonest. Aim for a defensible, obvious architecture with light tuning, and say what tuning you did.

## Method
- Inputs: peptide (9 residues) + `hla_pseudoseq` (34 residues), one-hot or BLOSUM62-encoded.
- Model: MLP, or a small 1D-CNN. A few hundred thousand parameters, not millions.
- Target: `log1p(thalf_hours)`, MSE. Invert with `expm1` to report hours.
- Train on the frozen split from `splits/`. No peptide in validation or test may share a cluster with
  a training peptide.
- Also record: training wall-clock, parameter count. The brief weighs "engineering & compute
  requirements", so a baseline that trains in 90 seconds is itself an argument.

## Metric and comparison
Spearman ρ as primary (rank correlation survives the assay-scale problem; see `../context/NOTES.md`).
Also report Pearson r on the log target and log-RMSE. This experiment *is* the comparison point.

## Cost
Negligible. Should run on a laptop.

---

## Result

**Outcome:** done, 2026-10-03. Full evidence in `../results/README.md` Run 2. Three seeds
(42/43/44); the reported model is the seed-ensemble mean. Validation rows only — the test split has
not been read by anything.

**Numbers:**

| Model | ρ pooled | ρ within-allele | Pearson r (log) | log-RMSE | params | train time |
|---|---|---|---|---|---|---|
| B0a global median | undefined | undefined | undefined | 1.143 | 1 | instant |
| B0b per-allele median | **0.563** | undefined | 0.499 | 0.977 | 75 | instant |
| B1 one-hot + pseudoseq + allele | **0.780** | **0.633** | 0.791 | 0.666 | 256,129 | 38–47 s/seed |
| B1′ one-hot peptide + allele id | 0.748 | 0.557 | 0.758 | 0.712 | 82,049 | 11–21 s/seed |

**What actually happened:**

The baseline turned out to be the most important number in the project, for a reason that was not
anticipated when this was written.

**B0b scores 0.563 pooled Spearman with no peptide information whatsoever.** It predicts the
training median of whichever allele a row belongs to. Its within-allele correlation is *undefined by
construction* — it emits a constant inside each allele, so there is no variance to rank. Pooled
Spearman on this endpoint is therefore mostly measuring *which groove*, not *which peptide*, and
every pooled figure in this field needs that floor underneath it. B1's 0.780 is a gain of **+0.217
over knowing nothing about the peptide**, which is the honest framing.

Two secondary findings:

- **The HLA sequence beats the allele label, but half as much as one seed suggested.** B1 − B1′ is
  +0.032 pooled and +0.076 within-allele, against seed spreads of 0.014 and 0.010. A single-seed run
  had put those at +0.063 and +0.115 — roughly double. One run gives a value; three tell you whether
  it means anything. This is why every rung since has been run at three seeds.
- **Seed ensembling is free accuracy.** Individual B1 seeds score 0.751–0.760 pooled; their mean
  scores 0.780. About +0.025 for no extra training.

Caveat on the mechanism: the pseudosequence is *constant within an allele*, so it cannot directly
rank peptides inside one. The plausible explanation for the within-allele gain is parameter sharing
— the model learns a peptide-residue/groove-residue interaction and transfers it across similar
alleles — but that is a hypothesis, untested here.

**What I'd do with more time:** paired cluster bootstrap on the saved predictions for real intervals
(`paired_bootstrap()` in `../scripts/power_analysis.py`); and check the parameter-sharing hypothesis
by regressing per-allele gain against allele similarity.

**Does this change the team's answer to "are foundation models useful here"?** Decisively, and not
in their favour. B1 uses no protein language model and scores 0.780 / 0.633. As of experiment 001,
no frozen ESM-2 rung has matched it. See `001-esm2-embeddings-head.md`.
