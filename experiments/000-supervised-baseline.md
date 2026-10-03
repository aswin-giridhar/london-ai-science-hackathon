# 000 — Simple supervised baseline

**Owner:** — · **Status:** proposed · **Time box:** 2h · **Compute:** CPU or one small GPU, ~free

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
*Fill in after running.*

**Outcome:**

**Numbers:**

| Model | Spearman ρ | Pearson r (log) | log-RMSE | params | train time |
|---|---|---|---|---|---|
|  |  |  |  |  |  |

**What actually happened:**

**What I'd do with more time:**

**Does this change the team's answer to "are foundation models useful here"?**
