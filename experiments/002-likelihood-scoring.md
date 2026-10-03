# 002 — Zero-training likelihood scoring

**Owner:** — · **Status:** proposed · **Time box:** 3h · **Compute:** inference only, no training

> The brief names this explicitly: the models *"produce embeddings, log-likelihoods / perplexities,
> and both internal and external confidence metrics… run with different inputs masked and different
> seeds."* The SPEARMINT preprint does not try it. This is the cheapest route to a genuinely novel
> result in the folder.

## Hypothesis
A protein language model's own log-likelihood for the peptide, conditioned on the HLA groove
sequence, correlates with measured stability — **without any supervised training on half-lives at
all.** If a general protein model has internalised what a stable complex looks like, that knowledge
should be visible in the likelihood it assigns.

## What would prove me wrong
Spearman ρ near zero, or negative. Entirely possible: the model was trained on natural sequence
statistics, and a peptide sitting in a groove is not the kind of pairing it ever optimised for.
A clean null here is a *direct* answer to the brief's question and worth writing up carefully.

## Method
Several scoring variants, cheap enough to try all of them:
1. **Pseudo-log-likelihood of the peptide** conditioned on the HLA sequence in the same context —
   mask each peptide residue in turn, sum the log-probabilities of the true residues.
2. **Perplexity** of the concatenated peptide + HLA sequence.
3. **Difference** between the peptide's likelihood in the HLA context and alone — how much does the
   groove "explain" the peptide? Arguably the most meaningful of the three, since it isolates the
   interaction rather than the peptide's own typicality.
4. Repeat with varied masking patterns and seeds; use the spread as an uncertainty estimate.

No training, no head, no fitted parameters — so the whole dataset is effectively a test set, though
report on the same frozen split for comparability.

## Metric and comparison
Spearman ρ against measured `thalf_hours`. Compare to exp 000 and 001. Report all variants including
the ones that failed — the comparison between them is itself the finding.

## Cost
Masked scoring is ~9 forward passes per peptide for variant 1 (one per masked position). About
5,600 peptides × 9 ≈ 50k short forward passes. Batches well; under an hour on one GPU.

---

## Result
*Fill in after running.*

**Outcome:**

**Numbers:**

| Scoring variant | Spearman ρ | notes |
|---|---|---|
| peptide PLL in HLA context |  |  |
| full-complex perplexity |  |  |
| context minus no-context |  |  |
| seed/mask spread (uncertainty) |  |  |

**What actually happened:**

**What I'd do with more time:**

**Does this change the team's answer to "are foundation models useful here"?**
