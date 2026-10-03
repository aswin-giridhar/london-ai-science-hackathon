# 002 — Zero-training likelihood scoring

**Owner:** Claude · **Status:** **variant 1 (unconditioned) done; variants 2-4 not run** · **Time box:** 3h · **Compute:** measured - 91.5 s on an A10, shared with the Run 5 anchor probe

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

**Outcome:** partial, 2026-10-03. The masked-position scoring ran; the HLA-conditioned variants did
not. Evidence in `../results/README.md` Run 5 and `../results/anchors.json`.

**Numbers:**

| Scoring variant | Spearman rho | notes |
|---|---|---|
| peptide PLL, **no HLA context** | **+0.002 pooled, -0.006 within-allele** | a clean null |
| peptide PLL **in HLA context** | not run | needs a two-chain model, see below |
| context minus no-context | not run | same blocker |
| seed/mask spread | not run | — |

For scale on the identical rows: B0b 0.563 · F150 0.593 · X150 0.754 · B1 0.780.

**What actually happened:**

The hypothesis was that a protein language model has internalised what a stable complex looks like,
and that this would be visible in the likelihood it assigns. **It is not.** Not weakly - the
correlation is +0.002, which is zero to three decimal places.

This file said in advance that a clean null "is a *direct* answer to the brief's question and worth
writing up carefully". It is, and it gains force from a second, independent probe run at the same
time (Run 5, Probe A). Ablating each peptide position in the trained X150 gives a sharply peaked
profile - P9 > P2 > P1 > P3, with P4-P8 contributing essentially nothing, which is the textbook
B-pocket/F-pocket anchor picture recovered from data alone. ESM-2's own per-position likelihood
profile over the same peptides is **flat**, and the two profiles rank-correlate at **-0.03**.

So two different measurements agree on one conclusion: **ESM-2's intrinsic notion of which residues
matter is unrelated to which residues govern complex stability.** That is mechanistically
unsurprising in hindsight - a 9-mer is not a protein, and "peptide sitting in an MHC groove" is not
a configuration the masked-language objective was ever optimised for - but it is now measured rather
than assumed, and it explains the Run 4 result instead of merely restating it.

**Why the conditioned variants were not run.** Variants 1-proper and 3 need the peptide scored *in
the groove*. The obvious implementation is to concatenate peptide and HLA into one sequence, and the
repo README already cautions that concatenation does not establish biological conditioning - ESM-2
has no notion that these are two chains in contact. Running it would produce a number that looks
like an answer and is not one. A real conditional test needs a model that accepts two chains.

**What I'd do with more time:** run the conditioned variants with a genuinely multi-chain model;
add a cluster bootstrap CI on the null so "zero" comes with a width; and test whether PLL predicts
anything *within* the zero-label class, where a different mechanism may be at work.

**Does this change the team's answer to "are foundation models useful here"?** It sharpens it. The
answer is not just "the frozen embeddings underperform one-hot" but *why*: the model's salience is
orthogonal to the endpoint's. The useful signal in ESM-2 has to be extracted by a supervised head
with per-residue access - it is not sitting in the likelihood, and it is not sitting in a mean-pool.
