# 001 — ESM-2 embeddings + regression head

**Owner:** — · **Status:** proposed · **Time box:** 4h · **Compute:** 1 GPU for embedding extraction; head trains on CPU

## Hypothesis
Frozen ESM-2 embeddings of the peptide and the HLA α1/α2 domain contain stability signal beyond what
hand-crafted sequence features capture, so a shallow head on those embeddings beats experiment 000.

## What would prove me wrong
The head lands at or below exp 000's Spearman ρ. That is a publishable answer for this track, not a
failure — it would say a general protein model adds nothing here over domain features, which is
directly the question the brief asks.

A second, sharper falsification worth running: replace the HLA embedding with a **one-hot over the
75 alleles**. There are only 75 distinct HLA sequences in the whole dataset, so if one-hot matches
the embedding, the model is not using protein knowledge at all — it is memorising allele identity.
This is the single most informative control in the whole folder.

## Method
- Model: ESM-2 650M (`esm2_t33_650M_UR50D`). Note the brief also lists ESMC, ProtT5 and SaProt —
  if someone wants to compare backbones, that is a separate experiment.
- Encode peptide and HLA separately, mean-pool residue embeddings over each, concatenate.
- **Cache the HLA embeddings.** There are only 75 distinct α1/α2 sequences, so compute them once.
  The peptide side has 5,633. This makes the whole thing cheap. (Caching is only valid because the
  chains are encoded separately — it stops being valid for any cross-attention variant.)
- Head: 2-layer MLP → scalar. Target `log1p(thalf_hours)`, MSE.
- Same frozen split as 000.

## Metric and comparison
Spearman ρ against exp 000 on the identical split. Report the one-hot-allele control alongside.

## Cost
~5,700 ESM-2 forward passes for peptides plus 75 for alleles — minutes on one GPU, not hours.
Fits inside HF ZeroGPU or a small Modal instance.

---

## Result
*Fill in after running.*

**Outcome:**

**Numbers:**

| Variant | Spearman ρ | vs baseline 000 |
|---|---|---|
| exp 000 baseline |  | — |
| ESM-2 peptide + ESM-2 HLA |  |  |
| ESM-2 peptide + one-hot allele |  |  |
| one-hot peptide + one-hot allele |  |  |

**What actually happened:**

**What I'd do with more time:**

**Does this change the team's answer to "are foundation models useful here"?**
