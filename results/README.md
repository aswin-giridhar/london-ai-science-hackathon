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

## Run 2 — baselines, three seeds · 2026-10-03 21:42 — CURRENT

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

## Files

| File | What |
|---|---|
| `baselines_val_metrics.json` | All metrics per rung, including per-seed values and spread |
| `baselines_val_predictions.json` | Validation predictions per rung, aligned with peptide / allele / cluster_id — the input `paired_bootstrap()` needs |
