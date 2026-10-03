# Does our design actually answer the challenge question?

**Short answer: yes by construction — but our test split cannot resolve the two gaps we care about
most, unless we choose the metric deliberately.**

Measured 2026-10-03 with `scripts/power_analysis.py`. Nothing is trained yet; these are properties
of the *evaluation*, not of any model.

---

## Where the design does answer the brief

The challenge question is *"can existing foundation models improve performance of peptide-HLA
stability prediction?"* The **B1 → F0 gap in the ablation ladder is literally that question**: a
simple supervised neural net — which the brief asks for by name — versus frozen ESM-2, on the same
split, with the same head.

- **B1′, the one-hot allele control**, separates "protein knowledge" from "memorised allele
  identity". There are only 75 distinct HLA sequences in the dataset, so this matters.
- The frozen, audited split makes any answer honest.
- Compute discipline and a negative-result path are built in.

## The limit

On the frozen test split — **2,817 rows across 540 peptide clusters** — the minimum detectable paired
difference in pooled Spearman is **≈0.04–0.05**:

| true gap | 95% CI on Δ | resolvable? |
|---|---|---|
| +0.01 | [−0.037, +0.034] | no |
| +0.02 | [−0.003, +0.055] | no |
| +0.03 | [−0.015, +0.041] | no |
| **+0.05** | [+0.022, +0.075] | **yes** |
| +0.10 | [+0.057, +0.111] | yes |

Mapped onto the ablation rows:

| Ablation row | Expected gap | Resolvable? |
|---|---|---|
| F0 → F1, affinity pre-training | +0.171 published | ✅ comfortably |
| **B1 → F0, "does a foundation model help"** — *the actual challenge question* | unknown, plausibly small | ⚠️ maybe not |
| **mean → mean + std**, the novel structural claim | unknown, plausibly small | ⚠️ maybe not |

If a well-built one-hot-plus-allele baseline lands near 0.55 and frozen ESM-2 near 0.57, that +0.02
sits below our resolution. **The single most likely outcome on the two rows that matter most is an
honest "inconclusive at this sample size".**

## The metric choice changes the answer

Same test split, same simulated improvement, different metric. This is the part worth acting on:

| Metric | +0.01 | +0.02 | +0.03 | +0.05 |
|---|---|---|---|---|
| Spearman, pooled | no | no | no | **YES** |
| Pearson on log | no | no | **YES** | **YES** |
| **log-RMSE** | no | **YES** | **YES** | **YES** |
| Spearman, within-allele | no | no | no | no |

**log-RMSE resolves a +0.02 improvement that pooled Spearman cannot** — roughly 2.5× finer.

### This corrects something we had been assuming

We had been recommending *"always report within-allele Spearman, because pooled is confounded by
between-allele differences"*. That confound is real — unrelated pairs score 0.335 pooled against
0.027 within-allele. But **within-allele Spearman turns out to be the least sensitive metric of the
four**, unable to resolve even a +0.05 gap. The reason is sample size: test has ~38 rows per allele,
so each per-allele Spearman is noisy and averaging noisy estimates stays noisy.

Both facts are true at once, so the three metrics get three different jobs:

| Metric | Job |
|---|---|
| **log-RMSE** | **Detect** whether a feature block helps. Most sensitive, so this leads the ablation table |
| **Spearman, pooled** | **Compare** to the literature, which reports rank correlation. Confounded, so never alone |
| **Spearman, within-allele** | **Check the confound.** Directional only — it is underpowered, so a flat result here is not evidence of no effect |

## Why this is a strength, not a problem

The brief says a well-supported negative result counts as much as a positive one. **"Inconclusive,
with a stated minimum detectable effect" is a stronger result than an unqualified "+0.02".** Most
teams will report +0.02 as a win. Pre-computing what your own test set can and cannot distinguish is
precisely the *"watertight evaluation"* the evaluation criteria ask for.

Three things follow:

1. **Report Δ with its confidence interval on every ablation row.** Never a bare number.
2. **State the minimum detectable effect up front**, in the write-up, not an appendix.
3. **Do not fix this by re-splitting.** A larger test set would break the frozen split and cost more
   than it buys. Choose the sensitive metric instead.

## Caveats on the method

Predictions are simulated by adding Gaussian noise to the log target, tuned by bisection to hit a
target Spearman. Two limits worth stating:

- Real model errors are **structured** — heteroscedastic, allele-dependent, worse near the censored
  floor — while this noise is homoscedastic. Read the numbers as indicative of resolution, not exact.
- The simulated improvement is spread **uniformly across alleles**. If a real improvement is
  concentrated in particular alleles — a foundation model helping most on rare ones, for instance —
  the within-allele metric could be more sensitive to it than this shows.

Both caveats point the same way: re-run `scripts/power_analysis.py` against real residuals once the
first model exists, and replace these projections.

## What this does not cover

- **"Conclusions" and "evidence" cannot be scored yet.** Nothing is trained; everything so far is
  apparatus.
- **The demo is 20% of the event rubric** and currently has no owner.

---

Reproduce with `python scripts/power_analysis.py`. Split counts from
`splits/peptide_split.csv`; published gaps from Karthikeyan, Vincent & Rubinsteyn, bioRxiv 2026.
