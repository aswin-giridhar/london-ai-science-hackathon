# Can our evaluation resolve the differences we care about?

**Short answer: the honest one is "we will know once two models exist."** A first attempt to answer
it synthetically produced a per-metric table that did not reproduce, and that table is retracted
below rather than quietly removed.

Measured with `scripts/power_analysis.py`. Nothing is trained yet; everything here is a property of
the *evaluation*, not of any model.

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

## What is solidly established

On the frozen test split — **2,817 rows across 540 peptide clusters** — the 95% confidence interval
on a single Spearman is about **0.04–0.07 wide**, depending on accuracy:

| accuracy | 95% CI | width |
|---|---|---|
| ρ ≈ 0.45 | [0.428, 0.496] | 0.068 |
| ρ ≈ 0.60 | [0.582, 0.638] | 0.056 |
| ρ ≈ 0.75 | [0.739, 0.782] | 0.044 |

Reproduced across four independent runs. **What it supports:** two *independently reported* Spearman
values need to differ by roughly **0.05** before the difference means anything.

That alone justifies the practice: **report a confidence interval on every number**, and never claim
a 0.02 improvement from two separately quoted figures.

## ⚠ Retracted: the per-metric sensitivity table

An earlier version of this document carried a table claiming that log-RMSE could resolve a +0.02
improvement where pooled Spearman needed +0.05, and that within-allele Spearman was least sensitive.
**That table is withdrawn.** It came from a single run. Re-running it did not reproduce, and the way
it failed was diagnostic — results were **non-monotonic in effect size**:

```
metric                    +0.01  +0.02  +0.03  +0.05     (n of 5 realisations resolved)
Spearman, pooled           1/5    3/5    5/5    4/5       <- +0.03 beats +0.05
Pearson on log             4/5    1/5    2/5    4/5       <- incoherent
```

A sound power analysis must be monotonic: a larger true effect cannot be harder to detect. Two causes,
both confirmed:

**Flaw 1 — the metrics were never on a common footing.** Noise was tuned to hit a target *Spearman*,
then Pearson and RMSE were measured on the result. Those were uncontrolled. The tuning itself also
scattered: a target of 0.60 realised as **0.594 ± 0.019**, which is as large as the +0.01–0.02 gaps
the table claimed to detect. The "true gap" was never the stated gap.

**Flaw 2 — the simulated models were independent, and real ones are not.** Baseline and improved
predictions used separate noise draws, giving an error correlation of **+0.010**. Two real models
trained on a shared feature matrix typically correlate **above +0.8**, and a paired test on
correlated errors is far more powerful.

**The direction of that second error matters: the retracted table *understated* our resolution.**
The earlier warning — that the two rows we care about most might be unresolvable — was too
pessimistic. A paired comparison between two models that share features will resolve finer than the
0.05 implied by comparing two intervals. How much finer depends on how correlated their errors turn
out to be, which cannot be guessed in advance.

## What to do instead

`scripts/power_analysis.py` now exposes the function to call once two models exist:

```python
mean_diff, lo, hi = paired_bootstrap(y, pred_B1, pred_F0, clusters)
```

Both models scored on the *same* resampled clusters, every time. A CI excluding zero means the
difference survived; one spanning zero means **inconclusive**, which the brief explicitly accepts as
a result.

Three practices stand regardless:

1. **Report Δ with its confidence interval on every ablation row.** Never a bare number.
2. **State the resolution you actually achieved**, computed from real residuals, in the write-up.
3. **Do not fix a wide interval by re-splitting.** That would break the frozen split and cost more
   than it buys.

## On the three metrics

The retraction removes the evidence for ranking them, so the earlier recommendation to let log-RMSE
lead the ablation table is withdrawn too. What remains is the reasoning that does not depend on the
simulation:

| Metric | Role | Why |
|---|---|---|
| **Spearman, pooled** | Headline, comparable to the literature | Assays disagree on absolute hours but rank consistently. Confounded by between-allele differences, so never alone |
| **Spearman, within-allele** | Confound check | Unrelated pairs score 0.335 pooled against 0.027 within-allele, so the confound is real and measured. But with ~38 test rows per allele each estimate is noisy — treat a flat result as weak evidence, not as proof of no effect |
| **log-RMSE / MAE in log space** | Absolute accuracy | Rank metrics hide calibration failure: NetMHCstabpan ranks at 0.876 while its Pearson is 0.532 |

Which of these resolves a given difference most reliably is now an **open question to settle on real
residuals**, by running `paired_bootstrap` with each metric once B1 and F0 are trained. That is a
twenty-minute job at that point and it will be worth more than any amount of further simulation.

## What this does not cover

- **"Conclusions" and "evidence" cannot be scored yet.** Nothing is trained; everything so far is
  apparatus.
- **The demo is 20% of the event rubric** and currently has no owner.

---

Reproduce with `python scripts/power_analysis.py`. Split counts from `splits/peptide_split.csv`.
