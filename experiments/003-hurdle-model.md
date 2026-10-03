# 003 — Hurdle model for the zero class

**Owner:** Claude · **Status:** **done — negative** · **Time box:** 1h · **Compute:** measured, 287 s CPU

> A *declared experiment*, not an assumed correction. `../ARCHITECTURE.md` §6 is explicit: "Do not
> add a censored head by default. The CSV does not establish a detection limit, and a hurdle model
> is a separate empirical experiment rather than an automatic correction."

## Hypothesis
`t½ = 0` is a genuine point mass — 5,679 rows against 443 at 0.1 and 1,297 at 0.2, 13× its
neighbour and non-monotonic. A single regressor on `log1p(t½)` treats those as merely small values
and MSE drags its predictions toward them. Splitting the problem — a classifier for *is it zero?*
and a regressor for *how long, given it isn't* — should beat one regressor doing both jobs.

## What would prove me wrong
No improvement on either framing. That is a perfectly good answer and the one that should be
reported if it happens: it would say the point mass needs no special treatment, and it would
retire a modelling decision rather than leaving it open.

## Method
- Both arms use the **B1 feature set**, the same frozen split, the same three seeds, and the same
  manual early-stopping loop — so the only difference is the hurdle structure.
- Classifier selected on **AUC**, not Spearman: its job is separation, and selecting it on the
  regression metric would optimise the wrong thing.
- The non-zero regressor's epoch is chosen on non-zero validation rows only, so the zeros touch
  neither its fit nor its selection.
- Combined as `P(non-zero) × E[log1p t½ | non-zero]`.
- Scored three ways, because a hurdle model can win on one and lose on another.

---

## Result

**Outcome:** done, 2026-10-04. Negative. Evidence in `../results/README.md` Run 6.

**Numbers:**

| scored on | PLAIN | HURDLE | delta |
|---|---|---|---|
| all 2,817 rows, ρ pooled | **0.780** | 0.779 | −0.001 |
| all 2,817 rows, ρ within-allele | **0.633** | 0.621 | −0.012 |
| 2,238 non-zero rows, ρ pooled | **0.730** | 0.729 | −0.001 |
| 2,238 non-zero rows, ρ within-allele | **0.618** | 0.600 | −0.018 |

Zero vs non-zero classifier: **AUC 0.886** (0.889 / 0.885 / 0.882).

**What actually happened:**

The hypothesis was wrong, and the way it was wrong is the interesting part.

The classifier works well — AUC 0.886 means the features separate the point mass from the
measurable rows with little trouble. So it is not that the zeros are unpredictable. And yet
splitting the problem in two gains **nothing**: −0.001 pooled on both framings, and slightly worse
within-allele.

Taken together those say the single regressor has already absorbed whatever the classifier knows.
The zeros are not a separate regime that the plain model was failing to represent; they are part of
a continuum it already handles. The two-part model adds machinery and loses a little in the
recombination.

**What I'd do with more time:** a cluster bootstrap CI on the delta (it is far inside any plausible
interval, so the conclusion is safe, but the number should carry a width); and the same test on
X150 embeddings rather than B1 one-hots, in case a richer representation separates the regimes
where one-hots cannot.

**Does this change the team's answer to "are foundation models useful here"?** No — it is orthogonal
to that question. What it does is **close a modelling decision**: do not add a censored or hurdle
head, and say why, from a measurement rather than from caution.
