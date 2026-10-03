# Experiments

Anyone on the team can propose an experiment. Copy `TEMPLATE.md`, fill it in, commit it. You do not
need permission and you do not need to have run anything yet — **a proposal is an idea with a
falsifiable claim attached**, not a finished result.

## Why this folder exists

Serova's brief judges us on *"how you tackle this question, your approach to the data, chosen
evaluation metrics, your conclusions, and the evidence you provide to support them."* It also says:

> Negative results are just as good as positive results where well supported.

So the submission is an **argument**, and this folder is where its evidence accumulates. An
experiment that failed and was written up is worth more to us than one that quietly vanished.

## Ground rules

**1. Everything reports on the same frozen split.** This is the one rule that matters. The brief
withholds splits deliberately and the whole evaluation rests on ours. **The split now exists:
`../splits/peptide_split.csv`** — identity-clustered at 80%, verified to have zero cross-split
neighbours. Nobody re-splits. If you think the split is wrong, open an experiment arguing that,
don't silently use your own. Numbers from different splits cannot be compared and will sink the
submission. See `../splits/README.md` for how to load it and what it guarantees.

Use `val` for model selection and early stopping; touch `test` once, at the end.

**2. Compare against the baseline, not against the literature.** The brief rules NetMHCstabpan out
as a comparator. Our own simple supervised net (exp 000) is the number everything is measured
against.

**3. Claim it before you start.** Put your name in the file and move it to `running` in the table
below. Three people independently building an ESM-2 head is the most likely way we waste this day.

**4. Time-box it.** Put a wall-clock budget in the proposal. If you blow through it, stop and write
up what you have — a half-finished experiment with an honest "inconclusive, here's why" is a
legitimate entry.

**5. Record the negative result.** If it didn't work, fill in the Result section and say so. Do not
delete the file.

## How to add one

```
cp TEMPLATE.md NNN-short-name.md     # next free number
```

Then add a row to the table below. Keep the proposal under a page — if it needs more, it's probably
two experiments.

## Before you start: two planning tracks are being reconciled

A second plan — `../SCIENCE_SKILLS_PLAN.md`, the Stability Lens / Google DeepMind Science-Skills
revision — was developed independently and pushed to this repo. The two agree on most things that
matter (not a leaderboard, negative results are valid, freeze model selection before test, one
locked split). **`../RECONCILIATION.md` records where they differ and what still needs deciding.**

Two points from it affect anything you build here:

1. **The split threshold is settled at Hamming ≤ 2, not ≤ 1.** The other plan specifies one-edit
   groups, which is the published 80%-identity convention. We measured it: a test peptide 2 edits
   from a training peptide has its label rank predicted at Spearman 0.700 by that neighbour, against
   0.311 at distance 3 and 0.027 for random pairs. `splits/` already uses the stricter threshold.
2. **Never report a pooled correlation on its own.** Alleles differ systematically in stability, so
   pooled numbers are partly a between-allele artefact — unrelated pairs correlate at 0.335 pooled
   and 0.027 within-allele. Report within-allele alongside pooled, every time.

## Where to get ideas

**`BRAINSTORM.md`** catalogues ~40 directions across baselines, representation routes, zero-shot
scoring, structure, the label floor, evaluation, interpretability, uncertainty and the demo — each
with what it would prove, a cost estimate, and how it could fail. It also lists the attractive traps.
Start there rather than from a blank file.

## Index

| # | Experiment | Owner | Status | Headline result |
|---|---|---|---|---|
| 000 | [Baselines — B0 / B1 / B1′](000-supervised-baseline.md) | Claude | **done** | **B0b scores 0.563 pooled with no peptide information** — the floor every pooled figure must be read against. B1 0.780 / 0.633 |
| 001 | [ESM-2 — F150 / X150 / L150](001-esm2-embeddings-head.md) | Claude | **F150/X150 done, L150 running** | **The head beat the model.** Mean-pool → cross-attention is +0.161 pooled / +0.280 within. X150 0.754 / 0.558 ties B1′ and still loses to B1 |
| 002 | [Zero-training likelihood scoring](002-likelihood-scoring.md) | Claude | **partial** | **Clean null: ESM-2's own likelihood predicts stability at rho +0.002.** Its per-position profile is flat and rank-correlates -0.03 with what the trained model actually uses. Conditioned variants not run |
| 003 | [Hurdle model for the zero class](003-hurdle-model.md) | Claude | **done — negative** | **No gain: −0.001 pooled on both framings**, despite the zero/non-zero classifier reaching AUC 0.886. The plain regressor already absorbs what the classifier knows. Decision closed: no censored head |

Status: `proposed` → `running` → `done` / `abandoned` / `inconclusive`.

All measured numbers live in [`../results/README.md`](../results/README.md), which is append-only —
a superseded number stays, with the reason it was superseded. These result sections summarise; that
file is the evidence.

## Backlog — the short version

The full catalogue is in `BRAINSTORM.md`. If you want the four highest value-per-hour items from it:

| From BRAINSTORM | Why it is worth doing first |
|---|---|
| **B4** one-hot allele control | Only 75 distinct HLAs exist. If one-hot matches an embedding, the model is memorising allele identity, not using protein knowledge. Without this, no other number means anything |
| **E2** learning curves | The sharpest answer to the brief's question — if foundation models help most when labels are scarce, that is a precise, transferable finding |
| **I1** position occlusion | Does the model rediscover the P2/P9 anchors from sequence alone? Cheap, visual, and it is the biology criterion §6 asks for |
| **E1** within-allele reporting | Nearly free, and most teams will report the confounded pooled number instead |

`BRAINSTORM.md` holds the full catalogue — roughly 40 directions across baselines, representation
routes, zero-shot scoring, structure, the label floor, evaluation, interpretability, uncertainty and
the demo, each with what it would prove, a cost estimate and how it could fail. It also lists the
attractive traps. Read it rather than starting from a blank file.

## Where things live

| Path | What |
|---|---|
| `experiments/` | these proposals and their write-ups |
| `BRAINSTORM.md` | the full idea catalogue |
| `../splits/` | the frozen split — created, verified, never regenerated |
| `../scripts/` | split generation, independent audit, and the two data diagnostics |
| `../context/NOTES.md` | the brief, the dataset profile, the constraints |
| `../context/dataset.csv` | the data — **the only authoritative copy** |
| `../RECONCILIATION.md` | current state of the two planning tracks |
