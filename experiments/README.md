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

## Index

| # | Experiment | Owner | Status | Headline result |
|---|---|---|---|---|
| 000 | [Simple supervised baseline](000-supervised-baseline.md) | — | proposed | — |
| 001 | [ESM-2 embeddings + head](001-esm2-embeddings-head.md) | — | proposed | — |
| 002 | [Zero-training likelihood scoring](002-likelihood-scoring.md) | — | proposed | — |

Status: `proposed` → `running` → `done` / `abandoned` / `inconclusive`.

## Backlog — ideas nobody has written up yet

Grab any of these and turn it into a proposal, or ignore them entirely and propose your own. These
come from the brief's §5, which explicitly invites going beyond embeddings:

- **Model confidence as uncertainty.** The brief names "internal and external confidence metrics".
  Free uncertainty estimates, and the leading published model does none.
- **Seed and mask variation as a cheap ensemble.** Also from §5. Gives both a variance estimate and
  a possible accuracy gain for no training.
- **The `t½ = 0` floor.** 20% of labels are a detection limit, not a measurement. Censored
  likelihood, or classify-then-regress? Nobody in the literature handles this.
- **Does the pseudo-sequence lose anything?** 34 residues vs the full 182-residue α1/α2 domain —
  a direct, cheap ablation on information content.
- **Structure prediction on a subset.** Boltz-2 / Chai-1 co-folding. Expensive, so a subset, which
  the brief explicitly permits. A `boltz` Claude Code plugin exists but is not installed and needs
  an API key — check cost before committing.
- **Which alleles are hard?** Per-allele error vs training rows per allele. Cheap, and makes a good
  demo slide.
- **Does a general protein model beat a one-hot allele encoding at all?** The honest null hypothesis.
  If a lookup table of 75 alleles plus peptide features matches ESM-2, that is a real finding and
  directly answers the brief's question.

## Where things live

| Path | What |
|---|---|
| `experiments/` | these proposals and their write-ups |
| `splits/` | the frozen split — created, verified, never regenerated |
| `scripts/make_splits.py` | how the split was built, with its own assertions |
| `../context/NOTES.md` | the brief, the dataset profile, the constraints |
| `../context/dataset.csv` | the data |
