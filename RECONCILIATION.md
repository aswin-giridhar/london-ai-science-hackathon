# Reconciliation — two planning tracks meeting

**Written 2026-10-03 15:45 BST. ~23h to the 14:45 Sunday cutoff.**

This note reconciles `SCIENCE_SKILLS_PLAN.md` (the Stability Lens / Science-Skills revision) with the
work in `context/`, `splits/`, `scripts/` and `experiments/`. The two were produced independently and
mostly agree. Where they disagree, the specifics are below with evidence, so nobody has to re-derive
them.

Nothing here is a veto. Anything marked **decide** needs a human to pick.

---

## 1. Credit where it's due — the plan is right about things I under-weighted

Its factual claims check out. I verified independently:

- `google-deepmind/science-skills` is real, public, Apache-2.0; the pinned commit
  `68832757cbbf941c620b71df5756cf6e5cc287b0` resolves to **Science Skills v1.2.1 (2026-09-14)**, and
  every skill it names exists at that revision.
- 28,166 rows, the 182-aa α1/α2 domain vs full-chain distinction, and three `(C67S)` engineered
  constructs — all match what I measured from `context/dataset.csv` independently.
- The Sunday 14:45 BST cutoff matches the brief.

**Its strongest contribution is one I had under-weighted.** The brief's §6 asks for *"incorporation
of the biological background of the problem"* as a scoring criterion, not as context. The plan does
this properly — anchor-position exceptions, P3-dominant alleles, and a citation that improving anchor
residues can *fail* to improve measured stability. My notes treated biology as background. That was
a misread of the rubric and the plan corrects it.

Its evaluation hygiene should be adopted wholesale, in particular:

- *"Test results report the predeclared comparison; they do not select the winning model."*
- Freeze cohort membership before running; never swap a failed test pair for an easier one.
- Report full-cohort **and** valid-subset numbers with their denominators.
- Separate audit axes: label contamination, peptide leakage, structural-template exposure.

---

## UPDATE 16:05 — §2 is resolved, and the reviewer was right about something else

Since this note was written, the other track pushed an expanded `README.md` that adopts
**Hamming ≤ 2 as authoritative**. §2 below is settled; it is kept for the record.

It also made a criticism that was correct and that I had missed:

> *"Those diagnostics inspect evaluation labels, so disclose that the split design was informed by
> exploratory label analysis; it is not a benchmark designed without viewing any evaluation labels."*

That is true. The threshold was originally chosen by measuring how well a **test** peptide's label
was predicted by its nearest training peptide. However conservative the resulting change, a
benchmark whose design depended on evaluation labels is not blind.

**Re-derived from training labels only** (`measure_leakage_by_distance.py`, analysis A), the same
conclusion holds:

| edit distance | identity | Spearman between train neighbours | n |
|---|---|---|---|
| 1 | 0.889 | 0.612 | 437 |
| 2 | 0.778 | 0.636 | 297 |
| 3 | 0.667 | 0.406 | 728 |
| 4 | 0.556 | 0.354 | 5,501 |
| random | — | −0.000 | 14,650 |

Distances 1 and 2 behave alike; distance 3 drops sharply. The break is between 2 and 3 whether or
not eval labels are consulted, so the threshold stands on train-only evidence. The script now runs
the train-only analysis first and prints the train-vs-eval one beneath it, marked as confirmation
that must not drive design. The original mistake is disclosed permanently in `splits/README.md`
rather than edited out — the audit trail is worth more than a clean story.

**Two items from their README still open:**
- `experiments/001` names ESM-2 650M; their delivery baseline starts at 35M. A model-size ladder
  (BRAINSTORM `R4`) would settle it with evidence instead of a preference, and directly serves the
  brief's cost-benefit question.
- They flagged the zero-label censoring claim as a hypothesis pending source evidence. That was the
  right call, and it has now been **settled from the data** — see §8 below.

---

## 2. One concrete conflict: the split threshold *(resolved — see update above)*

**The plan specifies "exact one-edit global peptide groups" (Hamming ≤ 1). The split in `splits/` uses
Hamming ≤ 2.** This is not a style difference — it was measured.

Hamming ≤ 1 is exactly the published 80%-identity convention for 9-mers, so the plan is following
standard practice. I adopted the same convention, then checked whether it holds. It does not.

Measuring how well a test peptide's label is predicted by its nearest training peptide, as a
**within-allele** z-score of `log1p` (`scripts/measure_leakage_by_distance.py`):

| edit distance | identity | Spearman with neighbour |
|---|---|---|
| 2 | 0.778 | **0.700** |
| 3 | 0.667 | 0.311 |
| 4 | 0.556 | 0.364 |
| random pair | — | 0.027 |

The break between *near-duplicate* and *merely similar* sits between **2 and 3**, not between 1 and 2.
A Hamming ≤1 split leaves 13 test peptides whose labels are ~70% rank-predictable from a training
neighbour. Cost of the stricter threshold is negligible: 5,410 clusters instead of 5,494, largest
cluster 8 peptides instead of 5, so no single-linkage chaining.

**Also worth carrying across:** the first version of that measurement pooled pairs across alleles and
produced a null of 0.335 where ~0 was required — allele-level stability differences inflated every
correlation. This matters beyond the split. **Any pooled correlation on this dataset is partly a
between-allele effect.** Report within-allele numbers alongside pooled ones, or the headline figure
is measuring the wrong thing. The plan's §6 already says *"Pooled correlation can reflect
between-allele differences, so always show supported within-allele results"* — we agree, arrived at
separately.

**Decide:** adopt Hamming ≤ 2 (recommended — evidence above, already built and audited), or keep ≤ 1
for literature comparability and disclose the 13 peptides.

---

## 3. Two parallel structures

The plan edits `/Users/arkajyotisaha/Desktop/Hackathon/ARE/HACKATHON_PLAN.md` and a
`Peptide-HLA Stability Primer.md`. Neither is in this repo. This repo has:

| Path | What |
|---|---|
| `context/NOTES.md` | The brief, dataset profile, constraints, corrected reference numbers |
| `context/SPEARMINT_SUMMARY.md` | The preprint, framed as calibration not a target |
| `splits/` | Frozen, audited splits + `README.md` explaining what they guarantee |
| `scripts/` | `make_splits.py`, `audit_splits.py`, `measure_leakage_by_distance.py` |
| `experiments/` | Proposal template, ground rules, seeded experiments, `BRAINSTORM.md` |

**Decide:** which is canonical. My suggestion — this repo is the submission artifact (the brief wants
a GitHub link), so planning docs should land here, and `HACKATHON_PLAN.md` should be copied in rather
than maintained on one laptop. Not a strong opinion; a single location matters more than which one.

---

## 4. A gap in both plans worth closing

**The `t½ = 0` floor.** 5,679 rows — 20% of the dataset, 57% of the lowest histogram bin — sit at
exactly 0.0.

**What is measured:** the count and the exact value. **What is hypothesis:** that this represents an
assay floor rather than genuine zero-hour measurements. The CSV does not state a detection limit, so
this needs checking against Rasmussen et al. 2016 before any claim rests on it.

The plan gets close: it removes `max(t_half_hours, 0)` from the target contract, correctly noting it
*"silently turns bad data into valid zero-hour labels"*. But it then feeds 0 into `log1p` as a real
value. So does the published SPEARMINT work — no mention of censoring anywhere in it.

This is one of the few places where we could do something the literature hasn't. Options: a censored
(Tobit-style) likelihood, two-stage classify-then-regress, or an explicit zero-label class — but
confirm the assay semantics first.
Written up as a candidate experiment in `experiments/BRAINSTORM.md`.

---

## 5. Scope against the clock

Read from the system clock, not estimated: **23.2 hours remained at 15:35 BST Saturday.**

The plan's stated deliverable is *"an improved planning document — not an installed skill bundle,
trained model, new code repository, or deployment."* Its full scope — Boltz-2 co-folding, PDB
reference triage, chain/construct registries, β2-microglobulin, ProteinMPNN, conformal intervals,
Streamlit, seven test files — is substantially more than a weekend.

To its credit it gates this honestly (Gate A before B/C; *"skip optional searches, larger encoders,
and U1 before compromising the core comparison"*). I'm agreeing with the plan's own gates, not
arguing against it.

**Gate A is already unblocked.** Audited inputs and a locked split exist. The baseline (`experiments/
000`) and the frozen-embedding run (`001`) are the gate-A deliverables and need no structural work.

**Suggested division, if three people are working:**

| Owner | Track |
|---|---|
| Data/eval | Gate A: baseline, embeddings, the one-hot allele control, within-allele reporting |
| Structure | Gate B pilot on a handful of pairs — answer *is this feasible at all* before committing |
| Evidence/demo | Claim ledger, biology caveats, demo skeleton, the 2-min video |

---

## 6. Open questions for whoever wrote the plan

1. Are the "three domain mismatches" the same three `(C67S)` constructs, or a separate set? If
   separate, which alleles?
2. Did the dataset audit run against this `context/dataset.csv`, or a differently-exported copy? Row
   counts match, so probably the same file — worth confirming before we treat the audits as one.
3. Does the structural arm need the full heavy chain + β2m construct, given the sponsor supplies only
   the 182-aa α1/α2 domain? That reconstruction is the plan's §3 and looks like its largest single
   risk.
4. Was any part of the plan executed, or is it all proposal? §Verification implies nothing ran.

---

## 7. What is settled and shouldn't be relitigated

- Serova's track is **not a leaderboard**. NetMHCstabpan is ruled out as a comparator in the brief's
  own words. Both plans agree.
- A well-supported negative result is a valid submission. Both plans agree.
- Everything reports on one frozen split; test is touched once. Both plans agree.
- Model selection is frozen on train/dev before the test set is opened. Both plans agree.

That is a lot of agreement for two independently produced plans, and it is the part the brief
actually scores.

---

## 8. UPDATE 16:10 — the zero labels are settled, and both plans were wrong about them

Everyone's notes hedged on `thalf_hours == 0.0`. My first version asserted "detection floor" with no
evidence; their README correctly demanded source evidence; the official brief, the background primer
and the SPEARMINT preprint are all silent on it. It turns out the data answers the question on its
own — `scripts/investigate_zero_labels.py`.

**Two tests.**

1. *Is there a gap above zero?* A hard censoring threshold at some value leaves a point mass at 0
   and nothing until that value. There is no gap — 0.05 and 0.1 both occur. So "below the
   instrument's threshold" does not fit by itself.
2. *What is the reporting precision?* **98.4% of non-zero values are exact multiples of 0.1 h**
   (22,133 of 22,487), and **exactly one row in the entire dataset falls below 0.1 h**. The
   reporting scale is 0.1 h.

**Conclusion: `0.0` means "measured half-life below 0.05 h" — left-censored at three minutes, not a
complex with zero lifetime.** The censoring threshold is therefore *known* (0.05 h), which makes a
Tobit-style likelihood directly implementable rather than speculative. `log1p` treats that boundary
as a point observation, and so does the published work.

**The more immediately practical finding** is that the zeros are strongly allele-dependent:

| allele | rows | zero fraction |
|---|---|---|
| `HLA-B*39:06(C67S)` | 379 | 92.1% |
| `HLA-B*14:01(C67S)` | 374 | 89.0% |
| `HLA-B*14:02(C67S)` | 382 | 74.9% |
| `HLA-B*45:01` | 368 | 67.1% |
| `HLA-B*41:01` | 368 | 59.8% |
| `HLA-B*55:01` | 378 | 55.6% |
| `HLA-B*51:01` | 353 | 50.7% |

Median across alleles is 13.2%; five alleles have none at all. **Dropping zero rows would remove 20%
of the dataset and effectively delete the seven alleles above**, including all three engineered
constructs. Anyone filtering them needs to say so and report which alleles it costs.

This also supports the plan's existing instruction to quarantine the `(C67S)` constructs from the
structural arm: at 75–92% zeros, those three contribute very little gradient signal and their
measurements may reflect the construct rather than the biology.

**Reconciles a discrepancy too.** The primer and preprint both state **72** alleles; we measure 75.
75 − 3 engineered = 72 exactly, so the `(C67S)` constructs are additional to the published corpus,
not a counting error on either side.
