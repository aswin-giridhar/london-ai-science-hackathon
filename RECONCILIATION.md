# Reconciliation — where the two planning tracks stand

Two plans were produced independently: `SCIENCE_SKILLS_PLAN.md` / `HACKATHON_PLAN.md` (Stability
Lens) and the work in `context/`, `splits/`, `scripts/`, `experiments/`. They agree on most of what
matters. This file records **current state** — what is settled, what is open, what needs a human to
decide. It is maintained as state, not as a log.

Last updated 2026-10-03 16:20 BST.

---

## 1. Settled — agreed by both tracks, do not relitigate

- The track is **not a leaderboard**. The brief rules NetMHCstabpan out as a comparator in its own
  words, because it was trained on the whole dataset.
- A **well-supported negative result is a valid submission**, in the brief's own words.
- **One frozen split**; the test set is touched once.
- **Model selection is frozen on train/dev before the test set is opened.** Test results report the
  predeclared comparison; they do not choose the winner.
- Report full-cohort **and** valid-subset numbers, with denominators.
- Keep audit axes separate: label contamination, peptide leakage, structural-template exposure.

That is substantial agreement for two independent plans, and it is the part §6 of the brief scores.

## 2. Settled by measurement — evidence in `scripts/`

| Question | Answer | Evidence |
|---|---|---|
| Split threshold | **Hamming ≤ 2**, not the conventional ≤ 1 | `measure_leakage_by_distance.py` |
| Was the split designed blind? | **Now yes.** Re-derived from train labels only | same script, analysis A |
| What does `thalf_hours == 0.0` mean? | **Half-life < 0.05 h — left-censored**, not a true zero | `investigate_zero_labels.py` |
| 72 alleles (papers) vs 75 (our file) | 75 − 3 engineered `(C67S)` = 72. Both correct | same script |
| Is `dataset.xlsx` usable? | **No — removed.** 42% of the target was date-corrupted | see §4 |

**Split threshold.** The published 80%-identity convention is exactly Hamming ≤ 1 for 9-mers. Using
training labels only, neighbour label agreement is 0.612 at one edit and 0.636 at two, then drops to
0.406 at three, against a null of 0.000. The break between *near-duplicate* and *merely similar* sits
between 2 and 3. Cost of the stricter choice is negligible: 5,410 clusters instead of 5,494.

**Blind design.** The threshold was *originally* chosen using evaluation labels — a methodological
error the other track caught and was right about. The re-derivation above uses training labels only
and reaches the same conclusion, so the decision no longer depends on having seen eval labels. The
original mistake is disclosed permanently in `splits/README.md` rather than edited out.

**The zero labels.** 98.4% of non-zero values are exact multiples of 0.1 h and exactly one row in the
dataset falls below 0.1, so the reporting scale is 0.1 h and anything under 0.05 h rounds to 0.0.
There is no gap above zero, so this is a rounding floor rather than a hard instrument threshold.
Because the threshold is *known*, a Tobit-style censored likelihood is directly implementable rather
than speculative. The published work treats the boundary as a point observation.

The zeros are strongly allele-dependent — seven alleles are more than half zeros, including all three
`(C67S)` constructs at 75–92%, while five alleles have none. **Dropping zero rows removes 20% of the
data and effectively deletes those alleles.** This supports the existing instruction to quarantine
the engineered constructs from the structural arm.

## 3. Open — needs a human to decide

1. **Canonical location for planning docs.** `HACKATHON_PLAN.md` and the primer now live here, which
   resolves most of this. Remaining question is whether they are maintained here or on a laptop.
2. **A calibration partition.** `HACKATHON_PLAN.md` specifies four partitions (70/10/10/10) because
   split-conformal uncertainty needs a calibration set. The frozen split is three-way and has none.
   If U1 is wanted, carve calibration out of **train** — never from val or test.
3. **ESM-2 size.** `experiments/001` names 650M; the plan's delivery baseline starts at 35M on
   compute grounds. A size ladder (`BRAINSTORM` R4) would settle it with evidence, and directly
   serves the brief's cost-benefit question.
4. **Data redistribution.** `HACKATHON_PLAN.md` §3.1 says "store only a downloader/manifest in git
   until redistribution terms are checked". The dataset and both PDFs are already committed to a
   public repo on the repo owner's explicit instruction. The SPEARMINT preprint is CC-BY 4.0 so
   redistribution is fine; the Serova dataset's terms have not been checked. Worth a conversation.

## 4. Divergence found during cleanup: `context/dataset.xlsx` was corrupt

Removed from the repo. Excel date-parsed **11,907 of 28,166 `thalf_hours` values (42%)** on export:
`1.8` became 1 August 2026, `5.2` became 5 February 2026, and so on. Every one of the 11,907
reconstructs exactly as (day, month) from the CSV number, so the CSV holds the original values and
the xlsx was the damaged copy.

Nothing in `scripts/` ever read it, but it was linked from the README and would have silently
destroyed 42% of the prediction target for anyone who loaded it. **`context/dataset.csv` is the only
authoritative copy.**

## 4b. Track 3 Build Plan (@Edouard V) — two things to fix before building

`Track 3 Build Plan — Structure-Aware pHLA Stability Prediction.pdf` is the most operationally
realistic of the three plans: hard kill gates at 16:30 / 20:00 / 02:00 / 08:00, a guaranteed
sequence-only floor with structure as pure upside, one falsifiable original claim (does ensemble
*spread* add signal over ensemble *mean*), an ablation table as the deliverable rather than a
leaderboard number, and explicit vetoes on PyRosetta and 650M fine-tuning. Its thesis is sound and
genuinely novel: existing structural methods score a single static pose, which approximates a
thermodynamic quantity, while half-life is kinetic — so feature *variance* across an ensemble is the
kinetic signal.

Two corrections, both cheap, both before any code:

**1. The split already exists — do not rebuild it.** The plan specifies 80% identity (= Hamming ≤ 1
for 9-mers) targeting 21,600 / 2,700 / 2,700. The frozen split in `splits/` is **Hamming ≤ 2** at
22,532 / 2,817 / 2,817, and the project README already declares it authoritative. Rebuilding would
put every ablation row on a different split from the audited one.

The plan's own risk register names "a wrong split discovered at 13:00 Sunday" as one of two things
that will sink the submission, and proposes "have a second person sanity-check that no peptide
cluster appears on both sides". `scripts/audit_splits.py` already does exactly that, exhaustively and
independently of the generator. Use `splits/peptide_split.csv` and that half-hour is already spent.

The plan also wants a held-out-allele axis — "the real generalisation test, and where structure
should win". That exists too: `splits/allele_split.csv`, stratified by allele median half-life.

**2. Conformal intervals need their own calibration rows.** The plan says to take absolute residuals
"on the validation split". But validation is also doing model selection — the plan retrains in under
a minute specifically to "ablate aggressively". Selecting on the same rows used for conformal
calibration breaks exchangeability, and the distribution-free guarantee no longer holds. Since the
plan calls uncertainty "non-negotiable", this matters.

Fix: carve a calibration partition out of **train** — never from val or test. This is the same gap
`HACKATHON_PLAN.md` avoids with its 70/10/10/10 four-way split. Empirical coverage can still be
reported either way; it is the *guarantee* that needs clean rows.

**Two smaller notes.** The zero floor is now characterised precisely (§8): `0.0` means half-life
< 0.05 h, left-censored at a *known* threshold, which the two-head classify-then-regress design
handles fine and which can be stated exactly in Q&A. And NetMHCstabpan is described in places as
"the named baseline to beat" with the preprint's Table 2 as "our scoreboard" — the brief explicitly
rules it out as a comparator, and because it trained on the entire corpus our test peptides are in
its training data *regardless* of how we split. Running it is a useful reference; beating it is not
a goal the brief recognises.

## 5. Open questions for the Stability Lens authors

1. `training pipeline.pdf` shows Boltz-2 emitting an **Affinity** head alongside Structure. The Boltz
   docs state the affinity binder "must be a ligand chain (not a protein, DNA or RNA)", max ~56 atoms
   recommended — so a 9-mer peptide cannot be the binder. `HACKATHON_PLAN.md` already rejects this
   use. **Is the diagram superseded?** As drawn it cannot be built, and its Stage 2 also places
   Boltz-2 inside the trainer, which is far outside the compute budget.
2. Are the "three domain mismatches" the same three `(C67S)` constructs, or a separate set?
3. Does the structural arm need a full heavy chain + β2m construct, given the sponsor supplies only
   the 182-aa α1/α2 domain? That reconstruction looks like the largest single risk in the plan.

## 6. Known inaccuracies in `Peptide-HLA Stability Primer.md`

The primer predates the event and is retained as background. Three of its figures do not describe
this dataset, verified against `context/dataset.csv`:

| Primer says | This dataset |
|---|---|
| HLA heavy chain "about 365 amino acids" | `hla_seq` is **182** residues (α1/α2 only) |
| "about 27,000" / "21,626" measurements | **28,166** rows |
| "fall apart in minutes; others last half a day" | max is **256.7 h ≈ 10.7 days** |

The 365-residue figure matters because the primer's pooling argument depends on it: it claims a
9-mer is diluted to "about 2%" of a mean-pooled representation. With 182 residues it is 9/191 ≈
**4.7%**. The qualitative advice (pool the chains separately) survives; the number does not. The
primer also gives only P2/PΩ anchors with no allele-specific exceptions — `HACKATHON_PLAN.md` and the
README are more careful here, and anchor features built from the primer alone would hard-code a
simplification across all 75 alleles.
