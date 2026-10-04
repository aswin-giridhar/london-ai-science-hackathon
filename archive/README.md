# archive — source material that has been read and synthesised

Nothing here is deleted and nothing here is wrong. These are the inputs that fed
`../PLAN.md`, `../ARCHITECTURE.md` and `../RECONCILIATION.md`. They are archived so the working
root stays readable, and because the synthesis now carries their conclusions.

**If a conflict arises between a document here and `../PLAN.md`, the specifics are in
`../RECONCILIATION.md` with the evidence behind each resolution.**

## `plans/` — the three independent proposals

| File | Author | What was taken from it |
|---|---|---|
| `HACKATHON_PLAN.md` | Stability Lens | The evaluation discipline, chiefly *"test results report the predeclared comparison; they do not select the winning model"*; the four-partition split that exposed our missing calibration slice; construct-verification rigour |
| `SCIENCE_SKILLS_PLAN.md` | Stability Lens | The DeepMind Science Skills workflow and the evidence-ledger approach; the provenance discipline |
| `Track 3 Build Plan — Structure-Aware pHLA Stability Prediction.pdf` | @Edouard V | **The best scientific idea in the project** — ensemble feature *variance* as the kinetic signal. Also the kill-gate structure, the two-head zero handling, and the parquet interface contract |
| `training pipeline.pdf` | — | **Superseded.** It shows Boltz-2 emitting an Affinity head from peptide and HLA sequences, but the Boltz docs state the affinity binder *"must be a ligand chain (not a protein, DNA or RNA)"*. A 9-mer is a protein chain, so the architecture cannot be built as drawn. `HACKATHON_PLAN.md` independently rejects the same use |

## `background/` — reference material

| File | Note |
|---|---|
| `spearmint.pdf` | Karthikeyan, Vincent & Rubinsteyn, bioRxiv 2026. CC-BY 4.0, so redistribution is fine. Summarised in `../context/SPEARMINT_SUMMARY.md`; 12 MB, which is why it is here rather than in `context/` |
| ~~`Peptide-HLA Stability Primer.md`~~ | **Removed from the repository on 4 October 2026.** Pre-event, dated 2 October. See below. |

## The pre-event primer was removed - resolved 4 October 2026

`plans/HACKATHON_PLAN.md` said, in its own words:

> *"Do not commit or present the pre-event primer ... as a newly built submission"*
> *"because the primer predates kickoff, do not include it or copy its text into the judged
> artifact without organizer approval"*

The event rules say *"Build entirely during the event. No prior commits to the repo."* The primer
was written on 2 October; the event began on 3 October. Moving it into `archive/` did not resolve
that - it was still committed to the judged repository.

**Resolved by removing it.** `archive/background/Peptide-HLA Stability Primer.md` is no longer
tracked (`git rm --cached`), is listed in `.gitignore`, and is kept only on the author's machine.
The safest reading of the rule was removal, and that is what was done rather than asking for an
exception.

**One thing this does not do:** the file remains in the repository's *git history* and is
reachable through earlier commits. Erasing it from history would require a rewrite and a
force-push of a public repository, which is destructive and was not done unilaterally. If an
organiser requires that the file never appear in the repository at all, that is the remaining
step and it needs an explicit decision.

**What never depended on it.** No code, split, result or figure in this project derives from the
primer. Three of its figures are wrong for this dataset - it assumes a ~365-residue heavy chain
where ours is 182, gives 27,000/21,626 rows against our 28,166, and describes the range as
"minutes to half a day" where the measured maximum is 256.7 h. Those corrections are recorded in
`superseded-decisions/RECONCILIATION.md` section 6, which is itself an archived document. The
authoritative challenge description is and always was `../context/serova_primer.pdf`, the
official brief supplied by Serova to every participant, which is unaffected by this removal.

## `superseded-decisions/` — our own working documents, now resolved

The plan of record is **`../docs/GeoStab-FT-build-plan.pdf`**, and `../ARCHITECTURE.md` is its build
spec. These three did the work of getting there and are kept for the audit trail, not for guidance.

| File | What it was for | Why it is here |
|---|---|---|
| `PLAN.md` | Evaluated three competing plans and chose a path | Its job — choosing between plans — is done. GeoStab-FT is adopted |
| `RECONCILIATION.md` | Recorded every conflict between the plans with the evidence behind each resolution | Still the best record of *why* the split threshold, the zero-label reading and the Boltz/threading fork were decided as they were |
| `POWER_ANALYSIS.md` | Minimum detectable effect on the test split | Contains a **retraction**: a per-metric sensitivity table that did not reproduce, kept with its diagnosis rather than deleted. The surviving finding — a single Spearman has a CI ~0.05 wide — moved into `../ARCHITECTURE.md` |

`docs/architecture.html` was removed rather than archived: it was a styled copy of the old
parallel-track architecture, and a stale duplicate of a superseded diagram is worse than none.

## What deliberately stayed out of the archive

| Kept where it is | Why |
|---|---|
| `../context/dataset.csv` | **Live.** Five scripts read it by that path |
| `../context/serova_primer.pdf` | The official brief — the authoritative source, not superseded by anything |
| `../context/NOTES.md`, `../context/SPEARMINT_SUMMARY.md` | Working notes, still current |
| `../splits/`, `../scripts/`, `../experiments/` | Built artifacts and live code |

## `older_architecture.md` — a point-in-time snapshot, not a decision

A **verbatim byte copy of `../ARCHITECTURE.md` as of commit `8e722e1` (3 Oct, 19:39)** — the last
version before `087b5e8` (20:07) folded `HACKATHON_PLAN.pdf`'s findings in. Kept because that fold-in
was the point the eligibility, construct-quarantine and Boltz-2-vs-threading material entered the
spec, and it is useful to see the document without it.

It is **three commits stale** beyond that: `9661900` (GeoStab-FT adopted as plan of record),
`f37eaa5` (Boltz-2 scope in §5) and `e2ac9de` (baseline results) all landed after it. Its own
"Status at 16:45 Saturday" header is the honest marker. **Read `../ARCHITECTURE.md` for anything
current.**

Reproduce or re-verify it with:

```sh
git show 8e722e1:ARCHITECTURE.md | diff --strip-trailing-cr - archive/older_architecture.md
```
