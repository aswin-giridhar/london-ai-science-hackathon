# Serova Bio — Protein Engineering Track

Sources: `serova_primer.pdf` (the official 3-page brief, read in full 2026-10-03), `dataset.csv`,
and `spearmint.pdf` (background preprint, **not** part of the brief).

- Dataset: https://docs.google.com/spreadsheets/d/1NtZNvcF3u0KFn-1bbuA50CF3IXvs1l4HfbaR3KRLvso
- Brief: https://drive.google.com/file/d/11i-LSRZlgLS7T2IVM-12xuddwuryDmnu/view
- Explainers (private artifacts): [dataset + task](https://claude.ai/artifact/PrWLYagCR8ogYp4WyVJhhv) ·
  [the brief unpacked](https://claude.ai/artifact/MmYEWY9T1mKDTiPrU3cYRs)

## The question (verbatim, brief §1)
> The computational prediction of protein-protein interaction stability is a key step in protein
> design workflows. An example of this is modern immunotherapy design, which requires the prediction
> of peptide-HLA complex stability. Many teams use NetMHCstabpan, a sequence-based, domain-specific
> supervised learning model trained on peptide-HLA sequence pairs. This challenge asks if we can use
> existing protein foundation models to better predict the stability of peptide-HLA complexes.

## ⚠ This is NOT a leaderboard — the brief says so explicitly (§5)
> We are not looking for an approach that tops an arbitrary leaderboard. NetMHCstabpan, a competitive
> baseline for this task, is trained on the entirety of this dataset, hence is unfair to use as a
> direct comparator.

> Negative results are just as good as positive results where well supported. Foundation models are
> very capable, but computationally heavy and expensive to produce and run. We are asking the question
> **Are they useful for this problem**, not *Please prove they are useful for this problem*.

**An earlier version of this file said "the named baseline to beat is NetMHCstabpan". That was wrong
and is the opposite of what the brief asks.** There is no score to beat. The deliverable is a
well-evidenced answer to a research question, and "no, they don't help here" is a winning answer if
the evidence is clean.

## What the brief explicitly tells us to do (§5)
1. **Establish splits early, with a stated hypothesis.** No splits are supplied, and the brief says
   this is deliberate — "specifically to avoid a leaderboard approach". The split is part of the
   submission, not setup for it.
2. **Build a simple supervised NN on peptide+HLA pairs** as a baseline "to contextualise the results
   on your splits". Without it, a foundation-model number means nothing.
3. **Budget compute per model family** — "either through data subsetting or careful choice of which
   model(s) to evaluate". Both are acceptable; say which we chose and why.
4. **Explore under the hood.** The brief lists the usable outputs: embeddings, log-likelihoods /
   perplexities, internal *and* external confidence metrics, varying the masked inputs, varying
   seeds, fine-tuning, and prediction heads on embeddings.

## How it is judged (brief §6)
> We're interested in how you tackle this question, your approach to the data, chosen evaluation
> metrics, your conclusions, and the evidence you provide to support them. Winning submissions will
> have undertaken a watertight evaluation considering ML best practices, engineering & compute
> requirements during the timeframe, and incorporation of the biological background of the problem.

Note "**chosen** evaluation metrics" — the metric is ours to pick and justify, not a given.
This sits alongside the event-wide rubric (Technicality / Creativity / Usefulness / Demo /
Track alignment, 20 each) in `../support/HACKATHON.md`.

## The three model families the brief names (§3)
| Family | Direction | Models listed |
|---|---|---|
| Structure prediction | sequence → 3D coordinates | Boltz-2, Chai-1, Protenix, ESMFold, ESMFold2 |
| Inverse folding | 3D coordinates → sequence | ProteinMPNN, LigandMPNN, ESM-IF |
| Protein language model | sequence → embedding | ESM-2, ESMC, ProtT5, SaProt |

Only the language models hand us something directly usable as a feature. Structure models produce
coordinates we would then have to featurise ourselves — a much bigger job than it looks in 24h.
**MINT (the SPEARMINT paper's backbone) is not on this list.**

## Dataset (brief §4 + measured from `dataset.csv` 2026-10-03)
28,166 rows × 5 cols, no nulls, no duplicate (allele, peptide) pairs.

| Column | The brief's definition | Measured |
|---|---|---|
| `allele` | IPD-IMGT/HLA nomenclature; three engineered constructs keep a substitution suffix | 75 distinct; the three are `B*14:01(C67S)`, `B*14:02(C67S)`, `B*39:06(C67S)` ✓ |
| `peptide` | one-letter code, **all 9-mers** | 5,633 distinct, all length 9 ✓ |
| `thalf_hours` | measured half-life in hours — **the target** | median 1.1, mean 5.45, max 256.7, skew 5.87 |
| `hla_seq` | **the α1/α2 domain** — the part forming the groove, *not* the whole molecule | 182 aa, 75 distinct (1:1 with allele) |
| `hla_pseudoseq` | the 34-residue pseudosequence of peptide-contacting positions used by NetMHCpan | 34 aa, **74** distinct |

Also measured: only genes **A (36 alleles) and B (39)** are present — **no HLA-C at all**.
Class I binds 8–11-mers in general, but this dataset is 9-mers only.

**Citation obligation (§4):** the data is from Rasmussen et al. 2016, *Pan-Specific Prediction of
Peptide–MHC Class I Complex Stability, a Correlate of T Cell Immunogenicity*, J. Immunol.
197(4):1517–1524, doi:10.4049/jimmunol.1600582. The brief says submissions using it **should cite
their work** — put this in the README from the first commit.

## Biology that actually affects the model (brief §2)
- Cells chop up internal proteins and display the fragments in the HLA groove for T-cell inspection.
  A fragment that falls out before a T cell passes is never audited — that dwell time is the target.
- Different people carry different HLA alleles and so present different peptide sets. This is why the
  downstream therapy has to be personalised.
- **Affinity vs stability, stated more precisely than usual:** affinity is a function of *both* the
  association (on) rate and the dissociation (off) rate; stability "measures how long a peptide stays
  in the binding groove once bound, and is largely a function of the dissociation rate". So affinity
  is a composite containing the signal we want plus one we don't — useful as transfer, not a substitute.
- The groove is formed by two conserved α-helices over a β-sheet floor (brief Figure 1, PDB 4MJI).
  The helices are conserved across alleles; what differs is mostly the residues pointing into the
  trench — which is why 34 contact positions characterise a groove.

## Four things measured in the data that shape the modelling
1. **A random row split leaks.** Each peptide appears with a median of 4 alleles (max 36); 70% of
   peptides appear with more than one, and **94% of rows involve such a peptide**. A random split puts
   the same peptide on both sides. Split by peptide group, better by sequence-identity cluster.
   (The identity-clustering recipe is from the SPEARMINT preprint, not the brief — the brief only
   says to establish splits early with a sensible hypothesis.)
2. **20% of labels are exactly 0.0** (5,679 rows). **Settled from the data** by
   `scripts/investigate_zero_labels.py`: 98.4% of non-zero values are exact multiples of 0.1 h and
   only one row in the entire dataset falls below 0.1, so the reporting scale is 0.1 h and anything
   under 0.05 h rounds to 0.0. **`0.0` means "half-life < 0.05 h" — left-censored at 3 minutes, not
   a true zero.** There is no gap above zero, so this is a rounding floor rather than a hard
   instrument threshold. **The SPEARMINT paper does not address it** — `log(1+0)=0` straight into
   MSE — so a censored likelihood is a genuine open gap.
   The zeros are strongly allele-dependent: seven alleles are more than half zeros, including all
   three engineered `(C67S)` constructs at 75–92%. Dropping zero rows removes 20% of the data and
   effectively deletes those alleles.
3. **The target needs a transform.** `log1p` drops skew from 5.87 to 0.91.
4. **`hla_pseudoseq` cannot separate two alleles.** `B*14:01(C67S)` and `B*14:02(C67S)` share one,
   producing 368 colliding (peptide, pseudoseq) pairs. Key on `hla_seq` or the allele name.

## Reference numbers — from the SPEARMINT preprint, as CONTEXT not as a target
Full breakdown in `SPEARMINT_SUMMARY.md`. Table 2, n=2,700, identity-clustered split.

| Model | Spearman ρ | Pearson r | Caveat |
|---|---|---|---|
| NetMHCstabpan | 0.876 | 0.532 | trained on its own test data — the brief rules it out as a comparator |
| TLStab | 0.698 | 0.306 | heavy peptide overlap with test set |
| ESM-2 Direct | 0.574 | 0.515 | **what a naive frozen-embedding baseline gets** |
| ESM-2 Transfer | 0.745 | 0.714 | + affinity pre-training |
| MINT Direct | 0.732 | 0.679 | cross-chain attention, no transfer |
| MINT Transfer (SPEARMINT) | 0.791 | 0.761 | published state of the art |
| all models, independent IEDB data | <0.4 | — | assay distribution shift |

Useful as calibration: a plain frozen ESM-2 head lands near **0.574**, and anything above ~0.85 on a
random split is leakage rather than a result. Note their `hla_seq` was the **full heavy chain**;
ours is only the α1/α2 domain, so this is not a like-for-like reproduction.

## Candidate approaches
From the brief's own §5 list (authoritative):
1. Embeddings + prediction head — cheapest, the baseline everyone will build.
2. **Log-likelihood / perplexity scoring — no training at all.** Ask the model how "normal" the
   complex looks. Nobody in the SPEARMINT paper tried this.
3. **Internal/external confidence metrics — uncertainty for near-free**, which §6 rewards and the
   published model does not attempt.
4. Masking variations and multiple seeds — a cheap ensemble, and another uncertainty route.
5. Fine-tuning — most expensive; likely out of scope on our GPU budget.

From the preprint (background, not asked for): affinity → stability transfer curriculum;
structure co-folding; conformal prediction.

## Decisions still open
- [ ] The hypothesis we state up front (the brief asks for one)
- [ ] Which 2–3 foundation models, and from which families
- [ ] Clustering method + identity threshold for the split
- [ ] How to handle the 0.0 floor (censored likelihood vs two-stage)
- [ ] Which metrics we choose, and the justification (§6 judges this)
- [ ] Whether to attempt structure prediction at all in the time

## Hard deadlines
Submission closes **Sun 4 Oct 14:45 BST**. Needs: 2-min demo video, GitHub repo link, short
description. Round 1 is 1:30 pitch + 1:30 live demo + 2:00 Q&A.
