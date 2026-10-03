# Track 3 — Serova: peptide-HLA complex stability

## The challenge (verbatim from the organisers)
> The computational prediction of protein-protein interaction stability is a key step in protein
> design workflows. An example of this is modern immunotherapy design, which requires the prediction
> of peptide-HLA complex stability. Many teams use NetMHCstabpan, a sequence-based, domain-specific
> supervised learning model trained on peptide-HLA sequence pairs. This challenge asks if we can use
> existing protein foundation models to better predict the stability of peptide-HLA complexes.

- Dataset: https://docs.google.com/spreadsheets/d/1NtZNvcF3u0KFn-1bbuA50CF3IXvs1l4HfbaR3KRLvso
- Primer: https://drive.google.com/file/d/11i-LSRZlgLS7T2IVM-12xuddwuryDmnu/view (saved as `serova_primer.pdf`)

**Task shape:** regression on `thalf_hours` from (peptide, HLA) sequence pairs. The named baseline to
beat is NetMHCstabpan. "Use protein foundation models" is the explicit ask, so the sponsor-alignment
20 points favour an approach built on a pretrained protein model rather than a bespoke supervised net.

**Stability is not affinity.** Affinity = how tightly it binds at equilibrium; stability = how long
the complex survives. Pretrained affinity predictors are a useful transfer source, not a substitute.

## Dataset profile (measured 2026-10-03 from `dataset.csv`)
28,166 rows × 5 cols, no nulls, no duplicate (allele, peptide) pairs.

| Column | Notes |
|---|---|
| `allele` | 75 distinct, e.g. `HLA-A*02:01`. 7 alleles have <50 rows |
| `peptide` | **every peptide is exactly 9 residues** — 5,633 distinct |
| `thalf_hours` | target. median 1.1, mean 5.45, max 256.7, raw skew 5.87 |
| `hla_seq` | 182 aa, 75 distinct (one per allele) |
| `hla_pseudoseq` | 34 aa binding-groove subset, **74** distinct |

### Four things that shape the modelling
1. **A random row split leaks.** Each peptide appears with a median of 4 alleles (max 36); 70% of
   peptides appear with more than one, and **94% of rows involve such a peptide**. A random split puts
   the same peptide on both sides. Split by peptide group, and better, by sequence-identity cluster —
   the primer says identity-filtered splits are what separates the honest numbers from the inflated ones.
2. **20% of labels are exactly 0.0** (5,679 rows). That is a floor/detection-limit artifact, not a
   measurement of zero. Options: censored-regression likelihood, a two-stage classify-then-regress, or
   an explicit "unstable" class. Silently `log1p`-ing them treats a censoring boundary as a real value.
   **The SPEARMINT paper does exactly that** — `log(1+0)=0` straight into MSE, no mention of censoring.
   So this is a genuine open gap, not something we'd be re-treading.
3. **The target needs a transform.** `log1p` drops skew from 5.87 to 0.91. Train in log space, report
   in hours. The paper uses the same: `log(1 + t½)`, MSE, inverted as `exp(ŷ) − 1`.
4. **`hla_pseudoseq` cannot separate two alleles.** `HLA-B*14:01(C67S)` and `HLA-B*14:02(C67S)` share
   a pseudo-sequence, producing 368 (peptide, pseudoseq) pairs that collide. Keying on pseudoseq alone
   silently merges them; use `hla_seq` or the allele name as the identity.

## Metric
Primary: **Spearman ρ** — the primer notes assays disagree on absolute values but rank more
consistently. Also worth reporting: Pearson r on log, and a top-k shortlist metric
(precision@10 / recall at fixed FPR), since the clinical use is re-ranking a shortlist.

## Reference numbers — VERIFIED against the preprint (Table 2, n=2,700 clean test split)
Full breakdown in `SPEARMINT_SUMMARY.md`; paper at `spearmint.pdf`.

| Model | Spearman ρ | Pearson r | Caveat |
|---|---|---|---|
| NetMHCstabpan | 0.876 | 0.532 | trained on its own test data — not a fair target |
| TLStab | 0.698 | 0.306 | heavy peptide overlap with test set |
| ESM-2 Direct | 0.574 | 0.515 | **what a naive frozen-embedding baseline gets** |
| ESM-2 Transfer | 0.745 | 0.714 | + affinity pre-training |
| MINT Direct | 0.732 | 0.679 | cross-chain attention, no transfer |
| **MINT Transfer (SPEARMINT)** | **0.791** | **0.761** | honest state of the art |
| all models, independent IEDB data | <0.4 | — | assay distribution shift |

**0.79 is the number to aim at, not 0.876.** Anything above ~0.85 on a random split is the leak in
point 1 above, not a result. Affinity pre-training is worth more than the architecture
(ESM-2: 0.574 → 0.745), so the two-stage curriculum is the cheap win.

Our dataset is this paper's Stage-2 corpus (they report 27,034 rows / 72 alleles vs our 28,166 / 75),
so their setup is directly reproducible and their numbers are the right yardstick.

## Candidate approaches (from the primer)
1. Frozen/fine-tuned protein LM (ESM-2, ESM-C) embeddings + shallow regression head. Cheapest baseline.
2. Multi-task transfer: pretrain on binding affinity (~170k examples) then fine-tune on stability (~27k).
3. Structure-aware: co-fold the complex (Boltz-2 / AlphaFold3), featurise the interface. Most expensive.
4. Uncertainty: ensembles, conformal prediction, or a Gaussian likelihood head.

Open problems the authors list: uncertainty quantification, structure beyond sequence, joint
pretraining, rare-allele coverage (72 of 30,511 known alleles), and class II MHC.

## Judging (20 pts each) and where this track earns them
Technicality · Creativity · Usefulness · **Demo** · Track/sponsor alignment.
Demo is a full fifth of the score and a regression table is not a demo — budget time for something
that takes a peptide and shows a ranked, uncertainty-aware answer.

## Decisions still open
- [ ] Which foundation model, and frozen embeddings vs fine-tuned (depends on GPU budget)
- [ ] How to handle the 0.0 floor (censored likelihood vs two-stage)
- [ ] Clustering method for the split (identity threshold on 9-mers)
- [ ] Whether to attempt structure (approach 3) at all in 24h
- [ ] Who runs the NetMHCstabpan baseline for a like-for-like comparison on our split

## Hard deadlines
Submission closes **Sun 4 Oct 14:45 BST**. Needs: 2-min demo video, GitHub repo link, short description.
Round 1 pitch is 1:30 pitch + 1:30 live demo + 2:00 Q&A.
