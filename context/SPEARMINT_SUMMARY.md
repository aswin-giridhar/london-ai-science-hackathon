# SPEARMINT paper — simplified (verified against the PDF)

Karthikeyan, Vincent & Rubinsteyn (UNC Chapel Hill), bioRxiv preprint posted **29 June 2026**,
doi `10.64898/2026.06.28.735023`. CC-BY 4.0. Local copy: `spearmint.pdf` (33 pages).
*"Peptide:MHC binding stability prediction using protein language models."*

## The one-paragraph version
Stability (how long a peptide-MHC complex survives, `t½` in hours) matters more for immunogenicity
than binding affinity does, but it has ~6× less data and only two public predictors
(NetMHCstabpan, TLStab). The authors argue published scores are **inflated by data leakage**, redo
the benchmark with similarity-aware splits, and show a protein language model beats the incumbents
honestly. They then show published half-lives aren't comparable across lab assays, and fix that by
conditioning the model on which assay produced the number.

## How this paper relates to our track — read this before using its numbers
Serova's brief asks the same question this paper answers, so assume the judges know it. **But the
brief is explicit that this is not a leaderboard** ("we are not looking for an approach that tops an
arbitrary leaderboard"; "negative results are just as good as positive results where well
supported"). So treat Table 2 as **calibration, not a target**.

Three concrete mismatches with our task:
- **Different HLA input.** The paper deliberately uses the *full-length MHC-I heavy chain*. Our
  `hla_seq` is only the **α1/α2 domain** (182 aa). Not a like-for-like reproduction.
- **MINT is not on Serova's model list.** The brief names twelve models across structure prediction,
  inverse folding and protein language models; MINT is in none of them.
- **The brief wants breadth of method, not one tuned pipeline** — embeddings, log-likelihoods,
  confidence metrics, masking and seed variation. This paper only does embeddings + fine-tuning.

## Our dataset is the same corpus (with caveats)
They describe the NetMHCstabpan corpus as **27,034 measurements, 72 HLA-I alleles, 9-mers, all from
scintillation proximity assay (SPA) at 37 °C**. Ours is 28,166 rows / 75 alleles / all 9-mers.

**The allele count reconciles exactly:** 75 − 3 engineered `(C67S)` constructs = 72. So the
engineered constructs are additional to the published corpus and both counts are correct.

The row count nearly reconciles too: those three constructs hold 1,135 rows, and 28,166 − 1,135 =
**27,031 against the paper's 27,034 — a residual of 3 rows**, cause unknown. Close enough to confirm
the same underlying corpus, not close enough to claim the files are identical. Both trace to
Rasmussen et al. 2016, which Serova asks us to cite.

**But the HLA column differs**: they map each allele to its full-length heavy chain from
IPD-IMGT/HLA; our file supplies only the α1/α2 groove domain. Any comparison to their numbers is
approximate for that reason alone.

## The three design choices we should copy
1. **Target transform:** `BS_score = log(1 + t½)`, MSE loss in log space, invert with `exp(ŷ) − 1`.
   (Confirms the `log1p` we'd planned.)
2. **Full MHC sequence, not the 34-mer pseudo-sequence.** They deliberately use the complete
   heavy-chain sequence — it carries structural context beyond the groove and handles mutant alleles.
   Our `hla_seq` (182 aa) is the right column; `hla_pseudoseq` is the one to avoid.
3. **The split:** cluster unique peptides at **80 % sequence identity (normalised Levenshtein)** and
   assign *whole clusters* to train/val/test → 21,626 / 2,705 / 2,700. Also decontaminated against
   the *affinity* training set (cross-task leakage).

## Results on the clean NetMHCstabpan test set (n = 2,700) — their Table 2
| Method | Spearman ρ | Pearson r | CCC | log-RMSE |
|---|---|---|---|---|
| NetMHCstabpan † | **0.876** | 0.532 | 0.500 | 0.629 |
| TLStab ‡ | 0.698 | 0.306 | 0.132 | 0.929 |
| ESM-2 Direct | 0.574 | 0.515 | 0.340 | 0.869 |
| ESM-2 Transfer | 0.745 | 0.714 | 0.682 | 0.704 |
| MINT Direct | 0.732 | 0.679 | 0.637 | 0.735 |
| **MINT Transfer** | **0.791** | **0.761** | 0.673 | 0.650 |

† trained on the test data — not a fair target. ‡ heavy peptide overlap with the test set.

**The honest state of the art is ρ ≈ 0.79.** Note NetMHCstabpan's ρ 0.876 comes with Pearson r of
only 0.532 — it ranks well but its absolute hours are poorly calibrated. MINT Transfer beats it on
every metric except ρ.

### The single most useful number here
Transfer learning matters as much as the fancy architecture:
- ESM-2: **0.574 → 0.745** by adding affinity pre-training (+0.171)
- MINT: 0.732 → 0.791 (+0.059)
- ESM-2 Direct → MINT Direct: 0.574 → 0.732 (+0.158) from cross-chain attention

So **a plain ESM-2 + regression head with no affinity pre-training lands at ρ ≈ 0.57.** That is the
number our naive baseline will produce. The cheap win is the two-stage curriculum, not a bigger model.

## Their architecture, briefly
- **MINT** = ESM-2 650M with *dual attention* per layer — separate intra-chain and cross-chain
  attention modules, pre-trained on ~96 M protein-protein interactions from STRING.
- Peptide and MHC are concatenated into one sequence with a parallel **chain-ID tensor** (0 = peptide,
  1 = MHC), no separator token. One forward pass for the whole complex.
- Head: mask out `<cls>/<eos>/<pad>`, mean-pool residues across both chains → 1280-d, then a
  2-layer MLP to a scalar.
- **Stage 1** affinity (NetMHCpan 4.1 corpus, 170,107 measurements, 109 alleles, `1 − log(IC50)/log(50000)` labels)
  → **Stage 2** stability (freeze more, lower LR; keeping the Stage-1 head weights beats re-initialising)
  → **Stage 3** assay conditioning.
- Hyperparameters: batch 16; LR 1e-4 direct / 3e-5 transfer; freeze 90 % of 33 layers for direct,
  70 % for transfer; 20 epochs BA, 200 epochs BS. Trained on 4× A6000 48 GB.

## The assay problem (their second contribution)
The same peptide-allele pair measured by different assays doesn't just shift — it **re-orders**:
| Assay pair | Spearman ρ |
|---|---|
| SPA vs purified fluorescence | 0.34 |
| SPA vs cellular fluorescence | 0.20 |
| cellular vs purified fluorescence | 0.72 |

On an independent IEDB holdout **every model drops below ρ 0.4**, even on SPA. Per-allele performance
is explained mostly by *what fraction of that allele's data was SPA* (ρ = 0.858) — i.e. it's
distribution shift, not biology.

**Fix:** freeze the whole backbone, bolt on a ~810 K-parameter **FiLM** conditioning layer taking an
assay embedding + a continuous temperature encoding, identity-initialised so it starts out
reproducing Stage 2. Huber loss (δ=1). That model is SPEARMINT. Purified fluorescence
ρ 0.259 → **0.567**; SPA preserved (0.346 → 0.359).

## Downstream
Affinity beats stability for *eluted ligand* (is it presented at all), but stability adds
complementary signal for *immunogenicity* and for per-patient neoantigen ranking across four clinical
cohorts. Framing: **affinity gates candidacy, stability ranks persistence within the gate.**

## Gaps they state or leave open — our opening
1. **No uncertainty quantification anywhere in the paper.** Serova's brief and the primer both want it.
2. **They don't treat the `t½ = 0` floor as censored** — `log(1+0) = 0` is fed in as a real value.
   We measured **5,679 of 28,166 rows (20 %) at exactly 0.0**. A censored likelihood or
   classify-then-regress is a defensible, novel-ish contribution nobody in the paper tried.
3. Assay-conditioned data sparse, especially cellular fluorescence (only 105 examples).
4. Only assay type + temperature are modelled; no other protocol covariates.
5. EL labels deliberately unused in pre-training.
6. Class II MHC untouched; rare-allele coverage thin.
7. No structure (co-folding) anywhere — the primer's approach 3 is still open.

## What this means for a 24-hour build
- Reproducing MINT Transfer end-to-end (two stages, 650M params, 200 epochs) is **not** a
  weekend-sized job on $150 of Modal credit. Don't promise it — and the brief doesn't ask for it.
- A frozen-embedding + head baseline is a few hours and lands near ρ 0.57.
- The gaps this paper leaves (uncertainty, the 0.0 censoring floor) line up almost exactly with what
  Serova's §5 asks us to explore (confidence metrics, log-likelihoods, masking and seed variation).
  That overlap is the opening: not a better score, but methods the published work didn't try.
- Keep the paper's *discipline* — identity-clustered splits, rank metrics, honest reporting of what
  leaked — and drop its *goal*, which was topping a benchmark.
