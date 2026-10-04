# Other models, other routes: a bioengineering read on where the signal might actually be

Companion to `IMPROVEMENTS.md`, which covers evaluation rigour. This one asks a different
question: **given what we measured, which untried approach is most likely to work, and why?**

Sources are cited inline. Claims are marked **[measured]** (by us), **[literature]**, or
**[hypothesis]**. The distinction matters because most of this document is the third kind.

---

## 1. The diagnosis: we tested a single-chain model on a two-chain problem

Three of our own measurements point the same way:

- **[measured]** F150 (mean-pooled ESM-2) scores 0.593 pooled, barely above the 0.563 no-peptide
  floor. X150 (per-residue cross-attention) reaches 0.754. The *head* recovered the signal, not
  the embedding.
- **[measured]** ESM-2's masked-position likelihood profile over these peptides is **flat**, and
  rank-correlates **−0.03** with the positions the trained model actually relies on. Its
  per-peptide likelihood predicts stability at **ρ +0.002**.
- **[measured]** LoRA adaptation of the encoder adds +0.003 pooled, interval [−0.012, +0.018].

Read together: ESM-2 *contains* usable information, but its own notion of salience is unrelated to
this endpoint, and adaptation at r=8 does not reorganise it.

**The mechanistic reason is architectural.** ESM-2 is trained on *single* sequences with a masked
token objective. Peptide–HLA stability is not a property of either chain; it is a property of the
**interface**. We fed the model two chains separately and asked a head to reconstruct the
interaction that the encoder never represented.

> ## ⚠ CORRECTION, 2026-10-04 09:10
>
> **The section below contained an error, and correcting it produced the better result.**
>
> I presented the 2026 preprint *"Peptide:MHC Binding Stability Prediction Using Protein Language
> Models"* as newly-discovered literature found by search. **It is SPEARMINT** — the paper already
> in `archive/background/spearmint.pdf` and summarised in `context/SPEARMINT_SUMMARY.md` since the
> first hours of this project. I wrote this section from a search snippet without checking our own
> context folder, where line 86 already said: *"ESM-2 Direct → MINT Direct: 0.574 → 0.732 (+0.158)
> from cross-chain attention."*
>
> Having now read the numbers, the recommendation below **reverses**:
>
> | | pooled ESM-2 | interaction-aware | gain |
> |---|---|---|---|
> | SPEARMINT, cross-chain **pretraining** (MINT) | 0.574 | 0.732 | **+0.158** |
> | **Us**, cross-attention **head** on vanilla ESM-2 | 0.593 | 0.754 | **+0.161** |
>
> The gain SPEARMINT attributes to MINT's interaction-aware pretraining — 650M parameters, a
> modified MLM objective, 96M protein–protein interactions from STRING-DB — is **reproduced to
> within 0.003 by a 561K-parameter head on unmodified ESM-2 150M.**
>
> Absolute scores are not comparable: their split clusters at 80% identity (≈1 substitution on a
> 9-mer) while ours uses Hamming ≤2, which is stricter. But each delta is measured against that
> study's **own** pooled baseline, so delta-against-delta is a fair comparison, and the two agree.
>
> **So MINT is no longer the top-ranked next experiment.** Its headline gain appears to be
> something the X150 head already captures. The sharper remaining question is whether cross-chain
> *attention* adds anything **on top of** a cross-attention head — which `src/concat_encoding.py`
> tests directly by encoding both chains as one sequence.
>
> The ranked plan in §8 should be read with MINT demoted accordingly.

### This is a known gap with a named fix

[MINT (Multimeric INteraction Transformer)](https://www.nature.com/articles/s41467-025-67971-3)
extends ESM-2 650M with **cross-chain attention**, trained with a modified MLM objective over ~96M
protein–protein interactions from STRING-DB. The paper states the problem in the same terms we
arrived at empirically: *concatenating input sequences risks degrading embedding quality by
treating all sequences as a single unified entity*. **[literature]**

And it has already been applied to exactly our endpoint. A 2026 preprint,
[*Peptide:MHC Binding Stability Prediction Using Protein Language Models*](https://www.biorxiv.org/content/10.64898/2026.06.28.735023v1)
(Karthikeyan, Vincent, Rubinsteyn), fine-tunes MINT for quantitative pMHC-I half-life and reports
that **MINT improves stability prediction over standard ESM-2**. **[literature]**

### Why this is good news for our submission, not bad

It does not contradict our result — it *explains* it, and it upgrades what we can claim:

> We did not find that foundation models are useless here. We found that a **single-chain**
> foundation model is, measured four different ways, and we identified the architectural reason:
> its salience is orthogonal to the interface. The literature's fix for precisely this gap is an
> interaction-aware encoder, and there is a published result applying one to this endpoint.

That is a sharper and more defensible position than either "FMs don't work" or "FMs work". **It is
also a falsifiable prediction we could test tonight** — and testing your own explanation is worth
more than any additional negative.

**Action:** run MINT with the X150 head. If MINT closes the gap to B1 and ESM-2 does not, the
whole project resolves into a single clean statement about *what kind* of pretraining matters.
Estimated ~2h (new weights, cross-chain input format). **Highest-information experiment available.**

---

## 2. Model families, ranked by expected information gain

| Model | What it would test | Cost | Expected value |
|---|---|---|---|
| **MINT** | interaction-aware pretraining — our own diagnosis | ~2h | **highest**: tests our explanation, and the literature says it should work |
| **ESM-2 650M** | is the negative an artefact of scale? | ~1h | high: cheap, and the published 0.574 used this size |
| **SaProt** | structure-aware tokens (3Di); we now have Boltz backbones | ~2h | high, and uniquely on-thesis: structure is the mechanism |
| **ESM-C** | newer objective, better compute/performance frontier | ~1.5h | medium |
| **ProtT5** | encoder–decoder, different corpus and objective | ~1.5h | medium: a genuinely different inductive bias |
| **ESM-IF1 / ProteinMPNN** | inverse folding — see §3 | ~2h | high, and the least explored route in this field |

**If only one: MINT.** If two: MINT and ESM-2 650M, because together they separate *architecture*
from *scale*, which is the question our single data point cannot answer.

---

## 3. Inverse folding — the route nobody in this field has taken seriously

The idea, stated for a non-specialist: an inverse-folding model answers *"given this 3D backbone,
which amino acid sequence would fold into it?"* Run it on a peptide–HLA complex and ask for the
log-likelihood of the peptide **that is actually there**. A peptide the model considers a natural
fit for that groove geometry scores high. No training, no labels.

**[literature]** This works elsewhere:
[zero-shot protein stability prediction by inverse folding](https://arxiv.org/pdf/2506.05596);
and on antibody–antigen binding affinity, ESM-IF1 reaches 0.334 correlation, ahead of ProteinMPNN's
0.301 ([AntiFold](https://arxiv.org/pdf/2405.03370)).
[LigandMPNN](https://www.nature.com/articles/s42004-023-01029-7) extends the family to
protein–ligand interfaces, and peptide binder design with inverse folding is established.

### The circularity trap, and the control that defuses it

`ARCHITECTURE.md` §5 already flags this and it is worth restating because it is subtle:

> If Boltz-2 generated the backbone **from** the peptide sequence, then scoring that same sequence
> against that backbone is partly circular. The model is being asked to recover an input it was
> given.

**The control is a fixed template.** Score every peptide for a given allele against **one
reference backbone** for that allele — a pose that does not depend on which peptide is being
scored. Then any discrimination is genuinely about sequence–structure compatibility, not about
recovering the generator's input.

Run both arms:

| arm | backbone | what a high score means |
|---|---|---|
| per-peptide | Boltz pose for *this* peptide | confounded with the generator |
| **fixed template** | one reference pose per allele | genuine compatibility — **this is the result** |

The *gap* between the two arms is itself a measurement of how much circularity was present, which
is a nice thing to be able to report.

**[hypothesis]** I would expect modest signal. Inverse folding scores side-chain packing
compatibility, which is plausibly related to kinetic stability but is not the same quantity — an
off-rate depends on the transition state, not only the bound state. Worth ~2h; worth reporting
either way.

---

## 4. Our structural null may have looked in the wrong place

**[measured]** Run 9: Boltz-2 poses vary with the peptide **4.13× above the model's own diffusion
noise**, but the variation does not track half-life — Mantel r = −0.111, p = 0.116, with **92%
power to detect r = 0.2** at n=24.

That is a real negative, but note precisely what it tested: the **mean |ΔD| averaged over all 280
contacting cells**. A stability-relevant signal could live in a handful of specific cells and be
diluted to nothing by that average.

**And we know exactly which cells to look at.** Run 5's ablation says the model relies on **P9 ≫
P2 > P1 > P3**, with P4–P8 contributing essentially nothing. Averaging over all nine peptide
positions therefore spends 5/9 of the signal budget on positions our own experiment says are
irrelevant.

### The follow-up this implies — cheap, and connects two results

Re-run the Mantel test using only the **P2 and P9 rows** of D, and separately only the P4–P8 rows
as a negative control. If anchor-position geometry tracks stability while bulge geometry does not,
that is a genuine structural finding, mechanistically coherent, and it rescues part of the
structural arm from an averaging artefact.

**Cost: ~20 min.** The matrices are already on disk in `results/boltz_variance.json`. No new folds.
This is the single best value-per-minute item in this document.

### Richer structural features, if that works

Raw distances are a crude summary. Physically motivated alternatives:

- **anchor burial depth** at P2 and P9 specifically — how deep into the B and F pockets
- **buried surface area** of the peptide
- **hydrogen-bond count** to the groove, especially the conserved N-/C-terminal network
- **backbone bulge height** across P4–P6 — the central arch is known to vary with peptide length
  and is implicated in presentation stability
- **interface pLDDT / PAE** as a confidence proxy (this was always `G_conf`'s job)

**[hypothesis]** Of these, anchor burial depth is the most likely to carry signal, because it is
the structural correlate of the anchor-residue motif that our own ablation recovered.

---

## 5. What would beat one-hot, without any foundation model?

Our best model is a one-hot MLP. That is an interesting result in itself, and it raises a question
worth one experiment: **is the ceiling the features, or the data?**

Cheap, biologically motivated encodings that are strictly richer than one-hot:

1. **BLOSUM62 rows instead of one-hot.** Encodes substitution similarity, so the model can
   generalise across chemically similar residues rather than treating every substitution as equally
   foreign. This is what NetMHCpan has always used. ~30 min, and a plausible free win.
2. **Physicochemical descriptors** per position — hydrophobicity, volume, charge, polarity. Five
   numbers per residue instead of twenty. ~30 min.
3. **Explicit pocket pairing.** Rather than hoping an attention head discovers it, directly pair
   the P2 residue with the B-pocket residues of the pseudosequence, and P9 with the F-pocket. An
   outer product restricted to the pairings that physically exist. **[hypothesis]** this is the
   most promising non-FM idea, because it hard-codes the interaction structure that X150 has to
   learn from 22,532 rows. ~1h.
4. **Per-allele anchor-motif match score** as a single engineered feature. ~45 min.

**[hypothesis]** If (3) matched or beat X150, the conclusion sharpens again: the useful content of
a foundation model here is recoverable from 34 pseudosequence residues plus the right inductive
bias — which is a strong, cheap, and slightly provocative claim.

---

## 6. Data-side levers (bigger than any model choice)

**[literature]** The largest published single gain on this endpoint is **affinity pre-training**:
ESM-2 0.574 → 0.745 in the SPEARMINT work. Nothing in our model ladder comes close to that
magnitude. The 2026 MINT paper follows the same recipe — pre-train on binding affinity, fine-tune
on half-life — and reports that predicted stability then provides **signal beyond binding
affinity** for eluted-ligand, immunogenicity, and neoantigen prioritisation.

| lever | why it is big | cost |
|---|---|---|
| **Affinity pre-training** (NetMHCpan 4.1 corpus) | the largest published effect, and the mechanism is transfer from ~100× more data | ~4h incl. decontamination |
| **Eluted-ligand (mass-spec) auxiliary task** | different measurement modality, same biology; multi-task regularisation | ~3h |
| **Joint affinity + stability multi-task head** | affinity and off-rate are related but non-identical; forcing both may sharpen each | ~2h |

**The decontamination is the hard part, and it is where a careless team loses.** Any external
corpus must be filtered against our **test clusters** at the same Hamming ≤2 threshold, and the
leakage report published, or the entire frozen-split discipline is void. That is why this is a
4-hour job and not a 1-hour job.

---

## 7. Literature worth citing in the write-up

| Work | Why it matters here |
|---|---|
| [MINT — *Learning the language of protein–protein interactions*](https://www.nature.com/articles/s41467-025-67971-3), Nat Commun 2025 | Names and fixes the exact architectural gap our results diagnose |
| [*Peptide:MHC Binding Stability Prediction Using Protein Language Models*](https://www.biorxiv.org/content/10.64898/2026.06.28.735023v1), bioRxiv 2026 | MINT applied to **our endpoint**; reports improvement over ESM-2 |
| [*Continued domain-specific pre-training of PLMs for pMHC-I binding*](https://arxiv.org/pdf/2507.13077) | Domain-adaptive pretraining as a cheaper alternative to a new architecture |
| [*Improved prediction of MHC-peptide binding using PLMs*](https://pmc.ncbi.nlm.nih.gov/articles/PMC10469926/) | Prior art on PLM embeddings for the adjacent affinity task |
| [NetMHCstab](https://pmc.ncbi.nlm.nih.gov/articles/PMC3893846/) / NetMHCstabpan | The incumbent; trained on this dataset, hence unusable as a comparator |
| [Zero-shot stability by inverse folding](https://arxiv.org/pdf/2506.05596) | The basis for the §3 route |
| [AntiFold](https://arxiv.org/pdf/2405.03370) | Inverse-folding likelihoods vs binding affinity, with concrete correlations |
| [Peptide binder design with inverse folding](https://www.nature.com/articles/s42004-023-01029-7) | Inverse folding on peptide complexes specifically |

One framing note for the write-up: **[literature]** the 2026 preprint observes that in-silico
stability prediction is *underexplored relative to affinity prediction, because half-life datasets
are sparse and expensive*. That is the strongest possible justification for this challenge existing
and for a careful negative result being worth publishing.

---

## 8. Ranked, given the time that is left

| # | Experiment | Cost | Why it earns the slot |
|---|---|---|---|
| 1 | **Anchor-restricted Mantel test** (§4) | **20 min** | Matrices already on disk. Connects Run 5 and Run 9. May rescue the structural arm from an averaging artefact |
| 2 | **ESM-2 650M** with the X150 head | ~1h | Separates architecture from scale; the published number used this size |
| 3 | **MINT** with the X150 head | ~2h | Tests our own explanation. Literature says it should work on this exact task |
| 4 | **Pocket-paired one-hot features** (§5.3) | ~1h | Hard-codes the interaction X150 must learn; cheap and provocative if it wins |
| 5 | **Inverse folding, fixed-template arm** (§3) | ~2h | Genuinely unexplored for this endpoint; circularity control already designed |
| 6 | BLOSUM62 encoding | ~30 min | Plausible free win over one-hot |
| 7 | Affinity pre-training | ~4h | Biggest published lever, but the decontamination makes it a next-day job |

**If the night allows only one: #1.** It costs twenty minutes, requires no new compute, and tests a
specific, mechanistically motivated hypothesis generated by our own prior result — which is the
best shape an experiment can have.
