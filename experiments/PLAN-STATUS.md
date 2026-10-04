# Plan versus delivery

An accounting of what the plan of record asked for, what was delivered, what was added that nobody
planned, and what is still open. Written 2026-10-04 09:40, with 5 hours to the deadline.

Sources: `../ARCHITECTURE.md` (the build spec), `../docs/GeoStab-FT-build-plan.pdf` (the plan of
record it serves), and the experiment proposals in this folder.

---

## 1. The sequence ladder — ARCHITECTURE.md §3

| planned rung | status | result |
|---|---|---|
| **B0** global then per-allele median | ✅ done | 0.563 pooled, within-allele *undefined by construction* — the floor every other number is read against |
| **B1** positional one-hot + pseudosequence + allele | ✅ done | **0.780 / 0.633 val · 0.806 / 0.645 test** — the best model in the project |
| **B1′** one-hot over 75 alleles instead of HLA sequence | ✅ done | 0.748 / 0.557 — the control fired exactly as designed |
| **F150** frozen ESM-2 150M, mean-pooled | ✅ done | 0.593 / 0.278 — barely clears the no-peptide floor |
| **X150** frozen ESM-2 150M, residue interaction head | ✅ done | 0.754 / 0.558 — **H2 answered**, the head is worth +0.161 |
| **L150** LoRA r=8 α=16 on K/V | ✅ done | 0.757 / 0.572 — **H1 answered**, +0.003 [−0.012, +0.018], spans zero |
| U150 partial unfreeze *(marked optional)* | ❌ not done | Deliberately skipped: L150 landed inside seed noise, so a more expensive adaptation had no hypothesis left to test |

**7 of 7 non-optional rungs delivered.** The one skipped was marked optional in the plan and was
rendered uninformative by L150's result.

---

## 2. Six ways to use a foundation model — ARCHITECTURE.md §4

| route | status | result |
|---|---|---|
| Embeddings → head | ✅ | F150 / X150 |
| **Log-likelihood / perplexity** | ◐ partial | Unconditioned PLL done: **ρ +0.002**, a clean null. HLA-conditioned variants *not* run — they need a model that accepts two chains, and concatenation does not establish conditioning |
| **Internal confidence** | ✅ | Seed disagreement and residual correlation (Run 13); ensemble +0.009 [+0.001, +0.017] |
| **Masked-input scoring** | ✅ | Per-position ablation (Run 5) and masked-position likelihood (Run 19), used for anchor attribution as the plan intended |
| **Seed ensemble** | ✅ | Three seeds throughout; spread reported on every figure |
| Fine-tuning | ✅ | L150 |

**5½ of 6.** The gap is conditioned likelihood, and it is a genuine architectural blocker rather
than something skipped for time.

---

## 3. The structural arm — ARCHITECTURE.md §5

The plan marked this **✂ killable** and scoped it as B1 (generate structures) plus B2 (the
geometry residual).

| planned | status | result |
|---|---|---|
| Boltz-2 frozen → coordinates | ✅ done | boltz 2.2.1 runs; 48 folds across 24 peptides |
| D[9,182] minimum heavy-atom distances | ✅ done | computed and shipped back |
| **B2** — the matched `G_conf` vs `G_pair` residual | ❌ **deliberately not run** | Underpowered by ~10× at any foldable cohort. Dropped on analysis, not on time, and the reasoning is recorded |
| β2m pinned, full construct | ⚠️ **deviated** | Only the 182-aa α1/α2 domain exists in the dataset. β2m packs under α3, which we also lack, so it would have had nothing to dock against. **Recorded as a deviation** |
| K=3 pose ensemble ablation | ◐ | 2 diffusion samples per peptide, used as a noise floor rather than as an ensemble |

The arm delivered a **cleaner negative than the plan anticipated**: geometry varies with the
peptide at 4.1× the model's own noise, but does not track half-life (Mantel r −0.111, p 0.116,
**92% power at r = 0.2**), including at the anchor positions where it is most precisely resolved.

---

## 4. The zero labels — ARCHITECTURE.md §6

| planned | status | result |
|---|---|---|
| Hurdle model as a **declared** experiment | ✅ done | No gain: −0.001 on both framings, despite the classifier reaching AUC 0.886. Decision closed: no censored head, for a measured reason |

---

## 5. Reporting — ARCHITECTURE.md §7

| planned | status |
|---|---|
| Spearman with paired peptide-component bootstrap, 2,000 draws, seed 2026 | ✅ Run 7 |
| Report interval **unavailable** below 95% valid draws | ✅ fired once, honestly (B1 − B0b within-allele) |
| log-MAE / RMSE | ✅ throughout |
| **operational hours-MAE** | ✅ Run 20 |
| within-allele metrics | ✅ primary metric throughout |
| counts, seed variation, GPU cost, latency | ✅ Run 18 |
| coverage and failures | ✅ per-allele table, Run 20 |

### The four P0 defects the plan flagged

| | status |
|---|---|
| **A1** audit asserted distance ≤1 while the contract is ≤2 | ✅ fixed |
| **M1** `peptide_lookup_score` returned 0.0 as a sentinel | ✅ fixed — returns `None` |
| **D1** leakage probe normalised over all rows | ✅ **fixed 2026-10-04** |
| **D4** detector plants a 1-edit positive against a 2-edit contract | ✅ **fixed 2026-10-04** |

**All four P0 defects are closed.** D1: statistics are now fitted on training rows only - the
break still falls between distance 2 and 3, so the Hamming <= 2 threshold stands, and is now
justified without touching evaluation labels. D4: the detector is probed at the boundary in both
directions, and `scripts/verify_split.py` re-runs the whole contract against the frozen files.

---

## 6. Done that nobody planned

These are not in `ARCHITECTURE.md`, the plan of record, or any experiment proposal. Several turned
out to carry more weight than planned items.

| | why it exists |
|---|---|
| **Layer sweep** (Run 11) | Every ESM result read `last_hidden_state`. "You used the wrong layer" was the strongest available attack and went unanswered |
| **Scale sweep 8M→650M** (Run 12) | One model size cannot distinguish "the approach fails" from "the model was too small" |
| **Learning curve** (Run 14) | Data efficiency is the standard defence of pretraining; nobody in this literature reports one |
| **Allele split** (Run 15) | `allele_split.csv` existed and had never been used; it tests generalisation to an unseen HLA |
| **Shuffled-label control** (Run 10) | No negative control existed, and its absence is conspicuous |
| **Empirical noise ceiling** (Run 16) | Zero replicates made this look impossible; near-identical allele pairs solved it |
| **Boltz vs pseudosequence overlap** (Run 17) | Turned "does structure help?" into "can a 2026 folding model rediscover 2000s crystallography?" — **31/31, p=0.0001** |
| **Complementarity** (Run 13) | "Worse alone" and "adds nothing" are different claims and only the second justifies the conclusion |
| **Cost-benefit table** (Run 18) | The brief asks whether foundation models are *worth it*; nothing had priced them |
| **Motif recovery, both families** (Run 19) | Whether either model learned real immunology, measured against published anchors |
| **Cross-chain concatenation** (Run 21) | A reachable test of the MINT hypothesis when MINT itself was not reachable |
| **BLOSUM62 encodings** (Run 23) | The peptide side was the one component validated zero ways |
| **`scripts/audit_results.py`** | Cross-checks every number quoted anywhere against the file that produced it. 81/81 clean |
| **Split sha256 assertion** | `features.load()` refuses to read a label against a changed split |
| **Test-read guard** | Re-reading the test set requires `--force` and prints the earlier result |

---

## 7. Still open

| | cost | note |
|---|---|---|
| **Calibration rows** → prediction intervals | ~1h | Needs rows carved from *train*. B1's slope of 0.978 is a set property, not per-prediction uncertainty |
| **L150 on test** | ~75 min, $6.59 | Omitted deliberately; the test table has no fine-tuned rung and says so |
| **Conditioned likelihood** | needs a two-chain model | Architectural blocker, not a time one |
| **MINT / ESM-C / ProtT5 / SaProt** | ~2h each | MINT demoted after the cross-study comparison showed our head reproduces its gain |
| **Affinity pre-training** | ~4h | Largest published lever (0.574 → 0.745); the decontamination is the work |
| **Pinned environment** | ~10 min | Run 14 found B1 differs by ~0.010 between Anaconda and PyPI scikit-learn |
| **Demo** | ~2h | Unowned |

---

## 8. The honest summary

**Delivered: every non-optional rung of the planned ladder, five and a half of six foundation-model
routes, the structural arm's generation half with a better-powered negative than planned, the
declared zero-label experiment, and the full reporting contract.**

**Dropped on analysis rather than time:** B2 (underpowered ~10×) and U150 (no hypothesis left
after L150).

**Added beyond the plan:** thirteen experiments and three structural guards, several of which —
the layer sweep, the scale sweep, the noise ceiling and the audit — are now load-bearing for the
conclusion in ways the original plan did not anticipate.

**The plan asked whether foundation models help. The answer is no, and the work since has mostly
been establishing that the answer survives every obvious objection to it.**
