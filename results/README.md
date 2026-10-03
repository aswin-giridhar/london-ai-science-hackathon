# Results log

Every run that produced a number, what it showed, and what it does **not** establish. Append to this
rather than overwriting — superseded numbers stay, with the reason they were superseded.

Regenerate any row with the command given. The test split has **not been read by anything yet**.

---

## Run 1 — baselines, single seed · 2026-10-03 21:36

`python src/baselines.py` with `SEEDS = (42,)`. **Superseded by Run 2** — kept because the
difference between them is itself a finding.

| Rung | ρ pooled | ρ within-allele |
|---|---|---|
| B1 peptide + pseudoseq + allele | 0.751 | 0.580 |
| B1′ peptide + allele only | 0.688 | 0.465 |
| **delta (B1 − B1′)** | **+0.063** | **+0.115** |

## Run 2 — baselines, three seeds · 2026-10-03 21:42

`python src/baselines.py` (seeds 42, 43, 44; reported model is the seed-ensemble mean).

| Rung | n | ρ pooled | ρ within-allele | r (log) | RMSE log |
|---|---|---|---|---|---|
| B0a global median | 2,817 | **undefined** | undefined | undefined | 1.143 |
| B0b per-allele median | 2,817 | **0.563** | **undefined** | 0.499 | 0.977 |
| B1 peptide + pseudoseq + allele | 2,817 | **0.780** | **0.633** | 0.791 | 0.666 |
| B1′ peptide + allele only | 2,817 | 0.748 | 0.557 | 0.758 | 0.712 |

Per-seed pooled ρ — B1: 0.751 / 0.760 / 0.755 · B1′: 0.688 / 0.692 / 0.702.

### What these runs establish

**1. Pooled Spearman must be read against 0.563, not against zero.**
B0b predicts the training median of whichever allele a row belongs to. It contains **no peptide
information whatsoever**, and its within-allele correlation is *undefined by construction* because it
predicts a constant inside each allele. It still scores **0.563 pooled**.

That is the between-allele confound made concrete. B1's 0.780 is a gain of **+0.217 over knowing
nothing about the peptide**, which is the honest framing. Any rung reported later without this
context overstates itself.

**2. The HLA sequence adds over the allele label — but half as much as one seed suggested.**

| | Run 1 (1 seed) | Run 2 (3 seeds) | seed spread |
|---|---|---|---|
| delta pooled | +0.063 | **+0.032** | 0.014 |
| delta within-allele | +0.115 | **+0.076** | 0.010 |

Both Run 2 deltas exceed the seed spread, so the effect is real. But **the single-seed estimate was
roughly double the three-seed one.** One run gives a value; several tell you whether it means
anything. Quote Run 2.

**3. Seed ensembling is a free gain.** Individual B1 seeds score 0.751–0.760 pooled; their mean
scores **0.780**. About +0.025 for no extra training, and it is why the ensemble is the reported model.

**4. Early stopping is unstable for B1′.** Best epoch varied 8 / 25 / 15 across seeds, against a
tight 8 / 9 / 10 for B1. The weaker feature set has a flatter validation curve, so its epoch choice
is noisier — a reason to prefer the ensemble over any single run.

### What these runs do NOT establish

- **Nothing about test performance.** These are validation numbers and validation drove early
  stopping, so they are mildly optimistic.
- **No confidence intervals.** Seed spread is not a CI. Use `paired_bootstrap()` from
  `scripts/power_analysis.py` on saved predictions for that; it resamples peptide clusters.
- **No mechanism for finding 2.** The pseudosequence is *constant within an allele*, so it cannot
  directly rank peptides inside one. The plausible explanation is parameter sharing — the model can
  learn a peptide-residue/groove-residue interaction and transfer it across similar alleles — but
  that is a **hypothesis**, untested. A direct check would compare per-allele gains against allele
  similarity.
- **Nothing about foundation models.** No ESM-2 rung has been run.

### Protocol actually used

- Trained on the 22,532 training rows; epoch selected on the 2,817 validation rows.
- sklearn's own `early_stopping` carves a *random* slice out of training and would ignore the
  peptide-cluster structure, so the epoch loop is stepped manually against the frozen split.
- Target `log1p(thalf_hours)`; features are fixed one-hot encodings with nothing fitted.
- Allele vocabulary derived from **training rows only**; an unseen allele encodes as all-zero.
  (0 alleles appear in val but not train, so this path was not exercised.)
- MLP 256→64, ReLU, alpha 1e-4, lr 1e-3, batch 256, max 60 epochs, patience 8.

## Run 3 — frozen ESM-2 150M embedding cache · 2026-10-03 22:15

`python src/embed.py`. Not a model result — the input every ESM rung reads. Logged because two
of its properties constrain what the later numbers can mean.

| | |
|---|---|
| Model | `facebook/esm2_t30_150M_UR50D`, 148M params, hidden 640 |
| Encodes | **5,708** (5,633 distinct peptides + 75 distinct HLA domains) instead of 28,166 — 4.9x fewer |
| Output | `peptide_residue (5633, 9, 640)`, `hla_residue (75, 182, 640)`, float16 |
| Cost | **188 s on CPU**, 4 threads. 83 MB on disk |

### What this run established

**1. The cache is only valid while the encoder is frozen and the chains are encoded separately.**
Recorded as `VALID_ONLY_IF` inside `cache/esm2_150m.manifest.json`. The moment LoRA trains (L150) or
the two chains attend to each other, an HLA embedding starts depending on which peptide it is paired
with, and every number computed from this file becomes wrong. L150 must re-encode.

**2. `pooler.dense` is randomly initialised.** `EsmModel.from_pretrained` reports it MISSING from the
checkpoint and newly initialises it. `embed.py` uses `last_hidden_state` and never touches
`pooler_output`, so this is harmless **here** — but anything that reaches for the pooler would be
mean-pooling an untrained projection. Noted so a later rung does not pick it up by reflex.

### Two defects this run exposed, both fixed

| | |
|---|---|
| **Unreadable cache, exit code 0** | `hla.allele.to_numpy()` returns dtype `object`; `np.load(allow_pickle=False)` refuses object arrays. The first encode wrote an 83 MB file that `esm_heads.py` could not open, and the script exited **0**. Fixed by forcing `dtype="U"`, and `embed.py` now reads its own output back with `allow_pickle=False` and asserts shapes and keys before claiming success. An exit code is not a check |
| **10.5 GB of duplicated tensors** | `esm_heads.py` originally gathered a per-row `(182, 640)` HLA tensor — 10.5 GB for the training split. It now holds the unique-sequence banks (**179 MB measured**) and indexes them per batch. Same 4.9x redundancy `embed.py` exploits, one layer down |

### Environment note

Anaconda's MKL and pip's torch each load `libiomp5md.dll`, aborting the process. `esm_heads.py` sets
`KMP_DUPLICATE_LIB_OK=TRUE` with `OMP_NUM_THREADS=MKL_NUM_THREADS=4` before importing numpy. The
flag's own documentation warns it "may silently produce incorrect results", so it was **checked, not
trusted**: torch matmul vs numpy max abs diff **0.0**, a seeded X150 forward repeats bit-identically,
and 4-thread vs 1-thread differs by **8.2e-08** (float32 reduction order). Recorded so the result is
reproducible and the risk is visible rather than buried in a shell variable.

## Run 4 - F150 / X150, frozen ESM-2 150M, two heads - 2026-10-03 22:53 - CURRENT

`python -m modal run modal_app.py` on an **NVIDIA A10**, seeds 42/43/44, reported model is the
seed-ensemble mean. The encoder is frozen throughout; only the heads train.

| Rung | n | rho pooled | rho within-allele | r (log) | RMSE log |
|---|---|---|---|---|---|
| F150 mean-pooled head | 2,817 | **0.593** | **0.278** | 0.593 | 0.875 |
| X150 residue interaction head | 2,817 | **0.754** | **0.558** | 0.765 | 0.704 |

Per-seed pooled - F150: 0.588 / 0.574 / 0.581 - X150: 0.715 / 0.710 / 0.690.

### H2 answered: the head design dominates

| | X150 | F150 | delta | seed spread | verdict |
|---|---|---|---|---|---|
| pooled | 0.754 | 0.593 | **+0.161** | 0.025 | larger than seed noise |
| within-allele | 0.558 | 0.278 | **+0.280** | 0.039 | larger than seed noise |

Nothing changed but the head. Same frozen embeddings, same rows, same split, same budget. Swapping
mean-pooling for residue-level cross-attention **doubles** within-allele Spearman.

Mean-pooling a 9-mer averages away position, and position is where anchor residues live - P2 and P9
dock into the B and F pockets. F150's within-allele 0.278 is what survives that averaging. This is
the cheapest result on the board and it indicts a standard practice: "ESM-2 embeddings + MLP" is
usually implemented as mean-pool, and on this endpoint that choice costs more than the model size.

### The ladder as it now stands (validation, same split, all internal)

| Rung | rho pooled | rho within | Uses a protein LM? |
|---|---|---|---|
| B0a global median | undefined | undefined | no |
| B0b per-allele median | 0.563 | undefined | no |
| F150 frozen ESM-2, mean-pooled | 0.593 | 0.278 | **yes** |
| X150 frozen ESM-2, cross-attention | 0.754 | 0.558 | **yes** |
| B1' one-hot peptide + allele ID | 0.748 | 0.557 | no |
| B1 one-hot peptide + pseudoseq + allele | **0.780** | **0.633** | no |

**The uncomfortable reading, and the one we should report.** X150 (0.754 / 0.558) and B1' (0.748 /
0.557) are the same number to within seed noise. A frozen 150M protein language model with a
residue-level interaction head performs **no better than a one-hot peptide encoding plus an allele
lookup**. And B1 - which adds the 34-residue HLA pseudosequence and still uses no protein language
model at all - beats both.

Note also that F150, the configuration closest to a conventional "ESM-2 embeddings" baseline,
clears the **no-peptide-information** floor of 0.563 by only +0.030 pooled.

### Cross-check: the GPU result is not a new system

Run 4 was produced remotely. The local CPU run was left going until the remote one returned, so the
two could be compared on a known-good reference before the CPU job was killed:

| | CPU | A10 | diff |
|---|---|---|---|
| F150 seed 42 | 0.581 | 0.588 | 0.007 |
| F150 seed 43 | 0.571 | 0.574 | 0.003 |
| F150 seed 44 | 0.571 | 0.581 | 0.010 |
| X150 seed 42 | 0.722 | 0.715 | 0.007 |

All inside the 0.025 seed spread. The residual difference is expected: CUDA reductions run in a
different order, so early stopping selects a different epoch (CPU 30, GPU 26 for X150 seed 42).

### Cost

| | |
|---|---|
| Remote compute | **64 s** total for both rungs, 3 seeds each |
| Wall clock | 77 s including image upload and container start |
| Same work on 4 CPU threads | ~65 min (X150 measured at 1,242 s for one seed) |
| Speed-up | **~65x** |
| Per optimiser step | 6.6 ms - the model is 561K parameters, so this is launch-bound, not FLOP-bound. A faster GPU would not help this rung |

### What Run 4 does NOT establish

- **Nothing about fine-tuning.** Every ESM rung here is frozen. L150 (LoRA r=8 on K/V) is the
  brief's actual question and has not been run. Adaptation could reorder this table entirely.
- **Nothing about test performance.** Validation numbers, and validation drove early stopping.
- **No confidence intervals.** Seed spread is not a CI. Use `paired_bootstrap()` from
  `scripts/power_analysis.py` on `esm_heads_val_predictions.json`.
- **Not a claim about ESM-2 in general.** It is a claim about ESM-2 **150M, frozen**, with these two
  heads, on this endpoint and this split. The 650M model is untested here.
- **No mechanism for why one-hot competes.** A plausible explanation is that ESM-2 embeddings are
  contextual - the vector at P2 already mixes the whole 9-mer - while one-hot is position-pure, and
  position-purity may suit an anchor-driven endpoint with 22,532 training rows. That is a
  **hypothesis**, untested.

## Files

| File | What |
|---|---|
| `baselines_val_metrics.json` | All metrics per rung, including per-seed values and spread |
| `baselines_val_predictions.json` | Validation predictions per rung, aligned with peptide / allele / cluster_id — the input `paired_bootstrap()` needs |
| `esm_heads_val_metrics.json` | F150 / X150 metrics, per-seed values and spread |
| `esm_heads_val_predictions.json` | F150 / X150 validation predictions, same alignment |

## How to reproduce

| Run | Command |
|---|---|
| 1, 2 | `python src/baselines.py` |
| 3 | `python src/embed.py` (writes `cache/esm2_150m.npz`; needs `SSL_CERT_FILE` set to the local bundle) |
| 4 | `python -m modal run modal_app.py` (GPU), or `python src/esm_heads.py` (CPU, ~65 min) |

`src/features.py` asserts the frozen split's sha256 before reading any label, so a run against a
changed split fails immediately rather than reporting a number against different data.
