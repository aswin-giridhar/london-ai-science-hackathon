"""When is a foundation model worth it? Learning curves for one-hot vs frozen ESM-2.

The question nobody in this literature answers
-----------------------------------------------
The standard defence of a pretrained encoder is **data efficiency**: it should win when labels are
scarce, because the pretraining substitutes for supervision. Our full-data result says one-hot
wins. That is compatible with ESM-2 still being the right choice at 2,000 rows instead of 22,532.

If the curves cross, the answer to "are foundation models useful for this problem" stops being yes
or no and becomes **a number of labels** -- which is far more useful to anyone deciding whether to
run one, and is exactly what the brief is asking.

Design
------
Subsample by **peptide cluster**, never by row. Rows inside a cluster are within 2 substitutions of
each other, so row subsampling would leave near-duplicates of the retained peptides in the training
set and quietly inflate the small-data points -- the same leakage the frozen split exists to
prevent, reintroduced through the back door.

Validation is **always the full validation split**, so every point on the curve is measured against
the same rows and the curve is internally comparable.

Both models see exactly the same subsample at each fraction and seed.

    python -m modal run modal_app.py::curve
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import json
import time
from pathlib import Path

import numpy as np

import baselines
import esm_heads
import features
import metrics

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))

FRACTIONS = (0.05, 0.10, 0.25, 0.50, 1.00)
SEEDS = (42, 43, 44)
BLOCKS = ("peptide", "pseudoseq", "allele")


def cluster_subsample(meta, frac, seed):
    """Keep whole peptide clusters until `frac` of the training rows are covered."""
    rng = np.random.default_rng(seed)
    clusters = meta.cluster_id.to_numpy()
    uniq = np.unique(clusters)
    rng.shuffle(uniq)
    target = frac * len(meta)
    keep, total = set(), 0
    for c in uniq:
        if total >= target:
            break
        keep.add(c)
        total += int((clusters == c).sum())
    return np.isin(clusters, list(keep))


def main():
    import torch
    from torch import nn

    esm_heads.DEVICE = device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    OUT.mkdir(exist_ok=True)
    t_start = time.time()

    z, pep_index, all_index = esm_heads.load_cache()
    m = features.load()
    tr_meta = m[m.split == "train"].reset_index(drop=True)
    va_meta = m[m.split == "val"].reset_index(drop=True)
    vocab = sorted(tr_meta.allele.unique())

    # one-hot features, built once and indexed per subsample
    Xtr_full = features.build(tr_meta, BLOCKS, vocab)
    Xva = features.build(va_meta, BLOCKS, vocab)
    ytr_full, yva = features.target(tr_meta), features.target(va_meta)
    allele_va = va_meta.allele.to_numpy()

    # frozen embeddings, LayerNormed exactly as in the scale sweep
    banks = {}
    for k, src in (("pep_res", "peptide_residue"), ("hla_res", "hla_residue")):
        t = torch.tensor(z[src].astype(np.float32)).to(device)
        banks[k] = torch.nn.functional.layer_norm(t, (t.shape[-1],))
    banks["pep_pool"] = banks["pep_res"].mean(1)
    banks["hla_pool"] = banks["hla_res"].mean(1)
    d_res = banks["pep_res"].shape[-1]
    X150 = esm_heads.build_models(torch, nn, d_in_pooled=2 * d_res,
                                  d_res=d_res)["X150_interaction_head"]

    pi_va, ai_va = esm_heads.row_indices(va_meta, pep_index, all_index)
    val_t = {"pi": torch.from_numpy(pi_va).to(device),
             "ai": torch.from_numpy(ai_va).to(device),
             "y": torch.tensor(yva.astype(np.float32)).to(device),
             "allele": allele_va, "meta": va_meta}

    print(f"train {len(tr_meta):,} rows in {tr_meta.cluster_id.nunique():,} clusters")
    print(f"val   {len(va_meta):,} rows (ALWAYS full, never subsampled)")
    print("test  NOT READ\n")

    results = {}
    for frac in FRACTIONS:
        row = {"fraction": frac}
        for seed in SEEDS:
            mask = cluster_subsample(tr_meta, frac, seed)
            sub = tr_meta[mask].reset_index(drop=True)
            row.setdefault("n_rows", []).append(int(mask.sum()))
            row.setdefault("n_clusters", []).append(int(sub.cluster_id.nunique()))

            # --- B1: one-hot MLP
            mdl, _, _ = baselines.fit_mlp(Xtr_full[mask], ytr_full[mask], Xva, yva, seed=seed)
            row.setdefault("B1_pred", []).append(mdl.predict(Xva))

            # --- X150: frozen ESM-2 with the interaction head
            pi, ai = esm_heads.row_indices(sub, pep_index, all_index)
            tr_t = {"pi": torch.from_numpy(pi).to(device),
                    "ai": torch.from_numpy(ai).to(device),
                    "y": torch.tensor(features.target(sub).astype(np.float32)).to(device),
                    "allele": sub.allele.to_numpy(), "meta": sub}
            _, _, _, pv = esm_heads.train_one(torch, nn, X150, banks,
                                              {"train": tr_t, "val": val_t}, seed)
            row.setdefault("X150_pred", []).append(pv)

        for name in ("B1", "X150"):
            ens = np.mean(row.pop(f"{name}_pred"), axis=0)
            r = metrics.evaluate(yva, ens, allele_va)
            row[name] = {"pooled": r["spearman_pooled"], "within": r["spearman_within_allele"]}
        results[f"{frac:.2f}"] = row
        print(f"  {frac:>5.0%}  n={np.mean(row['n_rows']):>7.0f} rows  "
              f"B1 {row['B1']['pooled']:.3f}/{row['B1']['within']:.3f}   "
              f"X150 {row['X150']['pooled']:.3f}/{row['X150']['within']:.3f}   "
              f"delta {row['X150']['pooled'] - row['B1']['pooled']:+.3f}", flush=True)

    print("\nLEARNING CURVE (validation always full, test not touched)\n")
    print(f"  {'train rows':>11}{'B1 pooled':>12}{'X150 pooled':>13}{'delta':>9}"
          f"{'B1 within':>12}{'X150 within':>13}{'delta':>9}")
    crossed = None
    for frac in FRACTIONS:
        r = results[f"{frac:.2f}"]
        dp = r["X150"]["pooled"] - r["B1"]["pooled"]
        dw = ((r["X150"]["within"] or 0) - (r["B1"]["within"] or 0))
        print(f"  {np.mean(r['n_rows']):>11,.0f}{r['B1']['pooled']:>12.3f}"
              f"{r['X150']['pooled']:>13.3f}{dp:>+9.3f}"
              f"{(r['B1']['within'] or float('nan')):>12.3f}"
              f"{(r['X150']['within'] or float('nan')):>13.3f}{dw:>+9.3f}")
        if dp > 0 and crossed is None:
            crossed = np.mean(r["n_rows"])

    print()
    if crossed is not None:
        print(f"  CROSSOVER: ESM-2 is ahead at about {crossed:,.0f} training rows and behind at")
        print(f"  the full {len(tr_meta):,}. The answer to 'is a foundation model useful here' is")
        print(f"  therefore a label budget, not a yes or no.")
    else:
        print(f"  NO CROSSOVER: one-hot leads at every fraction down to "
              f"{min(FRACTIONS):.0%} ({np.mean(results[f'{min(FRACTIONS):.2f}']['n_rows']):,.0f} rows).")
        print("  The data-efficiency defence of pretraining does not hold on this endpoint --")
        print("  which is a stronger negative than the full-data comparison alone.")

    (OUT / "learning_curve.json").write_text(json.dumps(results, indent=2, default=float),
                                             encoding="utf-8")
    print(f"\nwrote {OUT / 'learning_curve.json'}")
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
