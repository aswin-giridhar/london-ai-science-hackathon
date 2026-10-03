"""B0, B1 and B1' -- the supervised baselines every later rung is measured against.

    B0a  global median of log1p(t-half), training rows only
    B0b  per-allele median, falling back to the global median for an unseen allele
    B1   MLP on peptide one-hot (180) + pseudosequence one-hot (680) + allele one-hot (75)
    B1'  MLP on peptide one-hot (180) + allele one-hot (75)   -- HLA SEQUENCE REMOVED

Why B1' is the control that matters
-----------------------------------
There are only 75 distinct HLA sequences in the whole dataset, so "knowing the HLA sequence" and
"knowing which of 75 alleles this is" are nearly the same information. If B1' matches B1, the
pseudosequence adds nothing over a lookup table, and any later claim that a protein model
"understands the HLA" has to clear the lookup table first, not just the global mean.

Protocol
--------
Train on the training split. Select the epoch on validation. **The test split is not read.**
Predictions are written to results/ so paired_bootstrap() can compare rungs later on identical rows.

    python src/baselines.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor

import features
import metrics

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
SEEDS = (42, 43, 44)   # one run gives a value; several tell you whether it means anything
SEED = 42
MAX_EPOCHS = 60
PATIENCE = 8


def fit_mlp(Xtr, ytr, Xva, yva, seed=SEED):
    """One warm-started MLP, epoch by epoch, selected on OUR validation split.

    sklearn's own early_stopping carves a random slice out of the training rows, which would ignore
    the peptide-cluster structure. Stepping manually keeps selection on the frozen split.
    """
    model = MLPRegressor(
        hidden_layer_sizes=(256, 64),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        batch_size=256,
        max_iter=1,
        warm_start=True,
        random_state=seed,
        shuffle=True,
    )
    best = {"rho": -np.inf, "epoch": -1, "pred_va": None, "coefs": None}
    stale = 0
    for epoch in range(MAX_EPOCHS):
        model.fit(Xtr, ytr)  # warm_start=True -> one more pass each call
        pred_va = model.predict(Xva)
        rho = metrics._spearman(yva, pred_va)
        if rho is not None and rho > best["rho"] + 1e-5:
            best = {
                "rho": rho,
                "epoch": epoch,
                "pred_va": pred_va,
                "coefs": [c.copy() for c in model.coefs_],
                "inters": [b.copy() for b in model.intercepts_],
            }
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    # restore the selected epoch
    model.coefs_ = best["coefs"]
    model.intercepts_ = best["inters"]
    return model, best["epoch"], best["rho"]


def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    m = features.load()

    tr = m[m.split == "train"].reset_index(drop=True)
    va = m[m.split == "val"].reset_index(drop=True)
    print(f"train {len(tr):,} rows / {tr.peptide.nunique():,} peptides")
    print(f"val   {len(va):,} rows / {va.peptide.nunique():,} peptides")
    print("test  NOT READ\n")

    ytr, yva = features.target(tr), features.target(va)
    allele_vocab = sorted(tr.allele.unique())  # training rows only
    print(f"allele vocabulary from training rows: {len(allele_vocab)}")
    unseen = sorted(set(va.allele) - set(allele_vocab))
    print(f"alleles in val but not train: {len(unseen)}{' ' + str(unseen) if unseen else ''}\n")

    results, preds = {}, {"y_true_log": yva.tolist(), "allele": va.allele.tolist(),
                          "peptide": va.peptide.tolist(), "cluster_id": va.cluster_id.tolist()}

    # ---- B0a: global median -------------------------------------------------------------
    gmed = float(np.median(ytr))
    p = np.full(len(va), gmed)
    results["B0a_global_median"] = metrics.evaluate(yva, p, va.allele)
    preds["B0a_global_median"] = p.tolist()

    # ---- B0b: per-allele median, training rows only -------------------------------------
    by_allele = tr.assign(y=ytr).groupby("allele").y.median().to_dict()
    p = np.array([by_allele.get(a, gmed) for a in va.allele])
    results["B0b_per_allele_median"] = metrics.evaluate(yva, p, va.allele)
    preds["B0b_per_allele_median"] = p.tolist()

    # ---- B1 and B1' ----------------------------------------------------------------------
    rungs = {
        "B1_peptide_pseudoseq_allele": ("peptide", "pseudoseq", "allele"),
        "B1prime_peptide_allele_only": ("peptide", "allele"),
    }
    for name, blocks in rungs.items():
        Xtr = features.build(tr, blocks, allele_vocab)
        Xva = features.build(va, blocks, allele_vocab)
        per_seed, seed_preds = [], []
        for sd in SEEDS:
            t = time.time()
            model, epoch, _ = fit_mlp(Xtr, ytr, Xva, yva, seed=sd)
            pv = model.predict(Xva)
            seed_preds.append(pv)
            per_seed.append(metrics.evaluate(yva, pv, va.allele))
            print(f"  {name} seed {sd}: best epoch {epoch}, {time.time()-t:.1f}s, "
                  f"rho_pooled {per_seed[-1]['spearman_pooled']:.3f}")
        # the reported model is the seed ensemble mean, with per-seed spread recorded
        p = np.mean(seed_preds, axis=0)
        results[name] = metrics.evaluate(yva, p, va.allele)
        results[name]["features"] = list(blocks)
        results[name]["n_features"] = int(Xtr.shape[1])
        results[name]["seeds"] = list(SEEDS)
        for k in ("spearman_pooled", "spearman_within_allele"):
            vals = [r[k] for r in per_seed if r[k] is not None]
            results[name][f"{k}_per_seed"] = [round(v, 4) for v in vals]
            results[name][f"{k}_seed_spread"] = round(max(vals) - min(vals), 4) if vals else None
        preds[name] = p.tolist()

    print("\nVALIDATION RESULTS (test not touched)\n")
    for name, r in results.items():
        print(metrics.fmt(name, r))

    b1, b1p = results["B1_peptide_pseudoseq_allele"], results["B1prime_peptide_allele_only"]
    print()
    print("  The control: does the HLA SEQUENCE add anything over the allele LABEL?")
    for key, label in [("spearman_pooled", "pooled"), ("spearman_within_allele", "within-allele")]:
        a, b = b1[key], b1p[key]
        if a is None or b is None:
            print(f"    {label:<14} undefined"); continue
        spread = max(b1.get(f"{key}_seed_spread") or 0, b1p.get(f"{key}_seed_spread") or 0)
        verdict = "larger than seed noise" if abs(a - b) > spread else "WITHIN seed noise"
        print(f"    {label:<14} B1 {a:.3f}  vs  B1' {b:.3f}   delta {a - b:+.3f}"
              f"   (seed spread {spread:.3f} -> {verdict})")
    print("    (a small delta means the pseudosequence is close to a lookup table here)")

    (OUT / "baselines_val_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (OUT / "baselines_val_predictions.json").write_text(json.dumps(preds), encoding="utf-8")
    print(f"\nwrote {OUT/'baselines_val_metrics.json'} and predictions")
    print(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
