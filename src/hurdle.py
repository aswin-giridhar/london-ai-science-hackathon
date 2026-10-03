"""Experiment 003 -- hurdle model for the t-half = 0 point mass. A declared test, not a correction.

The situation
-------------
4,534 of 22,532 training rows (20.1%) and 579 of 2,817 validation rows (20.6%) have t-half = 0.
That is a genuine point mass, not a rounded tail: 0.0 holds 5,679 rows against 443 at 0.1 and
1,297 at 0.2 -- 13x its neighbour and non-monotonic.

Why this is an experiment and not a fix
---------------------------------------
The CSV does not establish a detection limit. The reporting grid is 0.1 h, so anything under
0.05 h lands on zero, but whether those rows are an assay floor (censored) or genuine non-binders
(truly zero) is undetermined. Assuming censoring and bolting on a censored likelihood would be
correcting for a mechanism we have not shown exists. So: build the hurdle model, compare it to a
plain regressor under identical conditions, and report whichever wins.

The comparison
--------------
    PLAIN    one regressor on all rows, target log1p(t-half)        <- this is B1 from Run 2
    HURDLE   classifier P(t-half > 0)  x  regressor on the 17,998 non-zero training rows

Scored three ways, because a hurdle model can win on one and lose on another:

    all validation rows          does modelling the point mass help overall?
    non-zero rows only           does the regressor improve when zeros stop dragging it?
    zero/non-zero separation     can the classifier tell them apart at all? (AUC)

Both arms get the same features, the same frozen split, the same seeds and the same early-stopping
rule, so the difference is the hurdle structure and nothing else.

    python src/hurdle.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import baselines
import features
import metrics

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__import__("os").environ.get("RESULTS_DIR", ROOT / "results"))

SEEDS = baselines.SEEDS
MAX_EPOCHS = baselines.MAX_EPOCHS
PATIENCE = baselines.PATIENCE
BLOCKS = ("peptide", "pseudoseq", "allele")      # the B1 feature set


def fit_classifier(Xtr, ztr, Xva, zva, seed):
    """Logistic MLP for P(t-half > 0), stepped on our validation split like fit_mlp.

    Selected on AUC rather than Spearman: this arm's job is separation, not ranking, and
    selecting it on the regression metric would optimise the wrong thing.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.neural_network import MLPClassifier

    clf = MLPClassifier(
        hidden_layer_sizes=(256, 64), activation="relu", alpha=1e-4,
        learning_rate_init=1e-3, batch_size=256, max_iter=1,
        warm_start=True, random_state=seed, shuffle=True,
    )
    best = {"auc": -np.inf, "epoch": -1, "p": None, "coefs": None}
    stale = 0
    for epoch in range(MAX_EPOCHS):
        clf.fit(Xtr, ztr)
        p = clf.predict_proba(Xva)[:, 1]
        auc = roc_auc_score(zva, p)
        if auc > best["auc"] + 1e-5:
            best = {"auc": auc, "epoch": epoch, "p": p,
                    "coefs": [c.copy() for c in clf.coefs_],
                    "inters": [b.copy() for b in clf.intercepts_]}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    clf.coefs_, clf.intercepts_ = best["coefs"], best["inters"]
    return clf, best["auc"], best["epoch"]


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)

    m = features.load()
    tr = m[m.split == "train"].reset_index(drop=True)
    va = m[m.split == "val"].reset_index(drop=True)
    vocab = sorted(tr.allele.unique())

    Xtr = features.build(tr, BLOCKS, vocab)
    Xva = features.build(va, BLOCKS, vocab)
    ytr, yva = features.target(tr), features.target(va)

    # z = 1 means "measurable", i.e. NOT on the point mass
    ztr = (tr.thalf_hours > 0).to_numpy().astype(int)
    zva = (va.thalf_hours > 0).to_numpy().astype(int)
    nz_tr = ztr == 1
    nz_va = zva == 1

    print(f"train {len(tr):,} rows, {len(tr) - nz_tr.sum():,} zeros ({100 * (1 - nz_tr.mean()):.1f}%)")
    print(f"val   {len(va):,} rows, {len(va) - nz_va.sum():,} zeros ({100 * (1 - nz_va.mean()):.1f}%)")
    print("test  NOT READ\n")

    plain_preds, hurdle_preds, aucs, reg_preds = [], [], [], []
    for seed in SEEDS:
        t = time.time()
        # --- PLAIN: one regressor on everything (this reproduces B1)
        model_p, ep_p, _ = baselines.fit_mlp(Xtr, ytr, Xva, yva, seed=seed)
        plain = model_p.predict(Xva)

        # --- HURDLE arm 1: can we tell zero from non-zero?
        clf, auc, ep_c = fit_classifier(Xtr, ztr, Xva, zva, seed)
        p_nonzero = clf.predict_proba(Xva)[:, 1]

        # --- HURDLE arm 2: regressor fitted on measurable rows only.
        # Selection uses only the non-zero validation rows, so the zeros never influence
        # either the fit or the epoch choice -- that is the whole point of the split.
        model_r, ep_r, _ = baselines.fit_mlp(Xtr[nz_tr], ytr[nz_tr],
                                             Xva[nz_va], yva[nz_va], seed=seed)
        reg_all = model_r.predict(Xva)

        # combine: expected value under the two-part model
        hurdle = p_nonzero * reg_all

        plain_preds.append(plain)
        hurdle_preds.append(hurdle)
        reg_preds.append(reg_all)
        aucs.append(auc)
        print(f"  seed {seed}: plain epoch {ep_p}, clf epoch {ep_c} (AUC {auc:.3f}), "
              f"reg epoch {ep_r}  [{time.time() - t:.0f}s]", flush=True)

    plain = np.mean(plain_preds, axis=0)
    hurdle = np.mean(hurdle_preds, axis=0)
    reg_only = np.mean(reg_preds, axis=0)

    results = {}
    print("\nVALIDATION RESULTS (test not touched)\n")
    print("  (a) all validation rows")
    for name, pred in (("PLAIN_regressor_all_rows", plain), ("HURDLE_clf_x_regressor", hurdle)):
        r = metrics.evaluate(yva, pred, va.allele.to_numpy())
        results[f"all__{name}"] = r
        print("   ", metrics.fmt(name, r))

    print("\n  (b) non-zero validation rows only -- does removing the point mass help the regressor?")
    for name, pred in (("PLAIN_regressor_all_rows", plain),
                       ("HURDLE_regressor_nonzero_only", reg_only)):
        r = metrics.evaluate(yva[nz_va], pred[nz_va], va.allele.to_numpy()[nz_va])
        results[f"nonzero__{name}"] = r
        print("   ", metrics.fmt(name, r))

    results["classifier_auc_per_seed"] = [round(a, 4) for a in aucs]
    results["classifier_auc_mean"] = float(np.mean(aucs))
    results["seeds"] = list(SEEDS)
    results["n_val"] = int(len(va))
    results["n_val_zero"] = int((~nz_va).sum())

    print(f"\n  (c) zero vs non-zero separation: AUC {np.mean(aucs):.3f} "
          f"(per seed {[round(a, 3) for a in aucs]})")
    print("      0.5 would mean the features cannot tell the point mass apart at all")

    # verdict, stated against seed spread rather than eyeballed
    a_all = results["all__HURDLE_clf_x_regressor"]["spearman_pooled"]
    p_all = results["all__PLAIN_regressor_all_rows"]["spearman_pooled"]
    a_nz = results["nonzero__HURDLE_regressor_nonzero_only"]["spearman_pooled"]
    p_nz = results["nonzero__PLAIN_regressor_all_rows"]["spearman_pooled"]
    print("\n  VERDICT")
    print(f"    all rows      hurdle {a_all:.3f}  vs  plain {p_all:.3f}   delta {a_all - p_all:+.3f}")
    print(f"    non-zero only hurdle {a_nz:.3f}  vs  plain {p_nz:.3f}   delta {a_nz - p_nz:+.3f}")
    print("    A positive delta on (b) and a negative one on (a) would say the point mass is")
    print("    worth modelling separately but the recombination is lossy -- report both, not one.")

    preds_out = {
        "y_true_log": yva.tolist(), "allele": va.allele.tolist(),
        "peptide": va.peptide.tolist(), "cluster_id": va.cluster_id.tolist(),
        "is_nonzero": zva.tolist(),
        "PLAIN_regressor_all_rows": plain.tolist(),
        "HURDLE_clf_x_regressor": hurdle.tolist(),
        "HURDLE_regressor_nonzero_only": reg_only.tolist(),
    }
    (OUT / "hurdle_val_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (OUT / "hurdle_val_predictions.json").write_text(json.dumps(preds_out), encoding="utf-8")
    print(f"\nwrote {OUT / 'hurdle_val_metrics.json'}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
