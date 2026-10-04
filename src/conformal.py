"""Honest per-prediction intervals: split-conformal calibration, carved from TRAIN.

Why this is the last reporting gap
-----------------------------------
`ARCHITECTURE.md` section 2: *"Separate calibration rows (carved from train, never val) are
required before any per-prediction interval is shown. Metric confidence intervals are not
prediction intervals."*

Run 20 reported B1's calibration slope as 0.978, which is a property of the whole validation set.
It says nothing about any single prediction. A clinician asking "how confident are you about
*this* peptide" needs an interval, and nothing in this project has produced one.

Split conformal prediction
---------------------------
Fit the model on part of train. On held-out calibration rows compute the absolute residuals, take
the (1 - alpha) quantile q, and report `prediction +/- q`. Under exchangeability between the
calibration rows and the rows being predicted, marginal coverage is at least 1 - alpha with no
distributional assumption whatsoever.

The exchangeability trap, which is the whole design question
-------------------------------------------------------------
If calibration rows were drawn **at random** from train, they would share peptide clusters with
the rows the model trained on. Their residuals would then measure "difficulty on a cluster I have
already seen", while validation and test rows are all from clusters the model has never seen. The
intervals would come out systematically too narrow and the coverage guarantee would be void --
silently, because nothing would error and coverage would simply fall short.

So calibration is carved as **whole clusters**, cluster-disjoint from the reduced training set.
Calibration residuals then reflect the same unseen-cluster condition that validation and test
rows are scored under.

Marginal versus conditional coverage
-------------------------------------
Conformal guarantees coverage **on average over all rows**. It does not guarantee it within any
subgroup. Per-allele coverage is therefore reported as well, because a method that covers 90%
overall while covering 60% for one allele is not usable for that allele's patients - and that
failure is invisible in the marginal number.

    python src/conformal.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import baselines
import features
import metrics

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
BLOCKS = ("peptide", "pseudoseq", "allele")
SEEDS = baselines.SEEDS
CAL_FRACTION = 0.15          # of training CLUSTERS, not rows
ALPHAS = (0.20, 0.10, 0.05)  # target coverage 80%, 90%, 95%
SEED = 2026


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    m = features.load()
    tr_all = m[m.split == "train"].reset_index(drop=True)
    va = m[m.split == "val"].reset_index(drop=True)
    te = m[m.split == "test"].reset_index(drop=True)

    # ---- carve calibration as whole clusters out of train
    rng = np.random.default_rng(SEED)
    clusters = tr_all.cluster_id.unique()
    rng.shuffle(clusters)
    n_cal = int(round(CAL_FRACTION * len(clusters)))
    cal_clusters = set(clusters[:n_cal])
    is_cal = tr_all.cluster_id.isin(cal_clusters).to_numpy()
    fit, cal = tr_all[~is_cal].reset_index(drop=True), tr_all[is_cal].reset_index(drop=True)

    assert not (set(fit.cluster_id) & set(cal.cluster_id)), "calibration shares clusters with fit"
    assert not (set(fit.peptide) & set(cal.peptide)), "calibration shares peptides with fit"
    print(f"train {len(tr_all):,} rows in {len(clusters):,} clusters")
    print(f"  fit         {len(fit):,} rows / {fit.cluster_id.nunique():,} clusters")
    print(f"  calibration {len(cal):,} rows / {cal.cluster_id.nunique():,} clusters "
          f"(cluster-disjoint from fit, asserted)")
    print(f"  val {len(va):,}   test {len(te):,}\n")

    vocab = sorted(fit.allele.unique())
    X = {k: features.build(d, BLOCKS, vocab) for k, d in
         (("fit", fit), ("cal", cal), ("val", va), ("test", te))}
    y = {k: features.target(d) for k, d in
         (("fit", fit), ("cal", cal), ("val", va), ("test", te))}

    # epoch selection uses VALIDATION, exactly as every other rung in this project
    preds = {k: [] for k in ("cal", "val", "test")}
    for s in SEEDS:
        mdl, ep, _ = baselines.fit_mlp(X["fit"], y["fit"], X["val"], y["val"], seed=s)
        for k in preds:
            preds[k].append(mdl.predict(X[k]))
    P = {k: np.mean(v, axis=0) for k, v in preds.items()}

    r = metrics.evaluate(y["val"], P["val"], va.allele.to_numpy())
    print(f"model refit on {len(fit):,} rows (85% of train clusters): "
          f"val pooled {r['spearman_pooled']:.3f} / within {r['spearman_within_allele']:.3f}")
    print(f"  (full-train B1 is 0.780 / 0.633 -- the drop is the price of the calibration set)\n")

    resid_cal = np.abs(P["cal"] - y["cal"])
    n = len(resid_cal)
    results = {"n_fit": int(len(fit)), "n_cal": int(n),
               "val_pooled_refit": r["spearman_pooled"],
               "val_within_refit": r["spearman_within_allele"], "alphas": {}}

    print("SPLIT-CONFORMAL INTERVALS")
    print(f"  {'target':>8}{'q (log)':>10}{'width (log)':>13}"
          f"{'val cover':>11}{'test cover':>12}{'median width, hours':>22}")
    for a in ALPHAS:
        # the finite-sample correct quantile, not the naive one
        k = int(np.ceil((n + 1) * (1 - a)))
        q = float(np.sort(resid_cal)[min(k, n) - 1])
        row = {"target_coverage": 1 - a, "q_log": q, "width_log": 2 * q}
        for split, d in (("val", va), ("test", te)):
            lo, hi = P[split] - q, P[split] + q
            cover = float(np.mean((y[split] >= lo) & (y[split] <= hi)))
            row[f"{split}_coverage"] = cover
            # width on the original scale is prediction-dependent, so report the median
            w = np.expm1(hi) - np.expm1(lo)
            row[f"{split}_median_width_hours"] = float(np.median(w))
        print(f"  {1 - a:>8.0%}{q:>10.3f}{2 * q:>13.3f}"
              f"{row['val_coverage']:>11.1%}{row['test_coverage']:>12.1%}"
              f"{row['test_median_width_hours']:>22.2f}")
        results["alphas"][f"{a}"] = row

    # ---- conditional coverage: the guarantee does NOT extend to subgroups
    a = 0.10
    k = int(np.ceil((n + 1) * (1 - a)))
    q = float(np.sort(resid_cal)[min(k, n) - 1])
    rows = []
    for al in np.unique(te.allele.to_numpy()):
        i = te.allele.to_numpy() == al
        if i.sum() < 12:
            continue
        c = float(np.mean((y["test"][i] >= P["test"][i] - q) & (y["test"][i] <= P["test"][i] + q)))
        rows.append({"allele": al, "n": int(i.sum()), "coverage": c,
                     "median_h": float(np.median(np.expm1(y["test"][i])))})
    per = pd.DataFrame(rows).sort_values("coverage")
    results["per_allele_coverage_90"] = per.to_dict("records")

    print(f"\nCONDITIONAL COVERAGE at a 90% marginal target, per allele on TEST")
    print(f"  {len(per)} alleles with at least 12 test rows")
    print(f"  median {per.coverage.median():.1%}, range {per.coverage.min():.1%} "
          f"to {per.coverage.max():.1%}")
    print(f"  alleles below 80%: {(per.coverage < 0.80).sum()} of {len(per)}")
    print(f"\n  worst five:")
    print("   " + per.head(5).to_string(index=False).replace("\n", "\n   "))

    print(f"\n  Marginal coverage is guaranteed; conditional coverage is not, and the spread")
    print(f"  above is what that costs in practice. An interval that covers {results['alphas']['0.1']['test_coverage']:.0%}")
    print(f"  overall can still cover {per.coverage.min():.0%} for a particular allele's patients.")

    (OUT / "conformal.json").write_text(json.dumps(results, indent=2, default=float),
                                        encoding="utf-8")
    print(f"\nwrote {OUT / 'conformal.json'}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
