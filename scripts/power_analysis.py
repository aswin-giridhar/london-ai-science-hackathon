"""Confidence intervals and paired comparisons on the frozen test split.

Two things live here:

  1. `single_metric_ci`  -- how wide is the CI on one metric, given our test set size? This is
     sound, reproduces across runs, and is reported below.

  2. `paired_bootstrap`  -- the function to call once TWO REAL MODELS exist. This is the correct
     way to decide whether an ablation row is interpretable. It is not exercised here because
     nothing has been trained yet.

A retracted synthetic power table
---------------------------------
An earlier version of this script simulated predictions by adding Gaussian noise to the log target,
tuned to hit a target Spearman, and reported a per-metric table of minimum detectable effects. That
table was published in POWER_ANALYSIS.md and has been retracted. Re-running it did not reproduce,
and the failure was diagnostic: results were non-monotonic in effect size (Pearson resolved a +0.01
gap in 4 of 5 realisations but a +0.02 gap in only 1 of 5), which is impossible if the method is
sound. Two causes:

  FLAW 1 -- the noise was tuned on Spearman, then Pearson and RMSE were measured. Those were never
  controlled, so the rows were not comparable across metrics. The tuning itself also scattered:
  a target Spearman of 0.60 realised as 0.594 +/- 0.019, comparable to the +0.01 to +0.02 gaps the
  table claimed to resolve.

  FLAW 2 -- baseline and improved predictions used independent noise draws, giving an error
  correlation near zero. Two real models trained on a shared feature matrix typically correlate
  above +0.8, and a paired test on correlated errors is far more powerful. The retracted table
  therefore UNDERSTATED how fine a difference we can resolve.

The honest position: the single-metric CI below is real; the cross-metric sensitivity comparison
needs real residuals, which arrive with the first two trained models.

    python scripts/power_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPS = 2000


def load_test():
    df = pd.read_csv(ROOT / "context" / "dataset.csv")
    sp = pd.read_csv(ROOT / "splits" / "peptide_split.csv")
    te = df.merge(sp, on="peptide").query("split == 'test'")
    return te


def spearman(a, b):
    return np.corrcoef(pd.Series(a).rank().values, pd.Series(b).rank().values)[0, 1]


def cluster_resampler(clusters, rng):
    """Resample whole peptide clusters. Rows inside a cluster are not independent."""
    uniq = np.unique(clusters)
    idx = {c: np.where(clusters == c)[0] for c in uniq}

    def draw():
        return np.concatenate([idx[c] for c in rng.choice(uniq, len(uniq), replace=True)])

    return draw


def single_metric_ci(y, pred, clusters, metric=spearman, reps=REPS, seed=0):
    """95% CI on one metric, resampling clusters."""
    draw = cluster_resampler(clusters, np.random.default_rng(seed))
    vals = [metric(y[i], pred[i]) for i in (draw() for _ in range(reps))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap(y, pred_a, pred_b, clusters, metric=spearman, reps=REPS, seed=0):
    """CI on metric(b) - metric(a), both scored on the SAME resampled rows.

    This is what decides whether an ablation row is interpretable. Call it with two real models'
    predictions; a CI excluding zero means the difference survived, and one spanning zero means
    inconclusive -- which is a reportable result, not a failure.
    """
    draw = cluster_resampler(clusters, np.random.default_rng(seed))
    diffs = [metric(y[i], pred_b[i]) - metric(y[i], pred_a[i]) for i in (draw() for _ in range(reps))]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(lo), float(hi)


def main():
    te = load_test()
    y = np.log1p(te.thalf_hours.values)
    clusters = te.cluster_id.values
    rng = np.random.default_rng(0)

    print(f"test split: {len(te):,} rows across {len(np.unique(clusters))} peptide clusters")
    print(f"bootstrap:  {REPS} resamples of whole clusters\n")

    print("CONFIDENCE INTERVAL on a single Spearman, by accuracy level")
    print("  (predictions simulated only to set the accuracy level; the CI width is a property")
    print("   of the test set, not of any model)\n")
    for target in (0.45, 0.60, 0.75):
        lo_s, hi_s = 0.01, 20.0
        for _ in range(40):  # bisect the noise level to hit the target
            mid = (lo_s + hi_s) / 2
            if spearman(y, y + rng.normal(0, mid, len(y))) > target:
                lo_s = mid
            else:
                hi_s = mid
        pred = y + rng.normal(0, (lo_s + hi_s) / 2, len(y))
        lo, hi = single_metric_ci(y, pred, clusters)
        print(f"   rho ~{target:.2f}   95% CI [{lo:.3f}, {hi:.3f}]   width {hi - lo:.3f}")

    print("\nWhat this supports:")
    print("  Two INDEPENDENTLY reported Spearman values need to differ by roughly 0.05 before the")
    print("  difference means anything, because their intervals are ~0.05 wide.")
    print("\nWhat it does NOT support:")
    print("  Any claim about which metric is most sensitive, or about the minimum detectable")
    print("  PAIRED difference. A paired test on two models that share a feature matrix is much")
    print("  more powerful than comparing two intervals, and its resolution depends on how")
    print("  correlated the two models' errors actually are -- which cannot be guessed.")
    print("\n  Once B1 and F0 exist, call paired_bootstrap(y, pred_B1, pred_F0, clusters) and")
    print("  report the mean difference with its interval on every ablation row.")


if __name__ == "__main__":
    main()
