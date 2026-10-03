"""Confidence intervals on every rung, and on the comparisons the pitch rests on.

Why this exists
---------------
`results/README.md` reports point estimates and seed spreads. A seed spread is not a confidence
interval: it says how much the answer moves when the initialisation changes, not how much it would
move on a different sample of peptides. Two of our claims are specifically claims about *sampling*:

    "X150 ties B1-prime"     -- 0.754 vs 0.748. Is that gap real or noise?
    "B1 beats X150"          -- 0.780 vs 0.754. Same question.

A difference of 0.006 between two numbers is not evidence of anything until you know the width of
the interval around it. This script supplies that width.

Two things make it honest
-------------------------
**Resample clusters, not rows.** Peptides inside a cluster are within 2 substitutions of each
other, so they are not independent observations. Resampling rows would treat near-duplicates as
fresh information and produce an interval that is too narrow -- precision we have not earned.

**Pair the draws.** Each of the 2,000 replicates scores *both* models on the *same* resampled rows,
so "this draw happened to contain easy peptides" affects both equally and cancels. Unpaired
intervals on two correlated models are far wider than the question deserves.

A CI that spans zero means **inconclusive**, not equivalence -- and for the "X150 ties B1-prime"
claim, inconclusive IS the finding: we cannot distinguish a frozen protein language model from a
one-hot lookup.

    python scripts/bootstrap_ci.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import power_analysis as pa  # noqa: E402

RESULTS = ROOT / "results"
REPS = 2000
SEED = 2026          # the seed ARCHITECTURE.md section 7 pins for every reported bootstrap
MIN_ROWS_PER_ALLELE = 12


def fast_spearman(a, b):
    """Spearman via scipy rankdata. Verified identical to power_analysis.spearman to 2.2e-16.

    Worth the duplication: the pandas version costs 2.2 ms a call and this one 1.5 ms, but the
    real saving is downstream -- it lets the within-allele grouping below stay in numpy.
    Average ranks are not optional here: 2,466 of the 2,817 validation rows are tied, mostly on
    the t-half = 0 point mass, so a naive argsort rank would quietly give a different statistic.
    """
    ra = rankdata(a)
    rb = rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / d) if d else np.nan


def within_from_codes(y, pred, codes):
    """Mean Spearman inside each allele with at least MIN_ROWS_PER_ALLELE rows.

    Integer allele codes, sorted and split, instead of a per-draw pandas groupby on strings.
    That one change takes a paired within-allele draw from 156 ms to a few ms -- the earlier
    version could not finish 2,000 draws inside a 30 minute budget.
    """
    order = np.argsort(codes, kind="stable")
    c = codes[order]
    vals = []
    for g in np.split(order, np.flatnonzero(np.diff(c)) + 1):
        if len(g) < MIN_ROWS_PER_ALLELE:
            continue
        yy, pp = y[g], pred[g]
        if yy.min() != yy.max() and pp.min() != pp.max():
            vals.append(fast_spearman(yy, pp))
    return float(np.mean(vals)) if vals else np.nan


def within_allele_metric(allele):
    """(y, pred) -> within-allele Spearman, reading allele codes from an attribute.

    paired_bootstrap only passes (y, pred), so the resampled allele view travels on the function
    object. Set `metric.allele_view` to the codes for the current draw before calling.
    """

    def metric(y, pred):
        return within_from_codes(y, pred, metric.allele_view)

    metric.allele_view = allele
    return metric


def paired_within(y, pred_a, pred_b, clusters, codes, reps=REPS, seed=SEED):
    """Paired CI on within-allele Spearman. Needs its own loop so `allele` is resampled too."""
    draw = pa.cluster_resampler(clusters, np.random.default_rng(seed))
    m = within_allele_metric(None)
    diffs = []
    for _ in range(reps):
        i = draw()
        m.allele_view = codes[i]
        d = m(y[i], pred_b[i]) - m(y[i], pred_a[i])
        if not np.isnan(d):
            diffs.append(d)
    if len(diffs) < 0.95 * reps:
        # ARCHITECTURE.md section 7: report the interval unavailable rather than silently
        # resampling until enough draws succeed.
        return None
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(lo), float(hi)


def single_within(y, pred, clusters, codes, reps=REPS, seed=SEED):
    draw = pa.cluster_resampler(clusters, np.random.default_rng(seed))
    m = within_allele_metric(None)
    vals = []
    for _ in range(reps):
        i = draw()
        m.allele_view = codes[i]
        v = m(y[i], pred[i])
        if not np.isnan(v):
            vals.append(v)
    if len(vals) < 0.95 * reps:
        return None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def load():
    """Both prediction files, asserted to describe the same rows in the same order."""
    base = json.loads((RESULTS / "baselines_val_predictions.json").read_text(encoding="utf-8"))
    esm = json.loads((RESULTS / "esm_heads_val_predictions.json").read_text(encoding="utf-8"))

    # If these disagree, every comparison below is between misaligned vectors and would still
    # produce a plausible-looking number. Fail loudly instead.
    assert base["peptide"] == esm["peptide"], "prediction files describe different rows"
    assert base["allele"] == esm["allele"], "prediction files describe different alleles"
    assert np.allclose(base["y_true_log"], esm["y_true_log"]), "prediction files disagree on y"

    preds = {k: np.asarray(v, dtype=float)
             for k, v in {**base, **esm}.items()
             if k not in ("y_true_log", "allele", "peptide", "cluster_id")}
    extra = json.loads((RESULTS / "l150_val_metrics.json").read_text(encoding="utf-8")) \
        if (RESULTS / "l150_val_metrics.json").exists() else None
    l150 = RESULTS / "l150_val_predictions.json"
    if l150.exists():
        d = json.loads(l150.read_text(encoding="utf-8"))
        if d.get("peptide") == base["peptide"]:
            preds["L150_lora_kv"] = np.asarray(d["L150_lora_kv"], dtype=float)
        else:
            print("  (L150 predictions describe different rows; excluded)")
    return (np.asarray(base["y_true_log"], dtype=float),
            np.asarray(base["allele"]),
            np.asarray(base["cluster_id"]),
            preds, extra)


def main():
    y, allele, clusters, preds, _ = load()
    # integer codes once, so no draw ever groups on strings
    codes = pd.factorize(allele)[0]
    n_clusters = len(np.unique(clusters))
    print(f"{len(y):,} validation rows in {n_clusters:,} peptide clusters")
    print(f"{REPS:,} paired draws, seed {SEED}, percentile bounds 2.5/97.5")
    print("resampling CLUSTERS -- rows inside one are within 2 substitutions and not independent\n")

    order = [k for k in ("B0b_per_allele_median", "F150_pooled_head", "X150_interaction_head",
                         "B1prime_peptide_allele_only", "B1_peptide_pseudoseq_allele",
                         "L150_lora_kv") if k in preds]

    print("Per-rung 95% CI")
    print(f"  {'rung':<30} {'pooled rho':<24} {'within-allele rho'}")
    per_rung = {}
    for k in order:
        lo, hi = pa.single_metric_ci(y, preds[k], clusters, reps=REPS, seed=SEED)
        point = pa.spearman(y, preds[k])
        w = single_within(y, preds[k], clusters, codes)
        wpoint = within_from_codes(y, preds[k], codes)
        wtxt = "undefined" if np.isnan(wpoint) else (
            f"{wpoint:.3f} [{w[0]:.3f}, {w[1]:.3f}]" if w else f"{wpoint:.3f} [unavailable]")
        print(f"  {k:<30} {point:.3f} [{lo:.3f}, {hi:.3f}]    {wtxt}")
        per_rung[k] = {"pooled": [point, lo, hi],
                       "within": None if np.isnan(wpoint) else [wpoint, *(w or (None, None))]}

    comparisons = [
        ("X150_interaction_head", "B1prime_peptide_allele_only",
         "does a one-hot allele lookup match the frozen PLM?"),
        ("X150_interaction_head", "B1_peptide_pseudoseq_allele",
         "does the one-hot baseline beat the frozen PLM?"),
        ("F150_pooled_head", "X150_interaction_head",
         "H2 -- does the residue head beat mean pooling?"),
        ("B1prime_peptide_allele_only", "B1_peptide_pseudoseq_allele",
         "does the HLA sequence add over the allele label?"),
        ("B0b_per_allele_median", "B1_peptide_pseudoseq_allele",
         "does peptide information add over an allele median?"),
    ]
    if "L150_lora_kv" in preds:
        comparisons.insert(0, ("X150_interaction_head", "L150_lora_kv",
                               "H1 -- does adapting the encoder help?"))

    print("\nPaired comparisons, 95% CI on (B - A). Spanning zero = INCONCLUSIVE, not equivalence.")
    out = {}
    for a, b, question in comparisons:
        if a not in preds or b not in preds:
            continue
        mean, lo, hi = pa.paired_bootstrap(y, preds[a], preds[b], clusters, reps=REPS, seed=SEED)
        wres = paired_within(y, preds[a], preds[b], clusters, codes)
        spans = lo <= 0 <= hi
        print(f"\n  {question}")
        print(f"    {b}  -  {a}")
        print(f"      pooled        {mean:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
              f"{'SPANS ZERO -> inconclusive' if spans else 'excludes zero -> survives'}")
        if wres:
            wm_, wlo, whi = wres
            wspans = wlo <= 0 <= whi
            print(f"      within-allele {wm_:+.3f} [{wlo:+.3f}, {whi:+.3f}]  "
                  f"{'SPANS ZERO -> inconclusive' if wspans else 'excludes zero -> survives'}")
        else:
            wm_ = wlo = whi = None
            print("      within-allele unavailable (fewer than 95% of draws valid)")
        out[f"{b}_minus_{a}"] = {
            "question": question,
            "pooled": {"mean": mean, "lo": lo, "hi": hi, "spans_zero": bool(spans)},
            "within_allele": None if wres is None else
            {"mean": wm_, "lo": wlo, "hi": whi, "spans_zero": bool(wlo <= 0 <= whi)},
        }

    payload = {"reps": REPS, "seed": SEED, "n_rows": int(len(y)),
               "n_clusters": int(n_clusters), "resampled": "peptide clusters",
               "per_rung": per_rung, "comparisons": out}
    dest = RESULTS / "bootstrap_ci.json"
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
