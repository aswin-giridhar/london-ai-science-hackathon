"""Metrics a user would actually act on: hours of error, top-k retrieval, and per-allele behaviour.

Why these and not another correlation
--------------------------------------
Everything reported so far is Spearman on `log1p(t-half)`. That is the right primary metric -
rank correlation survives the assay-scale problem - but it answers none of the questions someone
deciding whether to use this model would ask:

    "how many hours out are you?"            -> hours-MAE, on the original scale
    "if I shortlist 10 peptides, how many
     of the truly best 10 do I get?"          -> top-k precision, the real decision
    "does it work for MY patient's allele?"   -> per-allele breakdown, not a pooled average
    "can I trust the number it gives me?"     -> calibration

Top-k precision is the one that can disagree most sharply with Spearman. A model can order the
bulk of a list well, which Spearman rewards, while missing the extreme tail - and the extreme
tail is the entire decision when you are picking candidates to synthesise.

All of this is computed on **validation**, with the Run 10 caveat that validation drove epoch
selection and is optimistic by roughly +0.08 pooled.

    python src/operational_metrics.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
R = ROOT / "results"
MIN_ROWS = 12
TOPK = (5, 10)


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / d) if d else np.nan


def topk_precision(y, pred, k):
    """Of the k truly highest-t-half peptides, how many are in the model's top k?"""
    if len(y) < 2 * k:
        return np.nan
    true_top = set(np.argsort(-y)[:k])
    pred_top = set(np.argsort(-pred)[:k])
    return len(true_top & pred_top) / k


def main():
    OUT.mkdir(exist_ok=True)
    base = json.loads((R / "baselines_val_predictions.json").read_text(encoding="utf-8"))
    esm = json.loads((R / "esm_heads_val_predictions.json").read_text(encoding="utf-8"))
    assert base["peptide"] == esm["peptide"], "prediction files describe different rows"

    y_log = np.asarray(base["y_true_log"], float)
    y_hours = np.expm1(y_log)
    allele = np.asarray(base["allele"])
    preds = {
        "B0b_per_allele_median": np.asarray(base["B0b_per_allele_median"], float),
        "B1_onehot": np.asarray(base["B1_peptide_pseudoseq_allele"], float),
        "X150_esm_frozen": np.asarray(esm["X150_interaction_head"], float),
    }
    l150 = R / "l150_val_predictions.json"
    if l150.exists():
        d = json.loads(l150.read_text(encoding="utf-8"))
        if d.get("peptide") == base["peptide"]:
            preds["L150_esm_lora"] = np.asarray(d["L150_lora_kv"], float)

    print(f"{len(y_log):,} validation rows, {len(np.unique(allele))} alleles")
    print(f"half-life range {y_hours.min():.1f} to {y_hours.max():.1f} h, "
          f"median {np.median(y_hours):.1f} h\n")

    results = {}

    # ---------------------------------------------------------------- 1. error in hours
    print("1. ERROR ON THE ORIGINAL SCALE (hours)")
    print(f"   {'model':<24}{'MAE h':>9}{'median AE h':>13}{'MAE log':>10}"
          f"{'MAE h, t<=2':>13}{'MAE h, t>24':>13}")
    lo, hi = y_hours <= 2, y_hours > 24
    for name, p in preds.items():
        ph = np.expm1(p)
        row = {
            "mae_hours": float(np.mean(np.abs(ph - y_hours))),
            "median_ae_hours": float(np.median(np.abs(ph - y_hours))),
            "mae_log": float(np.mean(np.abs(p - y_log))),
            "mae_hours_unstable": float(np.mean(np.abs(ph - y_hours)[lo])),
            "mae_hours_stable": float(np.mean(np.abs(ph - y_hours)[hi])),
        }
        results.setdefault(name, {}).update(row)
        print(f"   {name:<24}{row['mae_hours']:>9.2f}{row['median_ae_hours']:>13.2f}"
              f"{row['mae_log']:>10.3f}{row['mae_hours_unstable']:>13.2f}"
              f"{row['mae_hours_stable']:>13.2f}")
    print(f"   ({lo.sum():,} rows at or under 2 h, {hi.sum():,} rows over 24 h)")
    print("   A median much below the mean means a few large misses dominate the average.\n")

    # ---------------------------------------------------------------- 2. top-k retrieval
    print("2. TOP-K RETRIEVAL, within each allele -- the actual shortlisting decision")
    print(f"   of the k most stable peptides for an allele, how many are in the model's top k?\n")
    print(f"   {'model':<24}" + "".join(f"{'top-' + str(k):>10}" for k in TOPK)
          + f"{'alleles scored':>16}")
    for name, p in preds.items():
        row = {}
        for k in TOPK:
            vals = []
            for a in np.unique(allele):
                i = allele == a
                if i.sum() >= 2 * k:
                    v = topk_precision(y_log[i], p[i], k)
                    if not np.isnan(v):
                        vals.append(v)
            row[f"top{k}_precision"] = float(np.mean(vals))
            row[f"top{k}_n_alleles"] = len(vals)
        results[name].update(row)
        print(f"   {name:<24}" + "".join(f"{row[f'top{k}_precision']:>10.3f}" for k in TOPK)
              + f"{row[f'top{TOPK[0]}_n_alleles']:>16}")
    rand5 = TOPK[0] / np.median([(allele == a).sum() for a in np.unique(allele)])
    print(f"\n   random expectation for top-5 is about {rand5:.3f} "
          f"(k / median rows per allele)")
    print("   B0b predicts a constant inside an allele, so its ordering there is arbitrary --")
    print("   its top-k is a tie-break artefact, not a skill measurement.\n")

    # ---------------------------------------------------------------- 3. per-allele
    print("3. PER-ALLELE BEHAVIOUR -- a pooled average hides where a model fails")
    rows = []
    for a in np.unique(allele):
        i = allele == a
        if i.sum() < MIN_ROWS:
            continue
        rec = {"allele": a, "n": int(i.sum()),
               "median_h": float(np.median(y_hours[i]))}
        for name, p in preds.items():
            rec[name] = spearman(y_log[i], p[i])
        rows.append(rec)
    per = pd.DataFrame(rows).sort_values("B1_onehot")
    results["per_allele"] = per.to_dict("records")

    print(f"   {len(per)} alleles with at least {MIN_ROWS} validation rows\n")
    print(f"   worst five for the best model (B1):")
    print("   " + per.head(5)[["allele", "n", "median_h", "B1_onehot",
                               "X150_esm_frozen"]].to_string(index=False).replace("\n", "\n   "))
    print(f"\n   best five:")
    print("   " + per.tail(5)[["allele", "n", "median_h", "B1_onehot",
                               "X150_esm_frozen"]].to_string(index=False).replace("\n", "\n   "))
    print(f"\n   B1 per-allele Spearman: median {per.B1_onehot.median():.3f}, "
          f"range {per.B1_onehot.min():.3f} to {per.B1_onehot.max():.3f}")
    worse = (per.X150_esm_frozen > per.B1_onehot).sum()
    print(f"   X150 beats B1 in {worse} of {len(per)} alleles "
          f"({100 * worse / len(per):.0f}%)")

    # does performance track how much data the allele has?
    rho_n = spearman(per.n.to_numpy(), per.B1_onehot.to_numpy())
    rho_h = spearman(per.median_h.to_numpy(), per.B1_onehot.to_numpy())
    print(f"\n   correlation of per-allele skill with rows available: {rho_n:+.3f}")
    print(f"   correlation of per-allele skill with median half-life: {rho_h:+.3f}")
    results["skill_vs_n"] = rho_n
    results["skill_vs_median_halflife"] = rho_h

    # ---------------------------------------------------------------- 4. calibration
    print("\n4. CALIBRATION -- is the predicted number usable as hours, or only as a rank?")
    print(f"   {'model':<24}{'slope':>9}{'intercept':>11}{'bias h':>10}")
    for name, p in preds.items():
        A = np.vstack([p, np.ones_like(p)]).T
        slope, inter = np.linalg.lstsq(A, y_log, rcond=None)[0]
        bias = float(np.mean(np.expm1(p) - y_hours))
        results[name].update({"calib_slope": float(slope), "calib_intercept": float(inter),
                              "bias_hours": bias})
        print(f"   {name:<24}{slope:>9.3f}{inter:>11.3f}{bias:>10.2f}")
    print("   slope 1 and intercept 0 would be perfectly calibrated on the log scale.")
    print("   A slope below 1 means the model is under-confident at the extremes -- it")
    print("   compresses its predictions toward the middle, which is what MSE training does.")
    print("   NOTE: these are not prediction intervals. Honest intervals need calibration rows")
    print("   carved from TRAIN, which this project has not done.")

    (OUT / "operational_metrics.json").write_text(json.dumps(results, indent=2, default=float),
                                                  encoding="utf-8")
    print(f"\nwrote {OUT / 'operational_metrics.json'}")


if __name__ == "__main__":
    main()
