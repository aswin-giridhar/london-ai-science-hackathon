"""Is ESM-2 useless, or merely worse? Error correlation and ensembling.

The distinction this draws
--------------------------
We report that a one-hot MLP (B1, 0.780 / 0.633) beats every ESM-2 rung. "Worse alone" and
"carries no information the baseline lacks" are different claims, and only the second justifies
saying a foundation model is not useful here.

If the two models made *identical* errors, ESM-2 is strictly redundant. If their errors are
**uncorrelated**, ESM-2 sees something one-hot does not, and combining them should beat both --
in which case the honest headline becomes "ESM-2 adds nothing on its own but +X in combination",
which is a materially different and more useful statement.

And if the ensemble does **not** beat B1, that is a considerably stronger negative than anything
in the results file so far: the foundation model adds nothing even as a complement.

Method
------
Predictions are on different scales (log1p hours, but differently calibrated), so they are
z-scored before averaging. Spearman is rank-based and unaffected by that, but the ensemble mean
would be dominated by whichever model has the larger variance otherwise.

Error correlation is computed on **ranks**, matching the metric we report, and within-allele as
well as pooled -- two models can agree about which allele is stable while disagreeing about which
peptide, which is exactly the distinction that matters here.

    python scripts/complementarity.py
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
sys.path.insert(0, str(ROOT / "src"))

import metrics  # noqa: E402
import power_analysis as pa  # noqa: E402
from bootstrap_ci import within_from_codes  # noqa: E402

RESULTS = ROOT / "results"
REPS = 2000
SEED = 2026


def load():
    base = json.loads((RESULTS / "baselines_val_predictions.json").read_text(encoding="utf-8"))
    esm = json.loads((RESULTS / "esm_heads_val_predictions.json").read_text(encoding="utf-8"))
    assert base["peptide"] == esm["peptide"], "prediction files describe different rows"
    preds = {
        "B1_onehot": np.asarray(base["B1_peptide_pseudoseq_allele"], float),
        "B1prime_onehot_alleleid": np.asarray(base["B1prime_peptide_allele_only"], float),
        "X150_esm_frozen": np.asarray(esm["X150_interaction_head"], float),
        "F150_esm_pooled": np.asarray(esm["F150_pooled_head"], float),
    }
    l150 = RESULTS / "l150_val_predictions.json"
    if l150.exists():
        d = json.loads(l150.read_text(encoding="utf-8"))
        if d.get("peptide") == base["peptide"]:
            preds["L150_esm_lora"] = np.asarray(d["L150_lora_kv"], float)
    return (np.asarray(base["y_true_log"], float), np.asarray(base["allele"]),
            np.asarray(base["cluster_id"]), preds)


def z(x):
    return (x - x.mean()) / (x.std() or 1.0)


def main():
    y, allele, clusters, preds = load()
    codes = pd.factorize(allele)[0]
    print(f"{len(y):,} validation rows, {len(np.unique(clusters)):,} peptide clusters\n")

    # ---------------------------------------------------------------- error correlation
    print("Do the models make the same mistakes?")
    print("  (residual = rank(prediction) - rank(truth); correlation of those residuals)\n")
    ry = rankdata(y)
    resid = {k: rankdata(v) - ry for k, v in preds.items()}
    names = list(preds)
    print(f"  {'':<26}" + "".join(f"{n.split('_')[0]:>12}" for n in names))
    for a in names:
        row = "".join(f"{np.corrcoef(resid[a], resid[b])[0, 1]:>12.3f}" for b in names)
        print(f"  {a:<26}{row}")

    key_pair = ("B1_onehot", "X150_esm_frozen")
    r_err = float(np.corrcoef(resid[key_pair[0]], resid[key_pair[1]])[0, 1])
    print(f"\n  B1 vs X150 residual correlation: {r_err:+.3f}")
    print("    1.0 would mean ESM-2 is strictly redundant; 0.0 would mean fully complementary")

    # ---------------------------------------------------------------- ensembling
    print("\nDoes combining them beat either alone?")
    combos = {
        "B1 alone": ["B1_onehot"],
        "X150 alone": ["X150_esm_frozen"],
        "B1 + X150": ["B1_onehot", "X150_esm_frozen"],
        "B1 + F150": ["B1_onehot", "F150_esm_pooled"],
    }
    if "L150_esm_lora" in preds:
        combos["B1 + L150"] = ["B1_onehot", "L150_esm_lora"]
        combos["B1 + X150 + L150"] = ["B1_onehot", "X150_esm_frozen", "L150_esm_lora"]
    # a control: averaging B1 with a model that shares its features should help least
    combos["B1 + B1' (control)"] = ["B1_onehot", "B1prime_onehot_alleleid"]

    out = {"residual_correlation_B1_X150": r_err, "combos": {}}
    print(f"\n  {'combination':<24}{'pooled':>9}{'within':>9}{'vs B1 pooled':>15}")
    base_p = metrics.evaluate(y, preds["B1_onehot"], allele)["spearman_pooled"]
    for label, members in combos.items():
        ens = np.mean([z(preds[m]) for m in members], axis=0)
        r = metrics.evaluate(y, ens, allele)
        w = r["spearman_within_allele"]
        delta = r["spearman_pooled"] - base_p
        print(f"  {label:<24}{r['spearman_pooled']:>9.3f}"
              f"{(w if w is not None else float('nan')):>9.3f}"
              f"{delta:>+15.3f}")
        out["combos"][label] = {"pooled": r["spearman_pooled"], "within": w,
                                "delta_vs_B1": delta, "members": members}

    # ---------------------------------------------------------------- is the gain real?
    print("\nPaired cluster bootstrap on the combination that matters, 2,000 draws, seed 2026")
    ens = np.mean([z(preds["B1_onehot"]), z(preds["X150_esm_frozen"])], axis=0)
    m, lo, hi = pa.paired_bootstrap(y, preds["B1_onehot"], ens, clusters, reps=REPS, seed=SEED)
    spans = lo <= 0 <= hi
    print(f"  (B1 + X150) - B1   pooled {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
          f"{'SPANS ZERO -> no complementary signal' if spans else 'excludes zero -> ESM-2 adds something'}")
    out["ensemble_minus_B1_pooled"] = {"mean": m, "lo": lo, "hi": hi, "spans_zero": bool(spans)}

    draw = pa.cluster_resampler(clusters, np.random.default_rng(SEED))
    diffs = []
    for _ in range(REPS):
        i = draw()
        a = within_from_codes(y[i], preds["B1_onehot"][i], codes[i])
        b = within_from_codes(y[i], ens[i], codes[i])
        if not (np.isnan(a) or np.isnan(b)):
            diffs.append(b - a)
    if len(diffs) >= 0.95 * REPS:
        wm, wlo, whi = float(np.mean(diffs)), *np.percentile(diffs, [2.5, 97.5])
        wspans = wlo <= 0 <= whi
        print(f"  (B1 + X150) - B1   within {wm:+.3f} [{wlo:+.3f}, {whi:+.3f}]  "
              f"{'SPANS ZERO' if wspans else 'excludes zero'}")
        out["ensemble_minus_B1_within"] = {"mean": wm, "lo": float(wlo), "hi": float(whi),
                                           "spans_zero": bool(wspans)}
    else:
        print("  within-allele interval unavailable (fewer than 95% of draws valid)")

    print("\nVERDICT")
    if spans:
        print("  The ensemble does not beat B1. ESM-2 adds nothing even as a complement, which is")
        print("  a stronger negative than 'it scores lower' -- its information is a subset of what")
        print("  a one-hot encoding of the same sequences already provides.")
    else:
        print("  The ensemble beats B1. ESM-2 is worse alone but carries complementary signal, so")
        print("  the honest claim is 'adds nothing on its own, adds something in combination'.")

    dest = RESULTS / "complementarity.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
