"""Negative control: shuffle the training labels and confirm the pipeline learns nothing.

What this protects against
--------------------------
Every number in `results/` assumes the split does what it claims and that no label information
reaches the model except through the training rows it is supposed to see. That assumption has
never been tested directly. If some path leaks -- a mis-joined frame, a peptide appearing on both
sides, a target accidentally included among the features -- the pipeline would report a healthy
Spearman and nothing anywhere would error.

So: destroy the relationship between training features and training labels, keep the validation
labels intact, and retrain. A clean pipeline must now score **approximately zero**. Anything
appreciably above zero is leakage, and every result in this repository would need re-examining.

Two shuffles, because they fail differently
-------------------------------------------
    global    permute y across all training rows. Destroys everything, including allele identity.
    within    permute y *inside each allele*. Preserves the per-allele distribution, so a model
              can still learn "this allele is stable" and should score about B0b (0.563 pooled),
              but must retain nothing about which peptide.

The second is the sharper test. A pipeline that leaks peptide-level information will beat the
allele-median floor here, and the global shuffle would not reveal that.

    python src/shuffle_control.py
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
BLOCKS = ("peptide", "pseudoseq", "allele")
SEEDS = baselines.SEEDS


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)

    m = features.load()
    tr = m[m.split == "train"].reset_index(drop=True)
    va = m[m.split == "val"].reset_index(drop=True)
    vocab = sorted(tr.allele.unique())
    Xtr, Xva = features.build(tr, BLOCKS, vocab), features.build(va, BLOCKS, vocab)
    ytr, yva = features.target(tr), features.target(va)
    allele_va = va.allele.to_numpy()

    print(f"train {len(tr):,}  val {len(va):,}   (validation labels are NEVER shuffled)")
    print("test  NOT READ\n")

    def run(name, y_train, expectation):
        preds = []
        for seed in SEEDS:
            model, epoch, _ = baselines.fit_mlp(Xtr, y_train, Xva, yva, seed=seed)
            preds.append(model.predict(Xva))
        r = metrics.evaluate(yva, np.mean(preds, axis=0), allele_va)
        w = r["spearman_within_allele"]
        print(f"  {name:<34} rho_pooled {r['spearman_pooled']:+.4f}   "
              f"rho_within {'n/a' if w is None else f'{w:+.4f}'}")
        print(f"  {'':<34} expected: {expectation}")
        return r

    results = {}

    # the real thing, for scale
    results["intact"] = run("intact labels (= B1)", ytr, "0.780 pooled / 0.633 within")

    # global shuffle: nothing survives
    rng = np.random.default_rng(0)
    results["shuffled_global"] = run("labels shuffled GLOBALLY", rng.permutation(ytr),
                                     "~0.000 -- anything above ~0.05 means leakage")

    # within-allele shuffle: allele identity survives, peptide identity must not
    y_within = ytr.copy()
    for a in np.unique(tr.allele.to_numpy()):
        idx = np.flatnonzero(tr.allele.to_numpy() == a)
        y_within[idx] = rng.permutation(y_within[idx])
    results["shuffled_within_allele"] = run(
        "labels shuffled WITHIN each allele", y_within,
        "~0.563 pooled (the allele-median floor) and ~0.000 within-allele")

    print("\nVERDICT")
    g = results["shuffled_global"]["spearman_pooled"]
    wi = results["shuffled_within_allele"]
    ok_global = abs(g) < 0.05
    wi_within = wi["spearman_within_allele"]
    ok_within = wi_within is None or abs(wi_within) < 0.05
    print(f"  global shuffle pooled {g:+.4f}        "
          f"{'PASS' if ok_global else 'FAIL -- the pipeline learns from shuffled labels'}")
    print(f"  within-allele shuffle, within-allele rho "
          f"{'n/a' if wi_within is None else f'{wi_within:+.4f}'}   "
          f"{'PASS' if ok_within else 'FAIL -- peptide-level information is leaking'}")
    print(f"  within-allele shuffle, pooled {wi['spearman_pooled']:+.4f}  "
          f"(should land near the 0.563 allele-median floor, which is allele identity "
          f"surviving by design -- not leakage)")

    results["pass_global"] = bool(ok_global)
    results["pass_within_allele"] = bool(ok_within)
    results["seeds"] = list(SEEDS)
    (OUT / "shuffle_control.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'shuffle_control.json'}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
