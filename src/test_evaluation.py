"""THE TEST SET READ. Once.

Protocol, fixed before anything was run
---------------------------------------
    train on   the 22,532 training rows
    select on  the 2,817 validation rows (epoch choice only)
    report on  the 2,817 test rows, read exactly once

Validation drove every epoch choice and every comparison in Runs 1-21, so every number in this
repository is optimistic - Run 10 measured that optimism at roughly **+0.08 pooled** by training
on shuffled labels and watching epoch selection alone score +0.104. The test split has never been
read by anything. This is what it is for.

Why read it at all, and why now
--------------------------------
The model choice is frozen. Twenty-one runs established the ladder, and nothing further can change
which model we would ship: B1 leads on validation, every ESM-2 configuration trails it, and the
three obvious fixes (depth, scale, fine-tuning) all landed inside seed noise. Reading test earlier
would have risked tuning against it; reading it later would mean reporting no held-out number at
all, which is a hole in the middle of any evaluation claim.

The guard
---------
A read is recorded in `results/TEST_READ.json`. Running this again refuses unless `--force` is
passed, and the refusal prints the earlier result. That is deliberate: the danger is not reading
the test set, it is reading it repeatedly and quietly keeping the best number.

    python src/test_evaluation.py
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import baselines
import esm_heads
import features
import metrics

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
GUARD = ROOT / "results" / "TEST_READ.json"
SEEDS = baselines.SEEDS


def main(force=False):
    if GUARD.exists() and not force:
        prior = json.loads(GUARD.read_text(encoding="utf-8"))
        print("REFUSING TO RE-READ THE TEST SET")
        print(f"  already read {prior['read_at']}")
        print(f"  recorded: {json.dumps(prior['headline'], indent=2)}")
        print()
        print("  The danger is not reading the test set once; it is reading it repeatedly and")
        print("  keeping the best number. Pass --force only if you intend to overwrite the")
        print("  record of the first read, and say so in results/README.md.")
        raise SystemExit(1)

    import torch
    from torch import nn

    esm_heads.DEVICE = device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    print(f"device: {device}")

    m = features.load()
    tr = m[m.split == "train"].reset_index(drop=True)
    va = m[m.split == "val"].reset_index(drop=True)
    te = m[m.split == "test"].reset_index(drop=True)
    vocab = sorted(tr.allele.unique())
    y_tr, y_va, y_te = features.target(tr), features.target(va), features.target(te)
    allele_te = te.allele.to_numpy()

    print(f"train {len(tr):,}   val {len(va):,} (selection only)   "
          f"TEST {len(te):,} rows / {te.allele.nunique()} alleles")
    unseen = set(te.peptide) & set(tr.peptide)
    print(f"test peptides also in train: {len(unseen)} (must be 0)")
    assert not unseen, "test/train peptide overlap -- the split is broken"
    print()

    results, preds = {}, {}

    # ---------------------------------------------------------------- B0: medians from TRAIN
    gm = float(np.median(y_tr))
    preds["B0a_global_median"] = np.full(len(te), gm)
    med = tr.groupby("allele").apply(lambda g: np.median(features.target(g)), include_groups=False)
    preds["B0b_per_allele_median"] = te.allele.map(med).fillna(gm).to_numpy()

    # ---------------------------------------------------------------- B1 and B1'
    for name, blocks in (("B1_peptide_pseudoseq_allele", ("peptide", "pseudoseq", "allele")),
                         ("B1prime_peptide_allele_only", ("peptide", "allele"))):
        Xt = features.build(tr, blocks, vocab)
        Xv = features.build(va, blocks, vocab)
        Xs = features.build(te, blocks, vocab)
        ps = []
        for s in SEEDS:
            mdl, ep, _ = baselines.fit_mlp(Xt, y_tr, Xv, y_va, seed=s)   # epoch chosen on VAL
            ps.append(mdl.predict(Xs))
        preds[name] = np.mean(ps, axis=0)
        print(f"  {name} trained ({time.time() - t0:.0f}s)", flush=True)

    # ---------------------------------------------------------------- F150 / X150
    z, pep_index, all_index = esm_heads.load_cache()
    banks = {}
    for k, src in (("pep_res", "peptide_residue"), ("hla_res", "hla_residue")):
        t = torch.tensor(z[src].astype(np.float32)).to(device)
        banks[k] = t
    banks["pep_pool"] = banks["pep_res"].mean(1)
    banks["hla_pool"] = banks["hla_res"].mean(1)
    d_res = banks["pep_res"].shape[-1]

    tensors = {}
    for nm, sub in (("train", tr), ("val", va), ("test", te)):
        pi, ai = esm_heads.row_indices(sub, pep_index, all_index)
        tensors[nm] = {"pi": torch.from_numpy(pi).to(device),
                       "ai": torch.from_numpy(ai).to(device),
                       "y": torch.tensor(features.target(sub).astype(np.float32)).to(device),
                       "allele": sub.allele.to_numpy(), "meta": sub}

    models = esm_heads.build_models(torch, nn, d_in_pooled=2 * d_res, d_res=d_res)
    for name, Model in models.items():
        ps = []
        for s in SEEDS:
            mdl, ep, _, _ = esm_heads.train_one(torch, nn, Model, banks, tensors, s)
            need = Model.NEEDS_RESIDUES
            mdl.eval()
            with torch.no_grad():
                out = [mdl(*esm_heads.gather(banks, tensors["test"], slice(i, i + 1024), need))
                       for i in range(0, len(te), 1024)]
            ps.append(torch.cat(out).cpu().numpy())
        preds[name] = np.mean(ps, axis=0)
        print(f"  {name} trained ({time.time() - t0:.0f}s)", flush=True)

    # ---------------------------------------------------------------- report
    print("\n" + "=" * 78)
    print("TEST SET RESULTS -- first and only read")
    print("=" * 78 + "\n")
    # The validation comparison is a nicety; the test numbers are the point. If the reference
    # files are absent, say so and carry on -- crashing here would waste a GPU run and, worse,
    # could leave the test set read with nothing recorded.
    val_lookup = {}
    for fn in ("baselines_val_metrics.json", "esm_heads_val_metrics.json"):
        fp = ROOT / "results" / fn
        if fp.exists():
            val_lookup.update(json.loads(fp.read_text(encoding="utf-8")))
        else:
            print(f"  (no {fn}; validation-to-test comparison will be partial)")

    print(f"  {'rung':<30}{'TEST pooled':>13}{'TEST within':>13}"
          f"{'val pooled':>12}{'drop':>8}")
    for name, p in preds.items():
        r = metrics.evaluate(y_te, p, allele_te)
        vp = val_lookup.get(name, {}).get("spearman_pooled")
        drop = (r["spearman_pooled"] - vp) if (vp is not None and r["spearman_pooled"]) else None
        results[name] = r
        w = r["spearman_within_allele"]
        print(f"  {name:<30}"
              f"{(r['spearman_pooled'] if r['spearman_pooled'] is not None else float('nan')):>13.3f}"
              f"{(w if w is not None else float('nan')):>13.3f}"
              f"{(vp if vp is not None else float('nan')):>12.3f}"
              f"{(drop if drop is not None else float('nan')):>+8.3f}")

    b1 = results["B1_peptide_pseudoseq_allele"]
    x1 = results["X150_interaction_head"]
    print(f"\n  THE HEADLINE, ON HELD-OUT DATA")
    print(f"    B1 one-hot     {b1['spearman_pooled']:.3f} pooled / "
          f"{b1['spearman_within_allele']:.3f} within")
    print(f"    X150 ESM-2     {x1['spearman_pooled']:.3f} pooled / "
          f"{x1['spearman_within_allele']:.3f} within")
    print(f"    difference     {b1['spearman_pooled'] - x1['spearman_pooled']:+.3f} pooled / "
          f"{b1['spearman_within_allele'] - x1['spearman_within_allele']:+.3f} within")
    print(f"    -> {'one-hot still ahead on held-out data' if b1['spearman_pooled'] > x1['spearman_pooled'] else 'ORDERING FLIPPED ON TEST'}")

    drops = [results[n]["spearman_pooled"] - val_lookup[n]["spearman_pooled"]
             for n in results if n in val_lookup and results[n]["spearman_pooled"] is not None
             and val_lookup[n].get("spearman_pooled") is not None]
    print(f"\n  Mean validation-to-test drop: {np.mean(drops):+.3f} over {len(drops)} rungs.")
    print(f"  Run 10 predicted roughly -0.08 from epoch-selection optimism alone.")

    payload = {"read_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "n_test": int(len(te)), "n_test_alleles": int(te.allele.nunique()),
               "seeds": list(SEEDS), "results": results,
               "predictions": {k: v.tolist() for k, v in preds.items()},
               "y_true_log": y_te.tolist(), "allele": allele_te.tolist(),
               "peptide": te.peptide.tolist(), "cluster_id": te.cluster_id.tolist(),
               "mean_val_to_test_drop": float(np.mean(drops)) if drops else None}
    (OUT / "test_metrics.json").write_text(json.dumps(payload, indent=2, default=float),
                                           encoding="utf-8")
    GUARD.write_text(json.dumps({
        "read_at": payload["read_at"],
        "headline": {"B1_pooled": b1["spearman_pooled"], "B1_within": b1["spearman_within_allele"],
                     "X150_pooled": x1["spearman_pooled"],
                     "X150_within": x1["spearman_within_allele"]},
        "note": "The test split was read once, on this date, with the model choice already "
                "frozen. Re-reading requires --force and must be disclosed in results/README.md.",
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'test_metrics.json'} and the read guard {GUARD.name}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
