"""Which peptide positions does the model actually use? Two independent probes.

Run 4 showed X150 beating F150 by +0.161 pooled / +0.280 within-allele, and the explanation
offered was anchors: mean-pooling destroys position, and P2/P9 dock into the B and F pockets.
That explanation fits the numbers. It has not been tested. This file tests it.

Probe A -- position ablation on the trained model (tests OUR claim)
-------------------------------------------------------------------
Train X150 as in Run 4, then zero the peptide residue embedding at position k and re-score the
validation set. The drop in Spearman is how much the model relies on position k *for this
endpoint*. Nine ablations per seed, three seeds, all on cached embeddings -- seconds.

The anchor hypothesis makes a falsifiable prediction: P2 and P9 should drop most. A flat profile
would say the model spreads its attention evenly and the anchor story is wrong.

Probe B -- ESM-2 pseudo-likelihood (the brief's zero-training route)
--------------------------------------------------------------------
Mask position k, ask ESM-2 for the log-probability of the residue that is actually there. No
labels, no training, no leakage surface anywhere.

**These two measure different things and must not be conflated.** Probe B says what ESM-2 finds
predictable from protein statistics; Probe A says what predicts half-life. If they disagree, that
is itself the interesting result -- it would mean the stability signal is not where the language
model's confidence is.

    python -m modal run modal_app.py::anchors
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

import esm_heads
import features
import metrics

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
MODEL_ID = "facebook/esm2_t30_150M_UR50D"
PEPTIDE_LEN = 9
MLM_BATCH = 256


# --------------------------------------------------------------------------- probe A


def probe_ablation(torch, nn):
    """Train X150, then zero each peptide position in turn and measure the loss of skill."""
    z, pep_index, all_index = esm_heads.load_cache()
    m = features.load()

    banks = {
        "pep_pool": torch.tensor(z["peptide_pooled"].astype(np.float32)),
        "hla_pool": torch.tensor(z["hla_pooled"].astype(np.float32)),
        "pep_res": torch.tensor(z["peptide_residue"].astype(np.float32)),
        "hla_res": torch.tensor(z["hla_residue"].astype(np.float32)),
    }
    banks = {k: v.to(esm_heads.DEVICE) for k, v in banks.items()}

    tensors = {}
    for split in ("train", "val"):
        sub = m[m.split == split].reset_index(drop=True)
        pi, ai = esm_heads.row_indices(sub, pep_index, all_index)
        tensors[split] = {
            "pi": torch.from_numpy(pi).to(esm_heads.DEVICE),
            "ai": torch.from_numpy(ai).to(esm_heads.DEVICE),
            "y": torch.tensor(features.target(sub).astype(np.float32)).to(esm_heads.DEVICE),
            "allele": sub.allele.to_numpy(),
            "meta": sub,
        }

    va = tensors["val"]
    y = va["y"].cpu().numpy()
    X150 = esm_heads.build_models(torch, nn, d_in_pooled=1280, d_res=640)["X150_interaction_head"]

    def predict(model, bank_set):
        model.eval()
        with torch.no_grad():
            out = []
            for i in range(0, len(y), 1024):
                out.append(model(*esm_heads.gather(bank_set, va, slice(i, i + 1024), True)))
            return torch.cat(out).cpu().numpy()

    per_seed_full, per_seed_drop = [], []
    for seed in esm_heads.SEEDS:
        t = time.time()
        model, epoch, _, _ = esm_heads.train_one(torch, nn, X150, banks, tensors, seed)
        full = predict(model, banks)
        full_pooled = metrics._spearman(y, full)
        full_within = metrics.evaluate(y, full, va["allele"])["spearman_within_allele"]

        drops = []
        for k in range(PEPTIDE_LEN):
            ablated = dict(banks)
            pep = banks["pep_res"].clone()
            pep[:, k, :] = 0.0          # position k carries no information
            ablated["pep_res"] = pep
            p = predict(model, ablated)
            r = metrics.evaluate(y, p, va["allele"])
            drops.append({
                "position": k + 1,
                "pooled": r["spearman_pooled"],
                "within": r["spearman_within_allele"],
                "drop_pooled": full_pooled - r["spearman_pooled"],
                "drop_within": (None if (full_within is None or r["spearman_within_allele"] is None)
                                else full_within - r["spearman_within_allele"]),
            })
        per_seed_full.append({"seed": seed, "epoch": epoch,
                              "pooled": full_pooled, "within": full_within})
        per_seed_drop.append(drops)
        print(f"  ablation seed {seed}: {time.time() - t:.0f}s, "
              f"intact rho_pooled {full_pooled:.3f}", flush=True)

    summary = []
    for k in range(PEPTIDE_LEN):
        dp = [s[k]["drop_pooled"] for s in per_seed_drop]
        dw = [s[k]["drop_within"] for s in per_seed_drop if s[k]["drop_within"] is not None]
        summary.append({
            "position": f"P{k + 1}",
            "drop_pooled_mean": float(np.mean(dp)),
            "drop_pooled_spread": float(max(dp) - min(dp)),
            "drop_within_mean": float(np.mean(dw)) if dw else None,
            "drop_within_spread": float(max(dw) - min(dw)) if dw else None,
        })
    return {"intact": per_seed_full, "per_position": summary, "per_seed": per_seed_drop}


# --------------------------------------------------------------------------- probe B


def probe_likelihood(torch):
    """Masked-position pseudo-log-likelihood from ESM-2. No training, no labels."""
    from transformers import AutoTokenizer, EsmForMaskedLM

    m = features.load()
    peptides = sorted(m.peptide.unique())
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = EsmForMaskedLM.from_pretrained(MODEL_ID).to(esm_heads.DEVICE).eval()
    mask_id = tok.mask_token_id

    enc = tok(peptides, return_tensors="pt", padding=True)
    ids = enc["input_ids"].to(esm_heads.DEVICE)
    att = enc["attention_mask"].to(esm_heads.DEVICE)
    # token 0 is <cls>, so peptide position k (0-based) sits at token index k + 1
    assert ids.shape[1] == PEPTIDE_LEN + 2, f"unexpected token length {ids.shape[1]}"

    per_pos = np.zeros((len(peptides), PEPTIDE_LEN), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for k in range(PEPTIDE_LEN):
            col = k + 1
            for i in range(0, len(peptides), MLM_BATCH):
                sl = slice(i, i + MLM_BATCH)
                batch = ids[sl].clone()
                true = batch[:, col].clone()
                batch[:, col] = mask_id
                logits = model(input_ids=batch, attention_mask=att[sl]).logits[:, col, :]
                lp = torch.log_softmax(logits.float(), dim=-1)
                per_pos[sl, k] = lp.gather(1, true[:, None]).squeeze(1).cpu().numpy()
            print(f"  likelihood P{k + 1} done ({time.time() - t0:.0f}s)", flush=True)

    return {
        "peptides": peptides,
        "mean_logprob_per_position": [float(per_pos[:, k].mean()) for k in range(PEPTIDE_LEN)],
        "std_logprob_per_position": [float(per_pos[:, k].std()) for k in range(PEPTIDE_LEN)],
        "per_peptide_total": per_pos.sum(1).tolist(),
    }


def main():
    import torch
    from torch import nn

    esm_heads.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    print(f"device: {esm_heads.DEVICE}")
    OUT.mkdir(exist_ok=True)
    t0 = time.time()

    print("\nProbe A -- position ablation on trained X150")
    ablation = probe_ablation(torch, nn)

    print("\nProbe B -- ESM-2 masked-position pseudo-log-likelihood (no training)")
    likelihood = probe_likelihood(torch)

    print("\nPER-POSITION PROFILE (validation)\n")
    print(f"  {'pos':<5} {'ablation drop (pooled)':<26} {'ablation drop (within)':<26} "
          f"{'ESM-2 mean log P'}")
    lik = likelihood["mean_logprob_per_position"]
    for row, lp in zip(ablation["per_position"], lik):
        dw = ("n/a" if row["drop_within_mean"] is None
              else f"{row['drop_within_mean']:+.4f} +-{row['drop_within_spread']:.4f}")
        print(f"  {row['position']:<5} {row['drop_pooled_mean']:+.4f} "
              f"+-{row['drop_pooled_spread']:.4f}          {dw:<26} {lp:+.3f}")

    drops = [r["drop_pooled_mean"] for r in ablation["per_position"]]
    spreads = [r["drop_pooled_spread"] for r in ablation["per_position"]]
    top = int(np.argmax(drops)) + 1
    second = int(np.argsort(drops)[-2]) + 1
    print(f"\n  largest pooled drops: P{top} and P{second}")
    print(f"  anchor hypothesis predicts P2 and P9 -> "
          f"{'SUPPORTED' if {top, second} == {2, 9} else 'NOT the predicted pair'}")
    # a profile flatter than the seed noise cannot distinguish positions at all
    if max(drops) - min(drops) < max(spreads):
        print("  WARNING: the spread across positions is smaller than the seed spread. "
              "This profile does not separate positions; do not read anchors into it.")

    payload = {"ablation": ablation, "likelihood": likelihood,
               "seconds": round(time.time() - t0, 1)}
    (OUT / "anchors.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"\nwrote {OUT / 'anchors.json'}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
