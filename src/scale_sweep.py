"""Does the negative result survive scale? ESM-2 from 8M to 650M, identical head throughout.

The attack this answers
-----------------------
Our headline is "no ESM-2 rung beats a one-hot MLP". We tested **150M**. The claim a reader will
hear is "protein language models do not help", and the published figure that motivated this
endpoint used **650M**. A single size cannot distinguish:

    the approach does not work here          (flat scaling curve)
    we did not use a big enough model        (rising curve)

Four points across an 80x parameter range separates those.

Design
------
The **identical X150 head** on every encoder -- same split, same seeds, same budget, same selection
rule. Only `d_res` changes with the encoder's hidden width, which the head takes as an argument.
Embeddings are LayerNormed before the head so that models with different activation scales are
compared on equal terms; `layer_sweep.py` showed that skipping this produces an artefact.

The comparison that decides anything is **best model vs B1 = 0.780 pooled / 0.633 within-allele**.

    python -m modal run modal_app.py::scale
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

# (model id, short name, approximate parameter count) -- ascending
MODELS = [
    ("facebook/esm2_t6_8M_UR50D", "ESM-2 8M", 8),
    ("facebook/esm2_t12_35M_UR50D", "ESM-2 35M", 35),
    ("facebook/esm2_t30_150M_UR50D", "ESM-2 150M", 150),
    ("facebook/esm2_t33_650M_UR50D", "ESM-2 650M", 650),
]
BATCH = 32          # 650M on 182-token HLA needs a smaller batch than 150M did

B1_POOLED, B1_WITHIN = 0.780, 0.633


def encode(torch, model, tok, seqs, device, desc):
    """Final-layer per-residue embeddings, special tokens dropped."""
    out = []
    t0 = time.time()
    for i in range(0, len(seqs), BATCH):
        enc = tok(seqs[i:i + BATCH], return_tensors="pt", padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        mask = enc["attention_mask"].bool()
        for row in range(h.shape[0]):
            out.append(h[row][mask[row]][1:-1].cpu().numpy().astype(np.float16))
        if i % (BATCH * 40) == 0:
            print(f"    {desc} {min(i + BATCH, len(seqs)):,}/{len(seqs):,} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return np.stack(out)


def main():
    import torch
    from torch import nn
    from transformers import AutoTokenizer, EsmModel

    esm_heads.DEVICE = device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("refusing to encode four models on CPU")
    print(f"device: {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")
    OUT.mkdir(exist_ok=True)
    t_start = time.time()

    m = features.load()
    peptides = sorted(m.peptide.unique())
    hla = m.drop_duplicates("allele")[["allele", "hla_seq"]].sort_values("allele")
    pep_ix = {p: i for i, p in enumerate(peptides)}
    all_ix = {a: i for i, a in enumerate(hla.allele)}

    tensors = {}
    for split in ("train", "val"):
        sub = m[m.split == split].reset_index(drop=True)
        pi, ai = esm_heads.row_indices(sub, pep_ix, all_ix)
        tensors[split] = {
            "pi": torch.from_numpy(pi).to(device),
            "ai": torch.from_numpy(ai).to(device),
            "y": torch.tensor(features.target(sub).astype(np.float32)).to(device),
            "allele": sub.allele.to_numpy(),
            "meta": sub,
        }
    va = tensors["val"]
    y = va["y"].cpu().numpy()

    results = {}
    for model_id, short, params_m in MODELS:
        t0 = time.time()
        print(f"\n=== {short} ({model_id}) ===", flush=True)
        tok = AutoTokenizer.from_pretrained(model_id)
        enc_model = EsmModel.from_pretrained(model_id).to(device).eval()
        d_res = enc_model.config.hidden_size
        n_params = sum(p.numel() for p in enc_model.parameters())
        print(f"  {n_params / 1e6:.0f}M parameters, hidden {d_res}, "
              f"{enc_model.config.num_hidden_layers} layers", flush=True)

        pep = encode(torch, enc_model, tok, peptides, device, "peptides")
        hla_e = encode(torch, enc_model, tok, hla.hla_seq.tolist(), device, "hla")
        encode_s = time.time() - t0
        del enc_model
        torch.cuda.empty_cache()

        pr = torch.tensor(pep.astype(np.float32)).to(device)
        hr = torch.tensor(hla_e.astype(np.float32)).to(device)
        # identical normalisation for every model: activation scales differ between sizes, and
        # layer_sweep.py showed that comparing unnormalised representations reports a
        # preprocessing artefact rather than a property of the model.
        pr = torch.nn.functional.layer_norm(pr, (d_res,))
        hr = torch.nn.functional.layer_norm(hr, (d_res,))
        banks = {"pep_res": pr, "hla_res": hr, "pep_pool": pr.mean(1), "hla_pool": hr.mean(1)}

        X150 = esm_heads.build_models(torch, nn, d_in_pooled=2 * d_res,
                                      d_res=d_res)["X150_interaction_head"]

        # EVERY model gets the same two learning rates and the same selection rule. A single
        # shared LR is not neutral across an 80x range of hidden width -- 650M at lr 1e-3 scored
        # 0.526, below the 0.563 no-peptide floor, with a within-allele seed spread of 0.185.
        # A model cannot be worse than knowing nothing unless its head failed to converge, and
        # publishing that as "bigger is worse" would be reporting a tuning failure as a finding.
        # Giving the larger model a smaller LR *only* would be unfair the other way, so all four
        # get both and each keeps its best. Equal effort, chosen on validation like everything else.
        best = None
        lr_default = esm_heads.LR
        for lr in (1e-3, 3e-4):
            esm_heads.LR = lr
            sp, ps = [], []
            for seed in esm_heads.SEEDS:
                _, epoch, _, pv = esm_heads.train_one(torch, nn, X150, banks, tensors, seed)
                sp.append(pv)
                ps.append(metrics.evaluate(y, pv, va["allele"]))
            cand = metrics.evaluate(y, np.mean(sp, axis=0), va["allele"])
            print(f"    lr {lr:g}: pooled {cand['spearman_pooled']:.3f}", flush=True)
            if best is None or cand["spearman_pooled"] > best[0]["spearman_pooled"]:
                best = (cand, np.mean(sp, axis=0), ps, lr)
        esm_heads.LR = lr_default
        r, pm, per_seed, chosen_lr = best[0], best[1], best[2], best[3]
        r["lr"] = chosen_lr
        for k in ("spearman_pooled", "spearman_within_allele"):
            vals = [s[k] for s in per_seed if s[k] is not None]
            r[f"{k}_per_seed"] = [round(v, 4) for v in vals]
            r[f"{k}_seed_spread"] = round(max(vals) - min(vals), 4) if vals else None
        r.update({"model": model_id, "short": short, "params_millions": round(n_params / 1e6, 1),
                  "hidden": d_res, "encode_seconds": round(encode_s, 1),
                  "total_seconds": round(time.time() - t0, 1), "prediction": pm.tolist()})
        results[short] = r
        w = r["spearman_within_allele"]
        print(f"  -> pooled {r['spearman_pooled']:.3f}  "
              f"within {'n/a' if w is None else format(w, '.3f')}  "
              f"({time.time() - t0:.0f}s total)", flush=True)
        del banks, pr, hr
        torch.cuda.empty_cache()

    print("\nSCALE SWEEP (validation, X150 head throughout, test not touched)\n")
    print(f"  {'model':<14}{'params':>9}{'hidden':>8}{'pooled':>10}{'within':>10}{'encode s':>10}")
    for _, short, _ in MODELS:
        r = results[short]
        w = r["spearman_within_allele"]
        print(f"  {short:<14}{r['params_millions']:>8.0f}M{r['hidden']:>8}"
              f"{r['spearman_pooled']:>10.3f}"
              f"{(w if w is not None else float('nan')):>10.3f}{r['encode_seconds']:>10.0f}")

    pooled = {s: results[s]["spearman_pooled"] for _, s, _ in MODELS}
    within = {s: (results[s]["spearman_within_allele"] or -9) for _, s, _ in MODELS}
    bp, bw = max(pooled, key=pooled.get), max(within, key=within.get)
    spread = max(results[s]["spearman_pooled_seed_spread"] for _, s, _ in MODELS)
    rng = max(pooled.values()) - min(pooled.values())

    print(f"\n  range across 80x of scale: {rng:.3f} pooled, "
          f"worst seed spread {spread:.3f} -> "
          f"{'scale matters' if rng > spread else 'SCALE IS WITHIN SEED NOISE'}")
    print(f"\n  THE TEST THAT MATTERS: best model vs B1 ({B1_POOLED} pooled / {B1_WITHIN} within)")
    print(f"    pooled  best {bp} {pooled[bp]:.3f} vs B1 {B1_POOLED}   "
          f"{'B1 STILL AHEAD' if pooled[bp] < B1_POOLED else 'HEADLINE OVERTURNED'}")
    print(f"    within  best {bw} {within[bw]:.3f} vs B1 {B1_WITHIN}   "
          f"{'B1 STILL AHEAD' if within[bw] < B1_WITHIN else 'HEADLINE OVERTURNED'}")

    payload = {"models": [m[0] for m in MODELS], "seconds": round(time.time() - t_start, 1),
               "b1_reference": {"pooled": B1_POOLED, "within": B1_WITHIN}, "results": results}
    (OUT / "scale_sweep.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"\nwrote {OUT / 'scale_sweep.json'}")


if __name__ == "__main__":
    main()
