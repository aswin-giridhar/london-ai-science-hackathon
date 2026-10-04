"""Which layer of ESM-2 carries the stability signal? The attack our negative result must survive.

Why this exists
---------------
Every ESM-2 number in this project came from `last_hidden_state` -- the **final** layer. That is
the default, and it is the weakest defensible choice: the last layer of a masked language model is
specialised for predicting masked tokens, and intermediate representations are widely found to
transfer better to downstream tasks.

So the obvious attack on "a frozen protein language model does not beat one-hot here" is: *you
read the wrong layer*. This file runs that attack ourselves.

Design
------
Encode every distinct sequence once with `output_hidden_states=True`, keep a ladder of layers, and
train the **identical X150 head** on each. Nothing else changes -- same split, same seeds, same
budget, same selection rule -- so any difference is attributable to depth.

Both outcomes are useful:

    a flat profile    the negative result hardens considerably ("we checked every depth")
    a peak mid-stack  our headline was an artefact of a default, and we found it before publishing

The comparison that matters is not layer-vs-layer, it is **best layer vs B1 = 0.780 / 0.633**.

    python -m modal run modal_app.py::layers
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
BATCH = 64
# ESM-2 150M has 30 transformer layers. hidden_states has 31 entries: index 0 is the embedding
# layer before any transformer block, index 30 is the final layer we have been using all along.
LAYERS = (6, 12, 18, 24, 30)


def encode_layers(torch, model, tok, seqs, layers, device, desc):
    """Per-residue embeddings at several depths, special tokens dropped. One forward pass."""
    out = {l: [] for l in layers}
    t0 = time.time()
    for i in range(0, len(seqs), BATCH):
        enc = tok(seqs[i:i + BATCH], return_tensors="pt", padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        mask = enc["attention_mask"].bool()
        for l in layers:
            h = hs[l]
            for row in range(h.shape[0]):
                out[l].append(h[row][mask[row]][1:-1].cpu().numpy().astype(np.float16))
        if i % (BATCH * 20) == 0:
            print(f"    {desc}: {min(i + BATCH, len(seqs)):,}/{len(seqs):,} "
                  f"({(time.time() - t0):.0f}s)", flush=True)
    return {l: np.stack(v) for l, v in out.items()}


def main():
    import torch
    from torch import nn
    from transformers import AutoTokenizer, EsmModel

    esm_heads.DEVICE = device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("layer sweep encodes 5,708 sequences five times; refusing to run on CPU")
    print(f"device: {torch.cuda.get_device_name(0)}")
    OUT.mkdir(exist_ok=True)
    t_start = time.time()

    m = features.load()
    peptides = sorted(m.peptide.unique())
    hla = m.drop_duplicates("allele")[["allele", "hla_seq"]].sort_values("allele")
    pep_ix = {p: i for i, p in enumerate(peptides)}
    all_ix = {a: i for i, a in enumerate(hla.allele)}

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = EsmModel.from_pretrained(MODEL_ID).to(device).eval()
    n_layers = model.config.num_hidden_layers
    assert max(LAYERS) <= n_layers, f"model has {n_layers} layers, asked for {max(LAYERS)}"
    print(f"{n_layers} transformer layers, sweeping {LAYERS}\n")

    print("encoding peptides at all depths in one pass")
    pep = encode_layers(torch, model, tok, peptides, LAYERS, device, "peptides")
    print("encoding HLA domains")
    hla_emb = encode_layers(torch, model, tok, hla.hla_seq.tolist(), LAYERS, device, "hla")
    del model
    torch.cuda.empty_cache()
    print(f"encoded in {time.time() - t_start:.0f}s\n")

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
    X150 = esm_heads.build_models(torch, nn, d_in_pooled=1280, d_res=640)["X150_interaction_head"]

    results = {}
    for layer in LAYERS:
        for norm in (False, True):
            key = f"layer_{layer}" + ("_ln" if norm else "")
            pr = torch.tensor(pep[layer].astype(np.float32)).to(device)
            hr = torch.tensor(hla_emb[layer].astype(np.float32)).to(device)
            if norm:
                # ESM-2's residual stream grows in norm with depth, and the model applies a
                # final LayerNorm before `last_hidden_state`. So layer 30 arrives normalised
                # and the rest arrive raw, at very different scales. Comparing them without
                # fixing that would report a preprocessing artefact as a fact about depth.
                # F.layer_norm is parameter-free and fitted on nothing, so it adds no leakage
                # surface, and it is applied to EVERY layer including 30.
                pr = torch.nn.functional.layer_norm(pr, (pr.shape[-1],))
                hr = torch.nn.functional.layer_norm(hr, (hr.shape[-1],))
            banks = {"pep_res": pr, "hla_res": hr,
                     "pep_pool": pr.mean(1), "hla_pool": hr.mean(1)}

            seed_preds, per_seed = [], []
            for seed in esm_heads.SEEDS:
                _, epoch, _, pv = esm_heads.train_one(torch, nn, X150, banks, tensors, seed)
                seed_preds.append(pv)
                per_seed.append(metrics.evaluate(y, pv, va["allele"]))
            pm = np.mean(seed_preds, axis=0)
            r = metrics.evaluate(y, pm, va["allele"])
            for k in ("spearman_pooled", "spearman_within_allele"):
                vals = [sd[k] for sd in per_seed if sd[k] is not None]
                r[f"{k}_per_seed"] = [round(v, 4) for v in vals]
                r[f"{k}_seed_spread"] = round(max(vals) - min(vals), 4) if vals else None
            r["prediction"] = pm.tolist()
            r["layernorm"] = norm
            results[key] = r
            w = r["spearman_within_allele"]
            print(f"  layer {layer:>2} {'layernorm' if norm else 'raw      '}: "
                  f"pooled {r['spearman_pooled']:.3f}  "
                  f"within {'n/a' if w is None else format(w, '.3f')}", flush=True)
            del banks, pr, hr
            torch.cuda.empty_cache()

    print("\nLAYER SWEEP (validation, X150 head throughout, test not touched)\n")
    print(f"  {'layer':<7}{'raw pooled':>13}{'raw within':>13}{'LN pooled':>13}{'LN within':>13}")
    def cell(r, k):
        v = r.get(k)
        return '    n/a' if v is None else format(v, '7.3f')
    for layer in LAYERS:
        raw, ln = results[f'layer_{layer}'], results[f'layer_{layer}_ln']
        tag = '  <- Runs 4-8' if layer == max(LAYERS) else ''
        print(f"  {layer:<7}{cell(raw,'spearman_pooled'):>13}"
              f"{cell(raw,'spearman_within_allele'):>13}"
              f"{cell(ln,'spearman_pooled'):>13}"
              f"{cell(ln,'spearman_within_allele'):>13}{tag}")

    lp = {l: results[f'layer_{l}_ln']['spearman_pooled'] for l in LAYERS}
    lw = {l: (results[f'layer_{l}_ln']['spearman_within_allele'] or -9) for l in LAYERS}
    bp, bw = max(lp, key=lp.get), max(lw, key=lw.get)
    spread = max(results[f'layer_{l}_ln']['spearman_pooled_seed_spread'] for l in LAYERS)
    rng = max(lp.values()) - min(lp.values())
    print("\n  The LN columns are the fair comparison: ESM-2 applies a final LayerNorm")
    print("  before last_hidden_state, so layer 30 is normalised and the others are not.")
    print(f"\n  best normalised: layer {bp} pooled {lp[bp]:.3f}, layer {bw} within {lw[bw]:.3f}")
    print(f"  range across layers {rng:.3f} vs worst seed spread {spread:.3f} -> "
          f"{'depth matters' if rng > spread else 'DEPTH IS WITHIN SEED NOISE'}")
    print(f"\n  THE TEST THAT MATTERS: best layer vs B1 (0.780 pooled / 0.633 within)")
    print(f"    pooled  best {lp[bp]:.3f} vs B1 0.780   "
          f"{'B1 STILL AHEAD' if lp[bp] < 0.780 else 'HEADLINE OVERTURNED'}")
    print(f"    within  best {lw[bw]:.3f} vs B1 0.633   "
          f"{'B1 STILL AHEAD' if lw[bw] < 0.633 else 'HEADLINE OVERTURNED'}")

    payload = {"layers": list(LAYERS), "model": MODEL_ID, "n_transformer_layers": n_layers,
               "seconds": round(time.time() - t_start, 1), "results": results}
    (OUT / "layer_sweep.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"\nwrote {OUT / 'layer_sweep.json'}")


if __name__ == "__main__":
    main()
