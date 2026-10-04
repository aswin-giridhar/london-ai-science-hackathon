"""MINT's joint pMHC representation behind our own head: does cross-chain pretraining beat one-hot?

The experiment Run 21 asked for and Run 29 could not answer
------------------------------------------------------------
Run 21 let ESM-2 attend across both chains in one 191-residue sequence and measured -0.017,
within noise. FINDINGS section 4 concluded that *"the interaction has to be modelled by something
trained to model it; self-attention handed the opportunity does not discover an interface."*
That is an inference from two numbers in different studies, and it deserves a direct test.

MINT is that something: ESM2-650M with cross-chain multimer attention, pretrained on 96M STRING
protein-protein interactions, so its attention heads were trained on interacting pairs rather
than merely permitted to look at them.

Run 29 scored MINT zero-shot and found 0.421 within-allele with a pooled figure below the
no-peptide floor -- real signal, wrong across-allele calibration. That is a statement about the
*released affinity head*, not about the representation underneath it. Giving our own head the
same training labels every other rung received is what makes the comparison fair.

What is held fixed
------------------
The frozen split, the three seeds, the two-learning-rate choice, the early-stopping budget and
the normalisation are all identical to Run 12 and Run 27. The only thing that changes is the
encoder. The comparison is therefore against:

    F150  0.593 / 0.278   frozen ESM-2 150M, MEAN-POOLED -- the honest architectural analogue
    X150  0.754 / 0.558   frozen ESM-2 150M, residue cross-attention
    B1    0.780 / 0.633   one-hot

F150 is the fair architectural comparator: MINT's representation is mean-pooled over the joint
complex, exactly as F150's is mean-pooled over each chain. The difference is that MINT's pooling
happens *after* cross-chain attention, so if interaction pretraining carries the signal this
project has been unable to find, the gap from 0.593 should be large.

How the representation is extracted, and why it is checkable
-------------------------------------------------------------
Rather than guess module names, the head's first Linear is identified by its shape signature --
`in_features == embed_dim (1280)` and `out_features == hidden_dim (512)` -- and the match is
asserted unique. A forward hook captures that Linear's INPUT, which is the mean-pooled joint
representation the model uses for its own prediction. If the architecture is not what the config
describes, the assertion fails loudly instead of silently yielding the wrong tensor.

    python -m modal run modal_app.py::minthead
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
from pathlib import Path

import numpy as np

import features
import metrics

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
MODEL_ID = "dkarthikeyan1/mint-stage1-affinity"
EMBED_DIM, HIDDEN_DIM = 1280, 512
SEEDS = (42, 43, 44)
MAX_EPOCHS, PATIENCE, BATCH = 40, 6, 256
F150, F150_W = 0.593, 0.278
X150, X150_W = 0.754, 0.558
B1, B1_W = 0.780, 0.633


def find_head_linear(torch, model):
    """Locate the projection head's first Linear by shape, and assert the match is unique."""
    hits = [(n, m) for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear)
            and m.in_features == EMBED_DIM and m.out_features == HIDDEN_DIM]
    assert len(hits) == 1, (
        f"expected exactly one Linear({EMBED_DIM} -> {HIDDEN_DIM}); found {len(hits)}: "
        f"{[n for n, _ in hits]}. The architecture is not what config.json describes.")
    print(f"  pooled representation taken from the input of: {hits[0][0]}", flush=True)
    return hits[0][1]


def embed(torch, model, tokenizer, hook_target, peptides, mhcs, device, batch, tag):
    """Mean-pooled joint pMHC representations, captured from the head's input."""
    captured = {}

    def hook(_module, inputs, _output):
        captured["x"] = inputs[0].detach()

    handle = hook_target.register_forward_hook(hook)
    out, t0 = [], time.time()
    try:
        for i in range(0, len(peptides), batch):
            chains, chain_ids = tokenizer.prepare_batch(peptides[i:i + batch], mhcs[i:i + batch])
            with torch.no_grad():
                model(chains.to(device), chain_ids.to(device))
            x = captured.pop("x")
            assert x.ndim == 2 and x.shape[1] == EMBED_DIM, f"{tag}: got {tuple(x.shape)}"
            out.append(x.float().cpu().numpy().astype(np.float16))
            if i % (batch * 8) == 0:
                done = min(i + batch, len(peptides))
                rate = done / max(time.time() - t0, 1e-9)
                print(f"    {tag} {done:,}/{len(peptides):,} "
                      f"({time.time() - t0:.0f}s, {rate:.0f}/s, "
                      f"eta {(len(peptides) - done) / max(rate, 1e-9):.0f}s)", flush=True)
    finally:
        handle.remove()
    e = np.concatenate(out)
    assert e.shape == (len(peptides), EMBED_DIM), f"{tag}: {e.shape}"
    return e


def train_head(torch, nn, Xtr, ytr, Xva, yva, allele_va, seed, lr):
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = Xtr.shape[1]
    model = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 512), nn.ReLU(), nn.Dropout(0.1),
                          nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1)).to(Xtr.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    lossf = nn.SmoothL1Loss()
    n = Xtr.shape[0]
    best = {"rho": -2.0, "epoch": -1, "pred": None}
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for j in range(0, n, BATCH):
            idx = perm[j:j + BATCH]
            opt.zero_grad()
            lossf(model(Xtr[idx]).squeeze(-1), ytr[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva).squeeze(-1).cpu().numpy()
        rho = metrics.evaluate(yva.cpu().numpy(), pv, allele_va)["spearman_pooled"]
        if rho > best["rho"]:
            best = {"rho": rho, "epoch": epoch, "pred": pv}
        elif epoch - best["epoch"] >= PATIENCE:
            break
    return best


def main():
    import torch
    from torch import nn
    from transformers import AutoModel
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("refusing to run an 814M model on CPU")
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    OUT.mkdir(exist_ok=True)
    t_start = time.time()

    m = features.load()
    sub = {s: m[m.split == s].reset_index(drop=True) for s in ("train", "val")}
    for s, d in sub.items():
        print(f"{s}: {len(d):,} rows", flush=True)

    print(f"loading {MODEL_ID} ...", flush=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(device).eval()
    Tok = get_class_from_dynamic_module("modeling_mint_stability.MintTokenizer",
                                        MODEL_ID, trust_remote_code=True)
    tokenizer = Tok()
    target = find_head_linear(torch, model)

    emb = {}
    for s, d in sub.items():
        emb[s] = embed(torch, model, tokenizer, target, d.peptide.tolist(),
                       d.hla_seq.tolist(), device, 32, s)
        np.save(OUT / f"mint_emb_{s}.npy", emb[s])
    del model
    torch.cuda.empty_cache()
    encode_s = time.time() - t_start
    print(f"\nencoding done in {encode_s:.0f}s", flush=True)

    Xtr = torch.tensor(emb["train"].astype(np.float32)).to(device)
    Xva = torch.tensor(emb["val"].astype(np.float32)).to(device)
    ytr = torch.tensor(features.target(sub["train"]).astype(np.float32)).to(device)
    yva = torch.tensor(features.target(sub["val"]).astype(np.float32)).to(device)
    allele_va = sub["val"].allele.to_numpy()
    y_np = yva.cpu().numpy()

    best = None
    for lr in (1e-3, 3e-4):
        preds = [train_head(torch, nn, Xtr, ytr, Xva, yva, allele_va, s, lr)["pred"]
                 for s in SEEDS]
        cand = metrics.evaluate(y_np, np.mean(preds, axis=0), allele_va)
        per = [metrics.evaluate(y_np, p, allele_va) for p in preds]
        print(f"  lr {lr:g}: pooled {cand['spearman_pooled']:.3f}", flush=True)
        if best is None or cand["spearman_pooled"] > best[0]["spearman_pooled"]:
            best = (cand, per, lr)
    r, per, lr = best
    for k in ("spearman_pooled", "spearman_within_allele"):
        vals = [p[k] for p in per if p[k] is not None]
        r[f"{k}_per_seed"] = [round(v, 4) for v in vals]
        r[f"{k}_seed_spread"] = round(max(vals) - min(vals), 4) if vals else None
    r.update({"model": MODEL_ID, "lr": lr, "encode_seconds": round(encode_s, 1),
              "embed_dim": EMBED_DIM})

    w = r["spearman_within_allele"]
    print("\nMINT JOINT REPRESENTATION + OUR HEAD (validation)\n")
    print(f"  {'rung':<42}{'pooled':>9}{'within':>9}")
    print(f"  {'MINT pooled joint rep, our head':<42}{r['spearman_pooled']:>9.3f}{w:>9.3f}")
    print(f"  {'F150 ESM-2 mean-pooled (fair analogue)':<42}{F150:>9.3f}{F150_W:>9.3f}")
    print(f"  {'X150 ESM-2 residue cross-attention':<42}{X150:>9.3f}{X150_W:>9.3f}")
    print(f"  {'B1 one-hot':<42}{B1:>9.3f}{B1_W:>9.3f}")
    print(f"\n  vs F150, the matched mean-pooled architecture: "
          f"{r['spearman_pooled'] - F150:+.3f} pooled, {w - F150_W:+.3f} within")
    print(f"  vs B1, the baseline that has beaten everything: "
          f"{r['spearman_pooled'] - B1:+.3f} pooled, {w - B1_W:+.3f} within")
    print(f"    -> {'BEATS B1' if r['spearman_pooled'] > B1 else 'B1 STILL AHEAD'}")

    dest = OUT / "mint_head.json"
    dest.write_text(json.dumps({
        "result": r,
        "reference": {"F150": F150, "F150_within": F150_W, "X150": X150,
                      "X150_within": X150_W, "B1": B1, "B1_within": B1_W},
        "seconds": round(time.time() - t_start, 1)}, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
