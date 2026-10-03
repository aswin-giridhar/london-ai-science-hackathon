"""F150 and X150 -- frozen ESM-2 150M, two different heads.

    F150   mean-pool each chain -> concat (1280) -> MLP
    X150   per-residue peptide (9x640) and HLA (182x640) -> peptide->HLA cross-attention

The F150 -> X150 gap is hypothesis H2: does a residue-level interaction head beat mean pooling?
It should, because mean-pooling a 9-mer discards position, and position is exactly where anchor
residues live. If it does not, that is a reportable result about what the embedding encodes.

X150 geometry, following the plan of record so the structural bias drops in unchanged later:

    project 640 -> 128, four heads of width 32  (hence the /sqrt(32) in the attention)
    A_h = softmax(Q_h K_h^T / sqrt(32) + padding_mask)
    z = concat(flatten(9 attention outputs), mean(projected HLA))   -> 9*128 + 128 = 1280

The structural arm later adds a learned bias G_h inside that softmax. Nothing else changes, which is
why G_conf and G_pair end up genuinely matched.

Both rungs read the frozen cache from src/embed.py. The encoder is never trained here.

    python src/esm_heads.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import features
import metrics

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache" / "esm2_150m.npz"
OUT = ROOT / "results"

SEEDS = (42, 43, 44)
MAX_EPOCHS = 40
PATIENCE = 6
BATCH = 256
LR = 1e-3
D_MODEL = 128
N_HEADS = 4


def load_cache():
    if not CACHE.exists():
        raise SystemExit(f"missing {CACHE} -- run `python src/embed.py` first")
    z = np.load(CACHE, allow_pickle=False)
    pep_index = {p: i for i, p in enumerate(z["peptides"])}
    all_index = {a: i for i, a in enumerate(z["alleles"])}
    return z, pep_index, all_index


def row_indices(m: pd.DataFrame, pep_index, all_index):
    pi = m.peptide.map(pep_index).to_numpy()
    ai = m.allele.map(all_index).to_numpy()
    assert not np.isnan(pi.astype(float)).any(), "peptide missing from cache"
    assert not np.isnan(ai.astype(float)).any(), "allele missing from cache"
    return pi.astype(np.int64), ai.astype(np.int64)


# --------------------------------------------------------------------------- models


def build_models(torch, nn, d_in_pooled, d_res):
    class F150(nn.Module):
        """Mean-pooled baseline head."""

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in_pooled, 256), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(256, 64), nn.GELU(),
                nn.Linear(64, 1),
            )

        def forward(self, pep_pool, hla_pool, pep_res=None, hla_res=None):
            return self.net(torch.cat([pep_pool, hla_pool], dim=1)).squeeze(-1)

    class X150(nn.Module):
        """Residue-level peptide -> HLA cross-attention."""

        def __init__(self):
            super().__init__()
            self.pep_proj = nn.Linear(d_res, D_MODEL)
            self.hla_proj = nn.Linear(d_res, D_MODEL)
            self.pos = nn.Parameter(torch.zeros(9, D_MODEL))     # peptide positions matter
            self.q = nn.Linear(D_MODEL, D_MODEL)
            self.k = nn.Linear(D_MODEL, D_MODEL)
            self.v = nn.Linear(D_MODEL, D_MODEL)
            self.head = nn.Sequential(
                nn.LayerNorm(9 * D_MODEL + D_MODEL),
                nn.Linear(9 * D_MODEL + D_MODEL, 256), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(256, 64), nn.GELU(),
                nn.Linear(64, 1),
            )
            self.dh = D_MODEL // N_HEADS

        def forward(self, pep_pool, hla_pool, pep_res=None, hla_res=None):
            b = pep_res.shape[0]
            p = self.pep_proj(pep_res) + self.pos          # (b, 9, D)
            h = self.hla_proj(hla_res)                      # (b, 182, D)

            def split(x):
                return x.view(b, x.shape[1], N_HEADS, self.dh).transpose(1, 2)

            q, k, v = split(self.q(p)), split(self.k(h)), split(self.v(h))
            # scale by sqrt(head_dim) = sqrt(32); the structural bias G_h is added here later
            att = (q @ k.transpose(-2, -1)) / (self.dh ** 0.5)
            w = att.softmax(dim=-1)
            ctx = (w @ v).transpose(1, 2).reshape(b, 9, D_MODEL)   # (b, 9, D)
            z = torch.cat([ctx.reshape(b, -1), h.mean(1)], dim=1)  # 9*D + D = 1280
            return self.head(z).squeeze(-1)

    return {"F150_pooled_head": F150, "X150_interaction_head": X150}


# --------------------------------------------------------------------------- training


def train_one(torch, nn, Model, data, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    lossf = nn.MSELoss()

    tr, va = data["train"], data["val"]
    n = len(tr["y"])
    best = {"rho": -np.inf, "epoch": -1, "state": None}
    stale = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        order = np.random.permutation(n)
        for i in range(0, n, BATCH):
            idx = order[i : i + BATCH]
            opt.zero_grad()
            pred = model(tr["pep_pool"][idx], tr["hla_pool"][idx],
                         tr["pep_res"][idx], tr["hla_res"][idx])
            loss = lossf(pred, tr["y"][idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pv = []
            for i in range(0, len(va["y"]), 1024):
                s = slice(i, i + 1024)
                pv.append(model(va["pep_pool"][s], va["hla_pool"][s],
                                va["pep_res"][s], va["hla_res"][s]))
            pv = torch.cat(pv).numpy()
        rho = metrics._spearman(va["y"].numpy(), pv)
        if rho is not None and rho > best["rho"] + 1e-5:
            best = {"rho": rho, "epoch": epoch,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}, "pred": pv}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best["state"])
    return model, best["epoch"], best["rho"], best["pred"]


def main():
    import torch
    from torch import nn

    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    z, pep_index, all_index = load_cache()
    m = features.load()

    tensors = {}
    for split in ("train", "val"):
        sub = m[m.split == split].reset_index(drop=True)
        pi, ai = row_indices(sub, pep_index, all_index)
        tensors[split] = {
            "pep_pool": torch.tensor(z["peptide_pooled"][pi].astype(np.float32)),
            "hla_pool": torch.tensor(z["hla_pooled"][ai].astype(np.float32)),
            "pep_res": torch.tensor(z["peptide_residue"][pi].astype(np.float32)),
            "hla_res": torch.tensor(z["hla_residue"][ai].astype(np.float32)),
            "y": torch.tensor(features.target(sub).astype(np.float32)),
            "allele": sub.allele.to_numpy(),
            "meta": sub,
        }
        print(f"{split:<6} {len(sub):,} rows")
    print("test   NOT READ\n")

    d_res = z["peptide_residue"].shape[-1]
    models = build_models(torch, nn, d_in_pooled=2 * d_res, d_res=d_res)

    results, preds = {}, {}
    va = tensors["val"]
    preds["y_true_log"] = va["y"].numpy().tolist()
    preds["allele"] = va["allele"].tolist()
    preds["peptide"] = va["meta"].peptide.tolist()
    preds["cluster_id"] = va["meta"].cluster_id.tolist()

    for name, Model in models.items():
        seed_preds, per_seed = [], []
        for sd in SEEDS:
            t = time.time()
            _, epoch, rho, pv = train_one(torch, nn, Model, tensors, sd)
            seed_preds.append(pv)
            per_seed.append(metrics.evaluate(va["y"].numpy(), pv, va["allele"]))
            print(f"  {name} seed {sd}: epoch {epoch}, {time.time()-t:.0f}s, "
                  f"rho_pooled {per_seed[-1]['spearman_pooled']:.3f}", flush=True)
        p = np.mean(seed_preds, axis=0)
        r = metrics.evaluate(va["y"].numpy(), p, va["allele"])
        r["seeds"] = list(SEEDS)
        for k in ("spearman_pooled", "spearman_within_allele"):
            vals = [s[k] for s in per_seed if s[k] is not None]
            r[f"{k}_per_seed"] = [round(v, 4) for v in vals]
            r[f"{k}_seed_spread"] = round(max(vals) - min(vals), 4) if vals else None
        results[name] = r
        preds[name] = p.tolist()

    print("\nVALIDATION RESULTS (test not touched)\n")
    for name, r in results.items():
        print(metrics.fmt(name, r))

    f, x = results["F150_pooled_head"], results["X150_interaction_head"]
    print("\n  H2: does a residue interaction head beat mean pooling?")
    for key, label in (("spearman_pooled", "pooled"), ("spearman_within_allele", "within-allele")):
        a, b = x[key], f[key]
        if a is None or b is None:
            continue
        spread = max(x.get(f"{key}_seed_spread") or 0, f.get(f"{key}_seed_spread") or 0)
        verdict = "larger than seed noise" if abs(a - b) > spread else "WITHIN seed noise"
        print(f"    {label:<14} X150 {a:.3f}  vs  F150 {b:.3f}   delta {a-b:+.3f}"
              f"   (seed spread {spread:.3f} -> {verdict})")

    (OUT / "esm_heads_val_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (OUT / "esm_heads_val_predictions.json").write_text(json.dumps(preds), encoding="utf-8")
    print(f"\nwrote {OUT/'esm_heads_val_metrics.json'}")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
