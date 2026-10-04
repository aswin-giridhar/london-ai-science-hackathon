"""Does letting ESM-2 attend ACROSS the two chains help? A reachable test of the MINT hypothesis.

The hypothesis under test
--------------------------
Our results diagnose a single-chain limitation: ESM-2 is trained on individual sequences with a
masked-token objective, while peptide-HLA stability is a property of the **interface**. We encoded
the two chains separately and asked a head to reconstruct an interaction the encoder never
represented.

The literature's fix is an interaction-aware encoder - MINT (Nat Commun 2025) adds cross-chain
attention to ESM-2 650M, and a 2026 preprint applies it to this exact endpoint. MINT's weights are
not distributed through HuggingFace and it is a custom architecture, so it is out of reach today.

But the *mechanism* can be tested with what we have. Encode

    peptide (9) ++ HLA alpha1/alpha2 domain (182)  =  191 residues, as ONE sequence

so ESM-2's self-attention spans both chains, then split the per-residue output back into a 9-row
block and a 182-row block and feed the **identical X150 head**. Same split, same seeds, same
budget. The only difference anywhere is whether the encoder saw the two chains together.

What each outcome means
-----------------------
    concat clearly beats separate   cross-chain attention is the binding constraint, and
                                    MINT-style pretraining is the right next step. Our
                                    explanation is supported.

    no difference                   the limitation is not merely *seeing* both chains, it is
                                    that ESM-2's pretraining never learned what an interface
                                    IS. A stronger and more specific negative: it would mean
                                    interaction-aware PRE-TRAINING is required, not just
                                    interaction-aware input.

The honest caveat, stated up front
-----------------------------------
ESM-2 has no chain-break token. A concatenation is seen as one continuous protein, so the model
will try to fold residue 9 into residue 10 as though they were covalently bonded. That is a real
limitation of this proxy and it is precisely what MINT's architecture fixes. So this experiment
can show that cross-chain attention *helps*; if it does not help, it cannot fully distinguish
"attention across chains is useless here" from "ESM-2 cannot use it without a chain-break token".
Reported either way.

Caching note: the concatenation is row-specific, so the 4.9x unique-sequence saving does not
apply. Every (peptide, allele) pair needs its own encode.

    python -m modal run modal_app.py::concat
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
BATCH = 48
PEP_LEN, HLA_LEN = 9, 182


def main():
    import torch
    from torch import nn
    from transformers import AutoTokenizer, EsmModel

    esm_heads.DEVICE = device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("25k joint encodes; refusing to run on CPU")
    prop = torch.cuda.get_device_properties(0)
    print(f"device: {prop.name}, {prop.total_memory / 1e9:.0f} GB")
    OUT.mkdir(exist_ok=True)
    t0 = time.time()

    m = features.load()
    use = m[m.split.isin(("train", "val"))].reset_index(drop=True)
    print(f"{len(use):,} rows to encode jointly "
          f"(no unique-sequence saving: the concatenation is row-specific)")

    joint = (use.peptide + use.hla_seq).tolist()
    assert all(len(s) == PEP_LEN + HLA_LEN for s in joint), "unexpected concatenated length"

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    enc = EsmModel.from_pretrained(MODEL_ID).to(device).eval()

    # fp16 on the GPU: 25k x 191 x 640 x 2 bytes is about 6.2 GB, which fits on an A100
    n = len(joint)
    store = torch.empty((n, PEP_LEN + HLA_LEN, 640), dtype=torch.float16, device=device)
    for i in range(0, n, BATCH):
        chunk = joint[i:i + BATCH]
        e = tok(chunk, return_tensors="pt", padding=True)
        e = {k: v.to(device) for k, v in e.items()}
        with torch.no_grad():
            h = enc(**e).last_hidden_state[:, 1:-1, :]    # drop <cls> and <eos>
        store[i:i + len(chunk)] = h.half()
        if i % (BATCH * 100) == 0:
            print(f"  encoded {min(i + BATCH, n):,}/{n:,} ({time.time() - t0:.0f}s)", flush=True)
    del enc
    torch.cuda.empty_cache()
    print(f"joint encoding done in {time.time() - t0:.0f}s, "
          f"{store.element_size() * store.nelement() / 1e9:.1f} GB resident\n")

    # split back into the two blocks the X150 head expects, and normalise exactly as elsewhere
    pep_all = torch.nn.functional.layer_norm(store[:, :PEP_LEN, :].float(), (640,))
    hla_all = torch.nn.functional.layer_norm(store[:, PEP_LEN:, :].float(), (640,))
    del store
    torch.cuda.empty_cache()

    is_tr = (use.split == "train").to_numpy()
    idx_tr = torch.from_numpy(np.flatnonzero(is_tr)).to(device)
    idx_va = torch.from_numpy(np.flatnonzero(~is_tr)).to(device)
    tr_meta = use[is_tr].reset_index(drop=True)
    va_meta = use[~is_tr].reset_index(drop=True)
    y_va = features.target(va_meta)

    # the head indexes "banks" by row, so here the bank IS the row set
    banks = {"pep_res": pep_all, "hla_res": hla_all,
             "pep_pool": pep_all.mean(1), "hla_pool": hla_all.mean(1)}
    tensors = {
        "train": {"pi": idx_tr, "ai": idx_tr,
                  "y": torch.tensor(features.target(tr_meta).astype(np.float32)).to(device),
                  "allele": tr_meta.allele.to_numpy(), "meta": tr_meta},
        "val": {"pi": idx_va, "ai": idx_va,
                "y": torch.tensor(y_va.astype(np.float32)).to(device),
                "allele": va_meta.allele.to_numpy(), "meta": va_meta},
    }

    X150 = esm_heads.build_models(torch, nn, d_in_pooled=1280, d_res=640)["X150_interaction_head"]
    seed_preds, per_seed = [], []
    for seed in esm_heads.SEEDS:
        t = time.time()
        _, epoch, _, pv = esm_heads.train_one(torch, nn, X150, banks, tensors, seed)
        seed_preds.append(pv)
        per_seed.append(metrics.evaluate(y_va, pv, va_meta.allele.to_numpy()))
        print(f"  seed {seed}: epoch {epoch}, {time.time() - t:.0f}s, "
              f"rho {per_seed[-1]['spearman_pooled']:.3f}", flush=True)

    p = np.mean(seed_preds, axis=0)
    r = metrics.evaluate(y_va, p, va_meta.allele.to_numpy())
    for k in ("spearman_pooled", "spearman_within_allele"):
        vals = [s[k] for s in per_seed if s[k] is not None]
        r[f"{k}_per_seed"] = [round(v, 4) for v in vals]
        r[f"{k}_seed_spread"] = round(max(vals) - min(vals), 4) if vals else None
    r["prediction"] = p.tolist()
    r["peptide"] = va_meta.peptide.tolist()
    r["allele"] = va_meta.allele.to_numpy().tolist()
    r["seconds"] = round(time.time() - t0, 1)

    # compare against separately-encoded X150, read from its own result file
    xf = ROOT / "results" / "esm_heads_val_metrics.json"
    sep = json.loads(xf.read_text(encoding="utf-8"))["X150_interaction_head"] if xf.exists() else None

    print("\nCROSS-CHAIN ATTENTION: does encoding both chains together help?\n")
    print(f"  {'encoding':<34}{'pooled':>9}{'within':>9}")
    print(f"  {'separate chains (X150, Run 4)':<34}"
          f"{(sep['spearman_pooled'] if sep else float('nan')):>9.3f}"
          f"{(sep['spearman_within_allele'] if sep else float('nan')):>9.3f}")
    print(f"  {'concatenated, joint attention':<34}{r['spearman_pooled']:>9.3f}"
          f"{(r['spearman_within_allele'] or float('nan')):>9.3f}")
    if sep:
        dp = r["spearman_pooled"] - sep["spearman_pooled"]
        dw = (r["spearman_within_allele"] or 0) - (sep["spearman_within_allele"] or 0)
        spread = max(r["spearman_pooled_seed_spread"] or 0, sep.get("spearman_pooled_seed_spread") or 0)
        print(f"  {'delta':<34}{dp:>+9.3f}{dw:>+9.3f}")
        print(f"\n  seed spread {spread:.3f} -> "
              f"{'LARGER than seed noise' if abs(dp) > spread else 'WITHIN seed noise'}")
        print(f"  and against B1 (one-hot) at 0.780 / 0.633: "
              f"{'still behind' if r['spearman_pooled'] < 0.780 else 'AHEAD'}")
        r["delta_vs_separate_pooled"] = dp
        r["delta_vs_separate_within"] = dw

    (OUT / "concat_encoding.json").write_text(json.dumps(r, default=float), encoding="utf-8")
    print(f"\nwrote {OUT / 'concat_encoding.json'}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
