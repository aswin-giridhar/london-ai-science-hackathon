"""The last gap in ARCHITECTURE.md section 4: peptide likelihood IN the groove, and the difference.

What the plan asked for
-----------------------
    "Log-likelihood / perplexity -- score the peptide in HLA context, and context-minus-no-context
     to isolate what the groove explains."

Run 5 did the unconditioned half: masked-position pseudo-log-likelihood of the peptide alone,
which predicts stability at rho +0.002. The conditioned half needs the peptide scored *while the
groove is present*, and the obvious way to do that with a single-chain model is to concatenate.

Three scores per (peptide, allele) row
---------------------------------------
    PLL_alone     mask each of the 9 peptide positions in the peptide by itself
    PLL_context   mask each of the 9 positions in peptide ++ HLA (191 residues)
    delta         PLL_context - PLL_alone

`delta` is the quantity the plan actually wanted: how much more predictable the peptide becomes
once the groove is visible. If a groove "explains" a peptide it binds stably, delta should track
half-life. No training, no labels, no fitted parameters -- so no leakage surface at all.

The caveat, which is the whole difficulty
------------------------------------------
ESM-2 has **no chain-break token**. A concatenation is read as one continuous protein, so the
model treats peptide residue 9 as covalently bonded to HLA residue 1. The repository README has
said from the start that concatenation does not establish biological conditioning, and Run 21
showed that letting attention span both chains changes downstream performance by -0.017, inside
seed noise.

So a null here is the expected outcome and does not distinguish "the groove explains nothing
about stability" from "ESM-2 cannot condition on a chain it does not know is separate". Both
readings are reported. A *positive* result would be the surprising one.

    python -m modal run modal_app.py::condlik
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
from scipy.stats import rankdata

import features
import metrics

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
MODEL_ID = "facebook/esm2_t30_150M_UR50D"
PEP_LEN, HLA_LEN = 9, 182
BATCH = 32


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / d) if d else np.nan


def pll(torch, model, tok, seqs, n_positions, device, desc):
    """Masked-position pseudo-log-likelihood over the FIRST n_positions residues."""
    enc = tok(list(seqs), return_tensors="pt", padding=True)
    ids, att = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    total = np.zeros(len(seqs), dtype=np.float64)
    t0 = time.time()
    with torch.no_grad():
        for k in range(n_positions):
            col = k + 1                      # token 0 is <cls>
            for i in range(0, len(seqs), BATCH):
                sl = slice(i, i + BATCH)
                b = ids[sl].clone()
                true = b[:, col].clone()
                b[:, col] = tok.mask_token_id
                logits = model(input_ids=b, attention_mask=att[sl]).logits[:, col, :]
                lp = torch.log_softmax(logits.float(), dim=-1)
                total[sl] += lp.gather(1, true[:, None]).squeeze(1).cpu().numpy()
            print(f"    {desc} P{k + 1} done ({time.time() - t0:.0f}s)", flush=True)
    return total


def main():
    import torch
    from transformers import AutoTokenizer, EsmForMaskedLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("25k masked forwards over 191 tokens; refusing to run on CPU")
    print(f"device: {torch.cuda.get_device_name(0)}")
    OUT.mkdir(exist_ok=True)
    t0 = time.time()

    m = features.load()
    va = m[m.split == "val"].reset_index(drop=True)
    y = features.target(va)
    allele = va.allele.to_numpy()
    print(f"{len(va):,} validation rows, {va.peptide.nunique():,} distinct peptides\n")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = EsmForMaskedLM.from_pretrained(MODEL_ID).to(device).eval()

    # --- peptide alone: only distinct peptides need scoring
    peps = sorted(va.peptide.unique())
    print(f"scoring {len(peps):,} distinct peptides alone")
    alone_by_pep = dict(zip(peps, pll(torch, model, tok, peps, PEP_LEN, device, "alone")))
    alone = va.peptide.map(alone_by_pep).to_numpy()

    # --- peptide in the groove: row-specific, so every row is its own sequence
    joint = (va.peptide + va.hla_seq).tolist()
    assert all(len(s) == PEP_LEN + HLA_LEN for s in joint)
    print(f"\nscoring {len(joint):,} peptide++HLA concatenations")
    context = pll(torch, model, tok, joint, PEP_LEN, device, "in context")

    delta = context - alone

    print("\nCORRELATION WITH MEASURED HALF-LIFE (validation, no training anywhere)\n")
    print(f"  {'score':<34}{'pooled':>9}{'within':>9}")
    out = {}
    for name, v in (("PLL_alone (Run 5, reproduced)", alone),
                    ("PLL_in_HLA_context", context),
                    ("delta = context - alone", delta)):
        r = metrics.evaluate(y, v, allele)
        w = r["spearman_within_allele"]
        out[name] = {"pooled": r["spearman_pooled"], "within": w,
                     "mean": float(v.mean()), "std": float(v.std())}
        print(f"  {name:<34}{r['spearman_pooled']:>9.3f}"
              f"{(w if w is not None else float('nan')):>9.3f}")

    print(f"\n  for scale, on the same rows: B0b 0.563 | X150 0.754 | B1 0.780 pooled")

    # does the groove change the peptide's score at all?
    shift = float(np.mean(delta))
    corr = spearman(alone, context)
    print(f"\n  mean shift from adding the groove: {shift:+.2f} nats over 9 positions")
    print(f"  rank correlation between the two scores: {corr:.3f}")
    print(f"  -> {'the groove barely changes the ranking' if corr > 0.9 else 'the groove substantially reorders the peptides'}")
    out["mean_shift_nats"] = shift
    out["rank_corr_alone_vs_context"] = corr

    best = max(("PLL_alone (Run 5, reproduced)", "PLL_in_HLA_context", "delta = context - alone"),
               key=lambda k: abs(out[k]["pooled"]))
    print(f"\nVERDICT")
    print(f"  strongest of the three: {best} at {out[best]['pooled']:+.3f} pooled")
    if abs(out[best]["pooled"]) < 0.1:
        print("  All three are near zero. ESM-2's likelihood carries no usable stability signal,")
        print("  conditioned or not. Consistent with Run 5 (+0.002 unconditioned) and Run 21")
        print("  (joint attention worth -0.017, inside seed noise).")
        print("  NOTE this cannot separate 'the groove explains nothing about stability' from")
        print("  'ESM-2 cannot condition on a chain it does not know is separate'. ESM-2 has no")
        print("  chain-break token, so the concatenation is read as one continuous protein.")
    else:
        print("  A non-trivial signal appeared. This would be the surprising outcome and should")
        print("  be checked against a shuffled-HLA control before being believed.")

    (OUT / "conditioned_likelihood.json").write_text(
        json.dumps({"n_rows": int(len(va)), "scores": out,
                    "seconds": round(time.time() - t0, 1)}, indent=2, default=float),
        encoding="utf-8")
    print(f"\nwrote {OUT / 'conditioned_likelihood.json'}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
