"""Other protein language model families: is the negative about ESM-2, or about PLMs here?

What this closes
----------------
Every foundation-model result in this project uses **ESM-2**. Run 12 varied its size across 80x
and found 0.007; Run 11 varied its depth and found nothing outside seed noise. But both hold the
*family* fixed, so the honest claim has so far been "a frozen ESM-2 does not beat one-hot", while
a reader hears "protein language models do not".

Two genuinely different families are reachable:

    Rostlab/prot_bert                        BERT encoder, 420M, UniRef100
    Rostlab/prot_t5_xl_half_uniref50-enc     T5 encoder, ~1.2B used here, UniRef50

They differ from ESM-2 in architecture (BERT and encoder-decoder T5 versus ESM's transformer),
in tokenisation, and in training corpus. If all three families land in the same place, the result
is about the *approach*, not about one model.

ESM-C was not included: no official checkpoint is published on the Hub under a loadable name, and
substituting a third-party mirror would be a worse experiment than omitting it.
SaProt was not included: it needs foldseek 3Di structural tokens, which would require folding all
5,633 peptides and 75 domains, not the 24 peptides Run 9 folded.

Tokenisation, which is where this quietly goes wrong
-----------------------------------------------------
ProtBERT and ProtT5 expect **space-separated** residues ("M A S K"), not a bare string. Passing a
bare string tokenises it as subwords and produces embeddings that look fine and mean nothing.
Residue positions are recovered with the tokenizer's own `special_tokens_mask` rather than by
assuming where BOS and EOS sit, because the two families place them differently.

The head is the identical X150, with the same normalisation, split, seeds and budget as Run 12, so
the only difference is the encoder.

    python -m modal run modal_app.py::families
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

# (hf id, short name, loader kind, batch size)
FAMILIES = [
    ("Rostlab/prot_bert", "ProtBERT 420M", "bert", 32),
    ("Rostlab/prot_t5_xl_half_uniref50-enc", "ProtT5-XL enc", "t5", 8),
]
B1_POOLED, B1_WITHIN = 0.780, 0.633
ESM2_150M_POOLED, ESM2_150M_WITHIN = 0.771, 0.585     # Run 12, same head and normalisation


def load_encoder(torch, kind, model_id):
    """Load the model, and build a tokeniser by hand.

    transformers 5.x dropped slow tokenizers, and both Rostlab checkpoints are from 2020 and
    ship only slow-tokenizer files -- `vocab.txt` for ProtBERT (81 bytes), `spiece.model` for
    ProtT5. `AutoTokenizer.from_pretrained` therefore raises on both. Rather than pin an older
    transformers, the vocabularies are read directly: a protein alphabet is tiny, every residue
    is one token, and doing it by hand makes that assumption assertable instead of assumed.
    """
    from huggingface_hub import hf_hub_download
    from transformers import BertModel, T5EncoderModel

    # BertModel, not AutoModel: Rostlab/prot_bert's config.json predates the `model_type` key,
    # which older transformers inferred and 5.x requires, so AutoModel cannot dispatch on it.
    # Naming the class states what the checkpoint is instead of asking the library to guess.
    model = (T5EncoderModel if kind == "t5" else BertModel).from_pretrained(model_id)

    if kind == "bert":
        vocab_path = hf_hub_download(model_id, "vocab.txt")
        vocab = {t.strip(): i for i, t in
                 enumerate(Path(vocab_path).read_text(encoding="utf-8").splitlines())}
        cls, sep, unk = vocab["[CLS]"], vocab["[SEP]"], vocab["[UNK]"]

        def tokenize(seq):
            return [cls] + [vocab.get(a, unk) for a in seq] + [sep], 1

    else:
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor(model_file=hf_hub_download(model_id, "spiece.model"))

        def tokenize(seq):
            # ProtT5 expects space-separated residues; sentencepiece then gives one id each
            ids = sp.encode(" ".join(seq), out_type=int)
            assert len(ids) == len(seq), (
                f"ProtT5 tokenised {len(seq)} residues into {len(ids)} tokens -- the "
                f"one-token-per-residue assumption does not hold")
            return ids, 0          # no BOS, and we append no EOS

    return tokenize, model


def encode(torch, tokenize, model, seqs, device, batch, desc):
    """Per-residue embeddings. Every sequence here is the same length, so no padding is needed.

    `offset` is where the residues start in the token sequence: 1 for ProtBERT ([CLS] first),
    0 for ProtT5. The residue count is asserted on the way out rather than trusted.
    """
    n_res = len(seqs[0])
    assert all(len(s) == n_res for s in seqs), "encode() assumes equal-length sequences"
    toks = [tokenize(s) for s in seqs]
    offset = toks[0][1]
    ids_all = torch.tensor([t[0] for t in toks], dtype=torch.long)
    out, t0 = [], time.time()
    for i in range(0, len(seqs), batch):
        ids = ids_all[i:i + batch].to(device)
        att = torch.ones_like(ids)
        with torch.no_grad():
            h = model(input_ids=ids, attention_mask=att).last_hidden_state
        h = h[:, offset:offset + n_res, :]
        assert h.shape[1] == n_res, f"{desc}: got {h.shape[1]} residues, expected {n_res}"
        out.append(h.cpu().numpy().astype(np.float16))
        if i % (batch * 40) == 0:
            print(f"    {desc} {min(i + batch, len(seqs)):,}/{len(seqs):,} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return np.concatenate(out)


def main():
    import torch
    from torch import nn

    esm_heads.DEVICE = device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("refusing to encode a 1B+ model on CPU")
    print(f"device: {torch.cuda.get_device_name(0)}")
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
        tensors[split] = {"pi": torch.from_numpy(pi).to(device),
                          "ai": torch.from_numpy(ai).to(device),
                          "y": torch.tensor(features.target(sub).astype(np.float32)).to(device),
                          "allele": sub.allele.to_numpy(), "meta": sub}
    va = tensors["val"]
    y = va["y"].cpu().numpy()

    results = {}
    dest = OUT / "family_sweep.json"
    for model_id, short, kind, batch in FAMILIES:
        t0 = time.time()
        print(f"\n=== {short} ({model_id}) ===", flush=True)
        tokenize, enc_model = load_encoder(torch, kind, model_id)
        enc_model = enc_model.to(device).eval()
        n_params = sum(p.numel() for p in enc_model.parameters())
        d_res = enc_model.config.hidden_size if hasattr(enc_model.config, "hidden_size") \
            else enc_model.config.d_model
        print(f"  {n_params / 1e6:.0f}M parameters, hidden {d_res}", flush=True)

        pep = encode(torch, tokenize, enc_model, peptides, device, batch, "peptides")
        hla_e = encode(torch, tokenize, enc_model, hla.hla_seq.tolist(), device, batch, "hla")
        assert pep.shape[1] == 9, f"peptide residues came back as {pep.shape[1]}, expected 9"
        assert hla_e.shape[1] == 182, f"HLA residues came back as {hla_e.shape[1]}, expected 182"
        encode_s = time.time() - t0
        del enc_model
        torch.cuda.empty_cache()

        pr = torch.nn.functional.layer_norm(
            torch.tensor(pep.astype(np.float32)).to(device), (d_res,))
        hr = torch.nn.functional.layer_norm(
            torch.tensor(hla_e.astype(np.float32)).to(device), (d_res,))
        banks = {"pep_res": pr, "hla_res": hr, "pep_pool": pr.mean(1), "hla_pool": hr.mean(1)}
        X150 = esm_heads.build_models(torch, nn, d_in_pooled=2 * d_res,
                                      d_res=d_res)["X150_interaction_head"]

        # the same two learning rates every model got in Run 12, each keeping its best
        best, lr0 = None, esm_heads.LR
        for lr in (1e-3, 3e-4):
            esm_heads.LR = lr
            sp, ps = [], []
            for seed in esm_heads.SEEDS:
                _, _, _, pv = esm_heads.train_one(torch, nn, X150, banks, tensors, seed)
                sp.append(pv)
                ps.append(metrics.evaluate(y, pv, va["allele"]))
            cand = metrics.evaluate(y, np.mean(sp, axis=0), va["allele"])
            print(f"    lr {lr:g}: pooled {cand['spearman_pooled']:.3f}", flush=True)
            if best is None or cand["spearman_pooled"] > best[0]["spearman_pooled"]:
                best = (cand, ps, lr)
        esm_heads.LR = lr0
        r, per_seed, chosen = best
        for k in ("spearman_pooled", "spearman_within_allele"):
            vals = [s[k] for s in per_seed if s[k] is not None]
            r[f"{k}_per_seed"] = [round(v, 4) for v in vals]
            r[f"{k}_seed_spread"] = round(max(vals) - min(vals), 4) if vals else None
        r.update({"model": model_id, "short": short, "params_millions": round(n_params / 1e6, 1),
                  "hidden": int(d_res), "lr": chosen, "encode_seconds": round(encode_s, 1),
                  "total_seconds": round(time.time() - t0, 1)})
        results[short] = r
        dest.write_text(json.dumps({"results": results}, indent=2, default=float),
                        encoding="utf-8")
        w = r["spearman_within_allele"]
        print(f"  -> pooled {r['spearman_pooled']:.3f}  "
              f"within {(w if w is not None else float('nan')):.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        del banks, pr, hr
        torch.cuda.empty_cache()

    print("\nFAMILY SWEEP (validation, X150 head throughout, test not touched)\n")
    print(f"  {'family':<18}{'params':>9}{'hidden':>8}{'pooled':>9}{'within':>9}")
    for _, short, _, _ in FAMILIES:
        r = results[short]
        w = r["spearman_within_allele"]
        print(f"  {short:<18}{r['params_millions']:>8.0f}M{r['hidden']:>8}"
              f"{r['spearman_pooled']:>9.3f}{(w if w is not None else float('nan')):>9.3f}")
    print(f"  {'ESM-2 150M (Run 12)':<18}{148:>8.0f}M{640:>8}"
          f"{ESM2_150M_POOLED:>9.3f}{ESM2_150M_WITHIN:>9.3f}")
    print(f"  {'B1 one-hot':<18}{'-':>9}{'-':>8}{B1_POOLED:>9.3f}{B1_WITHIN:>9.3f}")

    pooled = {s: results[s]["spearman_pooled"] for _, s, _, _ in FAMILIES}
    pooled["ESM-2 150M"] = ESM2_150M_POOLED
    spread = max(results[s]["spearman_pooled_seed_spread"] or 0 for _, s, _, _ in FAMILIES)
    rng = max(pooled.values()) - min(pooled.values())
    best_f = max(pooled, key=pooled.get)
    print(f"\n  range across families {rng:.3f} vs worst seed spread {spread:.3f} -> "
          f"{'family matters' if rng > spread else 'FAMILY IS WITHIN SEED NOISE'}")
    print(f"\n  THE TEST THAT MATTERS: best family vs B1 ({B1_POOLED} / {B1_WITHIN})")
    print(f"    {best_f} at {pooled[best_f]:.3f}  -> "
          f"{'B1 STILL AHEAD' if pooled[best_f] < B1_POOLED else 'HEADLINE OVERTURNED'}")

    dest.write_text(json.dumps({
        "results": results, "esm2_150m_reference": {"pooled": ESM2_150M_POOLED,
                                                    "within": ESM2_150M_WITHIN},
        "b1_reference": {"pooled": B1_POOLED, "within": B1_WITHIN},
        "range_across_families": rng, "worst_seed_spread": spread,
        "seconds": round(time.time() - t_start, 1)}, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
