"""L150 -- ESM-2 150M with LoRA r=8 on K/V, trained end to end with the X150 head.

This is H1, and it is the brief's actual question: does adapting the encoder help?

The comparison is matched by construction
-----------------------------------------
L150 imports the **X150 head class itself** from esm_heads -- same architecture, same
initialisation, same optimiser, same seeds, same split, same early-stopping rule. The single
difference is that gradients reach the encoder. So `L150 - X150` is attributable to adaptation
and nothing else.

LoRA
----
    W_eff = W + (alpha / r) * B @ A        r = 8, alpha = 16, dropout 0.05
    B is zero-initialised, so at step 0 the model is *exactly* frozen X150.

Applied to the key and value projections of all 30 layers = 60 modules. Each holds
2 * 8 * 640 = 10,240 weights, so 614,400 trainable parameters against 148M frozen -- 0.41%.
The module count is asserted at build time rather than assumed.

Why the cache cannot be used
----------------------------
`cache/esm2_150m.npz` is valid only while the encoder is frozen; its manifest says so. Once LoRA
trains, an embedding changes every step, so every batch re-encodes. That is what makes this rung
need a real GPU.

The one saving that survives: a batch of 256 rows still contains at most 75 distinct HLA domains,
and the two chains are encoded independently, so the batch encodes `unique(alleles)` and indexes
the result back out. Exact, not an approximation, and it removes most of the 182-token cost.

    python -m modal run modal_app.py::l150_gpu
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
import pandas as pd

import esm_heads
import features
import metrics

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))

MODEL_ID = "facebook/esm2_t30_150M_UR50D"
# One seed per process, so seeds can run as parallel containers. L150_SEED selects it.
SEED = int(os.environ.get("L150_SEED", "42"))
BATCH = 64            # rows per step; the HLA forward is the memory driver, not this
EVAL_BATCH = 256

# Matched to X150's budget, which is what makes L150 - X150 attributable to adaptation.
# X150 selected epochs 26 / 20 / 20 out of 40 with patience 6 -- it was still improving past
# epoch 20. An earlier version of this file used 12 epochs and patience 3, which would have
# truncated L150 and understated adaptation. A shorter budget is not a matched comparison.
MAX_EPOCHS = 40
PATIENCE = 6
LR_LORA = 3e-4        # the adapter wants a larger step than the head
LR_HEAD = 1e-3
R = 8
ALPHA = 16
LORA_DROPOUT = 0.05


# --------------------------------------------------------------------------- LoRA


def make_lora(torch, nn):
    class LoRALinear(nn.Module):
        """W_eff = W + (alpha/r) B A, with W frozen and B zero-initialised."""

        def __init__(self, base, r=R, alpha=ALPHA, dropout=LORA_DROPOUT):
            super().__init__()
            self.base = base
            for p in self.base.parameters():
                p.requires_grad = False
            self.A = nn.Parameter(torch.empty(r, base.in_features))
            self.B = nn.Parameter(torch.zeros(base.out_features, r))
            nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
            self.drop = nn.Dropout(dropout)
            self.scale = alpha / r

        def forward(self, x):
            return self.base(x) + self.drop(x) @ self.A.T @ self.B.T * self.scale

    return LoRALinear


def inject_lora(torch, nn, encoder, verbose=True):
    """Replace every attention key/value projection with a LoRA-wrapped copy.

    Walks the module tree by name instead of hard-coding a path, because the attribute layout
    differs between transformers versions. Asserts the count so a silent miss -- which would make
    L150 a very slow re-run of X150 -- cannot pass.
    """
    LoRALinear = make_lora(torch, nn)
    targets = []
    for name, module in encoder.named_modules():
        if name.endswith("attention.self.key") or name.endswith("attention.self.value"):
            targets.append(name)

    n_layers = encoder.config.num_hidden_layers
    expected = 2 * n_layers
    assert len(targets) == expected, (
        f"expected {expected} key/value projections across {n_layers} layers, found "
        f"{len(targets)}. The module layout changed; fix the match rather than lowering this."
    )

    for name in targets:
        parent = encoder.get_submodule(name.rsplit(".", 1)[0])
        attr = name.rsplit(".", 1)[1]
        setattr(parent, attr, LoRALinear(getattr(parent, attr)))

    for p in encoder.parameters():
        p.requires_grad = False
    trainable = 0
    for n, p in encoder.named_parameters():
        if n.endswith(".A") or n.endswith(".B"):
            p.requires_grad = True
            trainable += p.numel()

    frozen = sum(p.numel() for p in encoder.parameters()) - trainable
    if verbose:
        print(f"  LoRA r={R} alpha={ALPHA} on {len(targets)} K/V modules "
              f"({n_layers} layers x 2)")
        print(f"  trainable {trainable:,} / {trainable + frozen:,} "
              f"({100 * trainable / (trainable + frozen):.2f}%)")
    assert trainable == expected * (2 * R * encoder.config.hidden_size), \
        f"unexpected trainable count {trainable:,}"
    return trainable


# --------------------------------------------------------------------------- model


def build_l150(torch, nn, encoder, head):
    class L150(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.head = head

        def encode(self, ids, mask):
            out = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
            return out[:, 1:-1, :]        # drop <cls> and <eos>; every sequence is full length

        def forward(self, pep_ids, pep_mask, hla_ids, hla_mask, hla_back):
            pep = self.encode(pep_ids, pep_mask)              # (b, 9, 640)
            # encode only the DISTINCT alleles in this batch, then scatter back to rows.
            # Exact: the two chains never attend to each other, so an HLA embedding does not
            # depend on which peptide it is paired with.
            hla_u = self.encode(hla_ids, hla_mask)            # (u, 182, 640)
            hla = hla_u[hla_back]                             # (b, 182, 640)
            return self.head(None, None, pep, hla)

    return L150


# --------------------------------------------------------------------------- data


def tokenise(tok, seqs, device, torch):
    enc = tok(list(seqs), return_tensors="pt", padding=True)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def main():
    import torch
    from torch import nn
    from transformers import AutoTokenizer, EsmModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("L150 re-encodes every batch; refusing to run on CPU")
    print(f"device: {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    OUT.mkdir(exist_ok=True)
    t_start = time.time()

    m = features.load()
    peptides = sorted(m.peptide.unique())
    hla = m.drop_duplicates("allele")[["allele", "hla_seq"]].sort_values("allele")
    pep_ix = {p: i for i, p in enumerate(peptides)}
    all_ix = {a: i for i, a in enumerate(hla.allele)}
    print(f"{len(peptides):,} distinct peptides, {len(hla)} distinct HLA domains")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    PEP_IDS, PEP_MASK = tokenise(tok, peptides, device, torch)
    HLA_IDS, HLA_MASK = tokenise(tok, hla.hla_seq, device, torch)
    print(f"token tensors: peptides {tuple(PEP_IDS.shape)}, hla {tuple(HLA_IDS.shape)}")

    split = {}
    for name in ("train", "val", "test"):
        sub = m[m.split == name].reset_index(drop=True)
        split[name] = {
            "pi": torch.tensor(sub.peptide.map(pep_ix).to_numpy(), device=device),
            "ai": torch.tensor(sub.allele.map(all_ix).to_numpy(), device=device),
            "y": torch.tensor(features.target(sub).astype(np.float32), device=device),
            "allele": sub.allele.to_numpy(),
            "meta": sub,
        }
        print(f"{name:<6} {len(sub):,} rows")
    print("test   NOT READ\n")

    heads = esm_heads.build_models(torch, nn, d_in_pooled=1280, d_res=640)
    X150 = heads["X150_interaction_head"]

    va = split["val"]
    y_val = va["y"].cpu().numpy()

    if True:
        seed = SEED
        torch.manual_seed(seed)
        np.random.seed(seed)

        encoder = EsmModel.from_pretrained(MODEL_ID).to(device)
        n_trainable = inject_lora(torch, nn, encoder, verbose=True)
        encoder.gradient_checkpointing_enable()
        model = build_l150(torch, nn, encoder, X150().to(device))().to(device)

        lora_params = [p for n, p in model.encoder.named_parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            [{"params": lora_params, "lr": LR_LORA},
             {"params": model.head.parameters(), "lr": LR_HEAD}],
            weight_decay=1e-2,
        )
        lossf = nn.MSELoss()
        scaler_dtype = torch.bfloat16

        def run_batch(d, idx):
            ai = d["ai"][idx]
            uniq, back = torch.unique(ai, return_inverse=True)
            return model(PEP_IDS[d["pi"][idx]], PEP_MASK[d["pi"][idx]],
                         HLA_IDS[uniq], HLA_MASK[uniq], back)

        n = len(split["train"]["y"])
        best = {"rho": -np.inf, "epoch": -1, "pred": None, "state": None}
        stale = 0
        for epoch in range(MAX_EPOCHS):
            model.train()
            t_ep = time.time()
            order = np.random.permutation(n)
            for i in range(0, n, BATCH):
                idx = torch.from_numpy(order[i:i + BATCH]).to(device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=scaler_dtype):
                    pred = run_batch(split["train"], idx)
                    loss = lossf(pred.float(), split["train"]["y"][idx])
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=scaler_dtype):
                pv = []
                for i in range(0, len(va["y"]), EVAL_BATCH):
                    idx = torch.arange(i, min(i + EVAL_BATCH, len(va["y"])), device=device)
                    pv.append(run_batch(va, idx).float())
                pv = torch.cat(pv).cpu().numpy()
            rho = metrics._spearman(y_val, pv)
            ep_secs = time.time() - t_ep
            print(f"  seed {seed} epoch {epoch}: {ep_secs:.0f}s  rho {rho:.4f}", flush=True)
            if epoch == 0:
                # Measure, then project. Not a guard that fails closed -- a number printed early
                # enough to act on. A guard built on a guessed constant is itself a hazard.
                worst = ep_secs * MAX_EPOCHS
                print(f"  [projection] {ep_secs:.0f}s/epoch -> at most {worst / 60:.0f} min for "
                      f"this seed x {MAX_EPOCHS} epochs (early stopping will cut this)",
                      flush=True)
                if worst > 9000:
                    print("  [projection] WARNING: that exceeds the container timeout. "
                          "Expect this to be cut short.", flush=True)
            if rho is not None and rho > best["rho"] + 1e-5:
                # Keep the weights, not a test prediction. Test is scored ONCE after
                # training, from the val-selected checkpoint, so nothing about the test split
                # can reach the selection decision even by accident.
                best = {"rho": rho, "epoch": epoch, "pred": pv,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
                stale = 0
            else:
                stale += 1
                if stale >= PATIENCE:
                    break

        # --- score TEST from the validation-selected checkpoint
        te = split["test"]
        y_te = te["y"].cpu().numpy()
        model.load_state_dict(best["state"])
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=scaler_dtype):
            tp = []
            for i in range(0, len(y_te), EVAL_BATCH):
                idx = torch.arange(i, min(i + EVAL_BATCH, len(y_te)), device=device)
                tp.append(run_batch(te, idx).float())
            test_pred = torch.cat(tp).cpu().numpy()
        test_res = metrics.evaluate(y_te, test_pred, te["allele"])
        print(f"  seed {seed} TEST pooled {test_res['spearman_pooled']:.3f} "
              f"within {test_res['spearman_within_allele']}", flush=True)

        res = metrics.evaluate(y_val, best["pred"], va["allele"])
        print(f"  seed {seed} best epoch {best['epoch']}, "
              f"rho_pooled {res['spearman_pooled']:.3f}, "
              f"rho_within {res['spearman_within_allele']}", flush=True)

    # One seed per process. The seed-ensemble mean and the H1 comparison are computed by the
    # caller once every seed has returned -- see modal_app.py::l150. Writing a single-seed file
    # here and merging later keeps the seeds genuinely independent: separate processes, separate
    # containers, no shared state to leak one run's choices into another's.
    out = {
        "rung": "L150_lora_kv",
        "seed": seed,
        "best_epoch": best["epoch"],
        "trainable_parameters": int(n_trainable),
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "metrics": res,
        "prediction": best["pred"].tolist(),
        "test_prediction": test_pred.tolist(),
        "test_metrics": test_res,
        "test_y_true_log": y_te.tolist(),
        "test_allele": te["allele"].tolist(),
        "test_peptide": te["meta"].peptide.tolist(),
        "y_true_log": y_val.tolist(),
        "allele": va["allele"].tolist(),
        "peptide": va["meta"].peptide.tolist(),
        "cluster_id": va["meta"].cluster_id.tolist(),
        "seconds": round(time.time() - t_start, 1),
    }
    path = OUT / f"l150_seed{seed}.json"
    path.write_text(json.dumps(out), encoding="utf-8")
    print(f"\nwrote {path}")
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
