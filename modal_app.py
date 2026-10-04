"""Run the sequence ladder on a Modal GPU.

What travels
------------
    src/                  the code
    splits/               the frozen split (its sha256 is asserted inside features.load)
    context/dataset.csv   labels
    cache/esm2_150m.npz   the frozen embeddings, 80 MB

Nothing is computed twice: the ESM-2 encode already happened locally and its output ships as a
file, so the GPU only ever sees the heads.

    python -m modal run modal_app.py                 # F150 + X150
    python -m modal run modal_app.py --gpu-kind T4   # cheaper, still ample for these heads

Results are written back into results/ locally, so a remote run and a local run are
interchangeable from the evidence log's point of view.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent
REMOTE = "/repo"

MODEL_ID = "facebook/esm2_t30_150M_UR50D"


def _bake_weights():
    """Download ESM-2 into the image so no run pays for it, and no run can be blocked by it."""
    from transformers import AutoTokenizer, EsmModel

    AutoTokenizer.from_pretrained(MODEL_ID)
    EsmModel.from_pretrained(MODEL_ID)


base = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "pandas", "transformers", "scipy", "scikit-learn")
    .run_function(_bake_weights)
)

repo_files = (
    lambda img: img
    .add_local_dir(REPO / "src", f"{REMOTE}/src", ignore=["__pycache__", "*.pyc"])
    .add_local_dir(REPO / "splits", f"{REMOTE}/splits")
    .add_local_file(REPO / "context" / "dataset.csv", f"{REMOTE}/context/dataset.csv")
)

image = repo_files(base).add_local_file(
    REPO / "cache" / "esm2_150m.npz", f"{REMOTE}/cache/esm2_150m.npz"
)

# L150 trains the encoder, so the embedding cache is void for it and is not shipped. X150's
# measured metrics are, so the H1 comparison reads the real file instead of a retyped number.
l150_image = repo_files(base).add_local_file(
    REPO / "results" / "esm_heads_val_metrics.json",
    f"{REMOTE}/results/esm_heads_val_metrics.json",
)

app = modal.App("stability-lens", image=image)


@app.function(gpu="A10G", timeout=3600)
def esm_heads_gpu() -> dict:
    """F150 and X150 on a GPU. Returns stdout plus the result files as text."""
    import os
    import subprocess
    import sys

    import torch

    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"device: {p.name}, {p.total_memory / 1e9:.1f} GB")
    else:
        # Never let a silent CPU fallback masquerade as a GPU run -- the timings would be
        # meaningless and we would not know which box produced the numbers.
        raise RuntimeError("no CUDA device in the container; refusing to report a GPU result")

    out = Path("/tmp/results")
    out.mkdir(exist_ok=True)
    env = {**os.environ, "RESULTS_DIR": str(out)}

    t0 = time.time()
    r = subprocess.run(
        [sys.executable, "-u", f"{REMOTE}/src/esm_heads.py"],
        capture_output=True, text=True, env=env,
    )
    elapsed = time.time() - t0

    return {
        "returncode": r.returncode,
        "stdout": r.stdout,
        "stderr": r.stderr[-4000:],
        "seconds": round(elapsed, 1),
        "gpu": torch.cuda.get_device_name(0),
        "files": {p.name: p.read_text(encoding="utf-8") for p in sorted(out.glob("*.json"))},
    }


SEEDS = (42, 43, 44)


@app.function(gpu="A10G", timeout=7200, image=image)
def allele_split_gpu() -> dict:
    """Re-run the ladder against the ALLELE split: every validation allele unseen in training."""
    import os
    import subprocess
    import sys

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device in the container; refusing to report a GPU result")
    out = Path("/tmp/results")
    out.mkdir(exist_ok=True)
    env = {**os.environ, "RESULTS_DIR": str(out)}
    t0 = time.time()
    r = subprocess.run([sys.executable, "-u", f"{REMOTE}/src/allele_split_run.py"], env=env)
    f = out / "allele_split_metrics.json"
    if r.returncode != 0 or not f.exists():
        raise RuntimeError(f"allele split failed: rc {r.returncode}, "
                           f"output {'missing' if not f.exists() else 'present'}")
    return {"seconds": round(time.time() - t0, 1), "gpu": torch.cuda.get_device_name(0),
            "payload": json.loads(f.read_text(encoding="utf-8"))}


@app.local_entrypoint()
def allele_split():
    res = allele_split_gpu.remote()
    dest = REPO / "results" / "allele_split_metrics.json"
    dest.write_text(json.dumps(res["payload"], indent=2), encoding="utf-8")
    print(f"gpu {res['gpu']}   remote {res['seconds']}s")
    print(f"wrote {dest}")


@app.function(gpu="A10G", timeout=7200, image=image)
def curve_gpu() -> dict:
    """Learning curves for one-hot vs frozen ESM-2, subsampled by peptide cluster."""
    import os
    import subprocess
    import sys

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device in the container; refusing to report a GPU result")
    print(f"learning curve on {torch.cuda.get_device_name(0)}", flush=True)
    out = Path("/tmp/results")
    out.mkdir(exist_ok=True)
    env = {**os.environ, "RESULTS_DIR": str(out)}
    t0 = time.time()
    r = subprocess.run([sys.executable, "-u", f"{REMOTE}/src/learning_curve.py"], env=env)
    f = out / "learning_curve.json"
    if r.returncode != 0 or not f.exists():
        raise RuntimeError(f"learning curve failed: rc {r.returncode}, "
                           f"output {'missing' if not f.exists() else 'present'}")
    return {"seconds": round(time.time() - t0, 1), "gpu": torch.cuda.get_device_name(0),
            "payload": json.loads(f.read_text(encoding="utf-8"))}


@app.local_entrypoint()
def curve():
    res = curve_gpu.remote()
    dest = REPO / "results" / "learning_curve.json"
    dest.write_text(json.dumps(res["payload"]), encoding="utf-8")
    print(f"{'rows':>9}{'B1':>9}{'X150':>9}{'delta':>9}")
    for k in sorted(res["payload"], key=float):
        r = res["payload"][k]
        n = sum(r["n_rows"]) / len(r["n_rows"])
        print(f"{n:>9,.0f}{r['B1']['pooled']:>9.3f}{r['X150']['pooled']:>9.3f}"
              f"{r['X150']['pooled'] - r['B1']['pooled']:>+9.3f}")
    print(f"gpu {res['gpu']}   remote {res['seconds']}s")
    print(f"wrote {dest}")



@app.function(gpu="A100", timeout=7200, image=l150_image)
def scale_gpu() -> dict:
    """ESM-2 8M -> 650M with the identical X150 head. A100 for the 650M encode."""
    import os
    import subprocess
    import sys

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device in the container; refusing to report a GPU result")
    print(f"scale sweep on {torch.cuda.get_device_name(0)}", flush=True)
    out = Path("/tmp/results")
    out.mkdir(exist_ok=True)
    env = {**os.environ, "RESULTS_DIR": str(out)}
    t0 = time.time()
    r = subprocess.run([sys.executable, "-u", f"{REMOTE}/src/scale_sweep.py"], env=env)
    f = out / "scale_sweep.json"
    if r.returncode != 0 or not f.exists():
        raise RuntimeError(f"scale sweep failed: rc {r.returncode}, "
                           f"output {'missing' if not f.exists() else 'present'}")
    return {"seconds": round(time.time() - t0, 1), "gpu": torch.cuda.get_device_name(0),
            "payload": json.loads(f.read_text(encoding="utf-8"))}


@app.local_entrypoint()
def scale():
    res = scale_gpu.remote()
    dest = REPO / "results" / "scale_sweep.json"
    dest.write_text(json.dumps(res["payload"]), encoding="utf-8")
    pay = res["payload"]
    print(f"\n{'model':<14}{'pooled':>10}{'within':>10}")
    for mid in pay["models"]:
        r = next(v for v in pay["results"].values() if v["model"] == mid)
        w = r["spearman_within_allele"]
        print(f"{r['short']:<14}{r['spearman_pooled']:>10.3f}"
              f"{(w if w is not None else float('nan')):>10.3f}")
    b = pay["b1_reference"]
    print(f"{'B1 (one-hot)':<14}{b['pooled']:>10.3f}{b['within']:>10.3f}")
    print(f"\ngpu {res['gpu']}   remote {res['seconds']}s")
    print(f"wrote {dest}")



# The layer sweep re-encodes from the model, so it needs the weights image, not the cache.
@app.function(gpu="A10G", timeout=5400, image=l150_image)
def layers_gpu() -> dict:
    """Train the X150 head on five depths of ESM-2 and see whether the default was the worst."""
    import os
    import subprocess
    import sys

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device in the container; refusing to report a GPU result")
    print(f"layer sweep on {torch.cuda.get_device_name(0)}", flush=True)
    out = Path("/tmp/results")
    out.mkdir(exist_ok=True)
    env = {**os.environ, "RESULTS_DIR": str(out)}
    t0 = time.time()
    r = subprocess.run([sys.executable, "-u", f"{REMOTE}/src/layer_sweep.py"], env=env)
    f = out / "layer_sweep.json"
    if r.returncode != 0 or not f.exists():
        raise RuntimeError(f"layer sweep failed: rc {r.returncode}, "
                           f"output {'missing' if not f.exists() else 'present'}")
    return {"seconds": round(time.time() - t0, 1), "gpu": torch.cuda.get_device_name(0),
            "payload": json.loads(f.read_text(encoding="utf-8"))}


@app.local_entrypoint()
def layers():
    res = layers_gpu.remote()
    dest = REPO / "results" / "layer_sweep.json"
    dest.write_text(json.dumps(res["payload"]), encoding="utf-8")
    pay = res["payload"]
    print(f"\n{pay['model']}, {pay['n_transformer_layers']} layers, swept {pay['layers']}")
    print(f"{'layer':<8}{'pooled':>10}{'within':>10}")
    for l in pay["layers"]:
        r = pay["results"][f"layer_{l}"]
        w = r["spearman_within_allele"]
        print(f"{l:<8}{r['spearman_pooled']:>10.3f}{(w if w is not None else float('nan')):>10.3f}")
    print(f"\ngpu    {res['gpu']}   remote {res['seconds']}s")
    print(f"wrote  {dest}")



# anchors needs the embedding cache (probe A) and the MLM head (probe B), so it gets the
# cache-carrying image, not the L150 one.
@app.function(gpu="A10G", timeout=3600, image=image)
def anchors_gpu() -> dict:
    """Position ablation on trained X150, plus ESM-2 masked-position likelihood."""
    import os
    import subprocess
    import sys

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device in the container; refusing to report a GPU result")
    print(f"anchors on {torch.cuda.get_device_name(0)}", flush=True)

    out = Path("/tmp/results")
    out.mkdir(exist_ok=True)
    env = {**os.environ, "RESULTS_DIR": str(out)}
    t0 = time.time()
    r = subprocess.run([sys.executable, "-u", f"{REMOTE}/src/anchors.py"], env=env)
    f = out / "anchors.json"
    if r.returncode != 0 or not f.exists():
        raise RuntimeError(
            f"anchors failed: returncode {r.returncode}, "
            f"output {'missing' if not f.exists() else 'present'}"
        )
    return {"seconds": round(time.time() - t0, 1), "gpu": torch.cuda.get_device_name(0),
            "payload": json.loads(f.read_text(encoding="utf-8"))}


@app.local_entrypoint()
def anchors():
    res = anchors_gpu.remote()
    dest = REPO / "results" / "anchors.json"
    dest.write_text(json.dumps(res["payload"]), encoding="utf-8")
    print(f"\ngpu    {res['gpu']}")
    print(f"remote {res['seconds']}s")
    print(f"wrote  {dest}")



@app.function(gpu="A100", timeout=10800, image=l150_image)
def l150_gpu(seed: int) -> dict:
    """One L150 seed: LoRA r=8 on K/V, encoder trains, X150 head.

    One seed per container, so the three run in parallel and stay genuinely independent --
    separate processes with no shared state, which is what makes the seed spread meaningful
    rather than three correlated draws from one run.
    """
    import os
    import subprocess
    import sys

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device in the container; refusing to report a GPU result")
    gpu = torch.cuda.get_device_properties(0)
    print(f"seed {seed} on {gpu.name}, {gpu.total_memory / 1e9:.1f} GB", flush=True)

    out = Path("/tmp/results")
    out.mkdir(exist_ok=True)
    src = Path(f"{REMOTE}/results/esm_heads_val_metrics.json")
    if src.exists():
        (out / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    env = {**os.environ, "RESULTS_DIR": str(out), "L150_SEED": str(seed)}
    t0 = time.time()
    # Not capture_output: this runs for tens of minutes and capturing would withhold every
    # per-epoch line until the end, making a slow run indistinguishable from a hung one.
    r = subprocess.run([sys.executable, "-u", f"{REMOTE}/src/l150.py"], env=env)

    f = out / f"l150_seed{seed}.json"
    if r.returncode != 0 or not f.exists():
        # Absent and broken must not look alike. Say which happened.
        raise RuntimeError(
            f"seed {seed} failed: returncode {r.returncode}, "
            f"output file {'missing' if not f.exists() else 'present'}"
        )
    return {"seed": seed, "seconds": round(time.time() - t0, 1),
            "gpu": gpu.name, "payload": json.loads(f.read_text(encoding="utf-8"))}


@app.local_entrypoint()
def l150():
    """Run every seed in parallel, then merge into the ensemble and compare against X150."""
    import sys

    import numpy as np

    sys.path.insert(0, str(REPO / "src"))
    import metrics

    t0 = time.time()
    # Persist each seed the moment it lands, before any merging. The seeds cost ~30 min of A100
    # each and the merge below is cheap; a bug in the cheap part must not destroy the expensive
    # part. Re-merging from these files needs no GPU.
    runs = []
    for r in l150_gpu.map(SEEDS, order_outputs=False):
        raw = REPO / "results" / f"l150_seed{r['seed']}.json"
        raw.write_text(json.dumps(r["payload"]), encoding="utf-8")
        print(f"  seed {r['seed']} returned in {r['seconds']:.0f}s -> {raw.name}", flush=True)
        runs.append(r)
    runs.sort(key=lambda r: r["seed"])

    first = runs[0]["payload"]
    y = np.array(first["y_true_log"])
    allele = np.array(first["allele"])
    for r in runs[1:]:
        # the rows must be identical across seeds or the ensemble mean is meaningless
        assert r["payload"]["peptide"] == first["peptide"], "seed runs disagree on the rows"

    per_seed = [metrics.evaluate(y, np.array(r["payload"]["prediction"]), allele) for r in runs]
    ens = np.mean([r["payload"]["prediction"] for r in runs], axis=0)
    out = metrics.evaluate(y, ens, allele)
    out["seeds"] = [r["seed"] for r in runs]
    out["best_epoch_per_seed"] = [r["payload"]["best_epoch"] for r in runs]
    out["trainable_parameters"] = first["trainable_parameters"]
    out["max_epochs"] = first["max_epochs"]
    out["patience"] = first["patience"]
    out["seconds_per_seed"] = [r["seconds"] for r in runs]
    for k in ("spearman_pooled", "spearman_within_allele"):
        vals = [p[k] for p in per_seed if p[k] is not None]
        out[f"{k}_per_seed"] = [round(v, 4) for v in vals]
        out[f"{k}_seed_spread"] = round(max(vals) - min(vals), 4) if vals else None

    print("\nVALIDATION RESULTS (test not touched)\n")
    print(metrics.fmt("L150_lora_kv", out))
    for r, p in zip(runs, per_seed):
        print(f"    seed {r['seed']}: best epoch {r['payload']['best_epoch']}, "
              f"{r['seconds']:.0f}s, rho_pooled {p['spearman_pooled']:.3f}")

    # H1, read from X150's own result file so it cannot drift from what was measured.
    xf = REPO / "results" / "esm_heads_val_metrics.json"
    if xf.exists():
        x = json.loads(xf.read_text(encoding="utf-8"))["X150_interaction_head"]
        print("\n  H1: does adapting the encoder help? (identical head, identical budget)")
        for key, label in (("spearman_pooled", "pooled"),
                           ("spearman_within_allele", "within-allele")):
            a, b = out[key], x[key]
            if a is None or b is None:
                continue
            spread = max(out.get(f"{key}_seed_spread") or 0, x.get(f"{key}_seed_spread") or 0)
            verdict = "larger than seed noise" if abs(a - b) > spread else "WITHIN seed noise"
            print(f"    {label:<14} L150 {a:.3f}  vs  X150 {b:.3f}   delta {a - b:+.3f}"
                  f"   (seed spread {spread:.3f} -> {verdict})")
    else:
        print("\n  X150 metrics not found; H1 skipped rather than guessed")

    (REPO / "results" / "l150_val_metrics.json").write_text(
        json.dumps({"L150_lora_kv": out}, indent=2), encoding="utf-8")
    (REPO / "results" / "l150_val_predictions.json").write_text(json.dumps({
        "y_true_log": first["y_true_log"], "allele": first["allele"],
        "peptide": first["peptide"], "cluster_id": first["cluster_id"],
        "L150_lora_kv": ens.tolist(),
        **{f"L150_seed_{r['seed']}": r["payload"]["prediction"] for r in runs},
    }), encoding="utf-8")
    print(f"\ngpu       {runs[0]['gpu']}")
    print(f"wall      {time.time() - t0:.0f}s for {len(runs)} seeds in parallel")
    print("wrote     l150_val_metrics.json, l150_val_predictions.json")


@app.local_entrypoint()
def main():
    t0 = time.time()
    res = esm_heads_gpu.remote()
    print(res["stdout"])
    if res["returncode"] != 0:
        print("--- stderr ---")
        print(res["stderr"])
        raise SystemExit(f"remote run failed with code {res['returncode']}")

    written = []
    for name, text in res["files"].items():
        # Writes to the canonical names, so a remote run replaces a local one. The cross-check
        # against CPU is done on the per-seed values printed in stdout, not on these files.
        dest = REPO / "results" / name
        dest.write_text(text, encoding="utf-8")
        written.append(dest.name)

    print(f"\ngpu       {res['gpu']}")
    print(f"remote    {res['seconds']}s compute")
    print(f"wall      {time.time() - t0:.0f}s including image build, upload and container start")
    print(f"wrote     {', '.join(written) or 'nothing'}")
