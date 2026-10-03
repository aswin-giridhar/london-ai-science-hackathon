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

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "pandas")
    .add_local_dir(REPO / "src", f"{REMOTE}/src", ignore=["__pycache__", "*.pyc"])
    .add_local_dir(REPO / "splits", f"{REMOTE}/splits")
    .add_local_file(REPO / "context" / "dataset.csv", f"{REMOTE}/context/dataset.csv")
    .add_local_file(REPO / "cache" / "esm2_150m.npz", f"{REMOTE}/cache/esm2_150m.npz")
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
