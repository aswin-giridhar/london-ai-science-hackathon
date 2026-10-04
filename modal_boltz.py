"""Boltz-2 structure generation for peptide-HLA pairs, on Modal.

Scope
-----
**B1 only: generate structures and measure what they contain.** The geometry residual (B2) is
deliberately not attempted -- at the planned cohort it is underpowered by roughly an order of
magnitude, and the honest version needs the full test partition plus four separate fixes.

The construct, and why it departs from the plan of record
---------------------------------------------------------
The plan assumed a verified full heavy chain plus beta-2 microglobulin. The dataset carries only
the **182-aa alpha1/alpha2 domain** -- no alpha3, no b2m. That domain is the complete
peptide-binding groove (alpha1 ~ 1-90, alpha2 ~ 91-182: two helices over a beta-sheet floor), so
every peptide contact is present. b2m packs under **alpha3** (183-274), which we also lack, so
including it would leave it nothing to dock against. Folding alpha1/alpha2 + peptide is therefore
both what the data supports and an exact match for the D[9,182] feature spec.

Recorded as a deviation, not passed over in silence.

The decisive measurement
------------------------
Boltz-2 was trained on MHC-peptide complexes. It will return a confident canonical pose for any
9-mer in any groove -- including the 20% of rows with t-half = 0, which do not form stable
complexes at all. So the failure mode is not bad structures; it is structures that barely differ
between peptides with very different half-lives.

`variance` therefore folds several peptides against the *same* allele and reports how much D moves.
If it barely moves, there is no signal for any downstream model to find, and the structural arm is
answered negatively for the price of a few folds.

    python -m modal run modal_boltz.py::probe        # does Boltz-2 run here at all?
    python -m modal run modal_boltz.py::variance     # does D vary across peptides?
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent
REMOTE = "/repo"

# Boltz downloads ~2 GB of weights on first use; a Volume keeps that out of every later run.
weights = modal.Volume.from_name("boltz-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "git")
    .pip_install("boltz[cuda]", "numpy", "pandas", "gemmi")
    .add_local_dir(REPO / "src", f"{REMOTE}/src", ignore=["__pycache__", "*.pyc"])
    .add_local_dir(REPO / "splits", f"{REMOTE}/splits")
    .add_local_file(REPO / "context" / "dataset.csv", f"{REMOTE}/context/dataset.csv")
)

app = modal.App("stability-lens-boltz", image=image)


def _yaml(hla_seq: str, peptide: str) -> str:
    """Boltz input: two protein chains, the groove and the peptide.

    The peptide gets an empty MSA on purpose -- a 9-mer has no meaningful homologs, and asking a
    search server for them wastes a request and can return noise. The groove gets a real MSA.
    """
    return (
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {hla_seq}\n"
        "  - protein:\n"
        "      id: B\n"
        f"      sequence: {peptide}\n"
        "      msa: empty\n"
    )


@app.function(gpu="A100", timeout=3600, volumes={"/root/.boltz": weights})
def probe() -> dict:
    """Can Boltz-2 fold one pair here? Reports the interface it found, not just success."""
    import subprocess
    import sys

    import pandas as pd

    info = {}
    v = subprocess.run([sys.executable, "-m", "pip", "show", "boltz"],
                       capture_output=True, text=True)
    info["pip_show"] = v.stdout[:600]
    h = subprocess.run(["boltz", "predict", "--help"], capture_output=True, text=True)
    info["cli_help"] = (h.stdout or h.stderr)[:2500]
    info["cli_returncode"] = h.returncode

    df = pd.read_csv(f"{REMOTE}/context/dataset.csv")
    row = df[df.allele == "HLA-A*02:01"].iloc[0]
    work = Path("/tmp/probe")
    work.mkdir(exist_ok=True)
    (work / "pair.yaml").write_text(_yaml(row.hla_seq, row.peptide), encoding="utf-8")
    info["peptide"] = row.peptide
    info["allele"] = row.allele
    info["hla_len"] = len(row.hla_seq)

    t0 = time.time()
    r = subprocess.run(
        ["boltz", "predict", str(work / "pair.yaml"),
         "--out_dir", str(work / "out"), "--use_msa_server",
         "--output_format", "mmcif", "--diffusion_samples", "1"],
        capture_output=True, text=True, cwd=str(work),
    )
    info["fold_seconds"] = round(time.time() - t0, 1)
    info["fold_returncode"] = r.returncode
    info["stdout_tail"] = r.stdout[-3000:]
    info["stderr_tail"] = r.stderr[-3000:]
    produced = sorted(str(p.relative_to(work)) for p in work.rglob("*")
                      if p.is_file() and p.suffix in (".cif", ".pdb", ".json", ".npz"))
    info["files"] = produced[:40]
    weights.commit()
    return info


def distance_matrix(cif_path, n_pep=9):
    """D[9, 182]: minimum heavy-atom distance between each peptide residue and each HLA residue.

    Heavy atoms only -- hydrogens are not resolved in the output and including them would make
    the matrix depend on a protonation guess rather than on the fold.
    """
    import gemmi
    import numpy as np

    st = gemmi.read_structure(str(cif_path))
    st.remove_hydrogens()
    model = st[0]
    chains = {ch.name: ch for ch in model}
    # the groove is whichever chain is long; the peptide is the 9-mer
    pep_name = min(chains, key=lambda c: len(chains[c]))
    hla_name = max(chains, key=lambda c: len(chains[c]))
    pep, hla = chains[pep_name], chains[hla_name]
    assert len(pep) == n_pep, f"peptide chain has {len(pep)} residues, expected {n_pep}"

    pep_xyz = [np.array([[a.pos.x, a.pos.y, a.pos.z] for a in r]) for r in pep]
    hla_xyz = [np.array([[a.pos.x, a.pos.y, a.pos.z] for a in r]) for r in hla]
    D = np.zeros((len(pep), len(hla)), dtype=np.float32)
    for i, pi in enumerate(pep_xyz):
        for j, hj in enumerate(hla_xyz):
            d = np.sqrt(((pi[:, None, :] - hj[None, :, :]) ** 2).sum(-1))
            D[i, j] = d.min()
    return D


@app.function(gpu="A100", timeout=7200, volumes={"/root/.boltz": weights})
def variance(allele: str = "HLA-A*02:01", n_peptides: int = 8,
             samples: int = 2, return_matrices: bool = False) -> dict:
    """Does D vary across peptides more than it varies across diffusion seeds?

    Two diffusion samples per peptide give the model's own noise floor for free. If different
    peptides separate no further than one peptide separates from itself, geometry carries no
    peptide-specific information and nothing downstream can recover any.
    """
    import subprocess
    import sys

    import numpy as np
    import pandas as pd

    df = pd.read_csv(f"{REMOTE}/context/dataset.csv")
    sub = df[df.allele == allele].drop_duplicates("peptide").sort_values("thalf_hours")
    # span the range deliberately: the zero class, the middle, and the most stable
    zeros = sub[sub.thalf_hours == 0].head(2)
    rest = sub[sub.thalf_hours > 0]
    # spread the sample evenly over the half-life range rather than clumping at the extremes:
    # a correlation test needs spread in the predictor, and the extremes alone would inflate it
    q = max(1, n_peptides // 4)
    idx = np.linspace(0, len(rest) - 1, n_peptides - min(q, len(zeros))).astype(int)
    picks = pd.concat([zeros.head(q), rest.iloc[np.unique(idx)]])         .drop_duplicates("peptide").head(n_peptides)
    hla_seq = sub.hla_seq.iloc[0]
    print(f"{allele}: {len(sub)} distinct peptides available, folding {len(picks)}", flush=True)
    for _, r in picks.iterrows():
        print(f"  {r.peptide}  t_half {r.thalf_hours}", flush=True)

    work = Path("/tmp/var")
    # the input directory must contain ONLY yaml files -- boltz rejects a nested out_dir
    inputs, outdir = work / "inputs", work / "out"
    inputs.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    for _, r in picks.iterrows():
        (inputs / f"{r.peptide}.yaml").write_text(_yaml(hla_seq, r.peptide), encoding="utf-8")

    t0 = time.time()
    run = subprocess.run(
        ["boltz", "predict", str(inputs), "--out_dir", str(outdir),
         "--use_msa_server", "--output_format", "mmcif", "--diffusion_samples", str(samples)],
        capture_output=True, text=True, cwd=str(work),
    )
    elapsed = time.time() - t0
    print(f"folded in {elapsed:.0f}s (rc {run.returncode})", flush=True)
    if run.returncode != 0:
        return {"ok": False, "returncode": run.returncode,
                "stdout": run.stdout[-4000:], "stderr": run.stderr[-4000:]}

    mats = {}
    for _, r in picks.iterrows():
        found = sorted(outdir.rglob(f"*{r.peptide}*model_*.cif"))
        if len(found) < samples:
            print(f"  WARNING {r.peptide}: {len(found)} models, expected {samples}", flush=True)
        mats[r.peptide] = [distance_matrix(f) for f in found[:samples]]

    peps = [p for p, v in mats.items() if len(v) == samples]
    # within-peptide: the same sequence, two diffusion seeds -> the model's own noise
    within = [float(np.abs(mats[p][0] - mats[p][1]).mean()) for p in peps]
    # across-peptide: different sequences, first sample each -> the signal, if any
    across = [float(np.abs(mats[a][0] - mats[b][0]).mean())
              for i, a in enumerate(peps) for b in peps[i + 1:]]

    # restrict to contacting cells: far-apart pairs move a lot and mean nothing
    contact = np.stack([mats[p][0] for p in peps]).min(0) < 8.0
    within_c = [float(np.abs(mats[p][0] - mats[p][1])[contact].mean()) for p in peps]
    across_c = [float(np.abs(mats[a][0] - mats[b][0])[contact].mean())
                for i, a in enumerate(peps) for b in peps[i + 1:]]

    out = {
        "ok": True, "allele": allele, "peptides": peps,
        "thalf": {r.peptide: float(r.thalf_hours) for _, r in picks.iterrows()},
        "seconds": round(elapsed, 1), "n_folds": len(peps) * 2,
        "within_peptide_mean_abs_delta_A": within,
        "across_peptide_mean_abs_delta_A": across,
        "within_contact_only": within_c,
        "across_contact_only": across_c,
        "contact_cells": int(contact.sum()), "total_cells": int(contact.size),
        "D_shape": list(mats[peps[0]][0].shape),
    }
    if return_matrices:
        # the full matrices travel back so the correlation test can be run, re-run and
        # permuted locally without paying for another fold
        out["contact_mask"] = contact.tolist()
        out["D"] = {p: [m.tolist() for m in mats[p]] for p in peps}
    weights.commit()
    return out


@app.local_entrypoint()
def variance_main(n_peptides: int = 24, allele: str = "HLA-A*02:01"):
    import numpy as np

    res = variance.remote(allele=allele, n_peptides=n_peptides,
                          samples=2, return_matrices=True)
    if not res.get("ok"):
        print(f"FAILED rc={res['returncode']}")
        print(res["stderr"][-3000:])
        raise SystemExit(1)

    w, a = np.mean(res["within_peptide_mean_abs_delta_A"]), np.mean(res["across_peptide_mean_abs_delta_A"])
    wc, ac = np.mean(res["within_contact_only"]), np.mean(res["across_contact_only"])
    print(f"\n{res['allele']}: {len(res['peptides'])} peptides, {res['n_folds']} folds, {res['seconds']}s")
    print(f"D shape {res['D_shape']}, contacting cells (<8 A) {res['contact_cells']}/{res['total_cells']}")
    print(f"\n                       all cells      contacting cells")
    print(f"  within-peptide (noise)  {w:7.3f} A        {wc:7.3f} A")
    print(f"  across-peptide (signal) {a:7.3f} A        {ac:7.3f} A")
    print(f"  ratio signal/noise      {a / w:7.2f}x        {ac / wc:7.2f}x")
    verdict = ("geometry varies with the peptide -- worth pursuing" if ac / wc > 2
               else "D is near-constant across peptides -- the structural arm has no signal to find")
    print(f"\n  VERDICT: {verdict}")
    (REPO / "results" / "boltz_variance.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("wrote results/boltz_variance.json")


@app.local_entrypoint()
def probe_main():
    res = probe.remote()
    print(f"boltz installed:\n{res['pip_show']}")
    print(f"\nCLI help (rc={res['cli_returncode']}):\n{res['cli_help'][:1200]}")
    print(f"\nfolded {res['allele']} + {res['peptide']} (hla {res['hla_len']} aa)")
    print(f"  returncode {res['fold_returncode']}, {res['fold_seconds']}s")
    print(f"  files produced: {res['files']}")
    if res["fold_returncode"] != 0:
        print(f"\nSTDOUT tail:\n{res['stdout_tail']}")
        print(f"\nSTDERR tail:\n{res['stderr_tail']}")
    (REPO / "results" / "boltz_probe.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\nwrote results/boltz_probe.json")
