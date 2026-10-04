"""Does predicted geometry differ more between peptides whose half-lives differ more?

The question, and why an ordinary correlation will not answer it
----------------------------------------------------------------
Take N peptides folded against the same allele. Build two N x N distance matrices:

    G[i,j] = mean |D_i - D_j| over contacting cells      how different the geometry is
    T[i,j] = |log1p(t_i) - log1p(t_j)|                   how different the stability is

A positive relationship between them is what the structural arm needs: it would say geometry
moves with the endpoint. Correlating the off-diagonal entries directly is tempting and wrong --
the N(N-1)/2 pairs are built from only N independent peptides, so every peptide appears in N-1
pairs and an ordinary p-value is badly anti-conservative.

The Mantel test fixes exactly this: the statistic is the same correlation, but significance comes
from permuting the **peptide labels** (N! arrangements sampled), which preserves the dependency
structure because each permutation rebuilds both matrices from the same N units.

A null result here is informative and cheap. It would say Boltz-2's poses carry peptide-specific
detail -- which the within/across variance ratio already establishes separately -- but no detail
that tracks stability, which is the only kind the downstream model could use.

    python scripts/mantel.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PERMS = 10000
SEED = 2026


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d else np.nan


def mantel(G, T, perms=PERMS, seed=SEED, method="spearman"):
    """Mantel r between two symmetric distance matrices, with a permutation p-value."""
    n = G.shape[0]
    iu = np.triu_indices(n, 1)

    def stat(g, t):
        gv, tv = g[iu], t[iu]
        if method == "spearman":
            gv, tv = rankdata(gv), rankdata(tv)
        return pearson(gv, tv)

    observed = stat(G, T)
    rng = np.random.default_rng(seed)
    null = np.empty(perms)
    for k in range(perms):
        p = rng.permutation(n)
        null[k] = stat(G[np.ix_(p, p)], T)
    # two-sided, with the observed value included in the null as is conventional
    pval = (np.sum(np.abs(null) >= abs(observed)) + 1) / (perms + 1)
    return observed, float(pval), null


def main():
    src = RESULTS / "boltz_variance.json"
    if not src.exists():
        raise SystemExit(f"missing {src} -- run `modal run modal_boltz.py::variance_main` first")
    d = json.loads(src.read_text(encoding="utf-8"))
    if "D" not in d:
        raise SystemExit("this variance file has no D matrices; rerun with return_matrices=True")

    peps = d["peptides"]
    thalf = d["thalf"]
    mask = np.array(d["contact_mask"], dtype=bool)
    # sample 0 of each peptide is the geometry; sample 1 is only used for the noise floor
    D = {p: np.array(d["D"][p][0], dtype=np.float32) for p in peps}
    n = len(peps)
    print(f"{n} peptides on {d['allele']}, {mask.sum()} contacting cells of {mask.size}")
    print(f"half-life range {min(thalf[p] for p in peps)} to {max(thalf[p] for p in peps)} h")

    G = np.zeros((n, n))
    T = np.zeros((n, n))
    for i, a in enumerate(peps):
        for j, b in enumerate(peps):
            if i < j:
                G[i, j] = G[j, i] = np.abs(D[a] - D[b])[mask].mean()
                T[i, j] = T[j, i] = abs(np.log1p(thalf[a]) - np.log1p(thalf[b]))

    # the noise floor, for scale: how far two folds of the SAME peptide sit apart
    noise = float(np.mean(d["within_contact_only"]))
    iu = np.triu_indices(n, 1)
    print(f"\ngeometry distance between peptides: mean {G[iu].mean():.3f} A, "
          f"range {G[iu].min():.3f}-{G[iu].max():.3f}")
    print(f"same peptide, different diffusion seed: {noise:.3f} A  "
          f"-> signal/noise {G[iu].mean() / noise:.2f}x")

    out = {"n_peptides": n, "allele": d["allele"], "perms": PERMS, "seed": SEED,
           "noise_floor_A": noise, "geometry_distance_mean_A": float(G[iu].mean()),
           "signal_to_noise": float(G[iu].mean() / noise)}

    print(f"\nMantel test, {PERMS:,} label permutations")
    for method in ("spearman", "pearson"):
        r, p, null = mantel(G, T, method=method)
        lo, hi = np.percentile(null, [2.5, 97.5])
        verdict = "relationship survives" if p < 0.05 else "NO evidence of a relationship"
        print(f"  {method:<9} r = {r:+.3f}   p = {p:.4f}   "
              f"(null 95% spans [{lo:+.3f}, {hi:+.3f}])   {verdict}")
        out[method] = {"r": r, "p": p, "null_lo": float(lo), "null_hi": float(hi)}

    # What size of effect could this many peptides actually have detected? Report it, so a null
    # is not mistaken for a demonstration of absence.
    rng = np.random.default_rng(SEED)
    detectable = []
    for target in (0.2, 0.3, 0.4, 0.5, 0.6):
        hits = 0
        for _ in range(200):
            # synthesise a T that genuinely correlates with G at roughly `target`
            gv = rankdata(G[iu])
            tv = target * (gv - gv.mean()) / gv.std() + rng.normal(size=len(gv))
            Tsim = np.zeros((n, n))
            Tsim[iu] = tv - tv.min()
            Tsim = Tsim + Tsim.T
            _, psim, _ = mantel(G, Tsim, perms=200, seed=int(rng.integers(1e6)))
            hits += psim < 0.05
        detectable.append((target, hits / 200))
    print(f"\n  power at n={n}: probability of detecting a true Mantel r of")
    for target, pw in detectable:
        print(f"    r = {target:.1f}  ->  {pw:.0%}")
    out["power"] = {str(t): pw for t, pw in detectable}

    dest = RESULTS / "mantel.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
