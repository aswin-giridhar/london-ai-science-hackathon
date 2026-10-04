"""Structure as a feature-selection prior: can Boltz-2 rediscover NetMHCpan's pseudosequence?

The idea
--------
Run 9 showed that Boltz-2 *distances* do not predict stability (Mantel r -0.111, 92% power at
r=0.2). But the contact map still says **which groove positions touch the peptide at all**, and
that is a different kind of information: not a measurement, a selection.

It turns out our winning baseline already depends on exactly that kind of prior. B1's HLA input is
the **34-residue pseudosequence** -- and those 34 positions were chosen by NetMHCpan from
*crystallographic* contacts. B1 is therefore not a naive baseline at all; it carries decades of
structural biology, compressed into a feature-selection decision.

So the sharp question is not "can structure help?" but:

    **Can a 2026 folding model rediscover, from scratch, the positions that crystallography gave
    NetMHCpan in the 2000s - and are they as good as features?**

Three things get measured
-------------------------
1. **Overlap.** How many of the 34 pseudosequence positions does the Boltz contact set recover,
   and is that more than chance?
2. **As features.** One-hot of the Boltz-selected groove positions, versus the pseudosequence,
   versus the full 182-aa domain, with the peptide block and allele id identical throughout.
3. **A control that matters.** The same *number* of groove positions chosen at random. Without it,
   "Boltz positions work" could just mean "more groove residues work".

Two caveats, stated up front
----------------------------
- We folded **HLA-A*02:01 only**, so the contact map is one allele's. Class I grooves are
  structurally highly conserved and the 182-aa domains are positionally aligned, so using it as a
  universal prior is defensible - but it is an assumption, not a measurement, and a per-allele map
  would be better.
- The mask is derived from structures of *training-set* peptides. It encodes no labels, so it is
  not label leakage, but it is not blind to the data either.

    python src/contact_features.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import baselines
import features
import metrics

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
SEEDS = baselines.SEEDS
CONTACT_A = 8.0        # the threshold already used for the contact mask in Run 9


def boltz_contact_positions():
    """Groove positions (0-based, within the 182-aa domain) that contact any peptide residue."""
    src = ROOT / "results" / "boltz_variance.json"
    if not src.exists():
        raise SystemExit(f"missing {src} -- run modal_boltz.py::variance_main first")
    d = json.loads(src.read_text(encoding="utf-8"))
    # min distance over peptides and samples, so a position counts as contacting if it ever does
    D = np.stack([np.array(m, dtype=np.float32) for p in d["peptides"] for m in d["D"][p]])
    mind = D.min(axis=0)                      # (9, 182)
    pos = np.flatnonzero((mind < CONTACT_A).any(axis=0))
    return pos, mind, d["allele"]


def pseudoseq_positions(m):
    """Recover which of the 182 domain positions the 34-residue pseudosequence corresponds to.

    The pseudosequence is given as a string, not as indices, so the mapping is inferred: for each
    pseudosequence slot, find the domain positions whose residue matches that slot in *every*
    allele. Positions that resolve uniquely are the mapping; ambiguous slots are reported.
    """
    sub = m.drop_duplicates("allele")[["allele", "hla_seq", "hla_pseudoseq"]]
    dom = np.array([list(s) for s in sub.hla_seq])          # (75, 182)
    pse = np.array([list(s) for s in sub.hla_pseudoseq])    # (75, 34)
    mapped, ambiguous = [], 0
    for k in range(pse.shape[1]):
        hits = [j for j in range(dom.shape[1]) if np.array_equal(dom[:, j], pse[:, k])]
        if len(hits) == 1:
            mapped.append(hits[0])
        elif len(hits) > 1:
            mapped.append(hits[0])
            ambiguous += 1
    return sorted(set(mapped)), ambiguous


def onehot_positions(seqs, positions):
    """One-hot of only the chosen domain positions: (n, len(positions) * 20)."""
    sel = np.array(positions)
    out = np.zeros((len(seqs), len(sel) * 20), dtype=np.float32)
    for row, s in enumerate(seqs.to_numpy()):
        for slot, j in enumerate(sel):
            idx = features.AA_INDEX.get(s[j])
            if idx is not None:
                out[row, slot * 20 + idx] = 1.0
    return out


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    m = features.load()
    tr = m[m.split == "train"].reset_index(drop=True)
    va = m[m.split == "val"].reset_index(drop=True)
    vocab = sorted(tr.allele.unique())
    y_tr, y_va = features.target(tr), features.target(va)
    allele_va = va.allele.to_numpy()

    contacts, mind, fold_allele = boltz_contact_positions()
    pseudo, ambiguous = pseudoseq_positions(m)
    print(f"Boltz contact map from {fold_allele}: {len(contacts)} of 182 groove positions "
          f"within {CONTACT_A:.0f} A of some peptide residue")
    print(f"NetMHCpan pseudosequence maps to {len(pseudo)} unique domain positions "
          f"({ambiguous} slots were ambiguous and took the first match)")

    # ------------------------------------------------------------------ 1. overlap
    inter = sorted(set(contacts) & set(pseudo))
    # chance overlap if Boltz had picked len(contacts) positions at random out of 182
    exp = len(contacts) * len(pseudo) / 182
    rng = np.random.default_rng(2026)
    null = [len(set(rng.choice(182, len(contacts), replace=False)) & set(pseudo))
            for _ in range(10000)]
    p_over = (np.sum(np.array(null) >= len(inter)) + 1) / 10001
    print(f"\nOVERLAP: Boltz recovers {len(inter)} of the {len(pseudo)} pseudosequence positions")
    print(f"  expected by chance {exp:.1f}, permutation p = {p_over:.4f}  "
          f"({'significant' if p_over < 0.05 else 'not better than chance'})")
    print(f"  recovered: {inter}")
    missed = sorted(set(pseudo) - set(contacts))
    print(f"  missed:    {missed}")

    # ------------------------------------------------------------------ 2 & 3. as features
    pep_tr = features.onehot_sequences(tr.peptide, 9)
    pep_va = features.onehot_sequences(va.peptide, 9)
    all_tr = features.onehot_alleles(tr.allele, vocab)
    all_va = features.onehot_alleles(va.allele, vocab)

    random_sets = [sorted(rng.choice(182, len(contacts), replace=False)) for _ in range(3)]

    results = {"n_contacts": int(len(contacts)), "n_pseudo": int(len(pseudo)),
               "overlap": int(len(inter)), "overlap_expected": float(exp),
               "overlap_p": float(p_over), "contacts": [int(x) for x in contacts],
               "pseudo_positions": [int(x) for x in pseudo], "fold_allele": fold_allele}
    dest = OUT / "contact_features.json"

    def run_set(label, pos):
        """Fit one feature set and persist immediately.

        Written after every set rather than at the end: the 182-position variant has 3,895
        features and dominates the runtime, and an earlier version of this script spent 9,000
        CPU-seconds without producing a single recoverable number. The decisive comparisons are
        cheap and now land first.
        """
        t = time.time()
        Xt = np.hstack([pep_tr, onehot_positions(tr.hla_seq, pos), all_tr])
        Xv = np.hstack([pep_va, onehot_positions(va.hla_seq, pos), all_va])
        preds = [baselines.fit_mlp(Xt, y_tr, Xv, y_va, seed=s)[0].predict(Xv) for s in SEEDS]
        r = metrics.evaluate(y_va, np.mean(preds, axis=0), allele_va)
        w = r["spearman_within_allele"]
        out = {"n_positions": len(pos), "n_features": int(Xt.shape[1]),
               "pooled": r["spearman_pooled"], "within": w,
               "seconds": round(time.time() - t, 1)}
        results[label] = out
        dest.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
        print(f"{label:<34}{len(pos):>10}{Xt.shape[1]:>9}{r['spearman_pooled']:>9.3f}"
              f"{(w if w is not None else float('nan')):>9.3f}{out['seconds']:>8.0f}s", flush=True)
        return out

    # cheapest and most decisive first; the full-domain variant is a nice-to-have and goes last
    print(f"\n{'feature set':<34}{'positions':>10}{'feats':>9}{'pooled':>9}{'within':>9}{'time':>9}")
    run_set("B1_pseudoseq_34 (reference)", pseudo)
    run_set(f"boltz_contacts_{len(contacts)}", list(contacts))

    rand_scores = []
    for i, pos in enumerate(random_sets):
        o = run_set(f"random_{len(pos)}_positions_{i + 1}", pos)
        rand_scores.append((o["pooled"], o["within"]))
    results["random_controls"] = [{"pooled": a, "within": b} for a, b in rand_scores]
    dest.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")

    # ------------------------------------------------------------------ verdict
    boltz_key = f"boltz_contacts_{len(contacts)}"
    b, ps = results[boltz_key]["pooled"], results["B1_pseudoseq_34 (reference)"]["pooled"]
    rnd = float(np.mean([a for a, _ in rand_scores]))
    print("\nVERDICT")
    print(f"  Boltz-selected {len(contacts)} positions  {b:.3f} pooled")
    print(f"  NetMHCpan pseudosequence 34   {ps:.3f} pooled")
    print(f"  random {len(contacts)} positions (mean of 3)  {rnd:.3f} pooled")
    if b > rnd + 0.01 and p_over < 0.05:
        print("  -> Boltz both recovers the known contact positions above chance AND its")
        print("     selection carries predictive value beyond an equally sized random set.")
    elif p_over < 0.05:
        print("  -> Boltz recovers the known contact positions above chance, but its selection")
        print("     is not worth more than a random set of the same size as features -- so the")
        print("     groove is informative broadly, not only at the contacts.")
    else:
        print("  -> Boltz does not recover the crystallographic contact positions above chance.")

    # last, because 3,895 features dominate the runtime and it answers a secondary question:
    # is the groove informative only at the contacts, or everywhere?
    print("\n(full 182-position domain last -- slowest, and a secondary question)")
    run_set("full_domain_182", list(range(182)))
    fd = results["full_domain_182"]["pooled"]
    print(f"\n  full domain {fd:.3f} vs pseudosequence {ps:.3f}  ->  "
          f"{'selection helps' if ps > fd else 'the extra positions add something'}")

    dest.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {dest}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
