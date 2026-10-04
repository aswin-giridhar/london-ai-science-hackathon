"""Cross-check every headline number in results/README.md against the file that produced it.

Why this exists
---------------
results/README.md is written by hand from run output. Hand-transcribed numbers drift: a digit
changes, a run is re-run and only some of the places it is quoted get updated, a figure from one
environment gets compared with a figure from another. None of that raises an error anywhere, and
all of it is the kind of thing a reviewer finds.

So the claims are listed here as literals, each paired with the JSON path that should produce it,
and compared. A mismatch is printed loudly with both values.

This file is itself a claim about where the numbers live; if a lookup path is wrong the entry is
reported as MISSING rather than silently skipped.

    python scripts/audit_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
TOL = 0.0006          # the README quotes 3 decimals


def load(name):
    p = R / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def dig(obj, path):
    """Walk a dotted path; return a sentinel rather than raising, so a bad path is visible."""
    cur = obj
    for part in path.split("."):
        if cur is None:
            return "<missing>"
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
                continue
            except (ValueError, IndexError):
                return "<missing>"
        if part not in cur:
            return "<missing>"
        cur = cur[part]
    return cur


# (label, file, dotted path, value claimed in results/README.md)
CLAIMS = [
    # --- Run 2 baselines
    ("B1 pooled", "baselines_val_metrics.json", "B1_peptide_pseudoseq_allele.spearman_pooled", 0.780),
    ("B1 within", "baselines_val_metrics.json", "B1_peptide_pseudoseq_allele.spearman_within_allele", 0.633),
    ("B1' pooled", "baselines_val_metrics.json", "B1prime_peptide_allele_only.spearman_pooled", 0.748),
    ("B1' within", "baselines_val_metrics.json", "B1prime_peptide_allele_only.spearman_within_allele", 0.557),
    ("B0b pooled", "baselines_val_metrics.json", "B0b_per_allele_median.spearman_pooled", 0.563),
    ("B0a rmse_log", "baselines_val_metrics.json", "B0a_global_median.rmse_log", 1.143),

    # --- Run 4 heads
    ("F150 pooled", "esm_heads_val_metrics.json", "F150_pooled_head.spearman_pooled", 0.593),
    ("F150 within", "esm_heads_val_metrics.json", "F150_pooled_head.spearman_within_allele", 0.278),
    ("X150 pooled", "esm_heads_val_metrics.json", "X150_interaction_head.spearman_pooled", 0.754),
    ("X150 within", "esm_heads_val_metrics.json", "X150_interaction_head.spearman_within_allele", 0.558),

    # --- Run 8 L150
    ("L150 pooled", "l150_val_metrics.json", "L150_lora_kv.spearman_pooled", 0.757),
    ("L150 within", "l150_val_metrics.json", "L150_lora_kv.spearman_within_allele", 0.572),
    ("L150 trainable params", "l150_val_metrics.json", "L150_lora_kv.trainable_parameters", 614400),

    # --- Run 7 intervals
    ("B1 CI lo", "bootstrap_ci.json", "per_rung.B1_peptide_pseudoseq_allele.pooled.1", 0.756),
    ("B1 CI hi", "bootstrap_ci.json", "per_rung.B1_peptide_pseudoseq_allele.pooled.2", 0.802),
    ("B1'-X150 pooled mean", "bootstrap_ci.json",
     "comparisons.B1prime_peptide_allele_only_minus_X150_interaction_head.pooled.mean", -0.005),
    ("X150-F150 within mean", "bootstrap_ci.json",
     "comparisons.X150_interaction_head_minus_F150_pooled_head.within_allele.mean", 0.276),
    ("B1-L150 pooled mean", "bootstrap_l150_vs_b1.json", "pooled.mean", 0.023),
    ("B1-L150 within mean", "bootstrap_l150_vs_b1.json", "within_allele.mean", 0.060),

    # --- Run 9 structures
    ("Mantel r (all positions)", "mantel.json", "spearman.r", -0.111),
    ("Mantel p (all positions)", "mantel.json", "spearman.p", 0.1158),
    ("signal/noise, all cells", "mantel.json", "signal_to_noise", 4.13),
    ("P2 Mantel r", "mantel_by_position.json", "P2_only.mantel_r", 0.002),
    ("P2 signal/noise", "mantel_by_position.json", "P2_only.signal_to_noise", 8.22),
    ("anchors Mantel r", "mantel_by_position.json", "anchors_P2_P9.mantel_r", -0.071),

    # --- Run 10 shuffle
    ("shuffle global pooled", "shuffle_control.json", "shuffled_global.spearman_pooled", 0.1036),
    ("shuffle within-allele, within", "shuffle_control.json",
     "shuffled_within_allele.spearman_within_allele", 0.0464),
    ("shuffle intact pooled", "shuffle_control.json", "intact.spearman_pooled", 0.7802),

    # --- Run 11 layers
    ("layer 6 LN pooled", "layer_sweep.json", "results.layer_6_ln.spearman_pooled", 0.726),
    ("layer 30 LN pooled", "layer_sweep.json", "results.layer_30_ln.spearman_pooled", 0.734),
    ("layer 30 raw pooled", "layer_sweep.json", "results.layer_30.spearman_pooled", 0.757),
    ("layer 6 raw pooled", "layer_sweep.json", "results.layer_6.spearman_pooled", 0.570),

    # --- Run 12 scale
    ("ESM-2 8M pooled", "scale_sweep.json", "results.ESM-2 8M.spearman_pooled", 0.764),
    ("ESM-2 650M pooled", "scale_sweep.json", "results.ESM-2 650M.spearman_pooled", 0.764),
    ("ESM-2 150M pooled", "scale_sweep.json", "results.ESM-2 150M.spearman_pooled", 0.771),

    # --- Run 13 complementarity
    ("B1/X150 residual corr", "complementarity.json", "residual_correlation_B1_X150", 0.751),
    ("ensemble-B1 pooled mean", "complementarity.json", "ensemble_minus_B1_pooled.mean", 0.009),
    ("ensemble-B1 pooled lo", "complementarity.json", "ensemble_minus_B1_pooled.lo", 0.001),

    # --- Run 15 allele split
    ("allele split B1 pooled", "allele_split_metrics.json",
     "B1_peptide_pseudoseq_allele.spearman_pooled", 0.642),
    ("allele split X150 pooled", "allele_split_metrics.json",
     "X150_interaction_head.spearman_pooled", 0.641),
    ("allele split B1' pooled", "allele_split_metrics.json",
     "B1prime_peptide_allele_only.spearman_pooled", 0.475),
    ("allele split B1 within", "allele_split_metrics.json",
     "B1_peptide_pseudoseq_allele.spearman_within_allele", 0.539),
]


# Numbers quoted in the experiment write-ups and the repo front door. These restate results/
# figures in prose, which is exactly where a stale number survives longest -- a run gets repeated,
# results/README.md is updated, and the summary three files away keeps the old value.
# (file, substring that must appear verbatim)
PROSE = [
    ("experiments/000-supervised-baseline.md", "| B0b per-allele median | **0.563** |"),
    ("experiments/000-supervised-baseline.md", "| B1 one-hot + pseudoseq + allele | **0.780** | **0.633** |"),
    ("experiments/001-esm2-embeddings-head.md", "| **X150** same embeddings, **residue cross-attention** | 0.754 | 0.558 |"),
    ("experiments/001-esm2-embeddings-head.md", "**+0.161 pooled and +0.280 within-allele**"),
    ("experiments/002-likelihood-scoring.md", "**+0.002 pooled, -0.006 within-allele**"),
    ("experiments/002-likelihood-scoring.md", "rank-correlate at **-0.03**"),
    ("experiments/003-hurdle-model.md", "| all 2,817 rows, ρ pooled | **0.780** | 0.779 | −0.001 |"),
    ("experiments/003-hurdle-model.md", "**AUC 0.886**"),
    ("README.md", "| **B1 one-hot peptide + HLA pseudosequence + allele** | **0.780 [0.756, 0.802]** | **0.633 [0.588, 0.654]** |"),
    ("README.md", "| L150 **LoRA-adapted** ESM-2 150M, same head | 0.757 [0.732, 0.780] | 0.572 [0.524, 0.599] |"),
    ("ARCHITECTURE.md", "| B1 one-hot + pseudoseq + allele | **0.780 [0.756, 0.802]** | **0.633 [0.588, 0.654]** |"),
]


def check_prose():
    """Confirm the figures restated in prose still match the ones in results/."""
    print("\nChecking figures restated in the write-ups and the front door.")
    bad = []
    for rel, needle in PROSE:
        p = ROOT / rel
        if not p.exists():
            bad.append((rel, "FILE MISSING", needle))
            continue
        if needle not in p.read_text(encoding="utf-8"):
            bad.append((rel, "not found", needle))
    if bad:
        print("  STALE OR CHANGED -- these quoted figures are no longer present:")
        for rel, why, needle in bad:
            print(f"    {rel}  [{why}]")
            print(f"      expected: {needle[:88]}")
    else:
        print(f"  all {len(PROSE)} restated figures still present and matching")
    return bad


def main():
    cache, ok, bad, missing = {}, 0, [], []
    print(f"Auditing {len(CLAIMS)} numbers quoted in results/README.md "
          f"against the files that produced them.\n")
    for label, fname, path, claimed in CLAIMS:
        if fname not in cache:
            cache[fname] = load(fname)
        if cache[fname] is None:
            missing.append((label, f"{fname} not found"))
            continue
        actual = dig(cache[fname], path)
        if actual == "<missing>":
            missing.append((label, f"{fname}:{path}"))
            continue
        if isinstance(actual, (int, float)) and abs(float(actual) - claimed) <= max(
                TOL, abs(claimed) * 0.004):
            ok += 1
        else:
            bad.append((label, claimed, actual, f"{fname}:{path}"))

    if bad:
        print("MISMATCHES -- the README says one thing and the data says another:")
        for label, claimed, actual, where in bad:
            print(f"  {label:<32} README {claimed}   file {actual}   ({where})")
        print()
    if missing:
        print("COULD NOT CHECK -- lookup path wrong or file absent:")
        for label, where in missing:
            print(f"  {label:<32} {where}")
        print()
    print(f"verified {ok} / {len(CLAIMS)}   mismatched {len(bad)}   uncheckable {len(missing)}")
    stale = check_prose()
    print()
    if not bad and not missing and not stale:
        print("CLEAN: every audited number matches its source, and every figure restated in the")
        print("write-ups is still present. Nothing has drifted.")
    else:
        print("ATTENTION: see above. A number somewhere says something the data does not.")


if __name__ == "__main__":
    main()
