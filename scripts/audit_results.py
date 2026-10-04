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


def tolerance(claimed):
    """Half a unit in the last decimal place the claim states, floored at TOL.

    A README that says 0.78 is asserting two decimals and is satisfied by anything rounding to
    it. Integers get a tolerance of 0.5, which is right for counts quoted exactly.
    """
    txt = repr(float(claimed))
    decimals = len(txt.split(".")[1].rstrip("0")) if "." in txt else 0
    return max(TOL, 0.5 * 10 ** -decimals) if decimals else 0.5


def load(name):
    p = R / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def dig(obj, path):
    """Walk a path; return a sentinel rather than raising, so a bad path is visible.

    A path may be a dotted string for convenience, or an explicit list of keys when a key itself
    contains a dot -- motif_supported.json is keyed by support threshold ("0.01"), which a dotted
    string cannot address. Splitting those silently gave four 'uncheckable' entries, which is the
    right failure for an audit: visible, not skipped.
    """
    cur = obj
    for part in (path if isinstance(path, (list, tuple)) else path.split(".")):
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

    # --- Run 16 noise ceiling
    ("ceiling, pseudoseq dist 1", "noise_ceiling.json", "by_distance.1.median_spearman", 0.823),
    ("ceiling, dist 1 max", "noise_ceiling.json", "by_distance.1.max_spearman", 0.921),
    ("ceiling, dist 2", "noise_ceiling.json", "by_distance.2.median_spearman", 0.659),
    ("ceiling, dist 3", "noise_ceiling.json", "by_distance.3.median_spearman", 0.474),
    ("ceiling, dist 4", "noise_ceiling.json", "by_distance.4.median_spearman", 0.318),
    ("ceiling, dist 0 anomaly", "noise_ceiling.json", "by_distance.0.median_spearman", 0.643),

    # --- Run 17 contact features
    ("Boltz contact positions", "contact_features.json", "n_contacts", 97),
    ("pseudoseq unique positions", "contact_features.json", "n_pseudo", 31),
    ("overlap recovered", "contact_features.json", "overlap", 31),
    ("overlap expected by chance", "contact_features.json", "overlap_expected", 16.5),
    ("overlap permutation p", "contact_features.json", "overlap_p", 0.0001),
    ("pseudoseq as features", "contact_features.json",
     "B1_pseudoseq_34 (reference).pooled", 0.785),
    ("boltz contacts as features", "contact_features.json", "boltz_contacts_97.pooled", 0.782),

    # --- Run 18 cost
    ("B1 device-seconds", "cost_benefit.json", "rows.2.device_seconds", 127),
    ("L150 cost usd", "cost_benefit.json", "rows.5.usd", 6.59),
    ("project total usd", "cost_benefit.json", "project_total_usd", 7.88),
    ("project gpu seconds", "cost_benefit.json", "gpu_seconds", 14253),

    # --- Run 17 contact features, completed
    ("random 97 #2", "contact_features.json", "random_97_positions_2.pooled", 0.773),
    ("full domain 182", "contact_features.json", "full_domain_182.pooled", 0.768),
    ("full domain within", "contact_features.json", "full_domain_182.within", 0.591),

    # --- Run 19 motifs
    ("sequences scored", "motif_recovery.json", "n_sequences_scored", 10320),
    ("novel sequences", "motif_recovery.json", "n_novel", 10258),
    ("B1 anchor rank, raw", "motif_supported.json", ["0.0", "B1"], 3.78),
    ("X150 anchor rank, raw", "motif_supported.json", ["0.0", "X150"], 3.69),
    ("B1 anchor rank, 1% support", "motif_supported.json", ["0.01", "B1"], 1.94),
    ("X150 anchor rank, 1% support", "motif_supported.json", ["0.01", "X150"], 2.11),

    # --- Run 20 operational metrics
    ("B1 MAE hours", "operational_metrics.json", "B1_onehot.mae_hours", 3.45),
    ("L150 median AE hours", "operational_metrics.json", "L150_esm_lora.median_ae_hours", 0.87),
    ("L150 MAE unstable", "operational_metrics.json", "L150_esm_lora.mae_hours_unstable", 0.78),
    ("B1 MAE stable", "operational_metrics.json", "B1_onehot.mae_hours_stable", 24.59),
    ("B1 top-5 precision", "operational_metrics.json", "B1_onehot.top5_precision", 0.500),
    ("X150 top-5 precision", "operational_metrics.json", "X150_esm_frozen.top5_precision", 0.424),
    ("B1 calibration slope", "operational_metrics.json", "B1_onehot.calib_slope", 0.978),
    ("skill vs rows available", "operational_metrics.json", "skill_vs_n", -0.076),
    ("skill vs median half-life", "operational_metrics.json", "skill_vs_median_halflife", 0.475),

    # --- Run 21 cross-chain concatenation
    ("concat pooled", "concat_encoding.json", "spearman_pooled", 0.737),
    ("concat within", "concat_encoding.json", "spearman_within_allele", 0.523),
    ("concat delta pooled", "concat_encoding.json", "delta_vs_separate_pooled", -0.017),
    ("concat delta within", "concat_encoding.json", "delta_vs_separate_within", -0.036),
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
            missing.append((label, f"{fname}:{path if isinstance(path, str) else '.'.join(path)}"))
            continue
        # Tolerance must match the precision the README actually quotes. A claim written as
        # "0.78" asserts two decimals, so 0.7844 agrees with it; a claim written as "0.784"
        # asserts three and would not. A fixed tolerance flagged a correctly-rounded figure as
        # a mismatch, which is a false alarm an audit can least afford -- it trains the reader
        # to ignore it.
        if isinstance(actual, (int, float)) and abs(float(actual) - claimed) <= tolerance(claimed):
            ok += 1
        else:
            bad.append((label, claimed, actual, f"{fname}:{path if isinstance(path, str) else chr(46).join(path)}"))

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
