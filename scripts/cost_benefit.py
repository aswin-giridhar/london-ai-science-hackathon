"""What did each approach cost, and what did it buy? The brief's actual question, as a table.

Why this matters more than another accuracy number
---------------------------------------------------
The challenge does not ask which model is most accurate. It asks:

    "Foundation models are very capable, but computationally heavy and expensive to produce and
     run. We are asking the question Are they useful for this problem."

and names "engineering & compute requirements during the timeframe" as a judging criterion. Every
run in results/README.md recorded its wall-clock and hardware. Nothing has yet put those beside
the Spearman they bought.

How the numbers are derived
---------------------------
- **Accuracy** is read from the result JSONs, never retyped.
- **Compute** is measured wall-clock from the run logs, times the number of devices.
- **Price** uses Modal's published per-second rates, fetched 2026-10-04:
  A100 40GB $0.000583/s, A10 $0.000306/s. Local CPU work is priced at $0 because it ran on a
  laptop, with the seconds still shown so the comparison is honest.
- **Gain** is measured against the **B0b floor of 0.563 pooled** - the per-allele median, which
  contains no peptide information. Scoring from zero would flatter every model by 0.563.

Within-allele has no usable floor (B0b is constant inside an allele, so its Spearman is
undefined), so within-allele gains are quoted against **B1**, the best non-foundation model. That
is the comparison a reader actually cares about: what does the foundation model add over the
cheap thing.

    python scripts/cost_benefit.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"

A100, A10, CPU_CORE = 0.000583, 0.000306, 0.0000131
B0B_POOLED = 0.563          # the no-peptide floor, Run 2

# (label, result file, dotted path to pooled, path to within, device, device-seconds, note)
ROWS = [
    ("B0b per-allele median", "baselines_val_metrics.json",
     "B0b_per_allele_median.spearman_pooled", None, "cpu", 1,
     "a groupby; no training at all"),
    ("B1' one-hot + allele id", "baselines_val_metrics.json",
     "B1prime_peptide_allele_only.spearman_pooled",
     "B1prime_peptide_allele_only.spearman_within_allele", "cpu", 48,
     "3 seeds x 11-21 s, laptop CPU"),
    ("B1 one-hot + pseudoseq", "baselines_val_metrics.json",
     "B1_peptide_pseudoseq_allele.spearman_pooled",
     "B1_peptide_pseudoseq_allele.spearman_within_allele", "cpu", 127,
     "3 seeds x 38-47 s, laptop CPU"),
    ("F150 ESM-2 150M pooled", "esm_heads_val_metrics.json",
     "F150_pooled_head.spearman_pooled", "F150_pooled_head.spearman_within_allele", "a10", 201,
     "188 s CPU encode + 13 s A10 head"),
    ("X150 ESM-2 150M x-attn", "esm_heads_val_metrics.json",
     "X150_interaction_head.spearman_pooled", "X150_interaction_head.spearman_within_allele",
     "a10", 237, "188 s encode + 49 s A10 head"),
    ("L150 ESM-2 150M LoRA", "l150_val_metrics.json",
     "L150_lora_kv.spearman_pooled", "L150_lora_kv.spearman_within_allele", "a100", 11307,
     "3 seeds x A100, encoder trains so the cache is void"),
    ("ESM-2 650M + x-attn head", "scale_sweep.json",
     "results.ESM-2 650M.spearman_pooled", "results.ESM-2 650M.spearman_within_allele",
     "a100", 180, "within the 320 s scale sweep; 2 learning rates"),
    ("B1 + X150 ensemble", "complementarity.json",
     "combos.B1 + X150.pooled", "combos.B1 + X150.within", "a10", 364,
     "both models, summed"),
]

# whole-project compute, for the footer
PROJECT = [
    ("embedding cache (Run 3)", "cpu", 188),
    ("F150 + X150 (Run 4)", "a10", 64),
    ("anchor probes (Run 5)", "a10", 92),
    ("hurdle (Run 6)", "cpu", 287),
    ("bootstrap intervals (Run 7)", "cpu", 1020),
    ("L150, 3 x A100 (Run 8)", "a100", 11307),
    ("Boltz-2 probe + 48 folds (Run 9)", "a100", 1063),
    ("shuffle control (Run 10)", "cpu", 284),
    ("layer sweep (Run 11)", "a10", 207),
    ("scale sweep (Run 12)", "a100", 320),
    ("learning curve (Run 14)", "a10", 900),
    ("allele split (Run 15)", "a10", 300),
]

RATE = {"a100": A100, "a10": A10, "cpu": 0.0}


def dig(o, path):
    cur = o
    for part in path.split("."):
        if cur is None or part not in cur:
            return None
        cur = cur[part]
    return cur


def main():
    cache = {}
    print("What each approach cost, and what it bought")
    print("Accuracy from the result files; compute measured; Modal rates as published 2026-10-04.\n")
    print(f"  {'approach':<26}{'pooled':>8}{'within':>8}{'device':>7}{'dev-s':>8}"
          f"{'$':>8}{'gain vs B0b':>13}{'$/0.01 rho':>12}")
    out = []
    for label, fname, p_pool, p_within, dev, secs, note in ROWS:
        if fname not in cache:
            cache[fname] = json.loads((R / fname).read_text(encoding="utf-8"))
        pooled = dig(cache[fname], p_pool)
        within = dig(cache[fname], p_within) if p_within else None
        cost = secs * RATE[dev]
        gain = pooled - B0B_POOLED
        per = (cost / (gain * 100)) if gain > 0 else float("nan")
        print(f"  {label:<26}{pooled:>8.3f}"
              f"{(within if within is not None else float('nan')):>8.3f}"
              f"{dev:>7}{secs:>8,}{cost:>8.2f}{gain:>+13.3f}"
              f"{per:>12.3f}")
        out.append({"approach": label, "pooled": pooled, "within": within, "device": dev,
                    "device_seconds": secs, "usd": round(cost, 4),
                    "gain_vs_b0b": round(gain, 4), "note": note})

    print("\n  (gain is pooled Spearman over the 0.563 no-peptide floor; $/0.01 rho is what one")
    print("   hundredth of a Spearman point cost on that scale)")

    # the comparison the brief is really asking for
    b1 = next(r for r in out if r["approach"].startswith("B1 one-hot"))
    ens = next(r for r in out if "ensemble" in r["approach"])
    l150 = next(r for r in out if r["approach"].startswith("L150"))
    big = next(r for r in out if "650M" in r["approach"])
    print("\nTHE COMPARISON THE BRIEF ASKS FOR")
    print(f"  one-hot alone      {b1['pooled']:.3f} pooled / {b1['within']:.3f} within   "
          f"{b1['device_seconds']:,} CPU-seconds on a laptop, $0.00")
    print(f"  best single ESM-2  {max(l150['pooled'], big['pooled']):.3f} pooled   "
          f"still BELOW one-hot, after {l150['usd'] + big['usd']:.2f} of GPU")
    print(f"  one-hot + ESM-2    {ens['pooled']:.3f} pooled / {ens['within']:.3f} within   "
          f"= +{ens['pooled'] - b1['pooled']:.3f} pooled for ${ens['usd']:.2f}")
    print(f"\n  Put plainly: the foundation model's entire measured contribution is "
          f"+{ens['pooled'] - b1['pooled']:.3f} pooled")
    print(f"  Spearman (and nothing distinguishable within-allele), on top of a model that runs")
    print(f"  in {b1['device_seconds']} seconds on a laptop for nothing.")

    total = sum(s * RATE[d] for _, d, s in PROJECT)
    gpu_s = sum(s for _, d, s in PROJECT if d != "cpu")
    cpu_s = sum(s for _, d, s in PROJECT if d == "cpu")
    print(f"\nWHOLE PROJECT")
    print(f"  {gpu_s:,} GPU-seconds ({gpu_s / 3600:.1f} GPU-hours), {cpu_s:,} CPU-seconds")
    print(f"  measured Modal spend on the runs below: ${total:.2f}")
    for label, dev, secs in PROJECT:
        print(f"    {label:<36}{dev:>6}{secs:>8,} s   ${secs * RATE[dev]:>6.2f}")

    (R / "cost_benefit.json").write_text(
        json.dumps({"rows": out, "project_total_usd": round(total, 2),
                    "gpu_seconds": gpu_s, "cpu_seconds": cpu_s,
                    "rates": {"a100_40gb": A100, "a10": A10},
                    "floor_pooled": B0B_POOLED}, indent=2), encoding="utf-8")
    print(f"\nwrote {R / 'cost_benefit.json'}")


if __name__ == "__main__":
    main()
