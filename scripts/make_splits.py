"""Build leakage-controlled train/val/test splits for the peptide-HLA stability dataset.

The failure this guards against
-------------------------------
Rows are (peptide, allele) pairs. A peptide appears with a median of 4 alleles and up to 36,
and 94% of rows involve a peptide measured against more than one allele. Splitting *rows* at
random therefore puts the same peptide on both sides of the split, and the model recognises the
fragment instead of predicting it. Published work on this dataset reports that near-duplicate
peptides between train and test inflate scores substantially.

So we split by *peptide*, and further group near-duplicate peptides so they travel together.

Identity threshold
------------------
The published convention is 80% sequence identity (normalised edit distance). Every peptide here
is a 9-mer, and for two equal-length strings a Levenshtein distance of 1 is reachable only by
substitution (insert+delete costs 2). So:

    identity = (9 - d) / 9   ->   d=1 gives 0.889 (>= 0.80), d=2 gives 0.778 (< 0.80)

80% identity on 9-mers is therefore exactly "Hamming distance <= 1". We cluster with single
linkage at that threshold and assign whole clusters. CLUSTER_MAX_DIST is configurable so the
team can test sensitivity; the report prints statistics for both 1 and 2.

Two splits are produced, answering different questions
------------------------------------------------------
  peptide_split  (PRIMARY)  unseen peptides, all alleles seen in training.
                            Generalisation to new fragments. Use this by default.
  allele_split   (SECONDARY) whole alleles held out.
                            Generalisation to an HLA the model never saw. This is where a protein
                            foundation model should beat a one-hot allele encoding, if anywhere.

They are separate, complete assignments of all rows. Never mix numbers from the two.

Usage
-----
    python scripts/make_splits.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "context" / "dataset.csv"
OUT = ROOT / "splits"

SEED = 42
# Hamming distance at which peptides are treated as near-duplicates and kept together.
#
# 1 would match the published 80%-identity convention exactly (see module docstring). We use 2,
# deliberately stricter, because measuring it showed the convention is too loose here: a test
# peptide 2 edits from a train peptide had its within-allele label rank predicted at Spearman 0.700
# by that neighbour, against 0.311 at distance 3 and 0.027 for random same-allele pairs. The break
# between "near-duplicate" and "merely similar" sits between 2 and 3, not between 1 and 2.
# Cost of the stricter choice is negligible: 5,410 clusters instead of 5,494.
# Evidence: scripts/measure_leakage_by_distance.py
CLUSTER_MAX_DIST = 2
TARGETS = {"train": 0.80, "val": 0.10, "test": 0.10}


# --------------------------------------------------------------------------- clustering


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def cluster_peptides(peptides, max_dist):
    """Single-linkage clusters at Hamming distance <= max_dist.

    Two strings within Hamming distance d of each other must agree after masking some set of d
    positions, so bucketing on masked keys finds every such pair without an O(n^2) scan. Exact,
    not approximate.
    """
    from itertools import combinations

    uf = UnionFind(peptides)
    length = len(peptides[0])
    for positions in combinations(range(length), max_dist):
        buckets = defaultdict(list)
        for p in peptides:
            key = "".join("*" if i in positions else c for i, c in enumerate(p))
            buckets[key].append(p)
        for group in buckets.values():
            if len(group) > 1:
                first = group[0]
                for other in group[1:]:
                    uf.union(first, other)

    clusters = defaultdict(list)
    for p in peptides:
        clusters[uf.find(p)].append(p)
    return list(clusters.values())


def hamming_le(a, b, k):
    diff = 0
    for x, y in zip(a, b):
        if x != y:
            diff += 1
            if diff > k:
                return False
    return True


# --------------------------------------------------------------------------- assignment


def greedy_assign_stratified(groups, weights, targets, seed, strata):
    """greedy_assign run independently within each stratum, then merged.

    Used for the allele split. Alleles differ systematically in how stable their complexes are, so
    an unstratified holdout can land six high-stability alleles in test and shift the median t-half
    far from train's -- which makes the metric incomparable even though the split is structurally
    valid. Stratifying on each allele's median half-life keeps the label distributions aligned while
    still holding out whole alleles.
    """
    assignment = {}
    by_stratum = defaultdict(list)
    for g in groups:
        by_stratum[strata[g]].append(g)
    for i, (_, members) in enumerate(sorted(by_stratum.items())):
        assignment.update(greedy_assign(members, weights, targets, seed + i))
    return assignment


def greedy_assign(groups, weights, targets, seed):
    """Assign whole groups to splits, largest first, to the *proportionally* neediest split.

    `groups` is a list of hashable group ids; `weights[g]` is that group's row count.
    Returns {group_id: split_name}.

    Why relative deficit and not absolute: with an 80/10/10 target, train's absolute deficit is
    8x the others', so an absolute-deficit rule hands train every large group before val or test
    receive any. That produced a test set whose peptides were each measured against at most 6
    alleles while train reached 36 -- structurally unrepresentative even though the row counts
    looked correct. Ranking by fraction-of-quota-remaining keeps all three splits filling at the
    same relative rate, so large and small groups are spread proportionally.
    """
    import random

    rng = random.Random(seed)
    total = sum(weights[g] for g in groups)
    quota = {s: frac * total for s, frac in targets.items()}
    filled = {s: 0.0 for s in targets}
    assignment = {}

    ordered = sorted(groups, key=lambda g: (-weights[g], rng.random()))
    for g in ordered:
        split = max(targets, key=lambda s: ((quota[s] - filled[s]) / quota[s], rng.random()))
        assignment[g] = split
        filled[split] += weights[g]
    return assignment


# --------------------------------------------------------------------------- verification


def verify_no_cross_split_neighbours(df, peptide_split, max_dist):
    """Exhaustive check that no eval peptide is within max_dist of any train peptide.

    Independent of how the split was built: re-derives neighbours from the peptide strings.
    """
    train = sorted({p for p, s in peptide_split.items() if s == "train"})
    train_set = set(train)
    length = len(train[0])
    from itertools import combinations

    # index train peptides by every masked key up to max_dist
    index = defaultdict(set)
    for positions in combinations(range(length), max_dist):
        for p in train:
            key = (positions, "".join("*" if i in positions else c for i, c in enumerate(p)))
            index[key].add(p)

    violations = []
    for pep, split in peptide_split.items():
        if split == "train":
            continue
        if pep in train_set:
            violations.append((pep, split, pep, 0))
            continue
        for positions in combinations(range(length), max_dist):
            key = (positions, "".join("*" if i in positions else c for i, c in enumerate(pep)))
            for cand in index.get(key, ()):
                if hamming_le(pep, cand, max_dist):
                    violations.append((pep, split, cand, sum(a != b for a, b in zip(pep, cand))))
                    break
            if violations and violations[-1][0] == pep:
                break
    return violations


def validate_detector(df, peptide_split, max_dist):
    """A check that cannot fail is not a check. Confirm the neighbour detector actually fires.

    Plant a known positive: take a training peptide, mutate one residue, and confirm it is
    reported as a violation when pretended to be a test peptide.
    """
    train = sorted({p for p, s in peptide_split.items() if s == "train"})
    victim = train[0]
    mutated = ("A" if victim[0] != "A" else "C") + victim[1:]
    planted = dict(peptide_split)
    planted[mutated] = "test"
    found = verify_no_cross_split_neighbours(df, planted, max_dist)
    fired = any(v[0] == mutated for v in found)
    return fired, victim, mutated


# --------------------------------------------------------------------------- reporting


def summarise(df, col, assignment, name):
    d = df.copy()
    d["split"] = d[col].map(assignment)
    rows = d.split.value_counts().to_dict()
    total = len(d)
    out = {"name": name, "keyed_on": col, "rows": {}, "label_stats": {}, "alleles": {}, "peptides": {}}
    for s in ["train", "val", "test"]:
        sub = d[d.split == s]
        out["rows"][s] = {"n": int(len(sub)), "pct": round(100 * len(sub) / total, 2)}
        out["label_stats"][s] = {
            "median_thalf": round(float(sub.thalf_hours.median()), 3),
            "mean_thalf": round(float(sub.thalf_hours.mean()), 3),
            "frac_zero": round(float((sub.thalf_hours == 0).mean()), 4),
        }
        out["alleles"][s] = int(sub.allele.nunique())
        out["peptides"][s] = int(sub.peptide.nunique())
        per_pep = sub.groupby("peptide").allele.nunique()
        out.setdefault("alleles_per_peptide", {})[s] = {
            "mean": round(float(per_pep.mean()), 2),
            "median": float(per_pep.median()),
            "max": int(per_pep.max()),
        }
    return out, d


def main(force=False):
    OUT.mkdir(exist_ok=True)

    # The split is a frozen benchmark that every experiment reports against. Regenerating it
    # silently would make every result produced so far incomparable, with no error anywhere.
    # Refuse by default; an overwrite has to be asked for explicitly.
    existing = [f for f in ("peptide_split.csv", "allele_split.csv") if (OUT / f).exists()]
    if existing and not force:
        print("REFUSING TO OVERWRITE THE FROZEN SPLIT")
        print(f"  {', '.join(existing)} already exist in {OUT}/")
        print()
        print("  Every experiment reports against these files. Regenerating them would make all")
        print("  prior results incomparable, and nothing downstream would raise an error.")
        print()
        print("  To verify the split instead:   python scripts/audit_splits.py")
        print("  To deliberately replace it:    python scripts/make_splits.py --force")
        raise SystemExit(1)

    df = pd.read_csv(DATA)
    peptides = sorted(df.peptide.unique())
    print(f"dataset: {len(df):,} rows | {len(peptides):,} peptides | {df.allele.nunique()} alleles")
    assert df.peptide.str.len().nunique() == 1, "expected uniform peptide length"

    # ---- sensitivity of the clustering threshold -------------------------------------
    sensitivity = {}
    for d in (1, 2):
        cl = cluster_peptides(peptides, d)
        sizes = sorted((len(c) for c in cl), reverse=True)
        sensitivity[d] = {
            "identity": round((9 - d) / 9, 3),
            "n_clusters": len(cl),
            "largest_cluster": sizes[0],
            "largest_cluster_pct_of_peptides": round(100 * sizes[0] / len(peptides), 2),
            "singletons": sum(1 for s in sizes if s == 1),
        }
        print(f"  Hamming<={d} (identity {(9-d)/9:.3f}): {len(cl):,} clusters, "
              f"largest {sizes[0]:,} peptides ({100*sizes[0]/len(peptides):.1f}%)")

    # ---- PRIMARY: peptide split -------------------------------------------------------
    clusters = cluster_peptides(peptides, CLUSTER_MAX_DIST)
    pep_to_cluster = {p: i for i, c in enumerate(clusters) for p in c}
    rows_per_peptide = df.peptide.value_counts().to_dict()
    cluster_weight = defaultdict(int)
    for p, ci in pep_to_cluster.items():
        cluster_weight[ci] += rows_per_peptide[p]

    cluster_split = greedy_assign(list(range(len(clusters))), cluster_weight, TARGETS, SEED)
    peptide_split = {p: cluster_split[ci] for p, ci in pep_to_cluster.items()}

    # ---- SECONDARY: allele split ------------------------------------------------------
    alleles = sorted(df.allele.unique())
    rows_per_allele = df.allele.value_counts().to_dict()
    allele_median = df.groupby("allele").thalf_hours.median()
    tercile = pd.qcut(allele_median, 3, labels=False, duplicates="drop").to_dict()
    allele_split = greedy_assign_stratified(
        alleles, rows_per_allele, TARGETS, SEED, strata=tercile
    )

    # ---- verification -----------------------------------------------------------------
    print("\nverification")
    fired, victim, mutated = validate_detector(df, peptide_split, CLUSTER_MAX_DIST)
    print(f"  detector self-test: planted {mutated} (1 edit from train peptide {victim}) "
          f"-> {'DETECTED' if fired else 'NOT DETECTED'}")
    assert fired, "neighbour detector failed its own known-positive test"

    violations = verify_no_cross_split_neighbours(df, peptide_split, CLUSTER_MAX_DIST)
    print(f"  cross-split neighbours at Hamming<={CLUSTER_MAX_DIST}: {len(violations)}")
    assert not violations, f"LEAKAGE: {violations[:5]}"

    # things that should still DIFFER (a check that only confirms sameness is half a check)
    sets = {s: {p for p, v in peptide_split.items() if v == s} for s in TARGETS}
    assert not (sets["train"] & sets["test"]), "peptide in both train and test"
    assert not (sets["train"] & sets["val"]), "peptide in both train and val"
    assert not (sets["val"] & sets["test"]), "peptide in both val and test"
    assert all(len(v) > 0 for v in sets.values()), "an empty split"
    assert sum(len(v) for v in sets.values()) == len(peptides), "peptides lost or duplicated"
    print(f"  peptide sets disjoint and non-empty: ok "
          f"(train {len(sets['train']):,} / val {len(sets['val']):,} / test {len(sets['test']):,})")

    a_sets = {s: {a for a, v in allele_split.items() if v == s} for s in TARGETS}
    assert all(len(v) > 0 for v in a_sets.values()), "an empty allele split"
    print(f"  allele split disjoint and non-empty: ok "
          f"(train {len(a_sets['train'])} / val {len(a_sets['val'])} / test {len(a_sets['test'])})")

    # ---- summaries --------------------------------------------------------------------
    pep_summary, pep_df = summarise(df, "peptide", peptide_split, "peptide_split")
    all_summary, all_df = summarise(df, "allele", allele_split, "allele_split")

    for s in (pep_summary, all_summary):
        print(f"\n{s['name']} (keyed on {s['keyed_on']})")
        for k in ["train", "val", "test"]:
            r, l = s["rows"][k], s["label_stats"][k]
            print(f"  {k:<5} {r['n']:>6,} rows ({r['pct']:>5.2f}%)  "
                  f"median t½ {l['median_thalf']:>6.2f}  zeros {l['frac_zero']:.3f}  "
                  f"alleles {s['alleles'][k]:>2}  peptides {s['peptides'][k]:>5,}")

    # unseen-allele check for the PRIMARY split: test rows should not need unseen alleles
    train_alleles = set(pep_df[pep_df.split == "train"].allele)
    for s in ("val", "test"):
        unseen = set(pep_df[pep_df.split == s].allele) - train_alleles
        print(f"  peptide_split {s}: alleles unseen in train = {len(unseen)}")

    # Guard the representativeness property. An earlier version ranked splits by ABSOLUTE quota
    # deficit, which sent every large cluster to train: test peptides were each measured against
    # at most 6 alleles while train reached 36, giving a structurally unrepresentative test set
    # despite correct row counts. That version scored ~0.36 on this ratio; a correct split scores
    # ~1.00, so the bound below separates the two cases by a wide margin.
    apr = pep_summary["alleles_per_peptide"]
    base = apr["train"]["mean"]
    for s in ("val", "test"):
        ratio = apr[s]["mean"] / base
        assert 0.85 <= ratio <= 1.15, (
            f"peptide_split {s} is unrepresentative: {apr[s]['mean']} alleles/peptide "
            f"vs train {base} (ratio {ratio:.2f})"
        )
    print(f"  alleles/peptide balanced across splits: "
          f"train {apr['train']['mean']} / val {apr['val']['mean']} / test {apr['test']['mean']} ok")

    # ---- write ------------------------------------------------------------------------
    pd.DataFrame(
        {"peptide": p, "cluster_id": pep_to_cluster[p], "split": peptide_split[p]}
        for p in peptides
    ).sort_values(["split", "cluster_id", "peptide"]).to_csv(
        OUT / "peptide_split.csv", index=False
    )
    pd.DataFrame(
        {"allele": a, "split": allele_split[a]} for a in alleles
    ).sort_values(["split", "allele"]).to_csv(OUT / "allele_split.csv", index=False)

    report = {
        "generated": "scripts/make_splits.py",
        "seed": SEED,
        "cluster_max_hamming": CLUSTER_MAX_DIST,
        "identity_threshold": round((9 - CLUSTER_MAX_DIST) / 9, 3),
        "targets": TARGETS,
        "dataset": {
            "rows": int(len(df)),
            "peptides": int(len(peptides)),
            "alleles": int(df.allele.nunique()),
        },
        "threshold_sensitivity": sensitivity,
        "peptide_split": pep_summary,
        "allele_split": all_summary,
        "verification": {
            "detector_self_test_passed": bool(fired),
            "cross_split_neighbours": len(violations),
        },
    }
    (OUT / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT/'peptide_split.csv'}, {OUT/'allele_split.csv'}, {OUT/'split_report.json'}")


if __name__ == "__main__":
    import sys

    main(force="--force" in sys.argv)
