"""Metrics, reported the way the plan of record requires.

Two rules enforced here rather than left to the caller:

  1. Pooled and within-allele rank correlation are always reported together. Pooled alone is
     inflated on this dataset by between-allele differences -- unrelated pairs score 0.335 pooled
     against 0.027 within-allele.

  2. An undefined statistic is reported as None, never as 0.0. A sentinel that looks like a
     measurement is how a broken evaluation passes unnoticed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_ROWS_PER_ALLELE = 12  # below this a per-allele rank correlation is noise


def _spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return None  # constant input: correlation is undefined, not zero
    return float(np.corrcoef(ra, rb)[0, 1])


def _pearson(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def evaluate(y_true_log, y_pred_log, alleles) -> dict:
    """y_* are in log1p space. `alleles` aligns row-for-row."""
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    alleles = np.asarray(alleles)

    per_allele, supported = [], 0
    for a in np.unique(alleles):
        mask = alleles == a
        if mask.sum() < MIN_ROWS_PER_ALLELE:
            continue
        r = _spearman(y_true_log[mask], y_pred_log[mask])
        if r is not None:
            per_allele.append(r)
            supported += 1

    hours_true = np.expm1(y_true_log)
    hours_pred = np.expm1(np.clip(y_pred_log, 0, None))  # a negative log1p is not a valid half-life

    return {
        "n": int(len(y_true_log)),
        "spearman_pooled": _spearman(y_true_log, y_pred_log),
        "spearman_within_allele": float(np.mean(per_allele)) if per_allele else None,
        "alleles_supported": supported,
        "pearson_log": _pearson(y_true_log, y_pred_log),
        "rmse_log": float(np.sqrt(np.mean((y_true_log - y_pred_log) ** 2))),
        "mae_log": float(np.mean(np.abs(y_true_log - y_pred_log))),
        "mae_hours": float(np.mean(np.abs(hours_true - hours_pred))),
    }


def fmt(name: str, m: dict) -> str:
    def s(v, nd=3):
        return "   n/a" if v is None else f"{v:>6.{nd}f}"

    return (
        f"  {name:<28} n={m['n']:>6,}  "
        f"rho_pooled {s(m['spearman_pooled'])}  "
        f"rho_within {s(m['spearman_within_allele'])}  "
        f"r_log {s(m['pearson_log'])}  "
        f"rmse_log {s(m['rmse_log'])}"
    )
