"""Feature construction for the supervised baselines.

Everything here is deterministic and fitted on nothing -- these are fixed encodings, so there is no
train/test leakage surface. Any transform that *is* fitted (scalers, imputers) must be fitted on
training rows only; none are needed at this stage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "context" / "dataset.csv"
SPLIT = ROOT / "splits" / "peptide_split.csv"

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AMINO_ACIDS)}

PEPTIDE_LEN = 9
PSEUDOSEQ_LEN = 34


def load() -> pd.DataFrame:
    """Dataset joined to the frozen split. Fails loudly if the contract is violated."""
    df = pd.read_csv(DATA)
    sp = pd.read_csv(SPLIT)
    m = df.merge(sp, on="peptide", validate="many_to_one")

    assert len(m) == len(df), "join changed the row count"
    assert m.peptide.str.len().eq(PEPTIDE_LEN).all(), f"expected all peptides to be {PEPTIDE_LEN}-mers"
    assert m.hla_pseudoseq.str.len().eq(PSEUDOSEQ_LEN).all(), f"expected {PSEUDOSEQ_LEN}-residue pseudosequences"
    assert set(m.split) == {"train", "val", "test"}, f"unexpected split labels: {set(m.split)}"
    assert not m.thalf_hours.isna().any(), "null target"
    assert (m.thalf_hours >= 0).all(), "negative half-life"
    return m


def onehot_sequences(seqs: pd.Series, length: int) -> np.ndarray:
    """Positional one-hot: (n, length * 20). Unknown residues encode as all-zero at that position."""
    n = len(seqs)
    out = np.zeros((n, length * 20), dtype=np.float32)
    for row, s in enumerate(seqs.to_numpy()):
        for pos, aa in enumerate(s):
            idx = AA_INDEX.get(aa)
            if idx is not None:
                out[row, pos * 20 + idx] = 1.0
    return out


def onehot_alleles(alleles: pd.Series, vocabulary: list[str]) -> np.ndarray:
    """One-hot over a FIXED allele vocabulary derived from training rows.

    An allele unseen in training encodes as all-zero, which is the honest representation: a lookup
    table has nothing to say about it. This is the whole point of the B1' control.
    """
    index = {a: i for i, a in enumerate(vocabulary)}
    out = np.zeros((len(alleles), len(vocabulary)), dtype=np.float32)
    for row, a in enumerate(alleles.to_numpy()):
        i = index.get(a)
        if i is not None:
            out[row, i] = 1.0
    return out


def build(m: pd.DataFrame, blocks: tuple[str, ...], allele_vocab: list[str]) -> np.ndarray:
    """Concatenate the named feature blocks.

    blocks may contain: 'peptide' (180), 'pseudoseq' (680), 'allele' (len(vocab)).
    """
    parts = []
    for b in blocks:
        if b == "peptide":
            parts.append(onehot_sequences(m.peptide, PEPTIDE_LEN))
        elif b == "pseudoseq":
            parts.append(onehot_sequences(m.hla_pseudoseq, PSEUDOSEQ_LEN))
        elif b == "allele":
            parts.append(onehot_alleles(m.allele, allele_vocab))
        else:
            raise ValueError(f"unknown feature block: {b}")
    return np.hstack(parts)


def target(m: pd.DataFrame) -> np.ndarray:
    """log1p(hours). Inverted with expm1 when reporting in hours."""
    return np.log1p(m.thalf_hours.to_numpy(dtype=np.float64))
