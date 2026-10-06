"""
code/gan/dataset.py
===================
Dataset interface for the GAN training pipeline.

What it does
------------
Reads ``dataset/sample_claims.csv`` (the only labeled dataset available in
this repository), encodes every categorical claim field as a one-hot vector,
and returns a PyTorch Dataset ready for the DataLoader.

Why one-hot encoding?
---------------------
The GAN operates on **structured/tabular data** — not raw images.  Each
real claim record is a combination of categorical labels (claim_object,
issue_type, claim_status, etc.).  One-hot encoding is the standard way to
feed discrete categories into a dense neural network.

⚠  DEMO DATA NOTE
-----------------
``sample_claims.csv`` contains **21 labeled rows provided by HackerRank**.
These are real example claims used for evaluation, NOT a large production
dataset.  21 rows is far too few to train a GAN that generalises well.
The implementation includes a ``SyntheticClaimDataset`` fallback that
generates DEMO/SYNTHETIC rows to demonstrate the training loop.

Do NOT use DEMO/SYNTHETIC rows as ground-truth safety labels.
"""

import csv
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

# Avoid a circular import — import config values explicitly
from gan.config import (
    CATEGORICAL_FEATURES,
    INPUT_DIM,
    REAL_DATA_CSV,
    SEED,
)


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _build_vocab() -> dict[str, dict[str, int]]:
    """Return {column_name: {value: index_within_column}} from config."""
    return {col: {v: i for i, v in enumerate(vals)}
            for col, vals in CATEGORICAL_FEATURES.items()}


VOCAB: dict[str, dict[str, int]] = _build_vocab()


def encode_row(row: dict[str, str]) -> Optional[torch.Tensor]:
    """
    Encode one CSV row as a concatenated one-hot vector.

    Returns ``None`` if any required field is missing or unmapped (so the
    caller can skip the row gracefully).

    Parameters
    ----------
    row : dict
        One row from the claims CSV as a plain string dict.

    Returns
    -------
    torch.Tensor of shape ``(INPUT_DIM,)`` with dtype ``float32``, or None.
    """
    parts: list[torch.Tensor] = []
    for col, values in CATEGORICAL_FEATURES.items():
        raw = row.get(col, "").strip().lower()
        vocab = VOCAB[col]
        dim   = len(values)
        vec   = torch.zeros(dim)
        if raw in vocab:
            vec[vocab[raw]] = 1.0
        else:
            # Unknown value — leave zero vector (the model sees missing info)
            pass
        parts.append(vec)
    return torch.cat(parts)


def decode_row(tensor: torch.Tensor) -> dict[str, str]:
    """
    Decode a one-hot tensor back into a human-readable dict.

    Picks the argmax within each feature's slice.

    Parameters
    ----------
    tensor : torch.Tensor of shape ``(INPUT_DIM,)``

    Returns
    -------
    dict mapping column name → decoded string value
    """
    result = {}
    offset = 0
    for col, values in CATEGORICAL_FEATURES.items():
        dim   = len(values)
        slice_ = tensor[offset : offset + dim]
        idx    = int(slice_.argmax().item())
        result[col] = values[idx] if 0 <= idx < dim else "unknown"
        offset += dim
    return result


# ---------------------------------------------------------------------------
# Real-data dataset
# ---------------------------------------------------------------------------

class ClaimDataset(Dataset):
    """
    PyTorch Dataset wrapping ``sample_claims.csv``.

    Each ``__getitem__`` call returns a one-hot tensor of shape
    ``(INPUT_DIM,)`` representing one encoded claim record.

    If the CSV file does not exist, raises ``FileNotFoundError`` with a clear
    message explaining what is needed.
    """

    def __init__(self, csv_path: Path = REAL_DATA_CSV) -> None:
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Real dataset not found at {csv_path}.\n"
                "The GAN requires 'dataset/sample_claims.csv' which is part of\n"
                "the HackerRank Orchestrate starter repository.  If you are\n"
                "running this outside that repo, supply the path explicitly."
            )

        self.samples: list[torch.Tensor] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                encoded = encode_row(row)
                if encoded is not None:
                    self.samples.append(encoded)

        if len(self.samples) == 0:
            raise ValueError(
                f"No encodeable rows found in {csv_path}. "
                "Check that CATEGORICAL_FEATURES in config.py matches the CSV columns."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.samples[idx]


# ---------------------------------------------------------------------------
# DEMO / Synthetic dataset — clearly labelled, NOT real safety data
# ---------------------------------------------------------------------------

class SyntheticClaimDataset(Dataset):
    """
    ⚠  DEMO / SYNTHETIC DATA — for testing the GAN training loop only.

    Generates random one-hot vectors that sample uniformly from each
    categorical feature's vocabulary.  These rows do NOT represent real
    damage claims and MUST NOT be used as ground-truth safety labels.

    Purpose: allow the GAN training loop to be smoke-tested without a real
    dataset, or to pad a very small real dataset for demonstration purposes.

    Parameters
    ----------
    n_samples : int
        Number of synthetic rows to generate.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, n_samples: int = 500, seed: int = SEED) -> None:
        rng = random.Random(seed)
        torch.manual_seed(seed)
        self.samples: list[torch.Tensor] = []
        for _ in range(n_samples):
            parts: list[torch.Tensor] = []
            for col, values in CATEGORICAL_FEATURES.items():
                dim = len(values)
                vec = torch.zeros(dim)
                vec[rng.randrange(dim)] = 1.0
                parts.append(vec)
            self.samples.append(torch.cat(parts))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Factory — choose real or demo dataset automatically
# ---------------------------------------------------------------------------

def get_dataset(prefer_real: bool = True, demo_size: int = 500) -> Dataset:
    """
    Return the best available dataset.

    1. If ``prefer_real=True`` and ``sample_claims.csv`` exists → ClaimDataset.
    2. Otherwise → SyntheticClaimDataset (with a printed warning).

    Parameters
    ----------
    prefer_real : bool
        Try the real CSV first.
    demo_size : int
        Number of synthetic rows to generate when falling back.

    Returns
    -------
    torch.utils.data.Dataset
    """
    if prefer_real:
        try:
            ds = ClaimDataset()
            print(f"[dataset] Loaded {len(ds)} real rows from {REAL_DATA_CSV}")
            return ds
        except FileNotFoundError as exc:
            print(f"[dataset] WARNING: {exc}")

    print("[dataset] WARNING: Using DEMO/SYNTHETIC data. "
          "Do NOT treat generated samples as real safety data.")
    return SyntheticClaimDataset(n_samples=demo_size)
