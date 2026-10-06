"""
code/gan/utils.py
=================
Utility functions shared across the GAN training pipeline.

Covers:
- Device selection (CUDA / Apple Silicon MPS / CPU)
- Weight initialisation (GAN convention)
- Checkpoint saving / loading
- Generated sample persistence
- Metric logging
"""

import csv
import json
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from gan.config import (
    CATEGORICAL_FEATURES,
    CHECKPOINT_DIR,
    SAMPLES_DIR,
    SEED,
)
from gan.dataset import decode_row


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """
    Return the best available compute device:
        1. CUDA (NVIDIA GPU)
        2. MPS  (Apple Silicon — macOS 12.3+, PyTorch ≥ 1.12)
        3. CPU  (fallback)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[utils] Using device: {device}")
    return device


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    """Set random seeds for Python, NumPy (if available), and PyTorch."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Weight initialisation (GAN convention — Radford et al. 2015)
# ---------------------------------------------------------------------------

def weights_init(module: nn.Module) -> None:
    """
    Initialise Linear layer weights from N(0, 0.02) and biases to 0.

    Apply via::

        generator.apply(weights_init)
        discriminator.apply(weights_init)

    This is the standard GAN initialisation that helps stabilise early
    training by preventing vanishingly small or exploding gradients.
    """
    classname = type(module).__name__
    if classname == "Linear":
        nn.init.normal_(module.weight.data, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif classname == "BatchNorm1d":
        nn.init.normal_(module.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(module.bias.data, 0.0)


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def save_checkpoint(
    generator:     nn.Module,
    discriminator: nn.Module,
    opt_g:         torch.optim.Optimizer,
    opt_d:         torch.optim.Optimizer,
    epoch:         int,
    metrics:       dict,
    path:          Optional[Path] = None,
) -> Path:
    """
    Save a full training checkpoint to disk.

    Parameters
    ----------
    generator     : Generator nn.Module
    discriminator : Discriminator nn.Module
    opt_g         : Generator's Adam optimiser
    opt_d         : Discriminator's Adam optimiser
    epoch         : current epoch (0-indexed)
    metrics       : dict of scalar training metrics at this epoch
    path          : explicit save path; defaults to CHECKPOINT_DIR/epoch_{epoch:04d}.pt

    Returns
    -------
    Path to the saved checkpoint file.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if path is None:
        path = CHECKPOINT_DIR / f"epoch_{epoch:04d}.pt"

    torch.save(
        {
            "epoch":              epoch,
            "generator_state":    generator.state_dict(),
            "discriminator_state":discriminator.state_dict(),
            "opt_g_state":        opt_g.state_dict(),
            "opt_d_state":        opt_d.state_dict(),
            "metrics":            metrics,
        },
        path,
    )
    return path


def load_checkpoint(
    path:          Path,
    generator:     nn.Module,
    discriminator: nn.Module,
    opt_g:         torch.optim.Optimizer,
    opt_d:         torch.optim.Optimizer,
) -> int:
    """
    Restore a checkpoint.  Returns the saved epoch number.

    Parameters
    ----------
    path          : Path to the ``.pt`` checkpoint file.
    generator     : Generator to restore weights into.
    discriminator : Discriminator to restore weights into.
    opt_g, opt_d  : Optimisers to restore states into.

    Returns
    -------
    int — epoch number at which the checkpoint was saved.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location="cpu")
    generator.load_state_dict(ckpt["generator_state"])
    discriminator.load_state_dict(ckpt["discriminator_state"])
    opt_g.load_state_dict(ckpt["opt_g_state"])
    opt_d.load_state_dict(ckpt["opt_d_state"])
    print(f"[utils] Loaded checkpoint from epoch {ckpt['epoch']}: {path}")
    return ckpt["epoch"]


# ---------------------------------------------------------------------------
# Generated sample saving
# ---------------------------------------------------------------------------

def save_generated_samples(
    tensors: torch.Tensor,
    epoch:   int,
    tag:     str = "synthetic",
) -> Path:
    """
    Decode generated tensors and save them as a CSV file.

    Each row is decoded back into human-readable categorical values.

    Parameters
    ----------
    tensors : torch.Tensor of shape ``(N, INPUT_DIM)`` — raw Generator output
    epoch   : int — used for the filename
    tag     : str — label prefix for the file

    Returns
    -------
    Path to the written CSV file.

    ⚠  Saved rows are SYNTHETIC.  They are clearly labelled with a
    ``_synthetic_tag`` column set to ``"GAN_GENERATED"`` so they cannot be
    accidentally treated as real claims.
    """
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAMPLES_DIR / f"{tag}_epoch_{epoch:04d}.csv"

    fieldnames = list(CATEGORICAL_FEATURES.keys()) + ["_synthetic_tag"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in tensors.detach().cpu():
            decoded = decode_row(t)
            decoded["_synthetic_tag"] = "GAN_GENERATED"
            writer.writerow(decoded)

    return out_path


# ---------------------------------------------------------------------------
# Metric logging
# ---------------------------------------------------------------------------

class MetricLogger:
    """
    Lightweight in-memory metric logger with JSON export.

    Accumulates (epoch, loss_d, loss_g) entries and can write a JSON
    history file for later plotting or review.
    """

    def __init__(self) -> None:
        self.history: list[dict] = []

    def log(self, epoch: int, loss_d: float, loss_g: float, **extra) -> None:
        """Record metrics for one epoch."""
        entry = {"epoch": epoch, "loss_d": round(loss_d, 6), "loss_g": round(loss_g, 6)}
        entry.update({k: round(float(v), 6) for k, v in extra.items()})
        self.history.append(entry)

    def save(self, path: Path) -> None:
        """Write history to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def print_last(self, prefix: str = "") -> None:
        """Print the most recent log entry."""
        if self.history:
            e = self.history[-1]
            print(
                f"{prefix}[epoch {e['epoch']:4d}]  "
                f"loss_D={e['loss_d']:.4f}  loss_G={e['loss_g']:.4f}"
            )
