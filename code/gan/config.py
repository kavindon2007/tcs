"""
code/gan/config.py
==================
Central configuration for the GAN-based Safety Data Generation pipeline.

This GAN targets **structured/tabular claim records** (not raw images), because
the HackerRank Orchestrate dataset is a CSV-based multi-modal evidence review
dataset.  The Generator learns to produce realistic synthetic claim records that
resemble the statistical distribution of ``dataset/sample_claims.csv``.

Usage
-----
Import this module anywhere in the GAN package::

    from gan.config import GAN_CONFIG

All training knobs live here; edit this file rather than individual scripts.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Repository layout (resolved relative to this file's location)
# code/gan/config.py → repo root is three levels up
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).parent.parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
GAN_DIR     = Path(__file__).parent

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
REAL_DATA_CSV  = DATASET_DIR / "sample_claims.csv"   # 21 labeled rows — real data
OUTPUT_DIR     = GAN_DIR / "output"                  # generated samples + checkpoints
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
SAMPLES_DIR    = OUTPUT_DIR / "samples"

# ---------------------------------------------------------------------------
# Feature encoding
# Categorical columns encoded as one-hot; continuous columns normalised to [0,1].
# These must match the columns present in sample_claims.csv.
# ---------------------------------------------------------------------------
CATEGORICAL_FEATURES = {
    "claim_object":   ["car", "laptop", "package"],
    "claim_status":   ["supported", "contradicted", "not_enough_information"],
    "issue_type":     [
        "dent", "scratch", "crack", "glass_shatter", "broken_part",
        "missing_part", "torn_packaging", "crushed_packaging",
        "water_damage", "stain", "none", "unknown",
    ],
    "object_part":    [
        # car
        "front_bumper", "rear_bumper", "door", "hood", "windshield",
        "side_mirror", "headlight", "taillight", "fender", "quarter_panel",
        # laptop
        "screen", "keyboard", "trackpad", "hinge", "lid", "corner",
        "port", "base",
        # package
        "box", "package_corner", "package_side", "seal", "label",
        "contents", "item",
        # shared
        "body", "unknown",
    ],
    "severity":       ["none", "low", "medium", "high", "unknown"],
    "valid_image":    ["true", "false"],
    "evidence_standard_met": ["true", "false"],
}

# Total one-hot dimension
INPUT_DIM = sum(len(v) for v in CATEGORICAL_FEATURES.values())

# ---------------------------------------------------------------------------
# GAN hyperparameters
# ---------------------------------------------------------------------------
LATENT_DIM    = 64      # size of the random noise / latent vector
HIDDEN_DIM    = 256     # width of hidden layers in G and D
GEN_LAYERS    = 4       # number of hidden layers in Generator
DISC_LAYERS   = 4       # number of hidden layers in Discriminator
DROPOUT_RATE  = 0.3     # dropout probability in Discriminator

BATCH_SIZE    = 16      # small batch fits the 21-row real dataset
EPOCHS        = 2000    # standard for tabular GANs on tiny datasets
LR_G          = 2e-4    # Generator Adam learning rate
LR_D          = 2e-4    # Discriminator Adam learning rate
BETA1         = 0.5     # Adam beta1 (GAN convention)
BETA2         = 0.999   # Adam beta2

N_DISC_STEPS  = 2       # discriminator update steps per generator step
LABEL_SMOOTH  = 0.1     # one-sided label smoothing for real samples (0 = off)

# ---------------------------------------------------------------------------
# Checkpointing & output
# ---------------------------------------------------------------------------
SAVE_EVERY    = 200     # save checkpoint every N epochs
SAMPLE_EVERY  = 200     # generate & save sample rows every N epochs
N_SAMPLES     = 50      # number of synthetic rows to generate per sample save
SEED          = 42      # global random seed for reproducibility

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_EVERY     = 50      # print loss every N epochs
