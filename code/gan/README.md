# GAN — Damage-Claim Safety Data Generator

A **Generative Adversarial Network (GAN)** for synthesising structured damage claim
records that resemble the HackerRank Orchestrate dataset distribution.

> **Status**: Experimental / Proposed — not integrated into the main claim-verification pipeline.  
> **Data type**: Structured / tabular (not pixel images).

---

## File structure

```
code/gan/
├── __init__.py        — Python package marker
├── config.py          — All hyperparameters and paths
├── dataset.py         — Dataset loader + DEMO fallback
├── generator.py       — Generator network (G)
├── discriminator.py   — Discriminator network (D)
├── utils.py           — Device, init, checkpoints, sample saving, logging
├── train.py           — Training loop entry point
└── README.md          — You are here

docs/
└── gan_architecture.md — Full architecture diagram + documentation
```

---

## Quick start

```bash
# From the repository root

# Install PyTorch (if not already installed)
pip install torch

# Run training on real data (sample_claims.csv — 21 rows)
python code/gan/train.py

# Run training on DEMO/SYNTHETIC data (no real data required)
python code/gan/train.py --no-real --epochs 500

# Override hyperparameters
python code/gan/train.py --epochs 5000 --latent 128 --batch 32

# Resume from a checkpoint
python code/gan/train.py --resume code/gan/output/checkpoints/epoch_1999.pt
```

Output is written to `code/gan/output/`:
- `checkpoints/` — `.pt` files every 200 epochs
- `samples/` — synthetic CSV rows (every 200 epochs, tagged `GAN_GENERATED`)
- `metrics.json` — training loss history

---

## Architecture summary

```
z ~ N(0,I) ∈ ℝ⁶⁴
     ↓ Generator (FC: 64→256×4→54, BatchNorm+LeakyReLU+Sigmoid)
x̂ ∈ ℝ⁵⁴  (generated claim record)
     ↓
     ├──────────────────────────┐
     ↓                          ↓
Real x ∈ ℝ⁵⁴              Fake x̂ ∈ ℝ⁵⁴
     └──────────┬───────────────┘
                ↓ Discriminator (FC: 54→256×4→1, LeakyReLU+Dropout+Sigmoid)
           P(real) ∈ [0,1]
                ↓
   Loss_D (real+fake) / Loss_G (fool D)
                ↓
       Update D / Update G  →  repeat
```

See [`docs/gan_architecture.md`](../../docs/gan_architecture.md) for the full
Mermaid diagram, feature-space breakdown, limitations, and mentor explanation.

---

## What data goes in

The GAN reads `dataset/sample_claims.csv` and encodes seven categorical fields
as one-hot vectors:

| Field | Dim |
|---|---|
| `claim_object` | 3 |
| `claim_status` | 3 |
| `issue_type` | 12 |
| `object_part` | 27 |
| `severity` | 5 |
| `valid_image` | 2 |
| `evidence_standard_met` | 2 |
| **Total** | **54** |

Free-text fields (`user_claim`, `image_paths`) are not modelled.

---

## ⚠ Important notes

1. **21 real rows is too small** for a production GAN. Use the DEMO dataset for
   smoke-testing the training loop; a real training run needs ≥ 500–1,000 rows.
2. Generated samples are tagged with `_synthetic_tag=GAN_GENERATED` in every
   output CSV — they cannot be confused with real claim records.
3. The existing claim-verification pipeline (`code/main.py`) is **not modified**.
4. No training has been completed yet. Do not cite generated samples as evidence
   of GAN quality until a training run has been inspected.
