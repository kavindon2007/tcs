"""
code/gan/train.py
=================
Training loop for the Safety Data GAN.

Run from the repository root::

    python code/gan/train.py

Optional flags::

    --epochs    N        Override EPOCHS from config (default: config value)
    --batch     N        Override BATCH_SIZE
    --latent    N        Override LATENT_DIM
    --lr-g      FLOAT    Override LR_G
    --lr-d      FLOAT    Override LR_D
    --no-cache           Use DEMO/SYNTHETIC data even if real CSV exists
    --resume    PATH     Resume from a checkpoint .pt file

What this script does
---------------------
1. Selects the best available device (CUDA → MPS → CPU).
2. Loads the real dataset (sample_claims.csv) or falls back to DEMO data.
3. Instantiates Generator + Discriminator and applies weight initialisation.
4. Runs the standard GAN training loop:
       for each epoch:
           for each batch:
               a. Train Discriminator on real + fake samples (N_DISC_STEPS times)
               b. Train Generator to fool the Discriminator
5. Logs losses every LOG_EVERY epochs.
6. Saves checkpoints every SAVE_EVERY epochs.
7. Generates and saves sample rows every SAMPLE_EVERY epochs.
8. Writes training history to output/metrics.json.

⚠  No claim is made that training has succeeded until an actual run
   completes.  Loss curves should be inspected visually.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Allow running from repo root: python code/gan/train.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from gan import config as CFG
from gan.config import (
    BATCH_SIZE, BETA1, BETA2, EPOCHS, LABEL_SMOOTH, LATENT_DIM,
    LOG_EVERY, LR_D, LR_G, N_DISC_STEPS, N_SAMPLES,
    SAMPLE_EVERY, SAVE_EVERY, SEED,
)
from gan.dataset import get_dataset
from gan.discriminator import Discriminator
from gan.generator import Generator
from gan.utils import (
    MetricLogger, get_device, load_checkpoint, save_checkpoint,
    save_generated_samples, set_seed, weights_init,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the Safety Data GAN on encoded claim records."
    )
    p.add_argument("--epochs",   type=int,   default=EPOCHS,     help=f"Training epochs (default: {EPOCHS})")
    p.add_argument("--batch",    type=int,   default=BATCH_SIZE,  help=f"Batch size (default: {BATCH_SIZE})")
    p.add_argument("--latent",   type=int,   default=LATENT_DIM, help=f"Latent dim (default: {LATENT_DIM})")
    p.add_argument("--lr-g",     type=float, default=LR_G,       help=f"Generator LR (default: {LR_G})")
    p.add_argument("--lr-d",     type=float, default=LR_D,       help=f"Discriminator LR (default: {LR_D})")
    p.add_argument("--no-real",  action="store_true",            help="Force DEMO/SYNTHETIC data")
    p.add_argument("--resume",   type=str,   default=None,       help="Path to checkpoint .pt file to resume from")
    p.add_argument("--seed",     type=int,   default=SEED,       help=f"Random seed (default: {SEED})")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    # ── Setup ──────────────────────────────────────────────────────────────
    set_seed(args.seed)
    device = get_device()

    # ── Dataset ────────────────────────────────────────────────────────────
    prefer_real = not args.no_real
    dataset = get_dataset(prefer_real=prefer_real)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        drop_last=False,      # dataset may be tiny; keep all rows
        pin_memory=(device.type == "cuda"),
    )
    print(f"[train] Dataset size: {len(dataset)} rows | Batch size: {args.batch}")

    # ── Models ─────────────────────────────────────────────────────────────
    generator     = Generator(latent_dim=args.latent).to(device)
    discriminator = Discriminator().to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # ── Optimisers ─────────────────────────────────────────────────────────
    opt_g = torch.optim.Adam(
        generator.parameters(),
        lr=args.lr_g,
        betas=(BETA1, BETA2),
    )
    opt_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.lr_d,
        betas=(BETA1, BETA2),
    )

    # ── Loss function ───────────────────────────────────────────────────────
    criterion = nn.BCELoss()

    # ── Resume from checkpoint ─────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(
            Path(args.resume), generator, discriminator, opt_g, opt_d
        ) + 1

    logger = MetricLogger()

    # ── Training loop ───────────────────────────────────────────────────────
    print(f"\n[train] Starting training: epochs={args.epochs}, device={device}")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        epoch_loss_d = 0.0
        epoch_loss_g = 0.0
        n_batches    = 0

        for real_batch in loader:
            real_batch = real_batch.to(device)
            bs = real_batch.size(0)

            # ── Labels ─────────────────────────────────────────────────────
            # One-sided label smoothing: real labels are (1 - LABEL_SMOOTH),
            # fake labels are 0.  This prevents the Discriminator from becoming
            # overconfident on real samples too early.
            real_labels = torch.full((bs, 1), 1.0 - LABEL_SMOOTH, device=device)
            fake_labels = torch.zeros(bs, 1, device=device)

            # ── Train Discriminator (N_DISC_STEPS per batch) ───────────────
            for _ in range(N_DISC_STEPS):
                discriminator.zero_grad()

                # Real samples
                d_real = discriminator(real_batch)
                loss_d_real = criterion(d_real, real_labels)

                # Fake samples (detach so G does not receive these gradients)
                z = Generator.sample_noise(bs, args.latent, device)
                fake_batch  = generator(z).detach()
                d_fake      = discriminator(fake_batch)
                loss_d_fake = criterion(d_fake, fake_labels)

                loss_d = (loss_d_real + loss_d_fake) / 2.0
                loss_d.backward()
                opt_d.step()

            # ── Train Generator ────────────────────────────────────────────
            generator.zero_grad()

            z = Generator.sample_noise(bs, args.latent, device)
            fake_batch = generator(z)
            d_fake_for_g = discriminator(fake_batch)

            # Generator wants D to output 1 (real) for its samples
            loss_g = criterion(d_fake_for_g, real_labels)
            loss_g.backward()
            opt_g.step()

            epoch_loss_d += loss_d.item()
            epoch_loss_g += loss_g.item()
            n_batches    += 1

        # ── Per-epoch bookkeeping ──────────────────────────────────────────
        avg_loss_d = epoch_loss_d / max(n_batches, 1)
        avg_loss_g = epoch_loss_g / max(n_batches, 1)

        logger.log(epoch, avg_loss_d, avg_loss_g)

        if (epoch + 1) % LOG_EVERY == 0 or epoch == 0:
            logger.print_last(prefix="")

        if (epoch + 1) % SAVE_EVERY == 0 or epoch == args.epochs - 1:
            ckpt_path = save_checkpoint(
                generator, discriminator, opt_g, opt_d,
                epoch=epoch,
                metrics={"loss_d": avg_loss_d, "loss_g": avg_loss_g},
            )
            print(f"  ✓ Checkpoint saved → {ckpt_path.relative_to(Path.cwd())}")

        if (epoch + 1) % SAMPLE_EVERY == 0 or epoch == args.epochs - 1:
            generator.eval()
            with torch.no_grad():
                z = Generator.sample_noise(N_SAMPLES, args.latent, device)
                samples = generator(z)
            generator.train()
            sample_path = save_generated_samples(samples, epoch=epoch)
            print(f"  ✓ Samples saved    → {sample_path.relative_to(Path.cwd())}")

    # ── Final metrics export ───────────────────────────────────────────────
    metrics_path = CFG.OUTPUT_DIR / "metrics.json"
    logger.save(metrics_path)
    print(f"\n[train] Training complete.  Metrics → {metrics_path}")
    print("[train] ⚠  Inspect loss curves before using generated samples.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
