"""
code/gan/generator.py
=====================
Generator (G) for the Safety Data GAN.

Architecture
------------
The Generator is a fully-connected feed-forward network that maps a random
latent vector z ~ N(0, I) into a vector in the same feature space as a real
claim record.

    z ∈ R^{LATENT_DIM}
         ↓  Linear + BatchNorm + LeakyReLU  ×  GEN_LAYERS
    h ∈ R^{HIDDEN_DIM}
         ↓  Linear + Sigmoid (per-feature softmax applied post-hoc at decode)
    x̂ ∈ R^{INPUT_DIM}

Why Sigmoid at the output?
    Each output dimension is in [0, 1].  During decoding, the argmax within
    each feature slice picks the "voted" category — the raw output does not
    need to be a valid one-hot vector for the loss to work, but the
    Sigmoid keeps gradients well-behaved.

Why no image convolutions?
    The data is tabular (structured CSV records), not pixel grids.  A
    fully-connected architecture is the standard choice for tabular GANs
    (CTGAN, TableGAN, etc.).
"""

import torch
import torch.nn as nn

from gan.config import LATENT_DIM, HIDDEN_DIM, INPUT_DIM, GEN_LAYERS


class Generator(nn.Module):
    """
    Fully-connected Generator for tabular claim record synthesis.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the input noise vector.  Defaults to ``LATENT_DIM``
        from config.
    hidden_dim : int
        Width of each hidden layer.  Defaults to ``HIDDEN_DIM`` from config.
    output_dim : int
        Dimensionality of the output (one-hot encoded feature vector).
        Defaults to ``INPUT_DIM`` from config.
    n_layers : int
        Number of hidden layers between input and output.

    Forward pass
    ------------
    Input  : z of shape ``(batch_size, latent_dim)``
    Output : x̂ of shape ``(batch_size, output_dim)``, values in [0, 1]
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        output_dim: int = INPUT_DIM,
        n_layers:   int = GEN_LAYERS,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []

        # Input projection
        layers.extend([
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        ])

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
            ])

        # Output projection — Sigmoid maps each logit to [0, 1]
        layers.extend([
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        ])

        self.network = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate a batch of synthetic claim feature vectors.

        Parameters
        ----------
        z : torch.Tensor of shape ``(batch_size, latent_dim)``
            Random noise sampled from N(0, I).

        Returns
        -------
        torch.Tensor of shape ``(batch_size, output_dim)``
            Continuous approximation of one-hot encoded claim records.
        """
        return self.network(z)

    # ------------------------------------------------------------------
    @staticmethod
    def sample_noise(batch_size: int, latent_dim: int, device: torch.device) -> torch.Tensor:
        """
        Convenience: sample a batch of standard Gaussian noise vectors.

        Parameters
        ----------
        batch_size : int
        latent_dim : int
        device     : torch.device

        Returns
        -------
        torch.Tensor of shape ``(batch_size, latent_dim)``
        """
        return torch.randn(batch_size, latent_dim, device=device)
