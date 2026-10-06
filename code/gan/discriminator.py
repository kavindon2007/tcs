"""
code/gan/discriminator.py
=========================
Discriminator (D) for the Safety Data GAN.

Architecture
------------
The Discriminator is a fully-connected classifier that takes a feature vector
(either a real encoded claim or a Generator output) and outputs a scalar
probability that the input is real.

    x ∈ R^{INPUT_DIM}   (real or generated)
         ↓  Linear + LeakyReLU + Dropout  ×  DISC_LAYERS
    h ∈ R^{HIDDEN_DIM}
         ↓  Linear + Sigmoid
    p ∈ [0, 1]           (P(real))

Why no BatchNorm in the Discriminator?
    The GAN literature (Radford et al., 2015) recommends omitting BatchNorm
    in the Discriminator's input layer and often throughout, as it can destabilise
    training when the batch contains both real and generated samples.  Dropout
    is used instead for regularisation.

Why Sigmoid at the output?
    Standard (non-Wasserstein) GAN uses Binary Cross-Entropy loss, which
    expects probabilities in [0, 1].
"""

import torch
import torch.nn as nn

from gan.config import HIDDEN_DIM, INPUT_DIM, DISC_LAYERS, DROPOUT_RATE


class Discriminator(nn.Module):
    """
    Fully-connected Discriminator for tabular claim records.

    Parameters
    ----------
    input_dim   : int   — dimension of the feature vector (real or generated)
    hidden_dim  : int   — width of hidden layers
    n_layers    : int   — number of hidden layers
    dropout_rate: float — dropout probability applied after each hidden layer

    Forward pass
    ------------
    Input  : x of shape ``(batch_size, input_dim)``
    Output : scalar probability of shape ``(batch_size, 1)``, values in [0, 1]
    """

    def __init__(
        self,
        input_dim:    int   = INPUT_DIM,
        hidden_dim:   int   = HIDDEN_DIM,
        n_layers:     int   = DISC_LAYERS,
        dropout_rate: float = DROPOUT_RATE,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []

        # Input layer (no BatchNorm — standard GAN practice)
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),
        ])

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout_rate),
            ])

        # Output: single probability score
        layers.extend([
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        ])

        self.network = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify a batch of feature vectors as real or fake.

        Parameters
        ----------
        x : torch.Tensor of shape ``(batch_size, input_dim)``

        Returns
        -------
        torch.Tensor of shape ``(batch_size, 1)``
            P(real) for each sample in the batch.
        """
        return self.network(x)
