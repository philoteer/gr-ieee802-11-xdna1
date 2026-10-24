"""Floating-point CNN architecture and training helpers.

The model predicts a residual correction to the two-LTS LS estimate.  It uses
frequency-axis convolutions so adjacent subcarriers contribute to each output
channel estimate while preserving the 64-bin FFT layout.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .dataset import ChannelCorpus, OCCUPIED


class ChannelEstimator(nn.Module):
    """Small bias-free residual CNN used by the channel-estimator kernel.

    The input has five channels: real and imaginary components for each LTS,
    followed by the occupied-subcarrier mask.  The output has two channels for
    the real and imaginary residual correction.
    """

    def __init__(self):
        """Build the three-layer same-padded Conv1d network."""
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv1d(6, 12, 7, padding="same", bias=False),
            nn.Conv1d(12, 12, 5, padding="same", bias=False),
            nn.Conv1d(12, 2, 3, padding="same", bias=False),
        ])

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Evaluate the CNN residual predictor.

        Args:
            values: Float tensor with shape ``(batch, 5, 64)``.  Inputs are
                expected to be scaled to approximately ``[-1, 1]``.

        Returns:
            Float residual tensor with shape ``(batch, 2, 64)``.
        """
        values = torch.relu(self.layers[0](values))
        values = torch.relu(self.layers[1](values))
        return self.layers[2](values)


def float_estimate(model: ChannelEstimator, corpus: ChannelCorpus) -> np.ndarray:
    """Apply a floating-point model and restore channel-scale units.

    Args:
        model: Trained floating-point channel estimator.
        corpus: Corpus providing normalized inputs, LS estimates, and shifts.

    Returns:
        Float channel estimates with shape ``(N, 64, 2)``.  Guard and DC bins
        are explicitly zeroed.
    """
    x = torch.from_numpy(corpus.inputs_q15.astype(np.float32) / 32768.0)
    with torch.no_grad():
        residual = model(x).numpy().transpose(0, 2, 1) * 32768.0
    normalized_ls = np.empty_like(residual)
    for index, shift in enumerate(corpus.normalization_shift):
        normalized_ls[index] = corpus.ls[index] * (2.0 ** int(shift))
    estimate = (normalized_ls + residual) / np.asarray(
        [2.0 ** int(v) for v in corpus.normalization_shift])[:, None, None]
    estimate[:, ~OCCUPIED] = 0
    return estimate
