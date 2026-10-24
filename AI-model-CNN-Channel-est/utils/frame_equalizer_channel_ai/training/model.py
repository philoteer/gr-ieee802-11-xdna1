from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from ..common.dataset import ChannelCorpus, OCCUPIED
from ..common.model import ChannelEstimator


def _augment_inputs(batch: torch.Tensor, noise_std: float,
                    generator: torch.Generator, signal_features: int = 4) -> torch.Tensor:
    if not noise_std:
        return batch
    strength = torch.rand((len(batch),) + (1,) * (batch.ndim - 1), generator=generator) * noise_std
    noise = torch.randn(batch[:, :signal_features].shape, generator=generator) * strength
    return torch.cat((batch[:, :signal_features] + noise, batch[:, signal_features:]), dim=1)


def train_model(corpus: ChannelCorpus, validation: ChannelCorpus, epochs: int,
                batch_size: int, learning_rate: float, seed: int,
                patience: int = 8, min_delta: float = 0.0,
                noise_std: float = 0.02) -> tuple[ChannelEstimator, list[float], list[float], int]:
    if epochs < 1 or batch_size < 1 or patience < 1 or min_delta < 0 or noise_std < 0:
        raise ValueError("epochs, batch size, and patience must be positive; min delta and noise must be nonnegative")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = ChannelEstimator()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    x = torch.from_numpy(corpus.inputs_q15.astype(np.float32) / 32768.0)
    y = torch.from_numpy(corpus.target_q15.transpose(0, 2, 1).astype(np.float32) / 32768.0)
    validation_x = torch.from_numpy(validation.inputs_q15.astype(np.float32) / 32768.0)
    validation_y = torch.from_numpy(validation.target_q15.transpose(0, 2, 1).astype(np.float32) / 32768.0)
    mask = torch.from_numpy(OCCUPIED).reshape(1, 1, 64)
    generator = torch.Generator().manual_seed(seed + 1)
    noise_generator = torch.Generator().manual_seed(seed + 2)
    losses: list[float] = []
    validation_losses: list[float] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for indices in torch.randperm(len(x), generator=generator).split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(_augment_inputs(x[indices], noise_std, noise_generator))
            loss = torch.mean((prediction.masked_select(mask) -
                               y[indices].masked_select(mask)) ** 2)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        losses.append(total / len(x))
        model.eval()
        with torch.no_grad():
            total = 0.0
            for indices in torch.arange(len(validation_x)).split(batch_size):
                prediction = model(validation_x[indices])
                loss = torch.mean((prediction.masked_select(mask) -
                                   validation_y[indices].masked_select(mask)) ** 2)
                total += float(loss) * len(indices)
        validation_loss = total / len(validation_x)
        validation_losses.append(validation_loss)
        if validation_loss < best_loss - min_delta:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = deepcopy(model.state_dict())
        elif epoch + 1 - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return model.eval(), losses, validation_losses, best_epoch
