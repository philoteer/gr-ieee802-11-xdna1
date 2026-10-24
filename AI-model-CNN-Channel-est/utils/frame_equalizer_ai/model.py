"""Bias-free joint-I/Q neural demapper architectures and CPU training helpers.

The demapper consumes three real features per data subcarrier
(``Re(Y*conj(H))``, ``Im(Y*conj(H))``, ``|H|^2``) and emits one logit per
coded bit.  Every architecture is a bias-free MLP
``3 -> hidden... -> bits`` with ReLU after each hidden layer, chosen so the
exported weights map directly onto the AIE integer pipeline (int8 MACs, acc64,
shifts).  Training is CPU-only and fully deterministic: fixed seeds plus
``torch.use_deterministic_algorithms(True)`` keep checkpoints reproducible.

Architectures are selected by name.  The chosen name fixes both the trained
topology here and the matching kernel header under
``kernel/demodulator/model/``; see :data:`ARCHITECTURES`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

# Number of input features per data subcarrier: matched-filter real and
# imaginary components plus the channel power.
FEATURE_COUNT = 3
FFT_SIZE = 64
DATA_BINS = tuple(i for i in range(6, 59) if i not in (11, 25, 32, 39, 53))
FEATURE_VECTOR_SIZE = FFT_SIZE * FEATURE_COUNT


def _normalize_feature_vectors(values: torch.Tensor) -> torch.Tensor:
    """Scale each symbol's features by a power of two.

    The per-symbol divisor is the smallest power of two above the largest
    absolute feature, and it is shared by the real, imaginary, and
    channel-power components so their ratios are preserved.  This matches the
    AIE input stage's per-symbol, all-lane scaling.

    Args:
        values: Feature tensor of shape ``(N, 64, 3)`` or ``(N, 192)``.

    Returns:
        ``(N, 64, 3)`` tensor whose absolute values are below one.

    Raises:
        ValueError: If ``values`` is neither of the accepted shapes.
    """
    if values.ndim == 2:
        if values.shape[1] != FEATURE_VECTOR_SIZE:
            raise ValueError("flattened features must contain 192 values")
        values = values.reshape(-1, FFT_SIZE, FEATURE_COUNT)
    if values.ndim != 3 or values.shape[1:] != (FFT_SIZE, FEATURE_COUNT):
        raise ValueError("features must have shape (N, 64, 3) or (N, 192)")
    maximum = values.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    exponent = torch.floor(torch.log2(maximum)) + 1.0
    return values / torch.pow(2.0, exponent)


@dataclass(frozen=True)
class Architecture:
    """One compile-time MLP topology shared by the trainer and the AIE kernel.

    Attributes:
        name: Selector value used by the CLI and by the checkpoint metadata.
        hidden_sizes: Width of each hidden layer, in order.
        header_basename: File name of the generated kernel weights header for
            this architecture inside ``kernel/demodulator/model/``.
    """

    name: str
    hidden_sizes: tuple[int, ...]
    header_basename: str


# The model axis stays ordered like constellations.MODULATIONS for every
# architecture.  Adding an entry here requires a matching kernel header
# (kernel/demodulator/model/frame_equalizer_model_<name>.h), a selector branch in
# kernel/demodulator/model/frame_equalizer_model.h, and a Makefile SUPPORTED_MODELS
# entry.
ARCHITECTURES = {
    "mlp_16x16": Architecture(
        "mlp_16x16", (16, 16),
        "frame_equalizer_model_mlp_16x16_weights.h"),
    "mlp_16": Architecture(
        "mlp_16", (16,),
        "frame_equalizer_model_mlp_16_weights.h"),
}
DEFAULT_ARCHITECTURE = "mlp_16x16"


class JointIQMLP(nn.Module):
    """Joint-I/Q bias-free MLP with a configurable number of hidden layers.

    Architecture: ``Linear(3, h1, bias=False) -> ReLU -> ... ->
    Linear(h_last, bit_count, bias=False)``.  Biases are omitted so the
    quantized integer reference and the AIE kernel only need MAC, ReLU, and
    arithmetic shifts.

    Attributes:
        layers: The ``nn.Sequential`` interleaving linear layers and ReLUs.
            Even entries are the weight-bearing linear layers.
        hidden_sizes: Width of each hidden layer, in order.

    Args:
        bit_count: Number of output logits (coded bits per symbol).
        hidden_sizes: Width of each hidden layer, in order.
        initialize_sign_paths: Seed the first layers to implement exact
            BPSK/QPSK sign decisions before training (see
            :meth:`_initialize_sign_paths`).
    """

    def __init__(self, bit_count: int, hidden_sizes: tuple[int, ...] = (16, 16),
                 initialize_sign_paths: bool = False):
        super().__init__()
        if not hidden_sizes or any(size <= 0 for size in hidden_sizes):
            raise ValueError("hidden_sizes must contain at least one positive layer")
        self.hidden_sizes = tuple(hidden_sizes)
        layers: list[nn.Module] = []
        in_features = FEATURE_COUNT
        for size in self.hidden_sizes:
            layers += [nn.Linear(in_features, size, bias=False), nn.ReLU()]
            in_features = size
        layers.append(nn.Linear(in_features, bit_count, bias=False))
        self.layers = nn.Sequential(*layers)
        if initialize_sign_paths:
            self._initialize_sign_paths(bit_count)

    def linear_layers(self) -> list[nn.Linear]:
        """Return the weight-bearing layers in forward order.

        Returns:
            The first layer, every intermediate hidden layer, and the output
            layer.
        """
        return [self.layers[index] for index in range(0, len(self.layers), 2)]

    def _initialize_sign_paths(self, bit_count: int) -> None:
        """Seed BPSK/QPSK sign decisions while leaving all weights trainable.

        ``ReLU(+x) - ReLU(-x)`` reconstructs any signed feature exactly, so
        two hidden units per feature plus a unit-weight output implement the
        maximum-likelihood sign decision for BPSK (feature 0, I) and add the
        Q (feature 1) pair for QPSK as a starting point.  Intermediate hidden
        layers, when present, pass the seeded units through unchanged.  All
        weights remain trainable; this only biases optimization toward the
        known optimum.

        Args:
            bit_count: Number of network outputs; paths are seeded only for
                1 (BPSK) and 2 (QPSK), other orders keep their random init.
        """
        if bit_count not in (1, 2):
            return
        seeds = [(0, 0, 1.0), (1, 0, -1.0)]           # (hidden, feature, sign)
        connections = [(0, 0, 1.0), (0, 1, -1.0)]     # (bit, hidden, sign)
        if bit_count == 2:
            seeds += [(2, 1, 1.0), (3, 1, -1.0)]
            connections += [(1, 2, 1.0), (1, 3, -1.0)]
        linears = self.linear_layers()
        if self.hidden_sizes[0] < len(seeds):
            return
        with torch.no_grad():
            first, output = linears[0], linears[-1]
            first.weight.zero_()
            output.weight.zero_()
            for hidden, feature, sign in seeds:
                first.weight[hidden, feature] = sign
            # ReLU(+x)-ReLU(-x) reconstructs each signed input exactly, so the
            # identity keeps the seeded units available at the output.
            for layer in linears[1:-1]:
                layer.weight.zero_()
                for hidden, _, _ in seeds:
                    layer.weight[hidden, hidden] = 1.0
            for bit, hidden, sign in connections:
                output.weight[bit, hidden] = sign

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Run the network over the 48 data subcarriers of each symbol.

        Args:
            values: Feature tensor of shape ``(N, 64, 3)`` or ``(N, 192)``.

        Returns:
            ``(N, 48, bit_count)`` logits.
        """
        values = _normalize_feature_vectors(values)
        return self.layers(values[:, DATA_BINS, :])


def train_model(features: np.ndarray, labels: np.ndarray, bit_count: int,
                 epochs: int, batch_size: int, learning_rate: float,
                 seed: int, hidden_sizes: tuple[int, ...] = (16, 16)
                 ) -> tuple[JointIQMLP, dict[str, float]]:
    """Train deterministically on CPU and report that every dense tensor moved.

    Applies the project-wide deterministic settings (seeded global RNG,
    deterministic algorithms, seeded shuffle generator) and optimizes the
    per-bit binary cross-entropy of the LSB-first bit labels with Adam. BPSK
    and QPSK networks start from the sign-path initialization so training
    cannot regress below the linear baseline. After training, every weight
    tensor must differ from its initial value; a tensor that never moved
    indicates a dead or unused layer and raises.

    Args:
        features: Float features of shape ``(N, 64, 3)`` or flattened
            ``(N, 192)``.
        labels: ``(N * 48, bit_count)`` float32 LSB-first bit labels, one row
            per data subcarrier; reshaped internally to
            ``(N, 48, bit_count)``.
        bit_count: Number of output logits.
        epochs: Number of passes over the training set.
        batch_size: Mini-batch size for the shuffled updates.
        learning_rate: Adam learning rate.
        seed: Base seed for parameter initialization and shuffling.
        hidden_sizes: Width of each hidden layer, in order.

    Returns:
        ``(model, metrics)`` where ``model`` is in eval mode and ``metrics``
        holds ``final_loss`` (last mini-batch loss of the last epoch) and
        ``minimum_layer_weight_change`` (the smallest max-abs parameter delta
        across all dense tensors).

    Raises:
        RuntimeError: If any dense weight tensor did not change during
            training.
    """
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = JointIQMLP(bit_count, hidden_sizes,
                       initialize_sign_paths=bit_count <= 2)
    initial = [p.detach().clone() for p in model.parameters()]
    x = torch.from_numpy(features)
    y = torch.from_numpy(labels).reshape(len(x), len(DATA_BINS), bit_count)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed + 1)
    symbol_batch_size = max(1, batch_size // len(DATA_BINS))
    final_loss = 0.0
    for _ in range(epochs):
        for indices in torch.randperm(len(x), generator=generator).split(symbol_batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x[indices]), y[indices])
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())
    changes = [float(torch.max(torch.abs(p.detach() - old)))
               for p, old in zip(model.parameters(), initial)]
    if any(change == 0.0 for change in changes):
        raise RuntimeError("at least one dense weight tensor did not train")
    return model.eval(), {"final_loss": final_loss,
                          "minimum_layer_weight_change": min(changes)}


def predict_classes(model: nn.Module, features: np.ndarray) -> np.ndarray:
    """Compose thresholded output logits as an LSB-first class index.

    Runs the float model under ``torch.no_grad``, thresholds each output logit
    at zero, and packs the resulting bits LSB-first into class codes, matching
    both :func:`quantize.integer_classes` and the AIE demapper.

    Args:
        model: Float model to evaluate.
        features: Float features of shape ``(N, 64, 3)`` or flattened
            ``(N, 192)``.

    Returns:
        ``(N * 48,)`` int64 class codes.
    """
    with torch.no_grad():
        bits = model(torch.from_numpy(features)).numpy() >= 0
    return np.sum(bits.astype(np.int64) << np.arange(bits.shape[2]), axis=2).reshape(-1)
