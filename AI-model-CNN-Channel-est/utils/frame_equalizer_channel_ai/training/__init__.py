"""Training and fixed-point inference utilities for the CNN channel estimator.

The package exposes the corpus, floating-point model, quantized model, and
host-side inference entry points used by the AIE export and parity tests.
"""

from ..common.dataset import ChannelCorpus
from ..common.model import ChannelEstimator
from ..common.quantized import QuantizedEstimator
from .quantize import quantize_model

# Keep the supported high-level API explicit for callers and generated docs.
__all__ = [
    "ChannelCorpus", "ChannelEstimator", "QuantizedEstimator",
    "quantize_model",
]
