"""Training and fixed-point inference utilities for the CNN channel estimator.

The package exposes the corpus, floating-point model, quantized model, and
host-side inference entry points used by the AIE export and parity tests.
"""

from .dataset import ChannelCorpus, generate_corpus
from .model import ChannelEstimator
from .quantize import QuantizedEstimator, integer_estimate, quantize_model

# Keep the supported high-level API explicit for callers and generated docs.
__all__ = [
    "ChannelCorpus", "ChannelEstimator", "QuantizedEstimator",
    "generate_corpus", "integer_estimate", "quantize_model",
]
