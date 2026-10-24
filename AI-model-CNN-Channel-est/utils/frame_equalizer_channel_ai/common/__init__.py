"""Corpus schema and AIE-parity integer primitives shared by all workflows.

These definitions are the numerical contract with the AIE channel-estimator
kernel.  Both the offline corpus generator and the training, export, and
parity tooling depend on them, so they live here rather than inside either
consumer.  Only NumPy is required.
"""

from .dataset import (
    ChannelCorpus,
    FFT_SIZE,
    LONG,
    OCCUPIED,
    average_lts,
    concatenate_corpora,
    corpus_digest,
    estimate_snr_ratio_x16,
    snr_feature_q15,
)

# Keep the shared contract explicit for callers and generated docs.
__all__ = [
    "ChannelCorpus", "FFT_SIZE", "LONG", "OCCUPIED", "average_lts",
    "concatenate_corpora", "corpus_digest", "estimate_snr_ratio_x16",
    "snr_feature_q15",
]
