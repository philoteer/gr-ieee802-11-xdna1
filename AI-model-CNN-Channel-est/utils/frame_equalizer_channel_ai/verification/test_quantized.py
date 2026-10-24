from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.frame_equalizer_channel_ai.common.dataset import ChannelCorpus, OCCUPIED
from utils.frame_equalizer_channel_ai.common.quantized import QuantizedEstimator
from utils.frame_equalizer_channel_ai.training.exporter import export_model
from utils.frame_equalizer_channel_ai.training.quantize import quantize_model
from utils.frame_equalizer_channel_ai.common.model import ChannelEstimator
from utils.frame_equalizer_channel_ai.verification.quantized import integer_estimate, load_npz


def _sample_corpus(samples: int) -> ChannelCorpus:
    inputs_q15 = np.zeros((samples, 6, 64), dtype=np.int32)
    inputs_q15[:, 4] = OCCUPIED.astype(np.int32)[None, :] * 32767
    observations = np.zeros((samples, 2, 64, 2), dtype=np.int32)
    snr_ratio_x16 = np.full(samples, 32, dtype=np.uint64)
    ls = np.zeros((samples, 64, 2), dtype=np.int32)
    ls[:, OCCUPIED, 0] = 16000
    truth = ls.copy()
    target_q15 = np.zeros_like(ls)
    normalization_shift = np.zeros(samples, dtype=np.int32)
    return ChannelCorpus(inputs_q15, observations, snr_ratio_x16, ls, truth,
                         target_q15, normalization_shift)


def _zero_model(gate: int = np.iinfo(np.uint64).max) -> QuantizedEstimator:
    weights = tuple(np.zeros(shape, dtype=np.int8) for shape in (
        (12, 6, 7), (12, 12, 5), (2, 12, 3)))
    return QuantizedEstimator(weights, gate)


def test_integer_shape_and_masking():
    corpus = _sample_corpus(5)
    estimate = integer_estimate(_zero_model(), corpus)
    assert estimate.shape == (5, 64, 2)
    assert estimate.dtype == np.int32
    assert np.all(estimate[:, ~OCCUPIED] == 0)


def test_gate_returns_exact_ls_estimate():
    corpus = _sample_corpus(5)
    estimate = integer_estimate(_zero_model(0), corpus)
    assert np.array_equal(estimate, corpus.ls)


def test_export_reload_parity(tmp_path: Path):
    corpus = _sample_corpus(4)
    quantized = quantize_model(ChannelEstimator())
    _, npz, json_path = export_model(quantized, tmp_path / "weights.h",
                                     tmp_path / "weights")
    reloaded = load_npz(npz)
    assert json_path.is_file()
    assert np.array_equal(integer_estimate(quantized, corpus),
                          integer_estimate(reloaded, corpus))
