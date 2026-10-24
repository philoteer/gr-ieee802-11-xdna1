"""Unit tests for deterministic corpus generation and integer export parity."""

from pathlib import Path

import numpy as np

from dataset import (LONG, OCCUPIED, concatenate_corpora, corpus_digest,
                     generate_corpus, generate_gnuradio_corpus)
from exporter import export_model
from model import ChannelEstimator
from quantize import integer_estimate, load_npz, quantize_model


def test_dataset_determinism_and_split_difference():
    """Verify identical seeds reproduce data and distinct seeds diverge."""
    first = generate_corpus(8, 17)
    second = generate_corpus(8, 17)
    other = generate_corpus(8, 18)
    assert corpus_digest(first) == corpus_digest(second)
    assert corpus_digest(first) != corpus_digest(other)
    assert np.array_equal(OCCUPIED, LONG != 0)
    assert OCCUPIED.sum() == 52


def test_noiseless_ls_matches_quantized_truth_with_rounding_tolerance():
    """Verify the two-LTS LS estimate is exact up to integer rounding noise."""
    corpus = generate_corpus(16, 31, snr_min_db=300.0, snr_max_db=300.0)
    error = np.abs(corpus.ls.astype(np.int64) - corpus.truth.astype(np.int64))
    assert error[:, OCCUPIED].max() <= 1
    assert np.all(corpus.ls[:, ~OCCUPIED] == 0)


def test_integer_shape_and_masking():
    """Verify integer inference preserves shape, dtype, and bin masking."""
    corpus = generate_corpus(5, 41)
    estimate = integer_estimate(quantize_model(ChannelEstimator()), corpus)
    assert estimate.shape == (5, 64, 2)
    assert estimate.dtype == np.int32
    assert np.all(estimate[:, ~OCCUPIED] == 0)


def test_export_reload_parity(tmp_path: Path):
    """Verify serialized weights reproduce the in-memory integer reference."""
    corpus = generate_corpus(4, 51)
    quantized = quantize_model(ChannelEstimator())
    _, npz, json_path = export_model(quantized, tmp_path / "weights.h",
                                     tmp_path / "weights")
    reloaded = load_npz(npz)
    assert json_path.is_file()
    assert np.array_equal(integer_estimate(quantized, corpus),
                          integer_estimate(reloaded, corpus))


def test_gnuradio_dataset_determinism_and_contract():
    first = generate_gnuradio_corpus(4, 61)
    second = generate_gnuradio_corpus(4, 61)
    assert corpus_digest(first) == corpus_digest(second)
    assert first.inputs_q15.shape == (4, 5, 64)
    assert first.observations.shape == (4, 2, 64, 2)
    assert np.all(first.observations[:, :, ~OCCUPIED] == 0)
    assert np.all(first.truth[:, ~OCCUPIED] == 0)


def test_concatenate_corpora_preserves_all_samples():
    original = generate_corpus(3, 71)
    combined = concatenate_corpora(original, generate_corpus(2, 72))
    assert combined.inputs_q15.shape[0] == 5
    assert np.array_equal(combined.inputs_q15[:3], original.inputs_q15)
