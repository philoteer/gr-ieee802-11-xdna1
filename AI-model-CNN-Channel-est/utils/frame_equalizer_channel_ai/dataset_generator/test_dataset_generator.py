"""Tests for offline channel-estimator corpus generation."""

import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.frame_equalizer_channel_ai.common.dataset import (
    ChannelCorpus,
    LONG,
    OCCUPIED,
    concatenate_corpora,
    corpus_digest,
    estimate_snr_ratio_x16,
    snr_feature_q15,
)
from utils.frame_equalizer_channel_ai.dataset_generator.corpus import (
    convert_gnuradio_archive,
    generate_corpus,
    generate_gnuradio_corpus,
    generate_gnuradio_raw_archive,
    split_corpus,
)
from utils.frame_equalizer_channel_ai.dataset_generator.generate_dataset import parser


def test_dataset_determinism_and_split_difference():
    first = generate_corpus(8, 17)
    second = generate_corpus(8, 17)
    other = generate_corpus(8, 18)
    assert corpus_digest(first) == corpus_digest(second)
    assert corpus_digest(first) != corpus_digest(other)
    assert np.array_equal(OCCUPIED, LONG != 0)
    assert OCCUPIED.sum() == 52


def test_noiseless_ls_matches_quantized_truth_with_rounding_tolerance():
    corpus = generate_corpus(16, 31, snr_min_db=300.0, snr_max_db=300.0)
    error = np.abs(corpus.ls.astype(np.int64) - corpus.truth.astype(np.int64))
    assert error[:, OCCUPIED].max() <= 1
    assert np.all(corpus.ls[:, ~OCCUPIED] == 0)


def test_measured_snr_matches_kernel_formula():
    corpus = generate_corpus(7, 42, snr_min_db=2.0, snr_max_db=40.0)
    assert np.array_equal(corpus.snr_ratio_x16,
                          estimate_snr_ratio_x16(corpus.observations))
    assert np.all(corpus.inputs_q15[:, 5] ==
                  snr_feature_q15(corpus.snr_ratio_x16)[:, None])


def test_raw_gnuradio_archive_can_be_converted_manually(tmp_path: Path):
    raw_path = tmp_path / "gnuradio-dataset.npz"
    generate_gnuradio_raw_archive(4, 62, 8.0, 32.0, raw_path)
    with np.load(raw_path, allow_pickle=False) as archive:
        for name in ("first", "second", "truth"):
            assert archive[name].shape == (4, 64)
            assert archive[name].dtype == np.complex64

    converted = convert_gnuradio_archive(raw_path)
    direct = generate_gnuradio_corpus(4, 62)
    assert corpus_digest(converted) == corpus_digest(direct)
    assert converted.inputs_q15.shape == (4, 6, 64)
    assert converted.observations.shape == (4, 2, 64, 2)
    assert np.all(converted.observations[:, :, ~OCCUPIED] == 0)
    assert np.all(converted.truth[:, ~OCCUPIED] == 0)

    args = parser().parse_args(["--convert", str(raw_path)])
    args.func(args)
    saved = ChannelCorpus.load(tmp_path / "gnuradio-dataset-converted.npz")
    assert corpus_digest(saved) == corpus_digest(direct)


def test_concatenate_corpora_preserves_all_samples():
    original = generate_corpus(3, 71)
    combined = concatenate_corpora(original, generate_corpus(2, 72))
    assert combined.inputs_q15.shape[0] == 5
    assert np.array_equal(combined.inputs_q15[:3], original.inputs_q15)


def test_offline_program_writes_reusable_split_archives(tmp_path: Path):
    args = parser().parse_args([
        "--train-samples", "8", "--validation-samples", "4", "--test-samples", "4",
        "--dataset-dir", str(tmp_path),
    ])
    args.func(args)
    manifest = json.loads(
        (tmp_path / "synthetic_original" / "manifest.json").read_text())
    assert manifest["dataset"] == "synthetic_original"
    assert manifest["source"] == "original"
    for split in ("train", "validation", "test"):
        corpus = ChannelCorpus.load(
            tmp_path / "synthetic_original" / f"{split}.npz")
        assert len(corpus.inputs_q15) == manifest["splits"][split]["samples"]
        assert corpus_digest(corpus) == manifest["splits"][split]["digest"]


def test_both_generators_write_separate_source_archives(tmp_path: Path):
    args = parser().parse_args([
        "--train-samples", "5", "--validation-samples", "4", "--test-samples", "4",
        "--dataset", "both", "--dataset-dir", str(tmp_path),
    ])
    args.func(args)
    assert not (tmp_path / "both").exists()
    for split, original_samples, gnuradio_samples in (
        ("train", 3, 2), ("validation", 2, 2), ("test", 2, 2)):
        original = ChannelCorpus.load(
            tmp_path / "synthetic_original" / f"{split}.npz")
        gnuradio = ChannelCorpus.load(
            tmp_path / "synthetic_gnuradio" / f"{split}.npz")
        assert len(original.inputs_q15) == original_samples
        assert len(gnuradio.inputs_q15) == gnuradio_samples


def test_split_corpus_uses_independent_reproducible_partitions():
    train = split_corpus("train", 4, 73)
    validation = split_corpus("validation", 4, 73)
    repeated_validation = split_corpus("validation", 4, 73)
    assert corpus_digest(train) != corpus_digest(validation)
    assert corpus_digest(validation) == corpus_digest(repeated_validation)
