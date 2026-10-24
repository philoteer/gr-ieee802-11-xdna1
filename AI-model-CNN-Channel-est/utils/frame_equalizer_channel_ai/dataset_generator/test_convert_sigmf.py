"""Regression tests for converting SigMF recordings into channel corpora."""

import json
from pathlib import Path

import numpy as np
import pytest

from utils.frame_equalizer_channel_ai.common.dataset import LONG, OCCUPIED
from utils.frame_equalizer_channel_ai.common.sample_counts import count_samples
from utils.frame_equalizer_channel_ai.dataset_generator.convert_sigmf import (
    frame_examples, read_frames,
)
from utils.frame_equalizer_channel_ai.dataset_generator.split_sigmf import generate, load_splits


def _frame(symbols=10):
    """Create a synthetic 52-tone frame with a known channel response.

    Args:
        symbols: Number of OFDM symbols in the frame.

    Returns:
        Complex received/transmitted samples shaped ``(symbols, 52, 2)``.
    """
    data = np.ones((symbols, 52, 2), dtype=np.complex64)
    channel = np.linspace(1 + 0.2j, 2 + 0.4j, 52, dtype=np.complex64)
    data[:, :, 0] = channel
    data[0, :, 0] += 0.2 + 0.1j
    return data


def test_payload_observations_and_leave_out_reference():
    """Check paired targets use independent surrounding reference symbols."""
    corpus = frame_examples(_frame(20), reference_radius=8, min_reference_symbols=4)
    assert corpus is not None
    assert corpus.inputs_q15.shape == (19, 6, 64)
    assert not np.any(corpus.observations[:, :, ~OCCUPIED])
    assert not np.any(corpus.truth[:, ~OCCUPIED])
    corrected_first = corpus.observations[0, 0, OCCUPIED, 0] * LONG[OCCUPIED]
    corrected_second = corpus.observations[0, 1, OCCUPIED, 0] * LONG[OCCUPIED]
    assert np.median(corrected_first - corrected_second) > 1000
    assert np.max(np.abs(corpus.truth[0, OCCUPIED, 0] - corrected_second)) <= 1
    assert np.mean(np.abs(corpus.ls[0, OCCUPIED] - corpus.truth[0, OCCUPIED])) > 100
    assert np.mean(np.abs(corpus.ls[15, OCCUPIED] - corpus.truth[15, OCCUPIED])) < 1


def test_reading_sigmf_and_frame_disjoint_splits(tmp_path: Path):
    """Check SigMF parsing, channel lookup, and frame-level split isolation."""
    frames = [_frame() for _ in range(8)]
    samples = np.concatenate([frame.reshape(-1, 2) for frame in frames])
    base = tmp_path / 'sample'
    samples.astype('<c8').tofile(base.with_suffix('.sigmf-data'))
    metadata = {
        'global': {'core:datatype': 'cf32_le', 'core:num_channels': 2, 'core:version': '1.2.6'},
        'captures': [{'core:sample_start': index * 520,
                      'core:frequency': 2432e6 if index < 4 else (2412e6 if index < 6 else 2472e6)}
                     for index in range(8)],
        'annotations': [{'core:sample_start': index * 520, 'core:sample_count': 520,
                         'core:label': 'wifi-frame'} for index in range(8)],
    }
    base.with_suffix('.sigmf-meta').write_text(json.dumps(metadata))
    converted, skipped = read_frames([base.with_suffix('.sigmf-meta')], 8, 4)
    assert not skipped and len(converted) == 8
    assert [frame.channel for frame in converted] == [5] * 4 + [1] * 2 + [13] * 2
    archive = tmp_path / 'archive' / 'example'
    archive.mkdir(parents=True)
    base.with_suffix('.sigmf-meta').rename(archive / 'sample.sigmf-meta')
    base.with_suffix('.sigmf-data').rename(archive / 'sample.sigmf-data')
    generate('example', tmp_path / 'archive', tmp_path / 'outputs')
    for mode in ('by_frame', 'by_channel'):
        root = tmp_path / 'outputs' / 'example' / mode
        manifest = json.loads((root / 'manifest.json').read_text())
        assignments = [set(manifest['splits'][part]['sample.sigmf-meta']['source_offsets'])
                       for part in ('train', 'val', 'test')]
        assert set.union(*assignments) == set(range(0, 4160, 520))
        assert all(not a & b for i, a in enumerate(assignments) for b in assignments[i + 1:])
        if mode == 'by_frame':
            assert [len(split) for split in assignments] == [6, 1, 1]
        else:
            assert [len(split) for split in assignments] == [4, 2, 2]
        for part in ('train', 'val', 'test'):
            path = root / part / 'sample.sigmf-meta'
            recording, _ = read_frames([path], 8, 4)
            assert len(recording) == len(assignments[('train', 'val', 'test').index(part)])
        loaded, _ = load_splits(tmp_path / 'outputs' / 'example', mode, 8, 4)
        assert sum(map(len, loaded.values())) == 8
        summary = count_samples(tmp_path / 'outputs' / 'example', mode)
        expected_frames = dict(zip(('train', 'validation', 'test'), map(len, assignments)))
        assert summary['frames'] == {'cnn': expected_frames, 'mlp': expected_frames}
        assert summary['samples'] == {
            'cnn': {name: count * 9 for name, count in expected_frames.items()},
            'mlp': {name: count * 8 for name, count in expected_frames.items()},
        }
        filtered = count_samples(tmp_path / 'outputs' / 'example', mode, recording='missing')
        assert all(count == 0 for counts in filtered['samples'].values() for count in counts.values())
        data = root / 'train' / 'sample.sigmf-data'
        with data.open('r+b') as stream:
            stream.write(b'corrupt!')
        with pytest.raises(ValueError, match='checksum mismatch'):
            load_splits(tmp_path / 'outputs' / 'example', mode, 8, 4)


def test_channel_lookup(tmp_path: Path):
    frames = [_frame() for _ in range(6)]
    samples = np.concatenate([frame.reshape(-1, 2) for frame in frames])
    base = tmp_path / 'sample'
    samples.astype('<c8').tofile(base.with_suffix('.sigmf-data'))
    metadata = {
        'global': {'core:datatype': 'cf32_le', 'core:num_channels': 2, 'core:version': '1.2.6'},
        'captures': [{'core:sample_start': index * 520,
                      'core:frequency': 2412e6 + 5e6 * (index // 2)}
                     for index in range(6)],
        'annotations': [{'core:sample_start': index * 520, 'core:sample_count': 520,
                         'core:label': 'wifi-frame'} for index in range(6)],
    }
    base.with_suffix('.sigmf-meta').write_text(json.dumps(metadata))
    converted, _ = read_frames([base.with_suffix('.sigmf-meta')], 8, 4)
    assert [frame.channel for frame in converted] == [1, 1, 2, 2, 3, 3]
