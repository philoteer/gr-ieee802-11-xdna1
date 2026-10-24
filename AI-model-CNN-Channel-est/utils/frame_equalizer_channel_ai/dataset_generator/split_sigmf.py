#!/usr/bin/env python3
"""Materialize and validate fixed, frame-disjoint SigMF dataset splits.

Each supported collection is written in two forms: a stable 80/10/10 split by
frame identity and a channel-held-out split. Outputs preserve each selected
frame's sample bytes and SigMF annotations while rebasing sample offsets to the
new concatenated recordings.
"""

import argparse
from bisect import bisect_right
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.frame_equalizer_channel_ai.dataset_generator.convert_sigmf import wifi_channel


ROOT = Path(__file__).resolve().parents[3] / 'data' / 'datasets'
# Originals stay outside the consumer-facing dataset directory.
ARCHIVE = ROOT / 'channel_estimator_sources_archive'
DESTINATION = ROOT / 'channel_estimator'
COLLECTIONS = ('20260927_room', '20260928_bus')
SPLITS = ('train', 'val', 'test')
# Frame identity hashing makes assignments independent of input traversal order.
SEED = 40
# Channels 1 and 13 are held out; all remaining channels are used for training.
CHANNEL_SPLITS = {1: 'val', 13: 'test'}


def digest(path):
    """Calculate a file's SHA-256 digest in bounded-memory chunks.

    Args:
        path: File whose contents should be hashed.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def plan(collection, archive=ARCHIVE):
    """Read source metadata and validate every annotated Wi-Fi frame.

    The returned frame records contain source basename, start offset, sample
    count, channel number, annotation index, and capture index. Validation
    rejects unsupported formats, malformed/overlapping annotations, frames
    crossing frequency-capture boundaries, and frames with unknown channels.

    Args:
        collection: Collection directory name below ``archive``.
        archive: Directory containing the unsplit source collections.

    Returns:
        A pair of source-recording metadata and frame records. Recording
        metadata is keyed by basename and stores ``(metadata, payload_path)``.

    Raises:
        ValueError: If the collection is missing, malformed, or contains fewer
            than three Wi-Fi frames.
        FileNotFoundError: If a source metadata or payload file is unavailable.
    """
    source = archive / collection
    paths = sorted(source.glob('*.sigmf-meta'))
    if not paths:
        raise ValueError(f'no archived recordings in {source}')
    recordings = {}
    frames = []
    for path in paths:
        meta = json.loads(path.read_text())
        payload = path.with_suffix('.sigmf-data')
        channels = meta['global'].get('core:num_channels', 1)
        if meta['global'].get('core:datatype') != 'cf32_le' or channels not in (2, 3):
            raise ValueError(f'{path}: expected 2- or 3-channel cf32_le')
        if not payload.is_file() or payload.stat().st_size % (8 * channels):
            raise ValueError(f'{payload}: missing or incomplete payload')
        samples = payload.stat().st_size // (8 * channels)
        captures = meta['captures']
        starts = [capture['core:sample_start'] for capture in captures]
        if not starts or starts != sorted(set(starts)) or starts[0] != 0:
            raise ValueError(f'{path}: invalid capture offsets')
        annotations = meta['annotations']
        previous_end = 0
        for index, annotation in enumerate(annotations):
            start = annotation['core:sample_start']
            count = annotation['core:sample_count']
            if (annotation.get('core:label') != 'wifi-frame' or start < previous_end or
                    count <= 0 or count % 52 or start + count > samples):
                raise ValueError(f'{path}: invalid/overlapping frame at {start}')
            capture_index = bisect_right(starts, start) - 1
            if capture_index < 0 or start + count > (starts[capture_index + 1]
                                                      if capture_index + 1 < len(starts) else samples):
                raise ValueError(f'{path}: frame crosses capture boundary at {start}')
            channel = wifi_channel(captures[capture_index].get('core:frequency'))
            if channel is None:
                raise ValueError(f'{path}: unknown channel at {start}')
            frames.append((path.name, start, count, channel, index, capture_index))
            previous_end = start + count
        recordings[path.name] = (meta, payload)
    if len(frames) < 3:
        raise ValueError(f'{source}: not enough frames for three partitions')
    return recordings, frames


def partition(frames, mode):
    """Assign every frame to one of the fixed train/val/test partitions.

    ``by_frame`` sorts stable hashes of source basename and original sample
    offset, then takes rounded 80%, 10%, and remaining fractions. ``by_channel``
    holds out channels listed in ``CHANNEL_SPLITS`` and assigns all other
    channels to training.

    Args:
        frames: Validated frame records returned by :func:`plan`.
        mode: Either ``by_frame`` or ``by_channel``.

    Returns:
        A mapping from split name to its selected frame records.

    Raises:
        ValueError: If a split is empty.
    """
    if mode == 'by_frame':
        #frames((path.name, start, count, channel, index, capture_index))
        ordered = sorted(frames, key=lambda frame: hashlib.sha256(
            f'{SEED}:{frame[0]}:{frame[1]}'.encode()).digest())
        train_end = round(0.8 * len(ordered))
        val_end = train_end + round(0.1 * len(ordered))
        assignments = { (frame[0], frame[1]): name
                        for name, selected in zip(SPLITS, (ordered[:train_end],
                                                           ordered[train_end:val_end], ordered[val_end:]))
                        for frame in selected }
    else:
        assignments = {(frame[0], frame[1]): CHANNEL_SPLITS.get(frame[3], 'train')
                       for frame in frames}
    groups = {name: [frame for frame in frames if assignments[(frame[0], frame[1])] == name]
              for name in SPLITS}
    if any(not group for group in groups.values()):
        raise ValueError('a partition is empty')
    return groups


def generate(collection, archive=ARCHIVE, destination=DESTINATION):
    """Write both SigMF split variants and their integrity manifests.

    Frame sample ranges are copied verbatim into per-split payloads. Capture
    starts and annotation starts are rebased to each output's concatenated
    sample offsets. The manifest stores source digests, output digests, and
    original per-frame offsets for subsequent validation.

    Args:
        collection: Collection directory name to split.
        archive: Directory containing the original SigMF recordings.
        destination: Parent directory for generated collection splits.

    Raises:
        ValueError: If the source collection fails validation or has an empty
            split.
        FileExistsError: If either split variant already exists for the
            collection; generation never overwrites an existing split.
    """
    recordings, frames = plan(collection, archive)
    root = destination / collection
    if any((root / mode).exists() for mode in ('by_frame', 'by_channel')):
        raise FileExistsError(f'{root}: split output already exists; remove both versions before regenerating')
    for mode in ('by_frame', 'by_channel'):
        groups = partition(frames, mode)
        directory = root / mode
        directory.mkdir(parents=True)
        manifest = {'mode': mode, 'seed': SEED, 'channel_splits': CHANNEL_SPLITS if mode == 'by_channel' else None,
                    'sources': {name: {'metadata_sha256': digest(archive / collection / name),
                                       'data_sha256': digest(payload),
                                       'frames': sum(frame[0] == name for frame in frames)}
                                for name, (_, payload) in recordings.items()}, 'splits': {}}
        for split, selected in groups.items():
            manifest['splits'][split] = {}
            for name, (meta, payload) in recordings.items():
                picked = sorted((frame for frame in selected if frame[0] == name), key=lambda frame: frame[1])
                if not picked:
                    continue
                output = directory / split
                output.mkdir(exist_ok=True)
                target = output / name
                copy_meta = {**meta, 'global': dict(meta['global']), 'captures': [], 'annotations': []}
                copy_meta['global'].pop('core:dataset', None)
                offsets = []
                cursor = 0
                with payload.open('rb') as src, target.with_suffix('.sigmf-data').open('xb') as dst:
                    for _, start, count, channel, index, capture_index in picked:
                        src.seek(start * 8 * meta['global']['core:num_channels'])
                        size = count * 8 * meta['global']['core:num_channels']
                        chunk = src.read(size)
                        if len(chunk) != size:
                            raise ValueError(f'{payload}: truncated at {start}')
                        dst.write(chunk)
                        capture = dict(meta['captures'][capture_index])
                        capture['core:sample_start'] = cursor
                        copy_meta['captures'].append(capture)
                        annotation = dict(meta['annotations'][index])
                        annotation['core:sample_start'] = cursor
                        copy_meta['annotations'].append(annotation)
                        offsets.append(start)
                        cursor += count
                target.write_text(json.dumps(copy_meta, indent=2) + '\n')
                manifest['splits'][split][name] = {'source_offsets': offsets,
                                                     'data_sha256': digest(target.with_suffix('.sigmf-data')),
                                                     'metadata_sha256': digest(target)}
        (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(f'{root / mode}: ' + ', '.join(
            f'{split}={len(selected)}' for split, selected in groups.items()))


def load_splits(root, mode, radius, minimum, retain_samples=False):
    """Load the checksummed SigMF partitions for a collection.

    Verify that the manifest lists the available files and that each metadata
    and payload checksum matches before converting the stored frames into
    channel-estimator examples.

    Args:
        root: Collection directory containing ``by_frame``/``by_channel``.
        mode: Split version to load, either ``by_frame`` or ``by_channel``.
        radius: Reference-symbol radius passed to :func:`read_frames`.
        minimum: Minimum independent reference symbols required per frame.
        retain_samples: Whether converted frames should retain raw samples.

    Returns:
        A pair: mapping from ``train``, ``val``, and ``test`` to converted
        frames, and mapping from split name to counts of skipped frames.

    Raises:
        ValueError: If the manifest or a file checksum does not match.
        FileNotFoundError: If the manifest or a referenced split file is
            missing.
    """
    from utils.frame_equalizer_channel_ai.dataset_generator.convert_sigmf import read_frames

    directory = root / mode
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['mode'] != mode or set(manifest['splits']) != set(SPLITS):
        raise ValueError(f'{directory}: invalid split manifest')
    result = {}
    skipped = {}
    for split in SPLITS:
        listed = manifest['splits'][split]
        paths = sorted((directory / split).glob('*.sigmf-meta'))
        if not paths or {p.name for p in paths} != set(listed):
            raise ValueError(f'{directory / split}: missing or unexpected SigMF files')
        for path in paths:
            entry = listed[path.name]
            if (digest(path) != entry['metadata_sha256'] or
                    digest(path.with_suffix('.sigmf-data')) != entry['data_sha256']):
                raise ValueError(f'{path}: split checksum mismatch')
        result[split], skipped[split] = read_frames(paths, radius, minimum, retain_samples)
    return result, skipped


def main():
    """Generate splits for all configured source collections.

    The command validates every source collection and checks that no output
    already exists before beginning generation, avoiding a partial update when
    a later collection cannot be processed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, default=ARCHIVE)
    parser.add_argument('--destination', type=Path, default=DESTINATION)
    args = parser.parse_args()
    for collection in COLLECTIONS:
        plan(collection, args.archive)
        if any((args.destination / collection / mode).exists()
               for mode in ('by_frame', 'by_channel')):
            parser.error(f'{args.destination / collection}: split output already exists')
    for collection in COLLECTIONS:
        generate(collection, args.archive, args.destination)


if __name__ == '__main__':
    main()
