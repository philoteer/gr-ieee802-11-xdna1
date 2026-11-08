#!/usr/bin/env python3
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from jsonschema import ValidationError
from sigmf import sigmffile, validate
from sigmf.error import SigMFError


RATES_MBPS = (6, 9, 12, 18, 24, 36, 48, 54)
METRICS = ('snr_db', 'evm_rms_percent', 'pre_fec_ber', 'rx_gain_db')


def comment_fields(comment):
    fields = {}
    for part in comment.split(';'):
        name, separator, value = part.strip().partition('=')
        if separator:
            fields[name.strip()] = value.strip()
    return fields


def number(fields, name):
    try:
        value = float(fields[name])
    except (KeyError, ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def channel_frequency(frequency):
    if not isinstance(frequency, (int, float)) or not math.isfinite(frequency):
        return None
    channel = round((frequency - 2412e6) / 5e6) + 1
    nominal = 2412e6 + (channel - 1) * 5e6
    return nominal if 1 <= channel <= 13 and abs(frequency - nominal) < 1e6 else frequency


def channel_label(frequency):
    if frequency is None:
        return 'unknown'
    channel = round((frequency - 2412e6) / 5e6) + 1
    if 1 <= channel <= 13 and frequency == 2412e6 + (channel - 1) * 5e6:
        return f'ch {channel:2d} ({frequency / 1e6:g} MHz)'
    return f'{frequency / 1e6:g} MHz'


class PacketStats:
    def __init__(self):
        self.packets = 0
        self.symbols = 0
        self.frame_bytes = 0
        self.encodings = Counter()
        self.sums = defaultdict(float)
        self.counts = Counter()

    def add(self, annotation):
        self.packets += 1
        self.symbols += annotation['core:sample_count']
        fields = comment_fields(annotation.get('core:comment', ''))
        length = number(fields, 'frame_bytes')
        if length is not None and length >= 0:
            self.frame_bytes += int(length)
        encoding = number(fields, 'encoding')
        if encoding is not None and encoding >= 0 and encoding.is_integer():
            self.encodings[int(encoding)] += 1
        for metric in METRICS:
            value = number(fields, metric)
            if value is not None:
                self.sums[metric] += value
                self.counts[metric] += 1

    def mean(self, metric):
        count = self.counts[metric]
        return f'{self.sums[metric] / count:.3g}' if count else '-'


def recordings(path):
    if path.is_dir():
        return sorted(path.rglob('*.sigmf-meta'))
    if path.suffix == '.sigmf-data':
        path = path.with_suffix('.sigmf-meta')
    return [path] if path.is_file() else []


def analyze(path):
    metadata = json.loads(path.read_text(encoding='utf-8'))
    validate.validate(metadata)
    data_path = path.with_suffix('.sigmf-data')
    has_data = data_path.is_file()
    record = (sigmffile.fromfile(path, skip_checksum=True) if has_data
              else sigmffile.SigMFFile(metadata=metadata))
    captures = record.get_captures()
    starts = [item['core:sample_start'] for item in captures]
    if starts != sorted(starts):
        raise ValueError('captures are not ordered by sample_start')
    total = PacketStats()
    channels = defaultdict(PacketStats)
    for annotation in record.get_annotations():
        if annotation.get('core:label') != 'wifi-frame':
            continue
        start = annotation['core:sample_start']
        count = annotation['core:sample_count']
        if start < 0 or count <= 0 or (has_data and start + count > len(record)):
            raise ValueError(f'frame annotation at sample {start} exceeds the dataset')
        capture_index = bisect_right(starts, start) - 1
        frequency = (captures[capture_index].get('core:frequency')
                     if capture_index >= 0 else None)
        frequency = channel_frequency(frequency)
        total.add(annotation)
        channels[frequency].add(annotation)
    return record.get_global_field('core:num_channels', 1), has_data, total, channels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='?', type=Path,
                        default=Path(__file__).resolve().parent / 'data',
                        help='SigMF metadata file, dataset file, or directory (default: collector data/)')
    args = parser.parse_args()
    paths = recordings(args.path)
    if not paths:
        parser.error(f'no SigMF metadata found at {args.path}')

    all_channels = defaultdict(PacketStats)
    total_packets = 0
    total_symbols = 0
    for path in paths:
        try:
            n_channels, has_data, total, channels = analyze(path)
        except (OSError, ValueError, KeyError, SigMFError, ValidationError) as error:
            parser.exit(1, f'{path}: {error}\n')
        print(f'{path}: {total.packets} packets, {total.symbols} complex symbols '
              f'per channel, {n_channels} channels'
              f'{"" if has_data else " (metadata only)"}')
        total_packets += total.packets
        total_symbols += total.symbols
        for frequency, stats in channels.items():
            aggregate = all_channels[frequency]
            aggregate.packets += stats.packets
            aggregate.symbols += stats.symbols
            aggregate.frame_bytes += stats.frame_bytes
            aggregate.encodings.update(stats.encodings)
            for metric in METRICS:
                aggregate.sums[metric] += stats.sums[metric]
                aggregate.counts[metric] += stats.counts[metric]

    print(f'\nTotal: {len(paths)} recordings, {total_packets} CRC-valid packets, '
          f'{total_symbols} complex symbols per channel ({total_symbols // 48} OFDM symbols)')
    if not total_packets:
        return
    print('Channel             Packets   Share   Symbols   Bytes   SNR dB   EVM %   BER   RX gain dB')
    for frequency in sorted(all_channels, key=lambda f: (f is None, f or 0)):
        stats = all_channels[frequency]
        print(f'{channel_label(frequency):19} {stats.packets:7d} '
              f'{stats.packets / total_packets:6.1%} {stats.symbols:9d} '
              f'{stats.frame_bytes:7d} {stats.mean("snr_db"):>8} '
              f'{stats.mean("evm_rms_percent"):>7} '
              f'{stats.mean("pre_fec_ber"):>6} {stats.mean("rx_gain_db"):>12}')

    encodings = Counter()
    for stats in all_channels.values():
        encodings.update(stats.encodings)
    if encodings:
        print('\nEncoding: ' + ', '.join(
            f'{encoding} ({RATES_MBPS[encoding]} Mbps): {count}'
            if 0 <= encoding < len(RATES_MBPS) else f'{encoding}: {count}'
            for encoding, count in sorted(encodings.items())))


if __name__ == '__main__':
    main()
