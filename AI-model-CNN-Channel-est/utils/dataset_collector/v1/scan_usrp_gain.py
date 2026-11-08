#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
import math
import time

from gnuradio import blocks, gr, uhd


@dataclass(frozen=True)
class Reading:
    channel: int
    requested_gain: float
    actual_gain: float
    rms: float
    near_full: float
    windows: int


def comma_separated_numbers(text, *, integer=False):
    try:
        values = [int(part) if integer else float(part) for part in text.split(',')]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not values or any(not math.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError('provide a comma-separated list of finite numbers')
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError('duplicate values are not allowed')
    if integer and any(value < 1 or value > 13 for value in values):
        raise argparse.ArgumentTypeError('Wi-Fi channels must be between 1 and 13')
    return values


def positive_float(text):
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('must be positive and finite')
    return value


def nonnegative_float(text):
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError('must be nonnegative and finite')
    return value


class GainScanner:
    def __init__(self, device_args, antenna, sample_rate, decimation=128, window=128):
        self.radio = uhd.usrp_source(
            device_args, uhd.stream_args(cpu_format='fc32', channels=[0]))
        self.radio.set_samp_rate(sample_rate)
        self.radio.set_antenna(antenna, 0)
        self.radio.set_bandwidth(sample_rate, 0)
        self.radio.set_rx_agc(False, 0)
        self.radio.set_center_freq(2412e6, 0)

        self.graph = gr.top_block('Wi-Fi channel gain survey')
        keep = blocks.keep_one_in_n(gr.sizeof_gr_complex, decimation)
        magnitude = blocks.complex_to_mag_squared(1)
        power_sum = blocks.integrate_ff(window, 1)
        near_full = blocks.threshold_ff(0.95 * 0.95, 0.95 * 0.95, 0)
        near_full_sum = blocks.integrate_ff(window, 1)
        self.power = blocks.vector_sink_f()
        self.near_full = blocks.vector_sink_f()
        self.window = window
        self.current_channel = None
        self.graph.connect(self.radio, keep, magnitude, power_sum, self.power)
        self.graph.connect(magnitude, near_full, near_full_sum, self.near_full)

    def measure(self, channel, gain, settle, duration):
        if channel != self.current_channel:
            self.radio.set_center_freq(2412e6 + (channel - 1) * 5e6, 0)
            self.current_channel = channel
        self.radio.set_gain(gain, 0)
        time.sleep(settle)
        self.power.reset()
        self.near_full.reset()
        time.sleep(duration)
        powers = self.power.data()
        peaks = self.near_full.data()
        count = min(len(powers), len(peaks))
        if not count:
            raise RuntimeError(f'no I/Q measurements on channel {channel} at {gain} dB')
        average_power = sum(powers[:count]) / (count * self.window)
        near_full_fraction = sum(peaks[:count]) / (count * self.window)
        return Reading(channel, gain, self.radio.get_gain(0),
                       math.sqrt(average_power), near_full_fraction, count)

    def scan(self, channels, gains, settle, duration):
        gain_range = self.radio.get_gain_range(0)
        if any(gain < gain_range.start() or gain > gain_range.stop() for gain in gains):
            raise ValueError(f'gains must be within {gain_range.start():g}–'
                             f'{gain_range.stop():g} dB for this USRP')
        readings = []
        self.graph.start()
        try:
            for channel in channels:
                for gain in gains:
                    reading = self.measure(channel, gain, settle, duration)
                    readings.append(reading)
                    print(f'ch {channel:2d} ({2412 + 5 * (channel - 1)} MHz)  '
                          f'gain {reading.actual_gain:5.1f} dB  '
                          f'RMS {reading.rms:.4f}  '
                          f'near full scale {reading.near_full:.3%}  '
                          f'windows {reading.windows}', flush=True)
        finally:
            self.graph.stop()
            self.graph.wait()
        return readings


def recommend(readings, channels, gains, max_near_full, max_rms):
    by_pair = {(reading.channel, reading.requested_gain): reading for reading in readings}
    recommendations = {}
    for channel in channels:
        safe = [gain for gain in gains if
                (channel, gain) in by_pair and
                by_pair[channel, gain].near_full <= max_near_full and
                by_pair[channel, gain].rms <= max_rms]
        recommendations[channel] = max(safe) if safe else None
    return recommendations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device-args', default='type=b200')
    parser.add_argument('--antenna', default='TX/RX')
    parser.add_argument('--sample-rate', type=positive_float, default=20e6)
    parser.add_argument('--channels', type=lambda s: comma_separated_numbers(s, integer=True),
                        default=list(range(1, 14)))
    parser.add_argument('--gains', type=comma_separated_numbers,
                        default=[26, 32, 38, 44, 50, 56, 62, 68])
    parser.add_argument('--settle', type=nonnegative_float, default=0.15)
    parser.add_argument('--measure', type=positive_float, default=0.6)
    parser.add_argument('--max-near-full', type=nonnegative_float, default=0.001)
    parser.add_argument('--max-rms', type=positive_float, default=0.3)
    args = parser.parse_args()
    scanner = GainScanner(args.device_args, args.antenna, args.sample_rate)
    try:
        readings = scanner.scan(args.channels, args.gains, args.settle, args.measure)
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f'{error}\n')
    except KeyboardInterrupt:
        parser.exit(130, '\nScan interrupted\n')

    suggested = recommend(readings, args.channels, args.gains,
                          args.max_near_full, args.max_rms)
    print(f'\nScreening rule: near-full-scale fraction ≤ {args.max_near_full:.3%}, '
          f'RMS ≤ {args.max_rms:g} on every scanned channel.')
    for channel in args.channels:
        gain = suggested[channel]
        if gain is None:
            print(f'Channel {channel}: no tested gain met the limits.')
        else:
            print(f'Channel {channel}: recommended initial gain {gain:g} dB.')

    if all(suggested.get(channel) is not None for channel in range(1, 14)):
        values = ', '.join(f'{suggested[channel]:g}' for channel in range(1, 14))
        print('\nSet initial_gains in usrp.grc or usrp-v2.grc to:')
        print(f'[{values}]')
    else:
        print('\nScan all 13 Wi-Fi channels and ensure each has a safe tested gain '
              'to generate a complete initial_gains array.')


if __name__ == '__main__':
    main()
