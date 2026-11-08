#!/usr/bin/env python3
"""Survey safe receive gains across the 2.4 GHz Wi-Fi channels.

The scanner measures signal power and near-full-scale samples for each
channel/gain pair, then recommends the highest gain that satisfies the given
limits.
"""

import argparse
from dataclasses import dataclass
import math
import time

from gnuradio import blocks, gr, uhd


@dataclass(frozen=True)
class Reading:
    """Measurement results for one channel and requested gain pair.

    Attributes:
        channel: Wi-Fi channel number.
        requested_gain: Gain requested from the USRP, in dB.
        actual_gain: Gain applied by the USRP, in dB.
        rms: Root-mean-square magnitude of the measured samples.
        near_full: Fraction of measurements near full scale.
        windows: Number of complete measurement windows.
    """

    channel: int
    requested_gain: float
    actual_gain: float
    rms: float
    near_full: float
    windows: int


def comma_separated_numbers(text, *, integer=False):
    """Parse and validate a comma-separated list of numbers.

    Args:
        text: Comma-separated numeric values.
        integer: Require integer values and restrict them to Wi-Fi channels
            1 through 13.

    Returns:
        A list of parsed integers or floating-point values.

    Raises:
        argparse.ArgumentTypeError: If a value is invalid, non-finite,
            duplicated, or outside the allowed channel range.
    """
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
    """Parse a strictly positive finite floating-point value.

    Args:
        text: Text representation of the value.

    Returns:
        The parsed floating-point value.

    Raises:
        argparse.ArgumentTypeError: If the value is invalid, non-finite, or
            not positive.
    """
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('must be positive and finite')
    return value


def nonnegative_float(text):
    """Parse a finite floating-point value that is not negative.

    Args:
        text: Text representation of the value.

    Returns:
        The parsed floating-point value.

    Raises:
        argparse.ArgumentTypeError: If the value is invalid, non-finite, or
            negative.
    """
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError('must be nonnegative and finite')
    return value


class GainScanner:
    """Run gain measurements using a GNU Radio USRP source flowgraph."""

    def __init__(self, device_args, antenna, sample_rate, decimation=128, window=128):
        """Configure the USRP and measurement flowgraph.

        Args:
            device_args: UHD device argument string.
            antenna: Receive antenna name.
            sample_rate: Input sample rate in samples per second.
            decimation: Keep one sample out of every this many input samples.
            window: Number of decimated samples integrated per measurement
                window.
        """
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
        """Measure one channel at one requested receive gain.

        Args:
            channel: Wi-Fi channel number.
            gain: Requested receive gain in dB.
            settle: Seconds to wait after changing the gain.
            duration: Seconds over which to collect measurements.

        Returns:
            A ``Reading`` containing power and near-full-scale statistics.

        Raises:
            RuntimeError: If no complete measurement windows are collected.
        """
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
        """Measure every requested channel and gain combination.

        Args:
            channels: Iterable of Wi-Fi channel numbers.
            gains: Iterable of requested gains in dB.
            settle: Seconds to wait after changing the gain.
            duration: Seconds to measure each combination.

        Returns:
            A list of ``Reading`` objects in scan order.

        Raises:
            ValueError: If a requested gain is outside the USRP's supported
                range.
            RuntimeError: If a measurement produces no data.
        """
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
    """Select the highest gain satisfying both safety limits per channel.

    Args:
        readings: Measurements returned by ``GainScanner.scan``.
        channels: Channels for which recommendations are required.
        gains: Tested gain values in dB.
        max_near_full: Maximum allowed near-full-scale fraction.
        max_rms: Maximum allowed RMS magnitude.

    Returns:
        A mapping from channel number to the recommended gain, or ``None`` if
        no tested gain is safe for that channel.
    """
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
    """Parse command-line arguments, scan the USRP, and print recommendations."""
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
