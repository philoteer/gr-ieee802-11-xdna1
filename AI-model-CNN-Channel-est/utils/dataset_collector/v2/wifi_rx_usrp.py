#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Wifi Rx Usrp
# GNU Radio version: 3.10.9.2

from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import mlir_aie
from gnuradio import uhd
import time
import ieee802_11
import wifi_rx_usrp_epy_xy_collector as epy_xy_collector  # embedded python block


def snipfcn_attach_collector_radio(self):
    self.epy_xy_collector.radio = self.uhd_usrp_source_0

def snipfcn_start_channel_sweep(self):
    import math
    import random
    import threading
    import time
    self.channel_sweep_stop = threading.Event()
    if len(self.initial_gains) != 13:
        raise ValueError('initial_gains must contain 13 channel values')
    self.channel_gain_lut = {channel: self.initial_gains[channel - 1] for channel in range(1, 14)}
    gain_range = self.uhd_usrp_source_0.get_gain_range(0)

    def observe(duration):
        warmup = min(max(0.0, float(self.gain_settle_s)), duration / 3)
        if self.channel_sweep_stop.wait(warmup):
            return None
        deadline = time.monotonic() + duration - warmup
        power_sum = 0.0
        near_full_sum = 0.0
        count = 0
        while not self.channel_sweep_stop.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if self.channel_sweep_stop.wait(min(0.1, remaining)):
                return None
            power = self.blocks_probe_power.level() / 2048
            near_full = self.blocks_probe_clip.level() / 2048
            if math.isfinite(power) and power > 0 and 0 <= near_full <= 1:
                power_sum += power
                near_full_sum += near_full
                count += 1
        if count == 0:
            return None
        return math.sqrt(power_sum / count), near_full_sum / count, count

    def sweep():
        while not self.channel_sweep_stop.is_set():
            for channel in range(1, 14):
                if self.channel_sweep_stop.is_set():
                    return
                frequency = 2.412e9 + (channel - 1) * 5e6
                dwell = max(0.02, float(self.popular_dwell_s if channel in (1, 5, 9, 13)
                                        else self.other_dwell_s))
                self.epy_xy_collector.mute_for(max(0.0, float(self.gain_settle_s)))
                if self.get_freq() != frequency:
                    self.set_freq(frequency)
                self.set_gain(self.channel_gain_lut[channel])
                self.epy_xy_collector.mute_for(max(0.0, float(self.gain_settle_s)))
                print('Wi-Fi channel {}: {:.0f} MHz, gain {:.1f} dB, dwell {} s'.format(
                    channel, frequency / 1e6, self.uhd_usrp_source_0.get_gain(0), dwell),
                    flush=True)
                level = observe(dwell / 2)
                if level is not None:
                    rms, near_full, count = level
                    if rms > 0 and math.isfinite(rms):
                        delta = 6.0 if rms < 0.14 else (-6.0 if rms > 0.28 else 0.0)
                        if near_full > 1e-4:
                            delta = min(delta, -6.0)
                        noise = random.gauss(0, max(0.0, float(self.gain_noise_std_db)))
                        candidate = max(gain_range.start(), min(gain_range.stop(),
                            self.channel_gain_lut[channel] + delta + noise))
                        self.epy_xy_collector.mute_for(max(0.0, float(self.gain_settle_s)))
                        self.set_gain(candidate)
                        self.epy_xy_collector.mute_for(max(0.0, float(self.gain_settle_s)))
                        self.channel_gain_lut[channel] = self.uhd_usrp_source_0.get_gain(0)
                        print('Wi-Fi channel {}: rms {:.3f}, near full scale {:.4%} of {} polls, gain {:.1f} dB'.format(
                            channel, rms, near_full, count, self.channel_gain_lut[channel]),
                            flush=True)
                if self.channel_sweep_stop.wait(dwell / 2):
                    return

    self.channel_sweep_thread = threading.Thread(target=sweep, daemon=True)
    self.channel_sweep_thread.start()

def snipfcn_stop_channel_sweep(self):
    self.channel_sweep_stop.set()
    self.channel_sweep_thread.join(timeout=1)


def snippets_main_after_init(tb):
    snipfcn_attach_collector_radio(tb)

def snippets_main_after_start(tb):
    snipfcn_start_channel_sweep(tb)

def snippets_main_after_stop(tb):
    snipfcn_stop_channel_sweep(tb)


class wifi_rx_usrp(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Wifi Rx Usrp", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        self.initial_gains = initial_gains = [30, 62, 62, 62, 50, 32, 32, 32, 32, 32, 68, 62, 56]
        self.samp_rate = samp_rate = 20e6
        self.popular_dwell_s = popular_dwell_s = 180
        self.other_dwell_s = other_dwell_s = 60
        self.gain_settle_s = gain_settle_s = 1.0
        self.gain_noise_std_db = gain_noise_std_db = 2
        self.gain = gain = initial_gains[0]
        self.freq = freq = 2.412e9

        ##################################################
        # Blocks
        ##################################################

        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join(("type=b200", '')),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
        )
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        self.uhd_usrp_source_0.set_time_unknown_pps(uhd.time_spec(0))

        self.uhd_usrp_source_0.set_center_freq(freq, 0)
        self.uhd_usrp_source_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_source_0.set_bandwidth(samp_rate, 0)
        self.uhd_usrp_source_0.set_rx_agc(False, 0)
        self.uhd_usrp_source_0.set_gain(gain, 0)
        self.uhd_usrp_source_0.set_min_output_buffer(1048576)
        self.mlir_aie_mlir_aie_cpp_tagged_int32_to_int64_0 = mlir_aie.mlir_aie_cpp_tagged_int32_to_int64("../../test/until_fft/kernel/build/final.xclbin", "../../test/until_fft/kernel/build/insts.bin", 'MLIR_AIE', 16384, 2, 16)
        self.mlir_aie_complex64_to_cint16_0 = mlir_aie.complex64_to_cint16(False)
        self.mlir_aie_complex64_to_cint16_0.set_min_output_buffer(1048576)
        self.mlir_aie_cint32_to_complex64_0 = mlir_aie.cint32_to_complex64(16, 15)
        self.ieee802_11_frame_equalizer_debug_0 = ieee802_11.frame_equalizer_debug(ieee802_11.LS, freq, samp_rate, False, False)
        self.ieee802_11_decode_mac_EVM_0 = ieee802_11.decode_mac_EVM(False, False)
        self.epy_xy_collector = epy_xy_collector.xy_collector(output_path="data/xy_usrp", report_interval=25, center_frequency=freq, sample_rate=samp_rate, gain=gain)
        self.blocks_threshold_clip = blocks.threshold_ff((0.95 * 0.95), (0.95 * 0.95), 0)
        self.blocks_stream_to_vector_0_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, 64)
        self.blocks_probe_power = blocks.probe_signal_f()
        self.blocks_probe_clip = blocks.probe_signal_f()
        self.blocks_keep_one_in_n_0 = blocks.keep_one_in_n(gr.sizeof_gr_complex*1, 4)
        self.blocks_integrate_power = blocks.integrate_ff(2048, 1)
        self.blocks_integrate_clip = blocks.integrate_ff(2048, 1)
        self.blocks_complex_to_mag_squared_0 = blocks.complex_to_mag_squared(1)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.ieee802_11_decode_mac_EVM_0, 'diag'), (self.epy_xy_collector, 'diag'))
        self.connect((self.blocks_complex_to_mag_squared_0, 0), (self.blocks_integrate_power, 0))
        self.connect((self.blocks_complex_to_mag_squared_0, 0), (self.blocks_threshold_clip, 0))
        self.connect((self.blocks_integrate_clip, 0), (self.blocks_probe_clip, 0))
        self.connect((self.blocks_integrate_power, 0), (self.blocks_probe_power, 0))
        self.connect((self.blocks_keep_one_in_n_0, 0), (self.blocks_complex_to_mag_squared_0, 0))
        self.connect((self.blocks_stream_to_vector_0_0, 0), (self.ieee802_11_frame_equalizer_debug_0, 0))
        self.connect((self.blocks_threshold_clip, 0), (self.blocks_integrate_clip, 0))
        self.connect((self.ieee802_11_frame_equalizer_debug_0, 0), (self.ieee802_11_decode_mac_EVM_0, 0))
        self.connect((self.mlir_aie_cint32_to_complex64_0, 0), (self.blocks_stream_to_vector_0_0, 0))
        self.connect((self.mlir_aie_complex64_to_cint16_0, 0), (self.mlir_aie_mlir_aie_cpp_tagged_int32_to_int64_0, 0))
        self.connect((self.mlir_aie_mlir_aie_cpp_tagged_int32_to_int64_0, 0), (self.mlir_aie_cint32_to_complex64_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_keep_one_in_n_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.mlir_aie_complex64_to_cint16_0, 0))


    def get_initial_gains(self):
        return self.initial_gains

    def set_initial_gains(self, initial_gains):
        self.initial_gains = initial_gains
        self.set_gain(self.initial_gains[0])

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.ieee802_11_frame_equalizer_debug_0.set_bandwidth(self.samp_rate)
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)
        self.uhd_usrp_source_0.set_bandwidth(self.samp_rate, 0)

    def get_popular_dwell_s(self):
        return self.popular_dwell_s

    def set_popular_dwell_s(self, popular_dwell_s):
        self.popular_dwell_s = popular_dwell_s

    def get_other_dwell_s(self):
        return self.other_dwell_s

    def set_other_dwell_s(self, other_dwell_s):
        self.other_dwell_s = other_dwell_s

    def get_gain_settle_s(self):
        return self.gain_settle_s

    def set_gain_settle_s(self, gain_settle_s):
        self.gain_settle_s = gain_settle_s

    def get_gain_noise_std_db(self):
        return self.gain_noise_std_db

    def set_gain_noise_std_db(self, gain_noise_std_db):
        self.gain_noise_std_db = gain_noise_std_db

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain = gain
        self.epy_xy_collector.gain = self.gain
        self.uhd_usrp_source_0.set_gain(self.gain, 0)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.epy_xy_collector.center_frequency = self.freq
        self.ieee802_11_frame_equalizer_debug_0.set_frequency(self.freq)
        self.uhd_usrp_source_0.set_center_freq(self.freq, 0)




def main(top_block_cls=wifi_rx_usrp, options=None):
    tb = top_block_cls()
    snippets_main_after_init(tb)
    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        snippets_main_after_stop(tb)
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    snippets_main_after_start(tb)
    tb.wait()
    snippets_main_after_stop(tb)

if __name__ == '__main__':
    main()
