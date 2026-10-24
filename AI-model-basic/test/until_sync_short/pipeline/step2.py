#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Step2
# GNU Radio version: 3.10.9.2

from gnuradio import blocks
from gnuradio import blocks, gr
from gnuradio import fft
from gnuradio.fft import window
from gnuradio import gr
from gnuradio.filter import firdes
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import gr, blocks
import foo
import ieee802_11




class step2(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Step2", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        self.window_size = window_size = 48
        self.sync_length = sync_length = 320
        self.samp_rate = samp_rate = 20e6
        self.freq = freq = 2.417e9
        self.chan_est = chan_est = 0

        ##################################################
        # Blocks
        ##################################################

        self.ieee802_11_sync_long_0 = ieee802_11.sync_long(sync_length, False, False)
        self.ieee802_11_parse_mac_0 = ieee802_11.parse_mac(False, True)
        self.ieee802_11_frame_equalizer_0 = ieee802_11.frame_equalizer(ieee802_11.Equalizer(chan_est), freq, samp_rate, False, False)
        self.ieee802_11_decode_mac_0 = ieee802_11.decode_mac(True, False)
        self.foo_wireshark_connector_0 = foo.wireshark_connector(127, False)
        self.fft_vxx_0 = fft.fft_vcc(64, True, window.rectangular(64), True, 1)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, 64)
        self.blocks_probe_rate_0 = blocks.probe_rate(gr.sizeof_gr_complex*1, 500.0, 0.15, '')
        self.blocks_message_debug_0 = blocks.message_debug(True, gr.log_levels.info)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_char*1, './out.pcap', False)
        self.blocks_file_sink_0.set_unbuffered(True)
        self.blocks_file_meta_source_0 = blocks.file_meta_source('./out.iqdata', False, True, './out.iqdata.hdr')
        self.blocks_tag_gate_rx_time = blocks.tag_gate(gr.sizeof_gr_complex*1, False)
        self.blocks_tag_gate_rx_time.set_single_key('rx_time')
        self.blocks_tag_gate_rx_rate = blocks.tag_gate(gr.sizeof_gr_complex*1, False)
        self.blocks_tag_gate_rx_rate.set_single_key('rx_rate')
        self.blocks_delay_0 = blocks.delay(gr.sizeof_gr_complex*1, sync_length)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.blocks_probe_rate_0, 'rate'), (self.blocks_message_debug_0, 'print'))
        self.msg_connect((self.ieee802_11_decode_mac_0, 'out'), (self.foo_wireshark_connector_0, 'in'))
        self.msg_connect((self.ieee802_11_decode_mac_0, 'out'), (self.ieee802_11_parse_mac_0, 'in'))
        self.connect((self.blocks_delay_0, 0), (self.ieee802_11_sync_long_0, 1))
        self.connect((self.blocks_file_meta_source_0, 0), (self.blocks_delay_0, 0))
        self.connect((self.blocks_file_meta_source_0, 0), (self.blocks_probe_rate_0, 0))
        self.connect((self.blocks_file_meta_source_0, 0), (self.blocks_tag_gate_rx_time, 0))
        self.connect((self.blocks_tag_gate_rx_time, 0), (self.blocks_tag_gate_rx_rate, 0))
        self.connect((self.blocks_tag_gate_rx_rate, 0), (self.ieee802_11_sync_long_0, 0))
        self.connect((self.blocks_stream_to_vector_0, 0), (self.fft_vxx_0, 0))
        self.connect((self.fft_vxx_0, 0), (self.ieee802_11_frame_equalizer_0, 0))
        self.connect((self.foo_wireshark_connector_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.ieee802_11_frame_equalizer_0, 0), (self.ieee802_11_decode_mac_0, 0))
        self.connect((self.ieee802_11_sync_long_0, 0), (self.blocks_stream_to_vector_0, 0))


    def get_window_size(self):
        return self.window_size

    def set_window_size(self, window_size):
        self.window_size = window_size

    def get_sync_length(self):
        return self.sync_length

    def set_sync_length(self, sync_length):
        self.sync_length = sync_length
        self.blocks_delay_0.set_dly(int(self.sync_length))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.ieee802_11_frame_equalizer_0.set_bandwidth(self.samp_rate)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.ieee802_11_frame_equalizer_0.set_frequency(self.freq)

    def get_chan_est(self):
        return self.chan_est

    def set_chan_est(self, chan_est):
        self.chan_est = chan_est
        self.ieee802_11_frame_equalizer_0.set_algorithm(ieee802_11.Equalizer(self.chan_est))




def main(top_block_cls=step2, options=None):
    tb = top_block_cls()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()

    tb.wait()


if __name__ == '__main__':
    main()
