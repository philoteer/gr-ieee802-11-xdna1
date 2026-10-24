#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Step1
# GNU Radio version: 3.10.9.2

from gnuradio import blocks
import pmt
from gnuradio import blocks, gr
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import gr, blocks
from gnuradio import mlir_aie
import step1_epy_block_0 as epy_block_0  # embedded python block




class step1(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Step1", catch_exceptions=True)

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

        self.mlir_aie_complex64_to_cint16_0 = mlir_aie.complex64_to_cint16()
        self.mlir_aie_cint16_to_complex64_0 = mlir_aie.cint16_to_complex64()
        self.epy_block_0 = epy_block_0.mlir_aie_python_int32(path_xclbin="../build/final.xclbin", path_insts_bin="../build/insts.bin", VECTOR_SIZE=4096)
        self.blocks_probe_rate_0 = blocks.probe_rate(gr.sizeof_gr_complex*1, 500.0, 0.15, '')
        self.blocks_message_debug_0 = blocks.message_debug(True, gr.log_levels.info)
        self.blocks_interleaved_char_to_complex_0 = blocks.interleaved_char_to_complex(False,127.0)
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_char*1, '/home/piloteer/바탕화면/wifi_iq_hackrf.bin', False, 0, 0)
        self.blocks_file_source_0.set_begin_tag(pmt.PMT_NIL)
        self.blocks_file_meta_sink_0 = blocks.file_meta_sink(gr.sizeof_gr_complex*1, './out.iqdata', samp_rate, 1, blocks.GR_FILE_FLOAT, True, 1000000, pmt.make_dict(), True)
        self.blocks_file_meta_sink_0.set_unbuffered(True)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.blocks_probe_rate_0, 'rate'), (self.blocks_message_debug_0, 'print'))
        self.connect((self.blocks_file_source_0, 0), (self.blocks_interleaved_char_to_complex_0, 0))
        self.connect((self.blocks_interleaved_char_to_complex_0, 0), (self.blocks_probe_rate_0, 0))
        self.connect((self.blocks_interleaved_char_to_complex_0, 0), (self.mlir_aie_complex64_to_cint16_0, 0))
        self.connect((self.epy_block_0, 0), (self.mlir_aie_cint16_to_complex64_0, 0))
        self.connect((self.mlir_aie_cint16_to_complex64_0, 0), (self.blocks_file_meta_sink_0, 0))
        self.connect((self.mlir_aie_complex64_to_cint16_0, 0), (self.epy_block_0, 0))


    def get_window_size(self):
        return self.window_size

    def set_window_size(self, window_size):
        self.window_size = window_size

    def get_sync_length(self):
        return self.sync_length

    def set_sync_length(self, sync_length):
        self.sync_length = sync_length

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq

    def get_chan_est(self):
        return self.chan_est

    def set_chan_est(self, chan_est):
        self.chan_est = chan_est




def main(top_block_cls=step1, options=None):
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
