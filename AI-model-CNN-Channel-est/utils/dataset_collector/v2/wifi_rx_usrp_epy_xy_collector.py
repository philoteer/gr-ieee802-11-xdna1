import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import threading
import time

import numpy as np
import pmt
from gnuradio import gr


PILOT_POLARITY = (
    1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1,
    -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1,
    1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1,
    -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1,
    -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1,
    -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1,
    -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1,
    -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1,
)


class xy_collector(gr.basic_block):
    def __init__(self, output_path='data/xy_usrp', report_interval=25,
                 center_frequency=2.417e9, sample_rate=20e6, gain=None):
        gr.basic_block.__init__(self, name='xy_collector', in_sig=[], out_sig=[])
        self.output_base = Path(output_path)
        self.data_path = None
        self.meta_path = None
        self.data_file = None
        self.report_interval = max(1, int(report_interval))
        self.frames = 0
        self.sample_count = 0
        self.metadata = {
            'global': {
                'core:datatype': 'cf32_le',
                'core:num_channels': 2,
                'core:version': '1.2.6',
                'core:recorder': 'wifi_rx_usrp xy_collector',
                'core:hw': 'USRP B210 (UHD)',
                'core:description': (
                    'CRC-valid Wi-Fi payload symbols; channel 0 unequalized received, '
                    'channel 1 transmitted ground truth; 52 occupied subcarriers per OFDM symbol.\n'
                    'RF center frequency: {} Hz\nADC sample rate: {} samples/s\n'
                    'RX antenna: TX/RX\nRequested RX gain: {} dB'.format(
                        center_frequency, sample_rate, gain)),
            },
            'captures': [],
            'annotations': [],
        }
        self.center_frequency = float(center_frequency)
        self.gain = float(gain) if gain is not None else None
        self.radio = None
        self.lock = threading.Lock()
        self.drop_until = 0.0
        self.message_port_register_in(pmt.intern('diag'))
        self.set_msg_handler(pmt.intern('diag'), self._collect)

    def mute_for(self, seconds):
        with self.lock:
            self.drop_until = max(self.drop_until, time.monotonic() + seconds)

    def _open_recording(self):
        self.output_base.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        base = self.output_base.with_name(self.output_base.name + '-' + stamp)
        self.data_path = base.with_suffix('.sigmf-data')
        self.meta_path = base.with_suffix('.sigmf-meta')
        self.data_file = self.data_path.open('xb')

    def _save_metadata(self):
        temporary = self.meta_path.with_name(self.meta_path.name + '.tmp')
        with temporary.open('w', encoding='utf-8') as output:
            json.dump(self.metadata, output, indent=2)
            output.write('\n')
        os.replace(temporary, self.meta_path)

    @staticmethod
    def _field(metadata, name):
        value = pmt.dict_ref(metadata, pmt.intern(name), pmt.PMT_NIL)
        return int(pmt.to_uint64(value)) if pmt.is_uint64(value) else None

    @staticmethod
    def _real_field(metadata, name):
        value = pmt.dict_ref(metadata, pmt.intern(name), pmt.PMT_NIL)
        if pmt.is_real(value):
            number = float(pmt.to_double(value))
            return number if math.isfinite(number) else None
        return None

    def _collect(self, message):
        if time.monotonic() < self.drop_until or not pmt.is_pair(message):
            return
        metadata, iq = pmt.car(message), pmt.cdr(message)
        if not pmt.is_dict(metadata) or not pmt.is_dict(iq):
            return
        vectors = []
        for name in ('received_iq', 'equalized_iq', 'transmitted_iq'):
            value = pmt.dict_ref(iq, pmt.intern(name), pmt.PMT_NIL)
            if not pmt.is_c32vector(value):
                return
            vectors.append(np.asarray(pmt.c32vector_elements(value), dtype=np.complex64))
        unequalized, y, x = vectors
        frame_bytes = self._field(metadata, 'frame bytes')
        encoding = self._field(metadata, 'encoding')
        if (frame_bytes is None or encoding is None or x.size == 0 or
                x.size % 48 or x.size != y.size or unequalized.size != x.size // 48 * 52 or
                not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(unequalized))):
            return
        expected = np.empty((x.size // 48, 52), dtype=np.complex64)
        expected[:, [i for i in range(52) if i not in (5, 19, 32, 46)]] = x.reshape(-1, 48)
        polarity = np.asarray([PILOT_POLARITY[(n - 2) % 127] for n in range(3, 3 + expected.shape[0])], dtype=np.float32)
        expected[:, (5, 19, 32, 46)] = polarity[:, None] * np.array([1, 1, 1, -1], dtype=np.float32)
        samples = np.empty((unequalized.size, 2), dtype='<c8')
        samples[:, 0] = unequalized
        samples[:, 1] = expected.ravel()
        frequency = self._real_field(metadata, 'nominal frequency')
        if frequency is None or frequency <= 0:
            frequency = self.center_frequency
        diagnostics = ['frame_bytes={}', 'encoding={}']
        values = [frame_bytes, encoding]
        # MAC addresses live in the diag payload, not the PDU metadata.
        for name in ('src_mac', 'dst_mac'):
            address = pmt.dict_ref(iq, pmt.intern(name), pmt.PMT_NIL)
            if pmt.is_u8vector(address):
                octets = pmt.u8vector_elements(address)
                if len(octets) == 6:
                    diagnostics.append(name + '={}')
                    values.append(':'.join('{:02x}'.format(byte) for byte in octets))
        for name, label in (('snr', 'snr_db'),
                            ('frequency offset', 'frequency_offset_hz'),
                            ('beta', 'pilot_phase_rad')):
            value = self._real_field(metadata, name)
            if value is not None:
                diagnostics.append(label + '={}')
                values.append(value)
        error = y.astype(np.complex128) - x
        error_power = float(np.vdot(error, error).real)
        reference_power = float(np.vdot(x.astype(np.complex128), x).real)
        if reference_power > 0:
            diagnostics.append('evm_rms_percent={}')
            values.append(100.0 * math.sqrt(error_power / reference_power))
        received_bits = pmt.dict_ref(iq, pmt.intern('pre_viterbi'), pmt.PMT_NIL)
        corrected_bits = pmt.dict_ref(iq, pmt.intern('corrected_coded'), pmt.PMT_NIL)
        if pmt.is_u8vector(received_bits) and pmt.is_u8vector(corrected_bits):
            received = np.asarray(pmt.u8vector_elements(received_bits), dtype=np.uint8)
            corrected = np.asarray(pmt.u8vector_elements(corrected_bits), dtype=np.uint8)
            if (received.size and received.size == corrected.size and
                    np.all(received <= 1) and np.all(corrected <= 1)):
                diagnostics.append('pre_fec_ber={}')
                values.append(float(np.count_nonzero(received != corrected) / received.size))
        with self.lock:
            if time.monotonic() < self.drop_until:
                return
            gain = self.radio.get_gain(0) if self.radio is not None else self.gain
            if self.data_file is None:
                self._open_recording()
            self.data_file.write(samples.tobytes())
            self.data_file.flush()
            self.metadata['captures'].append({
                'core:sample_start': self.sample_count,
                'core:frequency': frequency,
            })
            self.metadata['annotations'].append({
                'core:sample_start': self.sample_count,
                'core:sample_count': int(samples.shape[0]),
                'core:label': 'wifi-frame',
                'core:comment': '; '.join(diagnostics + ['rx_gain_db={}']).format(*(values + [gain])),
            })
            self.sample_count += samples.shape[0]
            self._save_metadata()
            self.frames += 1
            if self.frames == 1 or self.frames % self.report_interval == 0:
                print('Collected {} CRC-valid Wi-Fi frames ({} symbols): {}'.format(
                    self.frames, samples.shape[0], self.data_path), flush=True)

    def stop(self):
        with self.lock:
            if self.data_file is not None:
                self.data_file.close()
                self.data_file = None
        return True