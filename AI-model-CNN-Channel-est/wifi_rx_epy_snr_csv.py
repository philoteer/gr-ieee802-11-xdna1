import csv
import threading
from pathlib import Path

import numpy as np
import pmt
from gnuradio import gr


class snr_csv(gr.basic_block):
    """One unbinned row per CRC-valid diagnostic PDU; no cross-frame joins."""

    def __init__(self, output_path='data/tmp_out/snr_ber_evm_cpu.csv',
                 estimator_label='LS'):
        gr.basic_block.__init__(self, name='SNR BER EVM CSV', in_sig=None, out_sig=None)
        self.output_path = output_path
        self.estimator_label = str(estimator_label)
        self.output_file = None
        self.writer = None
        self.frame_index = 0
        self.lock = threading.Lock()
        self.message_port_register_in(pmt.intern('diag'))
        self.set_msg_handler(pmt.intern('diag'), self._handle_diag)

    def start(self):
        output = Path(self.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.output_file = output.open('w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.output_file)
        self.frame_index = 0
        self.writer.writerow([
            'frame_index', 'estimator', 'frame_bytes', 'encoding', 'snr_db',
            'bit_errors', 'bits_compared', 'ber', 'evm_rms_percent', 'evm_rms_db',
            'evm_error_power', 'evm_reference_power',
        ])
        self.output_file.flush()
        return True

    @staticmethod
    def _metadata(metadata, key):
        value = pmt.dict_ref(metadata, pmt.intern(key), pmt.PMT_NIL)
        if pmt.is_uint64(value):
            return int(pmt.to_uint64(value))
        if pmt.is_real(value):
            return float(pmt.to_double(value))
        return ''

    def _handle_diag(self, message):
        if not pmt.is_pair(message):
            return
        metadata, diagnostic = pmt.car(message), pmt.cdr(message)
        if not pmt.is_dict(metadata) or not pmt.is_dict(diagnostic):
            return
        bits = []
        for key in ('pre_viterbi', 'corrected_coded'):
            value = pmt.dict_ref(diagnostic, pmt.intern(key), pmt.PMT_NIL)
            if not pmt.is_u8vector(value):
                return
            bits.append(np.asarray(pmt.u8vector_elements(value), dtype=np.uint8))
        received, reference = bits
        if (not received.size or received.size != reference.size or
                np.any(received > 1) or np.any(reference > 1)):
            return
        errors = int(np.count_nonzero(received != reference))
        evm_percent = evm_db = error_power = reference_power = ''
        tx = pmt.dict_ref(diagnostic, pmt.intern('transmitted_iq'), pmt.PMT_NIL)
        eq = pmt.dict_ref(diagnostic, pmt.intern('equalized_iq'), pmt.PMT_NIL)
        if pmt.is_c32vector(tx) and pmt.is_c32vector(eq):
            tx = np.asarray(pmt.c32vector_elements(tx), dtype=np.complex128)
            eq = np.asarray(pmt.c32vector_elements(eq), dtype=np.complex128)
            if (tx.size and tx.size % 48 == 0 and tx.size == eq.size and
                    np.all(np.isfinite(tx)) and np.all(np.isfinite(eq))):
                power = float(np.vdot(tx, tx).real)
                if power > 0:
                    reference_power = power
                    error_power = float(np.vdot(eq - tx, eq - tx).real)
                    ratio = float(np.sqrt(error_power / reference_power))
                    evm_percent = 100.0 * ratio
                    evm_db = float(20 * np.log10(ratio)) if ratio else float('-inf')
        with self.lock:
            if self.output_file is None:
                return
            self.frame_index += 1
            self.writer.writerow([
                self.frame_index, self.estimator_label,
                self._metadata(metadata, 'frame bytes'), self._metadata(metadata, 'encoding'),
                self._metadata(metadata, 'snr'), errors, int(received.size),
                errors / received.size, evm_percent, evm_db, error_power, reference_power,
            ])
            self.output_file.flush()

    def stop(self):
        with self.lock:
            if self.output_file is not None:
                self.output_file.close()
                self.output_file = None
        return True
