/*
 * Copyright 2026 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */
#include "pydoc_macros.h"
#define D(...) DOC(gr, ieee802_11, __VA_ARGS__)
/*
  This file contains placeholders for docstrings for the Python bindings.
  Do not edit! These were automatically extracted during the binding process
  and will be overwritten during the build process
 */



 static const char *__doc_gr_ieee802_11_decode_mac_EVM = R"doc(
Decode IEEE 802.11 MAC frames and publish coded bits and IQ vectors.

For FCS-valid frames of at least two OFDM symbols, the `diag` port publishes a
PDU whose payload is a dictionary containing `pre_viterbi` and `corrected_coded`
u8vectors, plus `transmitted_iq`, `received_iq`, and `equalized_iq` complex
vectors when IQ data is available. `pre_viterbi` contains hard-decision coded
bits after deinterleaving, and `corrected_coded` contains the Viterbi-decoded
bits re-encoded and punctured at the frame's original rate. Both vectors have
the same bit order and length: (number of OFDM symbols - 1) * coded bits per
OFDM symbol. The final symbol is omitted to exclude padding bits. The
re-encoded tail is reset to zero because it may start before that symbol. The
IQ vectors are flattened, with the final OFDM symbol omitted. `transmitted_iq`
and `equalized_iq` contain 48 data tones per symbol; `received_iq` is the frame
equalizer's `unequalized_iq` data and contains all 52 occupied tones, including
pilots. Failed frames produce no messages; single-symbol frames produce only an
`out` PDU. `out` publishes CRC-valid MAC payload PDUs; no EVM is calculated.
)doc";


 static const char *__doc_gr_ieee802_11_decode_mac_EVM_decode_mac_EVM = R"doc(
Create a decoder for 48-byte OFDM symbol items.
)doc";


 static const char *__doc_gr_ieee802_11_decode_mac_EVM_make = R"doc(
Create a decoder.

Args:
    log: Enable logging.
    debug: Enable decoder debug output.
)doc";

  
