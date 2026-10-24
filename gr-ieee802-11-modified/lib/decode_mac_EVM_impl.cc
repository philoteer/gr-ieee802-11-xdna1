#include <ieee802_11/decode_mac_EVM.h>
#include <ieee802_11/constellations.h>

#include "utils.h"
#include "viterbi_decoder/viterbi_decoder.h"

#include <gnuradio/io_signature.h>
#include <boost/crc.hpp>
#include <iomanip>
#include <vector>

using namespace gr::ieee802_11;

#define LINKTYPE_IEEE802_11 105 /* http://www.tcpdump.org/linktypes.html */

class decode_mac_EVM_impl : public decode_mac_EVM
{

public:
    decode_mac_EVM_impl(bool log, bool debug)
        : block("decode_mac_EVM",
                gr::io_signature::make(1, 1, 48),
                gr::io_signature::make(0, 0, 0)),
          d_log(log),
          d_debug(debug),
          d_ofdm(BPSK_1_2),
          d_frame(d_ofdm, 0),
          d_frame_complete(true)
    {
        message_port_register_out(pmt::mp("out"));
        message_port_register_out(pmt::mp("diag"));

        d_bpsk = constellation_bpsk::make();
        d_qpsk = constellation_qpsk::make();
        d_16qam = constellation_16qam::make();
        d_64qam = constellation_64qam::make();
    }

    int general_work(int noutput_items,
                     gr_vector_int& ninput_items,
                     gr_vector_const_void_star& input_items,
                     gr_vector_void_star& output_items)
    {
        const uint8_t* in = (const uint8_t*)input_items[0];
        int i = 0;
        std::vector<gr::tag_t> tags;
        const uint64_t nread = this->nitems_read(0);

        dout << "Decode MAC: input " << ninput_items[0] << std::endl;

        while (i < ninput_items[0]) {
            get_tags_in_range(tags, 0, nread + i, nread + i + 1);

            const pmt::pmt_t frame_bytes_key = pmt::mp("frame bytes");
            const pmt::pmt_t iqdata_key = pmt::mp("iqdata");
            bool frame_start = false;
            pmt::pmt_t iqdata = pmt::PMT_NIL;
            for (const auto& tag : tags) {
                if (pmt::eq(tag.key, frame_bytes_key)) {
                    frame_start = true;
                } else if (pmt::eq(tag.key, iqdata_key)) {
                    iqdata = tag.value;
                }
            }

            if (frame_start) {
                if (d_frame_complete == false) {
                    dout << "Warning: starting to receive new frame before old frame was "
                            "complete"
                         << std::endl;
                    dout << "Already copied " << copied << " out of " << d_frame.n_sym
                         << " symbols of last frame" << std::endl;
                }
                d_frame_complete = false;

                d_meta = pmt::make_dict();
                d_received_iq.clear();
                d_equalized_iq.clear();
                for (const auto& tag : tags) {
                    if (!pmt::eq(tag.key, iqdata_key)) {
                        d_meta = pmt::dict_add(d_meta, tag.key, tag.value);
                    }
                }

                int len_data = pmt::to_uint64(pmt::dict_ref(
                    d_meta, pmt::mp("frame bytes"), pmt::from_uint64(MAX_PSDU_SIZE + 1)));
                int encoding = pmt::to_uint64(
                    pmt::dict_ref(d_meta, pmt::mp("encoding"), pmt::from_uint64(0)));

                ofdm_param ofdm = ofdm_param((Encoding)encoding);
                frame_param frame = frame_param(ofdm, len_data);

                if (frame.n_sym <= MAX_SYM && frame.psdu_size <= MAX_PSDU_SIZE) {
                    d_ofdm = ofdm;
                    d_frame = frame;
                    copied = 0;
                    dout << "Decode MAC: frame start -- len " << len_data << "  symbols "
                         << frame.n_sym << "  encoding " << encoding << std::endl;
                } else {
                    dout << "Dropping frame which is too large (symbols or bits)"
                         << std::endl;
                }
            }

            const pmt::pmt_t equalized_iq_key = pmt::mp("equalized_iq");
            if (pmt::is_dict(iqdata) && pmt::dict_has_key(iqdata, equalized_iq_key)) {
                const auto equalized_iq = pmt::c32vector_elements(
                    pmt::dict_ref(iqdata, equalized_iq_key, pmt::PMT_NIL));
                if (equalized_iq.size() == 48) {
                    d_equalized_iq.insert(
                        d_equalized_iq.end(), equalized_iq.begin(), equalized_iq.end());
                }
            }

            const pmt::pmt_t received_iq_key = pmt::mp("unequalized_iq");
            if (pmt::is_dict(iqdata) && pmt::dict_has_key(iqdata, received_iq_key)) {
                const auto received_iq = pmt::c32vector_elements(
                    pmt::dict_ref(iqdata, received_iq_key, pmt::PMT_NIL));
                if (received_iq.size() == 52) {
                    d_received_iq.insert(
                        d_received_iq.end(), received_iq.begin(), received_iq.end());
                }
            }

            if (copied < d_frame.n_sym) {
                dout << "copy one symbol, copied " << copied << " out of "
                     << d_frame.n_sym << std::endl;
                std::memcpy(d_rx_symbols + (copied * 48), in, 48);
                copied++;

                if (copied == d_frame.n_sym) {
                    dout << "received complete frame - decoding" << std::endl;
                    decode();
                    in += 48;
                    i++;
                    d_frame_complete = true;
                    break;
                }
            }

            in += 48;
            i++;
        }

        consume(0, i);
        return 0;
    }

    void decode()
    {
        for (int i = 0; i < d_frame.n_sym * 48; i++) {
            for (int k = 0; k < d_ofdm.n_bpsc; k++) {
                d_rx_bits[i * d_ofdm.n_bpsc + k] = !!(d_rx_symbols[i] & (1 << k));
            }
        }

        deinterleave();
        uint8_t* decoded = d_decoder.decode(&d_ofdm, &d_frame, d_deinterleaved_bits);
        descramble(decoded);
        print_output();

        boost::crc_32_type result;
        result.process_bytes(out_bytes + 2, d_frame.psdu_size);
        if (result.checksum() != 558161692) {
            dout << "checksum wrong -- dropping" << std::endl;
            return;
        }

        if (d_frame.n_sym > 1) {
            int n_ber_bits = (d_frame.n_sym - 1) * d_ofdm.n_cbps;
            std::vector<uint8_t> pre_viterbi_bits(d_deinterleaved_bits,
                                                  d_deinterleaved_bits + n_ber_bits);
            std::vector<char> corrected_bits(decoded, decoded + d_frame.n_data_bits);
            reset_tail_bits(corrected_bits.data(), d_frame);
            std::vector<char> encoded_bits(d_frame.n_data_bits * 2);
            std::vector<char> corrected_coded(d_frame.n_encoded_bits);
            convolutional_encoding(corrected_bits.data(), encoded_bits.data(), d_frame);
            puncturing(encoded_bits.data(), corrected_coded.data(), d_frame, d_ofdm);
            std::vector<uint8_t> corrected_coded_bits(corrected_coded.begin(),
                                                      corrected_coded.begin() + n_ber_bits);

            pmt::pmt_t diag = pmt::make_dict();
            diag = pmt::dict_add(diag,
                                 pmt::mp("pre_viterbi"),
                                 pmt::init_u8vector(pre_viterbi_bits.size(), pre_viterbi_bits));
            diag = pmt::dict_add(diag,
                                 pmt::mp("corrected_coded"),
                                 pmt::init_u8vector(corrected_coded_bits.size(),
                                                    corrected_coded_bits));

            const auto transmitted_iq = recover_transmitted_iq(corrected_coded_bits);
            const size_t frame_equalized_iq_size = static_cast<size_t>(d_frame.n_sym) * 48;
            const size_t frame_received_iq_size = static_cast<size_t>(d_frame.n_sym) * 52;
            const size_t equalized_iq_size = static_cast<size_t>(d_frame.n_sym - 1) * 48;
            const size_t received_iq_size = static_cast<size_t>(d_frame.n_sym - 1) * 52;
            if (transmitted_iq.size() == equalized_iq_size &&
                d_equalized_iq.size() == frame_equalized_iq_size) {
                std::vector<gr_complex> equalized_iq(d_equalized_iq.begin(),
                                                     d_equalized_iq.begin() + equalized_iq_size);

                diag = pmt::dict_add(diag,
                                     pmt::mp("transmitted_iq"),
                                     pmt::init_c32vector(equalized_iq_size, transmitted_iq));
                diag = pmt::dict_add(diag,
                                     pmt::mp("equalized_iq"),
                                     pmt::init_c32vector(equalized_iq_size, equalized_iq));
                if (d_received_iq.size() == frame_received_iq_size) {
                    std::vector<gr_complex> received_iq(d_received_iq.begin(),
                                                        d_received_iq.begin() + received_iq_size);
                    diag = pmt::dict_add(diag,
                                         pmt::mp("received_iq"),
                                         pmt::init_c32vector(received_iq_size, received_iq));
                }
            }
            message_port_pub(pmt::mp("diag"), pmt::cons(d_meta, diag));
        }

        mylog("encoding: {} - length: {} - symbols: {}",
              d_ofdm.encoding,
              d_frame.psdu_size,
              d_frame.n_sym);

        pmt::pmt_t blob = pmt::make_blob(out_bytes + 2, d_frame.psdu_size - 4);
        d_meta =
            pmt::dict_add(d_meta, pmt::mp("dlt"), pmt::from_long(LINKTYPE_IEEE802_11));
        message_port_pub(pmt::mp("out"), pmt::cons(d_meta, blob));
    }

    std::vector<gr_complex> recover_transmitted_iq(
        const std::vector<uint8_t>& corrected_coded_bits)
    {
        frame_param frame = d_frame;
        frame.n_sym--;

        std::vector<char> interleaved_bits(corrected_coded_bits.size());
        std::vector<char> symbols(frame.n_sym * 48);
        interleave(reinterpret_cast<const char*>(corrected_coded_bits.data()),
                   interleaved_bits.data(),
                   frame,
                   d_ofdm);
        split_symbols(interleaved_bits.data(), symbols.data(), frame, d_ofdm);

        std::shared_ptr<gr::digital::constellation> mapping;
        switch (d_ofdm.n_bpsc) {
        case 1:
            mapping = d_bpsk;
            break;
        case 2:
            mapping = d_qpsk;
            break;
        case 4:
            mapping = d_16qam;
            break;
        case 6:
            mapping = d_64qam;
            break;
        default:
            return {};
        }

        std::vector<gr_complex> transmitted_iq(symbols.size());
        for (size_t i = 0; i < symbols.size(); i++) {
            mapping->map_to_points(static_cast<unsigned char>(symbols[i]),
                                   &transmitted_iq[i]);
        }
        return transmitted_iq;
    }

    void deinterleave()
    {
        int n_cbps = d_ofdm.n_cbps;
        int first[MAX_BITS_PER_SYM];
        int second[MAX_BITS_PER_SYM];
        int s = std::max(d_ofdm.n_bpsc / 2, 1);

        for (int j = 0; j < n_cbps; j++) {
            first[j] = s * (j / s) + ((j + int(floor(16.0 * j / n_cbps))) % s);
        }

        for (int i = 0; i < n_cbps; i++) {
            second[i] = 16 * i - (n_cbps - 1) * int(floor(16.0 * i / n_cbps));
        }

        int count = 0;
        for (int i = 0; i < d_frame.n_sym; i++) {
            for (int k = 0; k < n_cbps; k++) {
                d_deinterleaved_bits[i * n_cbps + second[first[k]]] =
                    d_rx_bits[i * n_cbps + k];
            }
        }
    }

    void descramble(uint8_t* decoded_bits)
    {
        int state = 0;
        std::memset(out_bytes, 0, d_frame.psdu_size + 2);

        for (int i = 0; i < 7; i++) {
            if (decoded_bits[i]) {
                state |= 1 << (6 - i);
            }
        }
        out_bytes[0] = state;

        int feedback;
        int bit;
        for (int i = 7; i < d_frame.psdu_size * 8 + 16; i++) {
            feedback = ((!!(state & 64))) ^ (!!(state & 8));
            bit = feedback ^ (decoded_bits[i] & 0x1);
            out_bytes[i / 8] |= bit << (i % 8);
            state = ((state << 1) & 0x7e) | feedback;
        }
    }

    void print_output()
    {
        dout << std::endl;
        dout << "psdu size" << d_frame.psdu_size << std::endl;
        for (int i = 2; i < d_frame.psdu_size + 2; i++) {
            dout << std::setfill('0') << std::setw(2) << std::hex
                 << ((unsigned int)out_bytes[i] & 0xFF) << std::dec << " ";
            if (i % 16 == 15) {
                dout << std::endl;
            }
        }
        dout << std::endl;
        for (int i = 2; i < d_frame.psdu_size + 2; i++) {
            if ((out_bytes[i] > 31) && (out_bytes[i] < 127)) {
                dout << ((char)out_bytes[i]);
            } else {
                dout << ".";
            }
        }
        dout << std::endl;
    }

private:
    bool d_debug;
    bool d_log;
    pmt::pmt_t d_meta;
    frame_param d_frame;
    ofdm_param d_ofdm;
    viterbi_decoder d_decoder;

    constellation_bpsk::sptr d_bpsk;
    constellation_qpsk::sptr d_qpsk;
    constellation_16qam::sptr d_16qam;
    constellation_64qam::sptr d_64qam;

    std::vector<gr_complex> d_received_iq;
    std::vector<gr_complex> d_equalized_iq;

    uint8_t d_rx_symbols[48 * MAX_SYM];
    uint8_t d_rx_bits[MAX_ENCODED_BITS];
    uint8_t d_deinterleaved_bits[MAX_ENCODED_BITS];
    uint8_t out_bytes[MAX_PSDU_SIZE + 2]; // 2 for signal field

    int copied;
    bool d_frame_complete;
};

decode_mac_EVM::sptr decode_mac_EVM::make(bool log, bool debug)
{
    return gnuradio::get_initial_sptr(new decode_mac_EVM_impl(log, debug));
}
