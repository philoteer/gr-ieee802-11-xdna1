#include "mac_address_metadata.h"
#include <array>
#include <cassert>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

using gr::ieee802_11::add_mac_address_metadata;

int main()
{
    std::array<uint8_t, 40> frame{};
    for (size_t i = 4; i < frame.size(); ++i) {
        frame[i] = static_cast<uint8_t>(i);
    }

    using field = std::pair<const char*, size_t>;
    const auto check = [&](uint8_t fc, uint8_t flags, size_t size,
                           std::initializer_list<field> fields,
                           bool signaling_ta = false) {
        frame[0] = fc;
        frame[1] = flags;
        frame[10] = signaling_ta ? 11 : 10;
        const auto original = pmt::dict_add(pmt::make_dict(), pmt::mp("existing"),
                                            pmt::from_long(42));
        // Test every truncation boundary, not just the full header.
        for (size_t length = 0; length <= size; ++length) {
            // An exact-size allocation also lets ASan catch out-of-bounds reads.
            const std::vector<uint8_t> input(frame.begin(), frame.begin() + length);
            const auto diag = add_mac_address_metadata(original, input.data(), length);
            assert(pmt::to_long(pmt::dict_ref(diag, pmt::mp("existing"),
                                             pmt::PMT_NIL)) == 42);
            for (const auto key : { "src_mac", "dst_mac", "rx_mac", "tx_mac",
                                    "bssid", "addr1", "addr2", "addr3", "addr4" }) {
                size_t offset = 0;
                for (const auto& expected : fields) {
                    if (std::string(key) == expected.first) {
                        offset = expected.second;
                    }
                }
                // A truncated QoS header cannot rule out A-MSDU semantics.
                if ((fc & 0x8c) == 0x88 &&
                    length < ((flags & 3) == 3 ? 32u : 26u) &&
                    (std::string(key) == "src_mac" || std::string(key) == "dst_mac")) {
                    offset = 0;
                }
                if (!offset || length < offset + 6) {
                    assert(!pmt::dict_has_key(diag, pmt::mp(key)));
                    continue;
                }
                const auto bytes = pmt::u8vector_elements(
                    pmt::dict_ref(diag, pmt::mp(key), pmt::PMT_NIL));
                assert(bytes.size() == 6);
                for (size_t i = 0; i < 6; ++i) {
                    uint8_t expected = frame[offset + i];
                    if ((fc & 0x0c) == 4 && i == 0 &&
                        (std::string(key) == "tx_mac" ||
                         (std::string(key) == "bssid" && offset == 10))) {
                        expected &= 0xfe;
                    }
                    assert(bytes[i] == expected);
                }
            }
        }
    };

    for (const auto fc : { 0x80, 0x08, 0x88 }) { // Management, data, QoS data.
        check(fc, 0, 26, {{"addr1", 4}, {"addr2", 10}, {"addr3", 16},
                          {"rx_mac", 4}, {"tx_mac", 10}, {"bssid", 16},
                          {"src_mac", 10}, {"dst_mac", 4}});
    }
    check(0x08, 1, 24, {{"addr1", 4}, {"addr2", 10}, {"addr3", 16},
                        {"rx_mac", 4}, {"tx_mac", 10}, {"bssid", 4},
                        {"src_mac", 10}, {"dst_mac", 16}});
    check(0x08, 2, 24, {{"addr1", 4}, {"addr2", 10}, {"addr3", 16},
                        {"rx_mac", 4}, {"tx_mac", 10}, {"bssid", 10},
                        {"src_mac", 16}, {"dst_mac", 4}});
    for (const auto fc : { 0x08, 0x88 }) {
        check(fc, 3, 32, {{"addr1", 4}, {"addr2", 10}, {"addr3", 16},
                          {"addr4", 24}, {"rx_mac", 4}, {"tx_mac", 10},
                          {"src_mac", 24}, {"dst_mac", 16}});
    }
    for (const auto fc : { 0x74, 0xc4, 0xd4 }) { // Wrapper, CTS, ACK.
        check(fc, 0, 40, {{"addr1", 4}, {"rx_mac", 4}});
    }
    for (const auto fc : { 0x24, 0x44, 0x54, 0x84, 0x94, 0xb4 }) {
        for (const bool signaling : { false, true }) {
            check(fc, 0, 40, {{"addr1", 4}, {"rx_mac", 4},
                              {"addr2", 10}, {"tx_mac", 10}}, signaling);
        }
    }
    check(0xa4, 0, 16, {{"addr1", 4}, {"rx_mac", 4}, {"addr2", 10},
                        {"tx_mac", 10}, {"bssid", 4}}); // PS-Poll.
    check(0xa4, 0, 16, {{"addr1", 4}, {"rx_mac", 4}, {"addr2", 10},
                        {"tx_mac", 10}, {"bssid", 4}}, true);
    for (const auto fc : { 0xe4, 0xf4 }) {
        check(fc, 0, 16, {{"addr1", 4}, {"rx_mac", 4}, {"addr2", 10},
                          {"tx_mac", 10}, {"bssid", 10}});
        check(fc, 0, 16, {{"addr1", 4}, {"rx_mac", 4}, {"addr2", 10},
                          {"tx_mac", 10}, {"bssid", 10}}, true);
    }
    check(0x64, 0, 40, {}); // Control extension needs subtype-specific parsing.
    check(0x04, 0, 40, {}); // Reserved control subtype.
    check(0x0c, 0, 40, {}); // Extension frame.
    check(0x09, 0, 40, {}); // Unsupported protocol version.

    // Never clear group bits in receiver/raw fields or non-control TAs.
    frame[4] = 5;
    check(0xd4, 0, 10, {{"addr1", 4}, {"rx_mac", 4}});
    check(0x08, 0, 24, {{"addr1", 4}, {"addr2", 10}, {"addr3", 16},
                        {"rx_mac", 4}, {"tx_mac", 10}, {"bssid", 16},
                        {"src_mac", 10}, {"dst_mac", 4}}, true);

    // A-MSDU: never label header BSSIDs as MSDU endpoints. Protected frames
    // still expose QoS Control; HT Control does not move its offset.
    for (uint8_t ds = 0; ds < 4; ++ds) {
        for (const uint8_t extra_flags : { 0x00, 0x40, 0x80, 0xc0 }) {
            frame[0] = 0x88;
            frame[1] = ds | extra_flags;
            const size_t qos_offset = ds == 3 ? 30 : 24;
            frame[qos_offset] = 0x80;
            const auto original = frame;
            const auto diag = add_mac_address_metadata(
                pmt::make_dict(), frame.data(), qos_offset + 2);
            assert(frame == original); // Normalization must not modify input.
            assert(!pmt::dict_has_key(diag, pmt::mp("src_mac")));
            assert(!pmt::dict_has_key(diag, pmt::mp("dst_mac")));
            const auto bssid = pmt::u8vector_elements(
                pmt::dict_ref(diag, pmt::mp("bssid"), pmt::PMT_NIL));
            for (size_t i = 0; i < 6; ++i) {
                assert(bssid[i] == frame[16 + i]);
            }
            for (const auto& field : { field{"addr1", 4}, field{"addr2", 10},
                                      field{"addr3", 16}, field{"rx_mac", 4},
                                      field{"tx_mac", 10}, field{"addr4", 24} }) {
                if (std::string(field.first) == "addr4" && ds != 3) {
                    assert(!pmt::dict_has_key(diag, pmt::mp(field.first)));
                    continue;
                }
                const auto bytes = pmt::u8vector_elements(
                    pmt::dict_ref(diag, pmt::mp(field.first), pmt::PMT_NIL));
                for (size_t i = 0; i < 6; ++i) {
                    assert(bytes[i] == frame[field.second + i]);
                }
            }
            frame[qos_offset] = 0;
        }
    }
}
