#ifndef INCLUDED_IEEE802_11_MAC_ADDRESS_METADATA_H
#define INCLUDED_IEEE802_11_MAC_ADDRESS_METADATA_H

#include <pmt/pmt.h>
#include <cstddef>
#include <cstdint>

namespace gr {
namespace ieee802_11 {

// Preserve each complete address in known 802.11 layouts; size excludes FCS.
// addr1..4 are raw fields; named roles are added only when their meaning is known.
inline pmt::pmt_t add_mac_address_metadata(pmt::pmt_t dict,
                                          const uint8_t* frame,
                                          size_t size)
{
    // Read frame control first. ACK/CTS headers need only 10 bytes, not 24.
    // Bits 0-1 select protocol version (only version 0 is supported).
    if (size < 2 || (frame[0] & 0x03) != 0) {
        return dict;
    }

    // Check each field independently: a missing later address must not hide an
    // earlier one. Never read partial addresses or mistake FCS bytes for a MAC.
    const auto add = [&](const char* key, size_t offset, bool individual = false) {
        if (size >= offset + 6) {
            uint8_t address[6];
            for (size_t i = 0; i < 6; ++i) {
                address[i] = frame[offset + i];
            }
            // Control TAs may set I/G for bandwidth signaling, not multicast.
            // Normalize the named identity only; addr1..4 remain byte-exact.
            if (individual) {
                address[0] &= 0xfe;
            }
            dict = pmt::dict_add(dict, pmt::mp(key),
                                 pmt::init_u8vector(6, address));
        }
    };

    const unsigned type = (frame[0] >> 2) & 0x03;
    const unsigned subtype = frame[0] >> 4;
    if (type == 0 || type == 2) { // Management or data.
        add("addr1", 4);
        add("addr2", 10);
        add("addr3", 16);
        add("rx_mac", 4);
        add("tx_mac", 10);
        if (type == 0) {
            add("dst_mac", 4);
            add("src_mac", 10);
            add("bssid", 16);
            return dict;
        }

        // DS = distribution system: traffic going to/from an access point.
        // For a non-aggregated MSDU:
        // To DS  From DS   Source     Destination
        //   0       0     address 2  address 1
        //   1       0     address 2  address 3
        //   0       1     address 3  address 1
        //   1       1     address 4  address 3 (wireless bridge)
        const bool to_ds = (frame[1] & 0x01) != 0;
        const bool from_ds = (frame[1] & 0x02) != 0;
        if (to_ds && from_ds) {
            add("addr4", 24);
        } else {
            add("bssid", to_ds ? 4 : (from_ds ? 10 : 16));
        }

        // QoS Control follows address 4 (when present), before any HT Control.
        // Its A-MSDU bit changes address semantics: address 3 (and 4) can be
        // BSSID, while actual MSDU endpoints are in multiple subframe headers.
        // Keep raw fields/radio roles, but do not invent one src/dst pair.
        if (subtype & 0x08) {
            const size_t qos_offset = to_ds && from_ds ? 30 : 24;
            if (size < qos_offset + 2) {
                return dict; // Cannot distinguish MSDU from A-MSDU safely.
            }
            if (frame[qos_offset] & 0x80) {
                add("bssid", 16);
                return dict;
            }
        }
        add("dst_mac", to_ds ? 16 : 4);
        add("src_mac", from_ds ? (to_ds ? 24 : 16) : 10);
    } else if (type == 1) { // Control: layout depends on subtype.
        switch (subtype) {
        case 2: // Trigger.
        case 4: // Beamforming report poll.
        case 5: // VHT NDP announcement.
        case 8: // Block ACK request.
        case 9: // Block ACK.
        case 11: // RTS.
            add("addr2", 10);
            add("tx_mac", 10, true);
            break;
        case 10: // PS-Poll: address 1 is BSSID, address 2 is transmitter.
            add("addr2", 10);
            add("tx_mac", 10, true);
            add("bssid", 4);
            break;
        case 14: // CF-End and CF-End+CF-ACK: address 2 is BSSID.
        case 15:
            add("addr2", 10);
            add("tx_mac", 10, true);
            add("bssid", 10, true);
            break;
        case 7: // Control wrapper: bytes after address 1 are not address 2.
        case 12: // CTS.
        case 13: // ACK.
            break;
        default: // Reserved or variant layouts: do not guess.
            return dict;
        }
        add("addr1", 4);
        add("rx_mac", 4);
        // Radio endpoints need not be original source/destination: do not
        // manufacture src_mac/dst_mac for control frames.
    }

    return dict;
}

} // namespace ieee802_11
} // namespace gr

#endif // INCLUDED_IEEE802_11_MAC_ADDRESS_METADATA_H
