/*
 * Copyright (C) 2016 Bastian Bloessl <bloessl@ccs-labs.org>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */


#ifndef INCLUDED_IEEE802_11_FRAME_EQUALIZER_H
#define INCLUDED_IEEE802_11_FRAME_EQUALIZER_H

#include <gnuradio/block.h>
#include <ieee802_11/api.h>

namespace gr {
namespace ieee802_11 {

enum Equalizer {
    LS = 0,
    LMS = 1,
    COMB = 2,
    STA = 3,
    CNN_FRAME = 6,
    CNN_CHANNEL = 7,
    CNN_FRAME_MLP = 8,
    CNN_CHANNEL_MLP = 9,
    CNN_FRAME_INT8 = 10,
    CNN_CHANNEL_INT8 = 11,
    CNN_FRAME_MLP_INT8 = 12,
    CNN_CHANNEL_MLP_INT8 = 13,
    CNN_BUS_TO_ROOM = 14,
    CNN_ROOM_TO_BUS = 15,
    CNN_BUS_TO_ROOM_INT8 = 16,
    CNN_ROOM_TO_BUS_INT8 = 17,
    CNN_BUS_TO_ROOM_MLP = 18,
    CNN_ROOM_TO_BUS_MLP = 19,
    CNN_BUS_TO_ROOM_MLP_INT8 = 20,
    CNN_ROOM_TO_BUS_MLP_INT8 = 21,
};


class IEEE802_11_API frame_equalizer : virtual public gr::block
{

public:
    typedef std::shared_ptr<frame_equalizer> sptr;
    static sptr make(Equalizer algo, double freq, double bw, bool log, bool debug);
    virtual void set_algorithm(Equalizer algo) = 0;
    virtual void set_bandwidth(double bw) = 0;
    virtual void set_frequency(double freq) = 0;
};

} // namespace ieee802_11
} // namespace gr

#endif /* INCLUDED_IEEE802_11_FRAME_EQUALIZER_H */
