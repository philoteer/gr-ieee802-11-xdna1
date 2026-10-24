#pragma once

#include "base.h"
#include "channel_ai_model.h"
#include <limits>

namespace gr {
namespace ieee802_11 {
namespace equalizer {

class cnn : public base
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    explicit cnn(bool update = false,
                 channel_ai_model::weights model = channel_ai_model::weights::by_frame,
                 channel_ai_model::arithmetic mode = channel_ai_model::arithmetic::float32);
    void equalize(gr_complex* in, int n, gr_complex* symbols, uint8_t* bits,
                  std::shared_ptr<gr::digital::constellation> mod) override;
    double get_snr() override { return d_snr; }

private:
    bool d_update;
    double d_snr = -std::numeric_limits<double>::infinity();
    double d_ratio = 0;
    gr_complex d_first[64] = {};
    gr_complex d_initial[64] = {};
    channel_ai_model d_model;
};

}
}
}
