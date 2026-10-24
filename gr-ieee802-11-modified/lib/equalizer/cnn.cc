#include "cnn.h"
#include <algorithm>
#include <cmath>

namespace gr {
namespace ieee802_11 {
namespace equalizer {

cnn::cnn(bool update, channel_ai_model::weights model, channel_ai_model::arithmetic mode)
    : d_update(update), d_model(model, mode)
{
    std::fill(d_H, d_H + 64, gr_complex(0));
}

void cnn::equalize(gr_complex* in, int n, gr_complex* symbols, uint8_t* bits,
                   std::shared_ptr<gr::digital::constellation> mod)
{
    if (n == 0) {
        std::fill(d_H, d_H + 64, gr_complex(0));
        std::fill(d_initial, d_initial + 64, gr_complex(0));
        d_ratio = 0;
        d_snr = -std::numeric_limits<double>::infinity();
        for (int bin = 0; bin < 64; ++bin)
            d_first[bin] = channel_ai_model::occupied(bin) ? in[bin] * LONG[bin] : gr_complex(0);
    } else if (n == 1) {
        gr_complex second[64];
        for (int bin = 0; bin < 64; ++bin) {
            second[bin] = channel_ai_model::occupied(bin) ? in[bin] * LONG[bin] : gr_complex(0);
            d_H[bin] = d_first[bin] * 0.5f + second[bin] * 0.5f;
        }
        d_ratio = channel_ai_model::snr_ratio(d_first, second);
        d_snr = 10 * std::log10(d_ratio / 32);
        d_model.refine(d_first, second, d_ratio, d_H);
        std::copy(d_H, d_H + 64, d_initial);
    } else {
        if (d_update) {
            const gr_complex polarity = POLARITY[(n - 2) % 127];
            const gr_complex pilots[] = {in[11] * polarity, in[25] * polarity,
                                          in[39] * polarity, -in[53] * polarity};
            d_model.update(d_initial, pilots, d_ratio, d_H);
        }
        int out = 0;
        for (int bin = 6; bin <= 58; ++bin) {
            if (bin == 11 || bin == 25 || bin == 32 || bin == 39 || bin == 53)
                continue;
            symbols[out] = std::norm(d_H[bin]) > 0 ? in[bin] / d_H[bin] : gr_complex(0);
            bits[out] = mod->decision_maker(&symbols[out]);
            ++out;
        }
    }
}

}
}
}
