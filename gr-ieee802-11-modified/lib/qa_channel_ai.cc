#include "equalizer/channel_ai_model.h"
#include "equalizer/cnn.h"
#include <ieee802_11/constellations.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace gr::ieee802_11;
using namespace gr::ieee802_11::equalizer;

struct exposed_cnn : cnn {
    using cnn::cnn;
    using base::LONG;
};

void require(bool condition)
{
    if (!condition)
        throw std::runtime_error("channel estimator assertion failed");
}

void self_test(channel_ai_model::arithmetic mode)
{
    const std::shared_ptr<gr::digital::constellation> mods[] = {
        constellation_bpsk::make(), constellation_qpsk::make(),
        constellation_16qam::make(), constellation_64qam::make()};
    for (bool updater : {false, true}) {
        exposed_cnn estimator(updater, channel_ai_model::weights::by_frame, mode);
        for (const auto& mod : mods) {
            gr_complex input[64], symbols[48];
            uint8_t bits[48];
            for (float amplitude : {0.25f, 4.0f}) {
                for (int bin = 0; bin < 64; ++bin)
                    input[bin] = exposed_cnn::LONG[bin] * amplitude;
                estimator.equalize(input, 0, symbols, bits, mod);
                estimator.equalize(input, 1, symbols, bits, mod);
                require(std::isinf(estimator.get_snr()) && estimator.get_snr() > 0);
                for (int n : {2, 3, 128, 129, 256}) {
                    const gr_complex p = base::POLARITY[(n - 2) % 127];
                    input[11] = input[25] = input[39] = amplitude * p;
                    input[53] = -amplitude * p;
                    int count = 0;
                    for (int bin = 6; bin <= 58; ++bin) {
                        if (bin == 11 || bin == 25 || bin == 32 || bin == 39 || bin == 53)
                            continue;
                        gr_complex point;
                        mod->map_to_points(count % mod->arity(), &point);
                        input[bin] = amplitude * point;
                        ++count;
                    }
                    Eigen::internal::set_is_malloc_allowed(false);
                    estimator.equalize(input, n, symbols, bits, mod);
                    Eigen::internal::set_is_malloc_allowed(true);
                    for (int i = 0; i < 48; ++i) {
                        gr_complex point;
                        mod->map_to_points(i % mod->arity(), &point);
                        require(std::abs(symbols[i] - point) < 1e-6f);
                        require(bits[i] == i % mod->arity());
                    }
                }
            }
        }
    }
}

void benchmark(channel_ai_model::weights selection, channel_ai_model::arithmetic mode)
{
    channel_ai_model model(selection, mode);
    gr_complex first[64] = {}, second[64] = {}, baseline[64] = {}, channel[64];
    gr_complex pilots[4] = {{0.8f, 0.1f}, {0.9f, -0.1f}, {1, 0.2f}, {0.7f, 0}};
    for (int bin = 6; bin <= 58; ++bin) {
        if (channel_ai_model::occupied(bin)) {
            first[bin] = baseline[bin] = {0.75f, 0.1f};
            second[bin] = {0.8f, 0.15f};
        }
    }
    for (int kind = 0; kind < 2; ++kind) {
        std::vector<double> times;
        for (int run = 0; run < 1100; ++run) {
            std::copy(baseline, baseline + 64, channel);
            const auto start = std::chrono::steady_clock::now();
            Eigen::internal::set_is_malloc_allowed(false);
            if (kind == 0)
                model.refine(first, second, 1600, channel);
            else
                model.update(baseline, pilots, 1600, channel);
            Eigen::internal::set_is_malloc_allowed(true);
            const auto end = std::chrono::steady_clock::now();
            if (run >= 100)
                times.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        }
        std::sort(times.begin(), times.end());
        std::cout << (kind == 0 ? "CNN" : "MLP") << " median_us=" << times[500]
                  << " p99_us=" << times[990] << '\n';
    }
}

int main(int argc, char** argv)
{
    const bool integer = std::find_if(argv + 1, argv + argc, [](const char* arg) {
        return std::string(arg) == "--int8";
    }) != argv + argc;
    const auto mode = integer ? channel_ai_model::arithmetic::int8 : channel_ai_model::arithmetic::float32;
    auto selection = channel_ai_model::weights::by_frame;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "by_channel") selection = channel_ai_model::weights::by_channel;
        if (arg == "bus_to_room") selection = channel_ai_model::weights::bus_to_room;
        if (arg == "room_to_bus") selection = channel_ai_model::weights::room_to_bus;
    }
    if (argc > 1 && std::string(argv[1]) == "--benchmark") {
        benchmark(selection, mode);
        return 0;
    }
    self_test(mode);
    channel_ai_model model(selection, mode);
    exposed_cnn estimator(true, selection, mode);
    std::cout << std::setprecision(9);
    int cases;
    std::cin >> cases;
    for (int sample = 0; sample < cases; ++sample) {
        double ratio;
        int symbol_number;
        std::cin >> ratio >> symbol_number;
        gr_complex first[64], second[64], channel[64], pilots[4];
        for (auto* values : {first, second}) {
            for (int bin = 0; bin < 64; ++bin) {
                float real, imag;
                std::cin >> real >> imag;
                values[bin] = {real, imag};
            }
        }
        for (auto& value : pilots) {
            float real, imag;
            std::cin >> real >> imag;
            value = {real, imag};
        }
        for (int bin = 0; bin < 64; ++bin)
            channel[bin] = channel_ai_model::occupied(bin)
                               ? first[bin] * 0.5f + second[bin] * 0.5f : gr_complex(0);
        Eigen::internal::set_is_malloc_allowed(false);
        model.refine(first, second, ratio, channel);
        gr_complex baseline[64];
        std::copy(channel, channel + 64, baseline);
        model.update(baseline, pilots, ratio, channel);
        model.update(baseline, pilots, ratio, channel);
        Eigen::internal::set_is_malloc_allowed(true);
        for (const auto* values : {baseline, channel})
            for (int bin = 0; bin < 64; ++bin)
                std::cout << values[bin].real() << ' ' << values[bin].imag() << ' ';
        gr_complex raw[64], symbols[48];
        uint8_t bits[48];
        auto mod = constellation_bpsk::make();
        for (int bin = 0; bin < 64; ++bin)
            raw[bin] = first[bin] * exposed_cnn::LONG[bin];
        estimator.equalize(raw, 0, symbols, bits, mod);
        for (int bin = 0; bin < 64; ++bin)
            raw[bin] = second[bin] * exposed_cnn::LONG[bin];
        estimator.equalize(raw, 1, symbols, bits, mod);
        std::fill(raw, raw + 64, gr_complex(0));
        const auto p = base::POLARITY[(symbol_number - 2) % 127];
        raw[11] = pilots[0] * p;
        raw[25] = pilots[1] * p;
        raw[39] = pilots[2] * p;
        raw[53] = -pilots[3] * p;
        estimator.equalize(raw, symbol_number, symbols, bits, mod);
        for (auto value : estimator.get_csi())
            std::cout << value.real() << ' ' << value.imag() << ' ';
        std::cout << '\n';
    }
    return std::cin ? 0 : 1;
}
