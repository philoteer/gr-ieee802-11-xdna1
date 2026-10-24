#pragma once

#include <Eigen/Core>
#include <complex>
#include <cstdint>

namespace gr {
namespace ieee802_11 {
namespace equalizer {

class channel_ai_model
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using complex = std::complex<float>;
    enum class weights { by_frame, by_channel, bus_to_room, room_to_bus };
    enum class arithmetic { float32, int8 };
    explicit channel_ai_model(weights model = weights::by_frame,
                              arithmetic mode = arithmetic::float32);
    void refine(const complex* first, const complex* second, double ratio_x16,
                complex* channel);
    void update(const complex* baseline, const complex* pilots, double ratio_x16,
                complex* channel);
    static bool occupied(int bin) { return bin >= 6 && bin <= 58 && bin != 32; }
    static double snr_ratio(const complex* first, const complex* second);
    static float snr_feature(double ratio_x16);

private:
    void refine_int8(double scale, complex* channel);
    void update_int8(int group);
    arithmetic d_mode;
    const std::int8_t *d_qw1, *d_qw2, *d_qw3, *d_qmlp_w1, *d_qmlp_w2;
    Eigen::Matrix<std::int32_t, 6, 70, Eigen::RowMajor> d_qinput;
    Eigen::Matrix<std::int32_t, 12, 70, Eigen::RowMajor> d_qhidden0, d_qhidden1;
    Eigen::Matrix<std::int32_t, 2, 70, Eigen::RowMajor> d_qresidual;
    const float* d_w1;
    const float* d_w2;
    const float* d_w3;
    double d_gate;
    const float* d_mlp_w1;
    const float* d_mlp_w2;
    double d_mlp_gate;
    Eigen::Matrix<float, 6, 70, Eigen::RowMajor> d_input;
    Eigen::Matrix<float, 12, 70, Eigen::RowMajor> d_hidden0;
    Eigen::Matrix<float, 12, 70, Eigen::RowMajor> d_hidden1;
    Eigen::Matrix<float, 2, 70, Eigen::RowMajor> d_residual;
    Eigen::Matrix<float, 61, 1> d_features;
    Eigen::Matrix<float, 16, 1> d_hidden;
    Eigen::Matrix<float, 26, 1> d_output;
};

}
}
}
