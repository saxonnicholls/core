//
//  rope_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/rope.hpp>: a table of cosines and sines built by the compiler, a query and a key
//  rotated in both layouts, the score depending only on their distance, and the inverse rotation as the backward
//  pass; the long-context rescalings (position interpolation, NTK-aware, YaRN two ways) and YaRN's temperature. Exits
//  non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/rope.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>

namespace {

namespace ml = snicholls::ml;

//! [rope]
// A context of 16 positions and heads of 8 channels: the compiler builds the whole table, in float as a kernel reads it
constexpr ml::rope_table<float, 16, 8> table{};
static_assert(table.cos[0][0] == 1.0f && table.sin[0][0] == 0.0f);   // position 0 does not rotate

float score(std::array<float, 8> q, std::size_t pq, std::array<float, 8> k, std::size_t pk, ml::rope_layout layout)
{
    ml::rope(q, table.cos_at(pq), table.sin_at(pq), layout);
    ml::rope(k, table.cos_at(pk), table.sin_at(pk), layout);
    float s = 0.0f;
    for (std::size_t i = 0; i < 8; ++i)
        s += q[i] * k[i];
    return s;
}
//! [rope]

} // namespace

int main()
{
    std::printf("snicholls-core rope demo\n");
    bool ok = true;
    const std::array<float, 8> q{0.5f, -1.0f, 0.25f, 2.0f, 1.0f, 0.0f, -0.5f, 0.75f};
    const std::array<float, 8> k{1.0f, 0.5f, -0.25f, 0.5f, -1.0f, 1.5f, 0.0f, 0.25f};

    for (const auto layout : {ml::rope_layout::half, ml::rope_layout::interleaved}) {
        const float near = score(q, 5, k, 3, layout), far = score(q, 14, k, 12, layout);
        std::printf("  %-11s q@5.k@3 = %.6f, q@14.k@12 = %.6f\n", layout == ml::rope_layout::half ? "half" : "interleaved",
                    static_cast<double>(near), static_cast<double>(far));
        ok &= std::fabs(near - far) < 1e-5f;
    }

    // Rotating at run time without a table, then back: the inverse is also the backward pass
    std::array<double, 4> x{1.0, 2.0, 3.0, 4.0};
    std::array<double, 2> c{}, s{};
    ml::rope_angles(1000, c, s, 500000.0);
    ml::rope(x, c, s);
    std::printf("  [1 2 3 4] at position 1000: [%.4f %.4f %.4f %.4f]\n", x[0], x[1], x[2], x[3]);
    ml::rope_inverse(x, c, s);
    ok &= std::fabs(x[0] - 1.0) < 1e-12 && std::fabs(x[3] - 4.0) < 1e-12;
    std::printf("  frequency of pair 1 of 4 at base 10000: %.4f\n", ml::rope_frequency<double>(1, 4));
    ok &= std::fabs(ml::rope_frequency<double>(1, 4) - 0.01) < 1e-15;

    //! [rope_scaling]
    // Reaching 40 times past a trained context of 4096: the five rescalings of a head of 64 rotary channels, pair by pair
    const auto method = [](ml::rope_scaling_kind kind) { return ml::rope_scaling<double>{kind, 40.0, 4096, 32.0, 1.0, 1.0}; };
    const double plain = ml::rope_frequency<double>(31, 64), fast = ml::rope_frequency<double>(0, 64);
    const double linear = ml::rope_scaled_frequency(31, 64, 10000.0, method(ml::rope_scaling_kind::linear));   // θ / 40
    const double ntk = ml::rope_scaled_frequency(16, 64, 10000.0, method(ml::rope_scaling_kind::ntk));         // a larger base:
    const double mid = ml::rope_frequency<double>(16, 64);                                                     // between the two
    const double yarn = ml::rope_scaled_frequency(31, 64, 10000.0, method(ml::rope_scaling_kind::yarn));       // slow pairs: θ / 40
    const double yarn_fast = ml::rope_scaled_frequency(0, 64, 10000.0, method(ml::rope_scaling_kind::yarn_index_ramp));   // fast: θ
    const double none = ml::rope_scaled_frequency(31, 64, 10000.0, method(ml::rope_scaling_kind::none));
    const double ramp = ml::yarn_ramp(16.5, 1.0, 32.0), lo = ml::yarn_correction_dim(32.0, 64, 10000.0, 4096);
    // YaRN's temperature: DeepSeek-V3 multiplies its softmax scale by its square; a table can carry it as an amplitude
    const double t = ml::yarn_attention_factor(40.0);
    std::array<double, 2> yc{}, ys{};
    ml::rope_angles(0, yc, ys, 10000.0, ml::rope_scaling<double>{ml::rope_scaling_kind::yarn_index_ramp, 40.0, 4096, 32.0, 1.0, t});
    constexpr ml::rope_table<float, 4, 8> long_table{10000.0f, ml::rope_scaling<float>{ml::rope_scaling_kind::ntk, 4.0f}};
    //! [rope_scaling]
    std::printf("  slowest pair: plain %.3e, linear %.3e, YaRN %.3e; pair 16 under NTK %.3e (plain %.3e); fastest under YaRN %.3f\n", plain,
                linear, yarn, ntk, mid, yarn_fast);
    std::printf("  YaRN's ramp at r = 16.5: %.2f; first interpolated pair from %.2f; temperature factor %.4f\n", ramp, lo, t);
    ok &= std::fabs(linear - plain / 40) < 1e-18 && ntk < mid && ntk > mid / 40 && std::fabs(yarn - plain / 40) < 1e-18 && yarn_fast == fast;
    ok &= none == plain && ramp == 0.5 && lo > 10 && lo < 11 && std::fabs(t - (0.1 * std::log(40.0) + 1)) < 1e-15 && yc[0] == t;
    ok &= long_table.cos[0][0] == 1.0f;

    std::printf(ok ? "rope demo: ok\n" : "rope demo: FAILED\n");
    return ok ? 0 : 1;
}
