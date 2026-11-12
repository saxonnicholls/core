//
//  linear_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/linear.hpp>: one product with float, bfloat16 and Q4_0 weights, the input prepared
//  once per storage. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/linear.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>

int main()
{
    std::printf("snicholls-core linear demo\n");
    namespace ml = snicholls::ml;
    bool ok = true;

    //! [linear]
    // 2 outputs from 32 inputs, laid out as nn.Linear.weight: row i is output i's weights
    std::array<float, 2 * 32> w{};
    std::array<float, 32> x{};
    for (std::size_t j = 0; j < 32; ++j) {
        w[j] = 0.5f;                       // output 0 sums the inputs, halved
        w[32 + j] = j % 2 ? -1.0f : 0.5f;  // output 1 alternates
        x[j] = static_cast<float>(j) / 8.0f;
    }
    const std::array<float, 2> bias{1.0f, 0.0f};

    const ml::dense_weights<float> dense{w, 2, 32};
    std::array<float, 2> y{};
    ml::linear(dense, ml::prepare_input<32>(dense, x), y, bias);

    // The same matrix in bfloat16, and quantised to Q4_0: the input is quantised once, for every product sharing it
    std::array<ml::bfloat16, 2 * 32> wb{};
    for (std::size_t i = 0; i < w.size(); ++i)
        wb[i] = ml::to_minifloat<ml::bfloat16>(w[i]);
    std::array<ml::block_q4_0, 2> wq{};
    ml::quantize(w, std::span<ml::block_q4_0>(wq));
    const ml::dense_weights<ml::bfloat16> half{wb, 2, 32};
    const ml::quantized_weights<ml::block_q4_0> q4{wq, 2, 32};
    std::array<float, 2> yb{}, yq{};
    ml::linear(half, ml::prepare_input<32>(half, x), yb, bias);
    ml::linear(q4, ml::prepare_input<32>(q4, x), yq, bias);
    //! [linear]

    std::printf("  float [%.4f %.4f], bfloat16 [%.4f %.4f], Q4_0 [%.4f %.4f]\n", static_cast<double>(y[0]), static_cast<double>(y[1]),
                static_cast<double>(yb[0]), static_cast<double>(yb[1]), static_cast<double>(yq[0]), static_cast<double>(yq[1]));
    ok &= y[0] == 1.0f + 0.5f * 496.0f / 8.0f && y[1] == (0.5f * 240.0f - 256.0f) / 8.0f && yb == y;   // 0.5 and -1 are exact in bfloat16
    // Q4_0 holds 0.5 and -1 exactly too (its scale is 1/8), so what remains is the input's rounding to Q8_0
    ok &= std::abs(yq[0] - y[0]) < 0.05f && std::abs(yq[1] - y[1]) < 0.05f;
    std::printf(ok ? "linear demo: ok\n" : "linear demo: FAILED\n");
    return ok ? 0 : 1;
}
