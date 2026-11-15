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

#include <snicholls/core/linalg/reference.hpp>

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

    // Training: the backward pass of the float product adds W^T dy, dy x^T and dy into the gradients
    const std::array<float, 2> dy{1.0f, 0.0f};   // the gradient in output 0 only
    std::array<float, 32> dx{};
    std::array<float, 2 * 32> dw{};
    std::array<float, 2> db{};
    ml::linear_backward(dense, x, dy, dx, dw, db);
    //! [linear]

    std::printf("  float [%.4f %.4f], bfloat16 [%.4f %.4f], Q4_0 [%.4f %.4f]\n", static_cast<double>(y[0]), static_cast<double>(y[1]),
                static_cast<double>(yb[0]), static_cast<double>(yb[1]), static_cast<double>(yq[0]), static_cast<double>(yq[1]));
    ok &= y[0] == 1.0f + 0.5f * 496.0f / 8.0f && y[1] == (0.5f * 240.0f - 256.0f) / 8.0f && yb == y;   // 0.5 and -1 are exact in bfloat16
    // Q4_0 holds 0.5 and -1 exactly too (its scale is 1/8), so what remains is the input's rounding to Q8_0
    ok &= std::abs(yq[0] - y[0]) < 0.05f && std::abs(yq[1] - y[1]) < 0.05f;
    // Only output 0 had a gradient: dx is its row of weights, dw's row 0 is x and row 1 zero, db is dy
    std::printf("  backward: dx[0] %.2f, dw[0][8] %.2f, dw[1][8] %.2f, db [%.0f %.0f]\n", static_cast<double>(dx[0]),
                static_cast<double>(dw[8]), static_cast<double>(dw[32 + 8]), static_cast<double>(db[0]), static_cast<double>(db[1]));
    for (std::size_t j = 0; j < 32; ++j)
        ok &= dx[j] == 0.5f && dw[j] == x[j] && dw[32 + j] == 0.0f;
    ok &= db == dy;

    //! [linear_sequence]
    // Training and prefill: a sequence of rows at once, Y = X Wᵀ + b, as one matrix product through a linalg backend
    // (core's reference here; a BLAS or a GPU backend from another library the same way)
    const std::array<float, 6> m{1, 2, 3, 4, 5, 6};        // 2 x 3
    const std::array<float, 6> rows{1, 0, -1, 0, 1, 0};   // two inputs
    const std::array<float, 2> row_bias{10, 20};
    std::array<float, 4> outs{};
    ml::linear_sequence(snicholls::linalg::reference, ml::dense_weights<float>{m, 2, 3}, rows, outs, row_bias);
    const std::array<float, 4> douts{1, 0, 0, 1};          // the gradient in output 0 of row 0 and output 1 of row 1
    std::array<float, 6> drows{}, dm{};
    std::array<float, 2> dbias{};
    ml::linear_sequence_backward(snicholls::linalg::reference, ml::dense_weights<float>{m, 2, 3}, rows, douts, drows, dm, dbias);
    //! [linear_sequence]
    std::printf("  sequence: [%.0f %.0f] [%.0f %.0f]; d rows [%.0f %.0f %.0f] [%.0f %.0f %.0f]\n", static_cast<double>(outs[0]),
                static_cast<double>(outs[1]), static_cast<double>(outs[2]), static_cast<double>(outs[3]), static_cast<double>(drows[0]),
                static_cast<double>(drows[1]), static_cast<double>(drows[2]), static_cast<double>(drows[3]), static_cast<double>(drows[4]),
                static_cast<double>(drows[5]));
    ok &= outs == std::array<float, 4>{8, 18, 12, 25};                    // W x + b for each row
    ok &= drows == std::array<float, 6>{1, 2, 3, 4, 5, 6};                // row t of dY picks row t of W
    ok &= dm == std::array<float, 6>{1, 0, -1, 0, 1, 0} && dbias == std::array<float, 2>{1, 1};

    std::printf(ok ? "linear demo: ok\n" : "linear demo: FAILED\n");
    return ok ? 0 : 1;
}
