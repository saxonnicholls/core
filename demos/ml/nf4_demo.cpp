//
//  nf4_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/nf4.hpp>: a 4 x 128 matrix of normally distributed weights quantised to QLoRA's
//  NormalFloat4 with double-quantised scales, its size and round-trip error, and products through it that dequantise
//  on the fly and equal the dense products of the dequantised matrix to the bit. Exits non-zero if a result is not
//  what it documents.
//

#include <snicholls/core/ml/nf4.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/linear.hpp>
#include <snicholls/core/stats/random.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace ml = snicholls::ml;
namespace stats = snicholls::stats;

int main()
{
    std::printf("snicholls-core nf4 demo\n");
    bool ok = true;
    const auto& backend = snicholls::linalg::reference;

    constexpr std::size_t rows = 4, cols = 128, n = rows * cols;
    stats::wyrand g(2023);
    std::vector<float> w(n);
    for (auto& x : w)
        x = static_cast<float>(0.02 * snicholls::math::normal_quantile((static_cast<double>(g() >> 11) + 0.5) * 0x1.0p-53));

    //! [nf4]
    // Quantise: 4-bit codes, one E4M3 scale per 64 weights, one float per 256 scales, and their mean
    std::vector<std::uint8_t> codes(ml::nf4_code_bytes(n));
    std::vector<ml::float8_e4m3> block_scales(ml::nf4_block_count(n));
    std::vector<float> group_scales(ml::nf4_group_count(n));
    const float mean = ml::nf4_quantize(w, codes, block_scales, group_scales);
    const ml::nf4_weights q{codes, block_scales, group_scales, mean, rows, cols};
    // y = W x, each weight dequantised as it is read
    std::vector<float> x(cols, 0.5f), y(rows);
    ml::linear(q, ml::prepare_input<cols>(q, x), y);
    //! [nf4]
    const double bits = 8.0 * static_cast<double>(codes.size() + block_scales.size() + 4 * group_scales.size() + 4) / n;
    std::vector<float> back(n);
    ml::nf4_dequantize(q, back);
    double err = 0, norm = 0;
    for (std::size_t i = 0; i < n; ++i) {
        err += (back[i] - w[i]) * (back[i] - w[i]);
        norm += w[i] * w[i];
    }
    std::printf("  %.3f bits a weight; relative round-trip error %.4f; weight (0, 0) %.5f -> code %u -> %.5f\n", bits,
                std::sqrt(err / norm), w[0], q.code(0), q.at(0, 0));
    ok &= std::sqrt(err / norm) < 0.12 && q.block_scale(0) > 0 && ml::nf4_code(ml::nf4_values[9]) == 9;
    ok &= ml::nf4_values[7] == 0.0f;

    // The products through NF4 are the dense products of the dequantised matrix, to the bit
    const ml::dense_weights<float> dense{back, rows, cols};
    std::vector<float> yd(rows), dx(cols, 0.0f), dxd(cols, 0.0f), dw(n, 0.0f), dy{1.0f, -1.0f, 0.5f, 2.0f};
    ml::linear(dense, x, yd);
    ml::linear_backward_input(q, dy, dx);
    ml::linear_backward(dense, x, dy, dxd, dw);
    std::vector<float> ys(2 * rows), xs(2 * cols, 0.25f), dxs(2 * cols, 0.0f), dys(2 * rows, 1.0f);
    ml::linear_sequence(backend, q, xs, ys);
    ml::linear_sequence_backward_input(backend, q, dys, dxs);
    std::printf("  y = %.5f %.5f %.5f %.5f, the dense product's: %s\n", y[0], y[1], y[2], y[3], y == yd && dx == dxd ? "yes" : "no");
    ok &= y == yd && dx == dxd && std::isfinite(ys[0]) && std::isfinite(dxs[0]);

    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
