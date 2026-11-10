//
//  norm_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/norm.hpp>: one token's LayerNorm with learned parameters and its backward
//  pass, and a parameter-free RMSNorm over a batch of rows. Exits non-zero if a result is not what it
//  documents.
//

#include <snicholls/core/ml/norm.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>
#include <vector>

int main()
{
    std::printf("snicholls-core norm demo\n");
    namespace ml = snicholls::ml;
    bool ok = true;

    //! [norm]
    // One token of four channels through LayerNorm with a learned weight and bias (GPT-2).
    const std::array<float, 4> x{1.0f, 2.0f, 4.0f, 9.0f};
    const std::array<float, 4> weight{1.0f, 0.5f, 2.0f, 1.0f}, bias{0.0f, 0.1f, 0.0f, -0.1f};
    std::array<float, 4> y{};
    const ml::norm_stats<float> stats = ml::layer_norm(x, weight, bias, y);

    // The backward pass: dx is written, dweight and dbias add up over the rows of a batch.
    const std::array<float, 4> dy{0.1f, -0.2f, 0.3f, 0.0f};
    std::array<float, 4> dx{}, dweight{}, dbias{};
    ml::layer_norm_backward(dy, x, weight, stats, dx, dweight, dbias);

    // A batch of three rows through RMSNorm with no parameters (nanochat), in place.
    std::vector<float> batch{3.0f, 4.0f, -6.0f, 8.0f, 0.5f, 0.5f};
    std::vector<float> rstd;
    for (std::size_t r = 0; r < 3; ++r) {
        const std::span<float> row(batch.data() + 2 * r, 2);
        rstd.push_back(ml::rms_norm(row, std::span<const float>{}, row));
    }

    // And RMSNorm's backward pass, for one row with a weight.
    const std::array<float, 2> v{3.0f, 4.0f}, g{1.0f, 1.0f}, scale{2.0f, 1.0f};
    std::array<float, 2> out{}, dv{}, dscale{};
    const float r = ml::rms_norm(v, scale, out);
    ml::rms_norm_backward(g, v, scale, r, dv, dscale);
    //! [norm]

    std::printf("  layer_norm: mean %.2f, rstd %.4f, y %.4f %.4f %.4f %.4f\n", static_cast<double>(stats.mean),
                static_cast<double>(stats.rstd), static_cast<double>(y[0]), static_cast<double>(y[1]), static_cast<double>(y[2]),
                static_cast<double>(y[3]));
    ok &= stats.mean == 4.0f && std::abs(stats.rstd - 1.0f / std::sqrt(9.5f + 1e-5f)) < 1e-7f;
    std::printf("  backward: dx %.4f %.4f %.4f %.4f, dbias %.1f %.1f %.1f %.1f\n", static_cast<double>(dx[0]),
                static_cast<double>(dx[1]), static_cast<double>(dx[2]), static_cast<double>(dx[3]), static_cast<double>(dbias[0]),
                static_cast<double>(dbias[1]), static_cast<double>(dbias[2]), static_cast<double>(dbias[3]));
    ok &= std::abs(dx[0] + dx[1] + dx[2] + dx[3]) < 1e-6f && dbias[2] == 0.3f;   // LayerNorm's dx sums to zero
    std::printf("  rms_norm rows: %.4f %.4f | %.4f %.4f | %.4f %.4f\n", static_cast<double>(batch[0]), static_cast<double>(batch[1]),
                static_cast<double>(batch[2]), static_cast<double>(batch[3]), static_cast<double>(batch[4]), static_cast<double>(batch[5]));
    ok &= std::abs(batch[0] - 3.0f / std::sqrt(12.5f)) < 1e-6f && std::abs(batch[4] - 1.0f) < 1e-6f;
    std::printf("  rms_norm backward: dv %.4f %.4f, dscale %.4f %.4f\n", static_cast<double>(dv[0]), static_cast<double>(dv[1]),
                static_cast<double>(dscale[0]), static_cast<double>(dscale[1]));
    ok &= std::abs(dscale[0] - 3.0f / std::sqrt(12.5f)) < 1e-6f;

    std::printf(ok ? "norm demo: ok\n" : "norm demo: FAILED\n");
    return ok ? 0 : 1;
}
