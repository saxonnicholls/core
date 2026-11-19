//
//  lora_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/lora.hpp>: a rank-2 adapter on a frozen 8 x 64 base, first dense and then QLoRA's NF4,
//  initialised so that it changes nothing, trained with AdamW through the optimiser's view of the adapter alone to fit
//  a low-rank change of the base, then merged into the weights for inference and unmerged. Also the batched forms
//  over a sequence and head by head. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/lora.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/optim.hpp>
#include <snicholls/core/stats/random.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace ml = snicholls::ml;
namespace stats = snicholls::stats;

namespace {

constexpr std::size_t rows = 8, cols = 64, rank = 2, n = 16;

// The mean squared error of the adapted layer against targets, its gradient into the adapter
template <typename W>
double step(const ml::lora_weights<W, float>& lw, const std::vector<float>& xs, const std::vector<float>& targets, std::span<float> da,
            std::span<float> db)
{
    std::fill(da.begin(), da.end(), 0.0f);
    std::fill(db.begin(), db.end(), 0.0f);
    double loss = 0;
    for (std::size_t t = 0; t < n; ++t) {
        const auto x = std::span<const float>(xs).subspan(t * cols, cols);
        std::array<float, rows> y{}, dy{};
        std::array<float, rank> u{}, du{};
        ml::lora_linear(lw, x, y, u);
        for (std::size_t i = 0; i < rows; ++i) {
            const float e = y[i] - targets[t * rows + i];
            loss += 0.5 * e * e / n;
            dy[i] = e / n;
        }
        ml::lora_linear_backward(lw, x, u, dy, {}, da, db, du);
    }
    return loss;
}

} // namespace

int main()
{
    std::printf("snicholls-core lora demo\n");
    bool ok = true;
    const auto& backend = snicholls::linalg::reference;
    stats::wyrand g(7);
    const auto uniform = [&](double s) { return static_cast<float>(s * (2.0 * stats::uniform_real(g) - 1.0)); };
    std::vector<float> w(rows * cols), xs(n * cols), targets(n * rows);
    for (auto& v : w)
        v = uniform(0.1);
    for (auto& v : xs)
        v = uniform(1.0);
    // The targets: the base plus a rank-1 change
    for (std::size_t t = 0; t < n; ++t)
        for (std::size_t i = 0; i < rows; ++i) {
            float y = 0, proj = 0;
            for (std::size_t j = 0; j < cols; ++j) {
                y += w[i * cols + j] * xs[t * cols + j];
                proj += (j % 2 == 0 ? 0.05f : -0.05f) * xs[t * cols + j];
            }
            targets[t * rows + i] = y + (static_cast<float>(i) - 3.5f) * 0.2f * proj;
        }

    //! [lora]
    // A frozen base, an adapter A (r x cols, Kaiming-uniform) and B (rows x r, zero): at first the layer is the base
    std::vector<float> a(rank * cols), b(rows * rank), da(a.size()), db(b.size());
    ml::lora_initialise<float>(a, b, cols, g);
    const ml::lora_weights<ml::dense_weights<float>, float> lw{{w, rows, cols}, a, b, rank, ml::lora_scaling(4.0f, rank)};
    const ml::lora_parameters<float> adapter{a, b, da, db};   // what the optimiser sees: A and B, nothing of W
    std::vector<float> ma(a.size()), va(a.size()), mb(b.size()), vb(b.size());
    for (std::uint64_t t = 1; t <= 200; ++t) {
        step(lw, xs, targets, da, db);
        ml::adamw_step(adapter.parameters()[0], adapter.gradients()[0], ma, va, t, ml::adamw_config<float>{.learning_rate = 0.01f});
        ml::adamw_step(adapter.parameters()[1], adapter.gradients()[1], mb, vb, t, ml::adamw_config<float>{.learning_rate = 0.01f});
    }
    //! [lora]
    std::vector<float> fresh_a(a.size()), fresh_b(b.size());   // a new adapter, A Gaussian this time: B = 0, so the base alone
    ml::lora_initialise<float>(fresh_a, fresh_b, cols, g, ml::lora_init::gaussian);
    const double before = step(ml::lora_weights<ml::dense_weights<float>, float>{{w, rows, cols}, fresh_a, fresh_b, rank, 2.0f}, xs, targets, da, db);
    const double after = step(lw, xs, targets, da, db);
    std::printf("  dense base: loss %.6f with B = 0, %.2e after 200 AdamW steps on %zu of %zu numbers\n", before, after, adapter.size(),
                adapter.size() + w.size());
    ok &= after < before * 0.01;

    // Merge for inference: one dense product, the same outputs within rounding; unmerge to train on
    std::vector<float> merged = w, y(rows), ym(rows);
    std::array<float, rank> u{};
    ml::lora_merge<float>(merged, cols, a, b, rank, lw.scaling);
    ml::lora_linear(lw, std::span<const float>(xs).first(cols), y, u);
    ml::linear(ml::dense_weights<float>{merged, rows, cols}, std::span<const float>(xs).first(cols), ym);
    ml::lora_unmerge<float>(merged, cols, a, b, rank, lw.scaling);
    float worst = 0, drift = 0;
    for (std::size_t i = 0; i < rows; ++i)
        worst = std::max(worst, std::fabs(y[i] - ym[i]));
    for (std::size_t i = 0; i < w.size(); ++i)
        drift = std::max(drift, std::fabs(merged[i] - w[i]));
    std::printf("  merged: largest output difference %.1e; after unmerge the weights moved by at most %.1e\n", worst, drift);
    ok &= worst < 1e-5f && drift < 1e-7f;

    // QLoRA: the same adapter over the base in NF4, dequantised on the fly
    std::vector<std::uint8_t> codes(ml::nf4_code_bytes(w.size()));
    std::vector<ml::float8_e4m3> scales(ml::nf4_block_count(w.size()));
    std::vector<float> groups(ml::nf4_group_count(w.size()));
    const float mean = ml::nf4_quantize(w, codes, scales, groups);
    const ml::lora_weights<ml::nf4_weights, float> lq{{codes, scales, groups, mean, rows, cols}, a, b, rank, lw.scaling};
    const double quantised = step(lq, xs, targets, da, db);
    std::printf("  NF4 base with the trained adapter: loss %.6f (the base's quantisation error)\n", quantised);
    ok &= quantised < before;

    // The batched forms through the backend: a sequence, and two heads of four outputs
    std::vector<float> ys(n * rows), us(n * rank), dys(n * rows, 0.01f), dxs(n * cols, 0.0f), dus(n * rank), yh(n * rows), uh(n * rank);
    ml::lora_sequence(backend, lw, xs, ys, us);
    ml::lora_sequence_backward(backend, lw, xs, us, dys, dxs, da, db, dus);
    ml::lora_heads(backend, lw, xs, yh, uh, {}, 2);
    ok &= std::fabs(yh[0] - ys[0]) < 1e-6f && std::fabs(yh[n * 4] - ys[4]) < 1e-6f;   // head 1 starts at output 4
    ml::lora_heads_backward(backend, lw, xs, uh, dys, dxs, da, db, dus, 2);
    ml::lora_sequence(backend, lq, xs, ys, us);
    ml::lora_sequence_backward(backend, lq, xs, us, dys, dxs, da, db, dus);
    std::vector<float> dx1(cols, 0.0f);
    ml::linear_backward_input(lw.base, std::span<const float>(dys).first(rows), dx1);
    ml::linear_sequence_backward_input(backend, lw.base, std::span<const float>(dys).first(rows), dx1);
    ok &= std::isfinite(dxs[0]) && std::isfinite(dx1[0]);
    std::printf("  heads: output (head 0, token 0, 0) %.5f\n", yh[0]);

    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
