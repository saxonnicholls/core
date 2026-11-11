//
//  optim_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/optim.hpp>: a small least-squares fit trained with AdamW under a warm-up and
//  cosine schedule and gradient clipping, nanochat's trapezoid schedule, Muon's Newton-Schulz polynomial on a
//  singular value, and the FLOPs and utilisation of GPT-2 small. Exits non-zero if a result is not what it
//  documents.
//

#include <snicholls/core/ml/optim.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>

int main()
{
    std::printf("snicholls-core optim demo\n");
    namespace ml = snicholls::ml;
    bool ok = true;

    //! [optim]
    // Fit y = 2x - 1 from four points: two parameters, their gradients, and AdamW's two moments
    const std::array<float, 4> xs{0.0f, 1.0f, 2.0f, 3.0f}, ys{-1.0f, 1.0f, 3.0f, 5.0f};
    std::array<float, 2> w{0.0f, 0.0f}, grad{}, m{}, v{};
    constexpr std::uint64_t steps = 3000, warmup = 100;
    for (std::uint64_t t = 0; t < steps; ++t) {
        grad = {};
        for (std::size_t i = 0; i < 4; ++i) {
            const float e = w[0] * xs[i] + w[1] - ys[i];
            grad[0] += 0.5f * e * xs[i];
            grad[1] += 0.5f * e;
        }
        // Clip by the global norm (one shard here; shards' squared norms add), then step at the scheduled rate
        const float clip = ml::clip_coefficient(std::sqrt(ml::squared_norm(grad)), 1.0f);
        for (float& g : grad)
            g *= clip;
        const float lr = ml::warmup_cosine<float>(t, warmup, steps, 0.05f, 0.0f);
        ml::adamw_step(w, grad, m, v, t + 1, {.learning_rate = lr, .beta2 = 0.95f});
    }
    //! [optim]
    std::printf("  fitted y = %.4f x + %.4f\n", static_cast<double>(w[0]), static_cast<double>(w[1]));
    ok &= std::fabs(w[0] - 2.0f) < 1e-3f && std::fabs(w[1] + 1.0f) < 1e-3f;

    // nanochat's schedule: warm-up, flat, warm-down to a tenth
    std::printf("  trapezoid at 0, 50, 90, 99 of 100:");
    for (const std::uint64_t t : {0, 50, 90, 99})
        std::printf(" %.3f", ml::warmup_warmdown<double>(t, 100, 5, 20, 0.1));
    std::printf("\n");
    ok &= ml::warmup_warmdown<double>(50, 100, 5, 20, 0.1) == 1.0;

    // Muon: five Newton-Schulz steps carry a small singular value near 1
    double s = 0.05;
    for (int k = 0; k < ml::muon_newton_schulz<double>::steps; ++k)
        s = ml::newton_schulz_polynomial(s);
    std::printf("  Newton-Schulz: 0.05 -> %.4f after %d steps\n", s, ml::muon_newton_schulz<double>::steps);
    ok &= s > 0.68 && s < 1.2;

    // GPT-2 small at 400k tokens/s on one H100 in bfloat16
    const double f = ml::transformer_flops_per_token(85'000'000, 12, 12, 64, 1024);
    const double mfu = ml::model_flops_utilisation(f, 400e3, 989e12);
    std::printf("  GPT-2 small: %.3g FLOPs per token, MFU %.1f%% at 400k tokens/s\n", f, 100 * mfu);
    ok &= mfu > 0.2 && mfu < 0.3;

    std::printf(ok ? "optim demo: ok\n" : "optim demo: FAILED\n");
    return ok ? 0 : 1;
}
