//
//  latent_block_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/latent_block.hpp>: the block decoded token by token through a latent cache against the
//  batched training pass within rounding; the batched backward pass against central differences in double for the
//  input, both norms, the MLP and the attention; the overloads living beside the standard block's.
//

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/latent_block.hpp>

#include "../support/latent_attention_fixture.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
using snicholls::linalg::reference;
using snicholls::core_testing::latent_problem;
using snicholls::core_testing::positions;
using snicholls::core_testing::table;

using snicholls::core_testing::block_problem;

TEST(MlLatentBlock, DecodingTokenByTokenIsTheTrainingPassWithinRounding)
{
    constexpr auto C = ml::tiny_mla;
    block_problem<C, double> p(6, 1);
    const auto trained = p.forward();
    std::vector<double> storage(ml::latent_kv_cache_view<C, double>::required_size(1, positions));
    const ml::latent_kv_cache_view<C, double> cache(storage, 1, positions);
    for (std::size_t t = 0; t < 6; ++t) {
        std::vector<double> x(p.xs.begin() + std::ptrdiff_t(t * C.d_model), p.xs.begin() + std::ptrdiff_t((t + 1) * C.d_model));
        ml::block_forward<C>(p.weights(), x, t, cache, 0, table<C, double>());
        for (std::size_t i = 0; i < C.d_model; ++i)
            EXPECT_NEAR(x[i], trained[t * C.d_model + i], 1e-12 * (1 + std::abs(x[i])));
    }
}

TEST(MlLatentBlock, BackwardMatchesCentralDifferences)
{
    constexpr auto C = ml::tiny_mla;
    using P = block_problem<C, double>;
    P p(4, 2);
    const auto dout = p.dxs;
    p.forward();
    auto dx = dout;
    ml::block_backward<C>(reference, p.weights(), table<C, double>(), p.act(), std::span<double>(dx), p.grads(), 2);
    const auto loss = [&] {
        const auto y = p.forward();
        double l = 0;
        for (std::size_t i = 0; i < y.size(); ++i)
            l += dout[i] * y[i];
        return l;
    };
    const double h = 1e-6;
    const auto check = [&](std::vector<double>& x, const std::vector<double>& analytic, const char* name) {
        for (std::size_t i = 0; i < x.size(); ++i) {
            const double x0 = x[i];
            x[i] = x0 + h;
            const double up = loss();
            x[i] = x0 - h;
            const double down = loss();
            x[i] = x0;
            EXPECT_NEAR(analytic[i], (up - down) / (2 * h), 2e-7 * (1 + std::abs(analytic[i]))) << name << ' ' << i;
        }
    };
    check(p.xs, dx, "x");
    check(p.norm1, p.g_norm1, "norm1");
    check(p.norm2, p.g_norm2, "norm2");
    check(p.up, p.g_up, "up");
    check(p.gate, p.g_gate, "gate");
    check(p.down, p.g_down, "down");
    check(p.attn.kv_up, p.attn.g_kv_up, "kv_up");
    check(p.attn.q_down, p.attn.g_q_down, "q_down");
}

} // namespace
