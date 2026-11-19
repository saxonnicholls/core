//
//  latent_attention_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/latent_attention.hpp>: each head against attend over its explicit keys [k^C; k^R],
//  from the same weights, to the bit; the absorbed decode against the training form within the stated bound (double
//  and float), with sinks, a window, YaRN's scale and a sparse selection; a batch of sequences against the sequences
//  one by one; the backward pass against central differences in double for every weight, norm, sink and the input,
//  with and without query compression; the KV cache's size against multi-head attention's as static_asserts; the
//  latent cache's ring; the configuration's checks.
//

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/attention.hpp>
#include <snicholls/core/ml/attention_sink.hpp>
#include <snicholls/core/ml/latent_attention.hpp>
#include <snicholls/core/ml/norm.hpp>
#include <snicholls/core/ml/rope.hpp>
#include <snicholls/core/ml/sparse_attention.hpp>

#include "../support/latent_attention_fixture.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

namespace {

namespace ml = snicholls::ml;
namespace m = snicholls::math;
using snicholls::linalg::reference;

// The KV cache: DeepSeek-V3 keeps 576 numbers per token and layer, where multi-head attention with its 128 heads of
// 128 would keep 32,768 (DeepSeek-V2, Table 1: (d_c + d_h^R) l against 2 n_h d_h l)
constexpr ml::transformer_config v3_as_mha{.d_model = 7168, .n_heads = 128, .n_kv_heads = 128, .head_dim = 128, .d_ff = 18432,
                                           .rope = ml::rope_kind::half};
static_assert(ml::kv_cache_elements_per_token(ml::deepseek_v3) == 512 + 64);
static_assert(ml::kv_cache_elements_per_token(v3_as_mha) == 2 * 128 * 128);
static_assert(ml::kv_cache_elements_per_token(v3_as_mha) / ml::kv_cache_elements_per_token(ml::deepseek_v3) == 56);
static_assert(ml::kv_cache_elements(ml::deepseek_v3, 61, 131072) == std::size_t{576} * 61 * 131072);
static_assert(ml::kv_cache_elements_per_token(ml::llama3_8b) == 2 * 8 * 128);
static_assert(ml::latent_kv_cache_view<ml::deepseek_v3, float>::width == 576);
static_assert(ml::latent_kv_cache_view<ml::tiny_mla, float>::required_size(3, 7) == 3 * 7 * 10);
static_assert(std::is_trivially_copyable_v<ml::latent_kv_cache_view<ml::tiny_mla, float>>);
static_assert(noexcept(ml::kv_cache_elements_per_token(ml::tiny_mla)) && noexcept(ml::kv_cache_elements(ml::tiny_mla, 1, 1)));
static_assert(ml::deepseek_v3.qk_nope_dim() == 128 && ml::deepseek_v3.latent() && !ml::llama3_8b.latent());

// The configuration's checks
constexpr ml::transformer_config with(ml::transformer_config c, std::size_t rope_dim)
{
    c.qk_rope_dim = rope_dim;
    return c;
}
static_assert(!ml::valid(with(ml::tiny_mla, 3)) && !ml::valid(with(ml::tiny_mla, 6)) && !ml::valid(with(ml::tiny_mla, 0)));
static_assert(!ml::valid([] { auto c = ml::tiny_mla; c.n_kv_heads = 1; return c; }()));
static_assert(!ml::valid([] { auto c = ml::llama3_8b; c.attention_sinks = true; return c; }()));
static_assert(!ml::valid([] { auto c = ml::llama3_8b; c.v_head_dim = 64; return c; }()));
static_assert(!ml::valid([] { auto c = ml::tiny_mla; c.rope_scaling = ml::rope_scaling_kind::yarn; c.rope_factor = 4; return c; }()));
static_assert(!ml::valid([] { auto c = ml::tiny_mla; c.mtp_loss_weight = 0.3; return c; }()));

// The variants under test
constexpr ml::transformer_config direct = [] {
    auto c = ml::tiny_mla;
    c.q_lora_rank = 0;
    return c;
}();
constexpr ml::transformer_config sinks = [] {
    auto c = ml::tiny_mla;
    c.attention_sinks = true;
    c.window = 3;
    c.rope_scaling = ml::rope_scaling_kind::yarn_index_ramp;
    c.rope_factor = 4;
    c.rope_original_context = 4;
    return c;
}();
constexpr ml::transformer_config sparse = [] {
    auto c = ml::tiny_mla;
    c.attention_sinks = true;
    c.sparse_top_k = 3;
    return c;
}();
static_assert(ml::valid(direct) && ml::valid(sinks) && ml::valid(sparse));

using snicholls::core_testing::latent_problem;
using snicholls::core_testing::positions;
using snicholls::core_testing::table;

TEST(MlLatentAttention, EachHeadIsAttendOverItsExplicitKeysToTheBit)
{
    using T = float;
    constexpr auto C = ml::tiny_mla;
    using P = latent_problem<C, T>;
    P p(6, 1);
    p.forward();
    // The same weights, step by step: the latents, their norms, the rotation, the up-projections, then each head's keys
    // [k^C_{j,i}; k^R_j] laid out end to end for attend
    const auto w = p.weights();
    std::vector<T> cq(p.rows * P::dq), q(p.rows * P::h * P::dqk), c(p.rows * (P::dc + P::dr)), ckv(p.rows * P::dc), kv(p.rows * P::h * P::kvw);
    ml::linear_sequence(reference, w.q_down, p.xs, cq);
    for (std::size_t t = 0; t < p.rows; ++t)
        ml::rms_norm(std::span<T>(cq).subspan(t * P::dq, P::dq), w.q_norm, std::span<T>(cq).subspan(t * P::dq, P::dq), ml::norm_eps_v<C, T>);
    ml::linear_sequence(reference, w.q_up, cq, q);
    ml::linear_sequence(reference, w.kv_down, p.xs, c);
    for (std::size_t t = 0; t < p.rows; ++t) {
        ml::rms_norm(std::span<T>(c).subspan(t * (P::dc + P::dr), P::dc), w.kv_norm, std::span<T>(ckv).subspan(t * P::dc, P::dc),
                     ml::norm_eps_v<C, T>);
        ml::rope(std::span<T>(c).subspan(t * (P::dc + P::dr) + P::dc, P::dr), table<C, T>().cos_at(t), table<C, T>().sin_at(t),
                 ml::rope_layout::interleaved);
        for (std::size_t hh = 0; hh < P::h; ++hh)
            ml::rope(std::span<T>(q).subspan(t * P::h * P::dqk + hh * P::dqk + P::dn, P::dr), table<C, T>().cos_at(t),
                     table<C, T>().sin_at(t), ml::rope_layout::interleaved);
    }
    ml::linear_sequence(reference, w.kv_up, ckv, kv);
    const T scale = T(1) / std::sqrt(T(P::dqk));
    for (std::size_t t = 0; t < p.rows; ++t)
        for (std::size_t hh = 0; hh < P::h; ++hh) {
            std::vector<T> keys, values;
            for (std::size_t j = 0; j <= t; ++j) {
                const auto row = std::span<const T>(kv).subspan(j * P::h * P::kvw + hh * P::kvw, P::kvw);
                keys.insert(keys.end(), row.begin(), row.begin() + P::dn);
                keys.insert(keys.end(), c.begin() + std::ptrdiff_t(j * (P::dc + P::dr) + P::dc), c.begin() + std::ptrdiff_t((j + 1) * (P::dc + P::dr)));
                values.insert(values.end(), row.begin() + P::dn, row.end());
            }
            std::vector<T> qh(q.begin() + std::ptrdiff_t(t * P::h * P::dqk + hh * P::dqk), q.begin() + std::ptrdiff_t(t * P::h * P::dqk + (hh + 1) * P::dqk));
            std::vector<T> yh(P::dv);
            const T lse = ml::attend_with_sink(qh, keys, values, yh, scale);
            EXPECT_EQ(lse, p.lse[t * P::h + hh]);
            for (std::size_t i = 0; i < P::dv; ++i)
                EXPECT_EQ(yh[i], p.y[t * P::h * P::dv + hh * P::dv + i]) << t << ' ' << hh << ' ' << i;
            if (P::dv == P::dqk) {   // with equal widths attend itself
                std::vector<T> ya(P::dv);
                EXPECT_EQ(ml::attend(qh, keys, values, ya, scale), lse);
            }
        }
    EXPECT_EQ(scale, (ml::latent_attention_scale<C, T>()));
}

template <ml::transformer_config C, typename T>
void absorbed_equals_training(T tolerance)
{
    using P = latent_problem<C, T>;
    P p(7, 1);
    p.forward();
    std::vector<T> storage(ml::latent_kv_cache_view<C, T>::required_size(2, positions));
    const ml::latent_kv_cache_view<C, T> cache(storage, 2, positions);
    T worst = T(0);
    for (std::size_t t = 0; t < p.n; ++t) {
        std::vector<T> out(P::d);
        const auto sel = C.sparse_top_k > 0 ? std::span<const std::uint32_t>(p.selected).subspan(t * C.sparse_top_k, std::min(C.sparse_top_k, t + 1))
                                            : std::span<const std::uint32_t>{};
        ml::latent_attention_forward<C>(p.weights(), std::span<const T>(p.xs).subspan(t * P::d, P::d), t, cache, 1, table<C, T>(), out, sel);
        for (std::size_t i = 0; i < P::d; ++i) {
            const T want = p.out[t * P::d + i];
            worst = std::max(worst, std::abs(out[i] - want) / (T(1) + std::abs(want)));
        }
        // The cache holds the normalised latent and the rotated key, as the training form computed them
        for (std::size_t i = 0; i < P::dc; ++i)
            EXPECT_EQ(cache.row(1, t)[i], p.ckv[t * P::dc + i]);
        for (std::size_t i = 0; i < P::dr; ++i)
            EXPECT_EQ(cache.row(1, t)[P::dc + i], p.k_rope[t * P::dr + i]);
    }
    EXPECT_LT(worst, tolerance);
    EXPECT_GT(worst, T(0));   // a different order of the same sums: equal within rounding, not to the bit
}

TEST(MlLatentAttention, AbsorbedDecodeEqualsTheTrainingFormWithinRounding)
{
    absorbed_equals_training<ml::tiny_mla, double>(1e-12);
    absorbed_equals_training<direct, double>(1e-12);
    absorbed_equals_training<sinks, double>(1e-12);
    absorbed_equals_training<sparse, double>(1e-12);
    absorbed_equals_training<ml::tiny_mla, float>(2e-5f);
    absorbed_equals_training<sinks, float>(2e-5f);
}

TEST(MlLatentAttention, ABatchOfSequencesIsEachSequenceOnItsOwn)
{
    using P = latent_problem<sinks, float>;
    P batch(5, 3);
    batch.forward();
    for (std::size_t s = 0; s < 3; ++s) {
        P one(5, 1);
        std::copy_n(batch.xs.begin() + std::ptrdiff_t(s * 5 * P::d), 5 * P::d, one.xs.begin());
        one.forward();
        for (std::size_t i = 0; i < 5 * P::d; ++i)
            EXPECT_EQ(one.out[i], batch.out[s * 5 * P::d + i]);
    }
}

template <ml::transformer_config C>
void central_differences()
{
    using P = latent_problem<C, double>;
    P p(5, 2);
    p.forward();
    p.backward();
    const double h = 1e-6;
    const auto check = [&](std::vector<double>& x, const std::vector<double>& analytic, const char* name) {
        ASSERT_EQ(x.size(), analytic.size()) << name;
        for (std::size_t i = 0; i < x.size(); ++i) {
            const double x0 = x[i];
            x[i] = x0 + h;
            const double up = p.loss();
            x[i] = x0 - h;
            const double down = p.loss();
            x[i] = x0;
            EXPECT_NEAR(analytic[i], (up - down) / (2 * h), 2e-7 * (1 + std::abs(analytic[i]))) << name << ' ' << i;
        }
    };
    check(p.xs, p.dxs, "x");
    check(p.q_up, p.g_q_up, "q_up");
    check(p.kv_down, p.g_kv_down, "kv_down");
    check(p.kv_norm, p.g_kv_norm, "kv_norm");
    check(p.kv_up, p.g_kv_up, "kv_up");
    check(p.o, p.g_o, "o");
    if constexpr (C.q_lora_rank > 0) {
        check(p.q_down, p.g_q_down, "q_down");
        check(p.q_norm, p.g_q_norm, "q_norm");
    }
    if constexpr (C.attention_sinks)
        check(p.sink, p.g_sink, "sinks");
}

TEST(MlLatentAttention, BackwardMatchesCentralDifferences) { central_differences<ml::tiny_mla>(); }
TEST(MlLatentAttention, BackwardWithoutQueryCompressionMatchesCentralDifferences) { central_differences<direct>(); }
TEST(MlLatentAttention, BackwardWithSinksWindowAndYarnMatchesCentralDifferences) { central_differences<sinks>(); }
TEST(MlLatentAttention, BackwardOverASparseSelectionMatchesCentralDifferences) { central_differences<sparse>(); }

TEST(MlLatentAttention, ScaleCarriesYarnsTemperatureSquared)
{
    const double f = 0.1 * std::log(40.0) + 1;
    EXPECT_NEAR((ml::latent_attention_scale<ml::deepseek_v3, double>()), f * f / std::sqrt(192.0), 1e-15);
    EXPECT_EQ((ml::latent_attention_scale<ml::tiny_mla, double>()), 1 / std::sqrt(6.0));
}

TEST(MlLatentAttention, TheLatentCacheIsARing)
{
    ml::latent_kv_cache<ml::tiny_mla, float, 2, 3> owned;
    const auto cache = owned.view();
    EXPECT_EQ(cache.layers(), 2u);
    EXPECT_EQ(cache.capacity(), 3u);
    for (std::size_t pos = 0; pos < 5; ++pos) {
        std::vector<float> r(10, static_cast<float>(pos));
        cache.append(1, pos, r);
    }
    EXPECT_EQ(cache.row(1, 4)[0], 4.0f);
    EXPECT_EQ(cache.row(1, 2)[9], 2.0f);
    EXPECT_EQ(cache.row(1, 1)[0], 4.0f);   // position 4 took slot 1
    EXPECT_EQ(cache.row(0, 0)[0], 0.0f);
}

} // namespace
