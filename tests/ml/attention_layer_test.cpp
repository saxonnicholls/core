//
//  attention_layer_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/core/ml/attention_layer.hpp>

#include <snicholls/core/combinatorics/permutations.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
using W = ml::dense_weights<float>;

constexpr ml::transformer_config head8{.d_model = 8, .n_heads = 1, .n_kv_heads = 1, .head_dim = 8, .d_ff = 8};
constexpr ml::transformer_config capped{.d_model = 8, .n_heads = 1, .n_kv_heads = 1, .head_dim = 8, .d_ff = 8, .attention_softcap = 2};
constexpr ml::transformer_config gqa{.d_model = 16, .n_heads = 4, .n_kv_heads = 2, .head_dim = 4, .d_ff = 16};
constexpr ml::transformer_config windowed{.d_model = 8, .n_heads = 2, .n_kv_heads = 1, .head_dim = 4, .d_ff = 8,
                                          .rope = ml::rope_kind::half, .window = 3};

std::vector<float> pattern(std::size_t n, float scale, std::size_t salt = 0)
{
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = scale * (float((i * 37 + salt * 11) % 23) / 23.0f - 0.5f);
    return v;
}

ml::kv_runs<float> one_run(const std::vector<float>& k, const std::vector<float>& v)
{
    ml::kv_runs<float> r{};
    r.keys[0] = k;
    r.values[0] = v;
    return r;
}

// One head over a contiguous run is attend itself: the same fold, the same bits
TEST(MlAttentionLayer, OneHeadOverOneRunIsAttendToTheBit)
{
    const auto q = pattern(8, 2.0f), k = pattern(40, 1.5f, 1), v = pattern(40, 1.0f, 2);
    std::array<float, 8> got{}, want{};
    const float lse = ml::attention_head<head8, float>(q, one_run(k, v), got);
    const float want_lse = ml::attend(q, k, v, want, 1.0f / std::sqrt(8.0f));
    EXPECT_EQ(got, want);
    EXPECT_EQ(lse, want_lse);
}

TEST(MlAttentionLayer, TheSoftCapBoundsEveryScore)
{
    const auto q = pattern(8, 6.0f), k = pattern(24, 6.0f, 1), v = pattern(24, 1.0f, 2);
    std::array<float, 8> got{};
    ml::attention_head<capped, float>(q, one_run(k, v), got);
    double s[3], m = -INFINITY, sum = 0;
    for (int j = 0; j < 3; ++j) {
        double d = 0;
        for (int i = 0; i < 8; ++i)
            d += double(q[i]) * double(k[j * 8 + i]);
        s[j] = 2.0 * std::tanh(d / std::sqrt(8.0) / 2.0);
        EXPECT_LE(std::abs(s[j]), 2.0);
        m = std::max(m, s[j]);
    }
    for (double& e : s)
        sum += (e = std::exp(e - m));
    for (int i = 0; i < 8; ++i)
        EXPECT_NEAR(got[i], (s[0] * v[i] + s[1] * v[8 + i] + s[2] * v[16 + i]) / sum, 1e-6);
}

TEST(MlAttentionLayer, NoKeysGiveZeroAndMinusInfinity)
{
    const auto q = pattern(8, 1.0f);
    std::array<float, 8> out{};
    out.fill(7.0f);
    EXPECT_EQ((ml::attention_head<head8, float>(q, ml::kv_runs<float>{}, out)), -INFINITY);
    for (float x : out)
        EXPECT_EQ(x, 0.0f);
}

// Without positions or a mask, attention does not care in which order the keys were cached
TEST(MlAttentionLayer, AttentionIsPermutationEquivariantOverTheCachedKeys)
{
    constexpr std::size_t n = 5;
    const auto q = pattern(8, 2.0f), k = pattern(8 * n, 1.5f, 1), v = pattern(8 * n, 1.0f, 2);
    std::array<float, 8> base{};
    ml::attention_head<head8, float>(q, one_run(k, v), base);
    std::size_t orderings = 0;
    snicholls::for_each_permutation<n>([&](const std::array<std::size_t, n>& p) {
        std::vector<float> pk(k.size()), pv(v.size());
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t i = 0; i < 8; ++i) {
                pk[j * 8 + i] = k[p[j] * 8 + i];
                pv[j * 8 + i] = v[p[j] * 8 + i];
            }
        std::array<float, 8> got{};
        ml::attention_head<head8, float>(q, one_run(pk, pv), got);
        for (std::size_t i = 0; i < 8; ++i)
            EXPECT_NEAR(got[i], base[i], 1e-6f);
        ++orderings;
    });
    EXPECT_EQ(orderings, 120u);
}

// A layer whose projections are identities: q, k and v are the input itself
template <ml::transformer_config C>
struct identity_layer {
    std::vector<float> eye_q, eye_kv, eye_o;
    identity_layer() : eye_q(C.q_size() * C.d_model), eye_kv(C.kv_size() * C.d_model), eye_o(C.d_model * C.q_size())
    {
        for (std::size_t i = 0; i < C.q_size() && i < C.d_model; ++i)
            eye_q[i * C.d_model + i] = 1.0f;
        for (std::size_t i = 0; i < C.kv_size() && i < C.d_model; ++i)
            eye_kv[i * C.d_model + i] = 1.0f;
        for (std::size_t i = 0; i < C.d_model && i < C.q_size(); ++i)
            eye_o[i * C.q_size() + i] = 1.0f;
    }
    ml::attention_weights<W> weights() const
    {
        return {W{eye_q, C.q_size(), C.d_model}, W{eye_kv, C.kv_size(), C.d_model}, W{eye_kv, C.kv_size(), C.d_model},
                W{eye_o, C.d_model, C.q_size()}};
    }
};

// Query heads 0 and 1 share key-value head 0, heads 2 and 3 share head 1: given the same query, a pair agrees
TEST(MlAttentionLayer, GroupedQueryHeadsReadTheirSharedKeyValueHead)
{
    std::vector<float> keys(ml::kv_cache_view<gqa>::required_size(1, 4)), values(keys.size());
    const ml::kv_cache_view<gqa> cache(keys, values, 1, 4);
    for (std::size_t p = 0; p < 3; ++p)
        cache.append(0, p, pattern(8, 1.0f, p), pattern(8, 1.0f, p + 7));
    const auto q = pattern(4, 2.0f);
    std::array<std::array<float, 4>, 4> out{};
    for (std::size_t h = 0; h < 4; ++h)
        ml::attention_head<gqa, float>(q, cache.visible(0, h / gqa.group(), ml::attention_window(2)), out[h]);
    EXPECT_EQ(out[0], out[1]);
    EXPECT_EQ(out[2], out[3]);
    EXPECT_NE(out[0], out[2]);
}

// A sliding window over a ring of three slots and over a flat cache of every position: the same bits
TEST(MlAttentionLayer, ARingCacheGivesTheFlatCachesBits)
{
    const identity_layer<windowed> layer;
    static const ml::rope_table<float, 16, 4> table{};
    std::vector<float> rk(ml::kv_cache_view<windowed>::required_size(1, 3)), rv(rk.size());
    std::vector<float> fk(ml::kv_cache_view<windowed>::required_size(1, 10)), fv(fk.size());
    const ml::kv_cache_view<windowed> ring(rk, rv, 1, 3), flat(fk, fv, 1, 10);
    for (std::size_t p = 0; p < 10; ++p) {
        const auto x = pattern(8, 1.0f, p);
        std::array<float, 8> a{}, b{};
        ml::attention_forward<windowed>(layer.weights(), x, p, ring, 0, table, a);
        ml::attention_forward<windowed>(layer.weights(), x, p, flat, 0, table, b);
        EXPECT_EQ(a, b) << "position " << p;
    }
}

TEST(MlAttentionLayer, TheFirstTokenAttendsToItselfAlone)
{
    const identity_layer<head8> layer;
    std::vector<float> k(ml::kv_cache_view<head8>::required_size(1, 2)), v(k.size());
    const ml::kv_cache_view<head8> cache(k, v, 1, 2);
    const auto x = pattern(8, 1.0f);
    std::array<float, 8> out{};
    ml::attention_forward<head8>(layer.weights(), x, 0, cache, 0, ml::no_rope, out);
    for (std::size_t i = 0; i < 8; ++i)
        EXPECT_FLOAT_EQ(out[i], x[i]);   // softmax over one key is 1, so the output is the token's own value
}

constexpr bool compile_time()
{
    constexpr ml::transformer_config c{.d_model = 2, .n_heads = 1, .n_kv_heads = 1, .head_dim = 2, .d_ff = 2};
    std::array<float, 4> eye{1, 0, 0, 1};
    const ml::attention_weights<W> w{W{eye, 2, 2}, W{eye, 2, 2}, W{eye, 2, 2}, W{eye, 2, 2}};
    std::array<float, 4> k{}, v{};
    const ml::kv_cache_view<c> cache(k, v, 1, 2);
    std::array<float, 2> x{1, 2}, out{};
    ml::attention_forward<c>(w, x, 0, cache, 0, ml::no_rope, out);
    return out[0] == 1 && out[1] == 2 && k[0] == 1 && v[1] == 2;
}
static_assert(compile_time());

} // namespace

namespace {

// A paged cache and a flat cache of the same positions: attention gives the same bits
TEST(MlAttentionLayer, APagedCacheGivesTheFlatCachesBits)
{
    const identity_layer<windowed> layer;
    static const ml::rope_table<float, 16, 4> table{};
    using paged = ml::paged_kv_cache_view<windowed, float, 8>;
    std::vector<float> pk(paged::required_size(5, 1, 2)), pv(pk.size());
    std::vector<float> fk(ml::kv_cache_view<windowed>::required_size(1, 10)), fv(fk.size());
    const std::array<std::uint32_t, 5> pages{4, 0, 3, 1, 2};
    const paged p(pk, pv, 1, 2, pages);
    const ml::kv_cache_view<windowed> flat(fk, fv, 1, 10);
    for (std::size_t pos = 0; pos < 10; ++pos) {
        const auto x = pattern(8, 1.0f, pos);
        std::array<float, 8> a{}, b{};
        ml::attention_forward<windowed>(layer.weights(), x, pos, p, 0, table, a);
        ml::attention_forward<windowed>(layer.weights(), x, pos, flat, 0, table, b);
        EXPECT_EQ(a, b) << "position " << pos;
    }
}

} // namespace
