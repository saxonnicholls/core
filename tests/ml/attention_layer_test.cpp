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

// Training: the sequence forward, the head's and the layer's backward passes
using WD = ml::dense_weights<double>;

std::vector<double> wave(std::size_t n, double a, double b)
{
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = a * std::sin(0.43 * double(i) + b);
    return v;
}

constexpr ml::transformer_config t_plain{.d_model = 8, .n_heads = 2, .n_kv_heads = 2, .head_dim = 4, .d_ff = 8};
constexpr ml::transformer_config t_gqa_rope{.d_model = 8, .n_heads = 4, .n_kv_heads = 2, .head_dim = 4, .d_ff = 8,
                                            .rope = ml::rope_kind::half, .window = 3};
constexpr ml::transformer_config t_cap{.d_model = 8, .n_heads = 2, .n_kv_heads = 1, .head_dim = 4, .d_ff = 8,
                                       .attention_softcap = 1.5};
constexpr ml::transformer_config t_qk_before{.d_model = 8, .n_heads = 2, .n_kv_heads = 1, .head_dim = 4, .d_ff = 8,
                                             .qk_norm = ml::qk_norm_kind::before_rope, .rope = ml::rope_kind::interleaved};
constexpr ml::transformer_config t_qk_after{.d_model = 8, .n_heads = 2, .n_kv_heads = 2, .head_dim = 4, .d_ff = 8,
                                            .qk_norm = ml::qk_norm_kind::after_rope, .rope = ml::rope_kind::half,
                                            .window = 4};

template <ml::transformer_config C>
struct train_layer {
    static constexpr std::size_t n = 6;
    std::vector<double> wq, wk, wv, wo, bq, bk, bv, bo, nq, nk, x;
    train_layer()
        : wq(wave(C.q_size() * C.d_model, 0.6, 0.1)), wk(wave(C.kv_size() * C.d_model, 0.6, 1.1)),
          wv(wave(C.kv_size() * C.d_model, 0.7, 2.1)), wo(wave(C.d_model * C.q_size(), 0.5, 3.1)), bq(wave(C.q_size(), 0.2, 0.3)),
          bk(wave(C.kv_size(), 0.2, 1.3)), bv(wave(C.kv_size(), 0.2, 2.3)), bo(wave(C.d_model, 0.1, 0.8)),
          nq(C.qk_norm == ml::qk_norm_kind::none ? 0 : C.head_dim), nk(nq.size()), x(wave(n * C.d_model, 1.2, 0.5))
    {
        for (std::size_t i = 0; i < nq.size(); ++i) {
            nq[i] = 1.0 + 0.1 * double(i);
            nk[i] = 0.9 - 0.05 * double(i);
        }
    }
    ml::attention_weights<WD, double> weights() const
    {
        return {WD{wq, C.q_size(), C.d_model}, WD{wk, C.kv_size(), C.d_model}, WD{wv, C.kv_size(), C.d_model},
                WD{wo, C.d_model, C.q_size()}, bq, bk, bv, bo, nq, nk};
    }
};

template <ml::transformer_config C>
struct train_buffers {
    static constexpr std::size_t n = train_layer<C>::n;
    static constexpr bool raw = C.qk_norm != ml::qk_norm_kind::none;
    std::vector<double> q = std::vector<double>(n * C.q_size()), k = std::vector<double>(n * C.kv_size()), v = k,
                        qr = std::vector<double>(raw ? q.size() : 0), kr = std::vector<double>(raw ? k.size() : 0), y = q,
                        lse = std::vector<double>(n * C.n_heads), out = std::vector<double>(n * C.d_model);
    ml::attention_activations<double> act() { return {q, k, v, qr, kr, y, lse}; }
};

static const ml::rope_table<double, 16, 4> rope4{};

template <ml::transformer_config C>
const auto& rope_for()
{
    if constexpr (C.rope == ml::rope_kind::none)
        return ml::no_rope;
    else
        return rope4;
}

template <ml::transformer_config C>
void sequence_is_attention_forward_token_by_token()
{
    const train_layer<C> l;
    train_buffers<C> b;
    ml::attention_forward_sequence<C>(l.weights(), l.x, rope_for<C>(), b.act(), b.out);
    constexpr std::size_t n = train_layer<C>::n;
    std::vector<double> keys(ml::kv_cache_view<C, double>::required_size(1, n)), values(keys.size());
    const ml::kv_cache_view<C, double> cache(keys, values, 1, n);
    for (std::size_t t = 0; t < n; ++t) {
        std::vector<double> out(C.d_model);
        ml::attention_forward<C>(l.weights(), std::span<const double>(l.x).subspan(t * C.d_model, C.d_model), t, cache, 0,
                                 rope_for<C>(), out);
        for (std::size_t i = 0; i < C.d_model; ++i)
            EXPECT_EQ(out[i], b.out[t * C.d_model + i]) << t << " " << i;
    }
    EXPECT_EQ(keys, b.k);   // with capacity n the flat cache is the saved keys, head by head
    EXPECT_EQ(values, b.v);
}

TEST(MlAttentionLayer, TheSequenceForwardIsAttentionForwardTokenByTokenToTheBit)
{
    sequence_is_attention_forward_token_by_token<t_plain>();
    sequence_is_attention_forward_token_by_token<t_gqa_rope>();
    sequence_is_attention_forward_token_by_token<t_cap>();
    sequence_is_attention_forward_token_by_token<t_qk_before>();
    sequence_is_attention_forward_token_by_token<t_qk_after>();
}

template <ml::transformer_config C>
void backward_is_central_differences()
{
    train_layer<C> l;
    constexpr std::size_t n = train_layer<C>::n;
    const auto gout = wave(n * C.d_model, 0.9, 1.7);
    auto loss = [&] {
        train_buffers<C> b;
        ml::attention_forward_sequence<C>(l.weights(), l.x, rope_for<C>(), b.act(), b.out);
        double s = 0;
        for (std::size_t i = 0; i < b.out.size(); ++i)
            s += gout[i] * b.out[i];
        return s;
    };
    train_buffers<C> b;
    ml::attention_forward_sequence<C>(l.weights(), l.x, rope_for<C>(), b.act(), b.out);
    std::vector<double> dq(b.q.size()), dk(b.k.size()), dv(b.v.size()), dwq(l.wq.size()), dwk(l.wk.size()), dwv(l.wv.size()),
        dwo(l.wo.size()), dbq(l.bq.size()), dbk(l.bk.size()), dbv(l.bv.size()), dbo(l.bo.size()), dnq(l.nq.size()),
        dnk(l.nk.size()), dx(l.x.size());
    ml::attention_backward<C>(l.weights(), l.x, rope_for<C>(), b.act(), gout, dx,
                              ml::attention_gradients<double>{dq, dk, dv, dwq, dwk, dwv, dwo, dbq, dbk, dbv, dbo, dnq, dnk});
    auto check = [&](std::vector<double>& p, const std::vector<double>& grad, const char* what) {
        for (std::size_t i = 0; i < p.size(); ++i) {
            const double keep = p[i], e = 1e-6;
            p[i] = keep + e;
            const double hi = loss();
            p[i] = keep - e;
            const double lo = loss();
            p[i] = keep;
            EXPECT_NEAR(grad[i], (hi - lo) / (2 * e), 2e-7) << what << " " << i;
        }
    };
    check(l.x, dx, "x");
    check(l.wq, dwq, "wq");
    check(l.wk, dwk, "wk");
    check(l.wv, dwv, "wv");
    check(l.wo, dwo, "wo");
    check(l.bq, dbq, "bq");
    check(l.bk, dbk, "bk");
    check(l.bv, dbv, "bv");
    check(l.bo, dbo, "bo");
    check(l.nq, dnq, "q_norm");
    check(l.nk, dnk, "k_norm");
}

TEST(MlAttentionLayer, BackwardMatchesCentralDifferencesForMultiHeadAttention)
{
    backward_is_central_differences<t_plain>();
}

TEST(MlAttentionLayer, BackwardMatchesCentralDifferencesWithGroupedQueriesRopeAndAWindow)
{
    backward_is_central_differences<t_gqa_rope>();
}

TEST(MlAttentionLayer, BackwardMatchesCentralDifferencesWithASoftCap)
{
    backward_is_central_differences<t_cap>();
}

TEST(MlAttentionLayer, BackwardMatchesCentralDifferencesWithQkNormBeforeAndAfterRope)
{
    backward_is_central_differences<t_qk_before>();
    backward_is_central_differences<t_qk_after>();
}

TEST(MlAttentionLayer, HeadBackwardWithoutACapIsAttendBackwardToTheBit)
{
    const auto q = pattern(8, 2.0f), k = pattern(40, 1.5f, 1), v = pattern(40, 1.0f, 2), dout = pattern(8, 1.0f, 3);
    std::array<float, 8> out{};
    const float lse = ml::attention_head<head8, float>(q, one_run(k, v), out);
    std::array<float, 8> dq1{}, dq2{};
    std::vector<float> dk1(40), dv1(40), dk2(40), dv2(40);
    ml::attention_head_backward<head8, float>(q, k, v, out, dout, lse, dq1, dk1, dv1);
    ml::attend_backward(q, k, v, out, dout, lse, 1.0f / std::sqrt(8.0f), dq2, dk2, dv2);
    EXPECT_EQ(dq1, dq2);
    EXPECT_EQ(dk1, dk2);
    EXPECT_EQ(dv1, dv2);
}

template <typename Wt>
concept layer_trainable = requires(const ml::attention_weights<Wt, float>& w, std::span<const float> x,
                                   const ml::attention_activations<float>& a, const ml::attention_gradients<float>& g,
                                   std::span<float> dx) { ml::attention_backward<head8>(w, x, ml::no_rope, a, x, dx, g); };
static_assert(layer_trainable<W> && !layer_trainable<ml::quantized_weights<ml::block_q8_0>>);

} // namespace
