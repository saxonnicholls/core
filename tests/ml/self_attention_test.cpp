//
//  self_attention_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/self_attention.hpp>: one head against the naive softmax(Q K^T / sqrt(d) + mask) V in
//  double, with and without biases, a sliding window, RoPE in both layouts, the log-sum-exps, a prefix that gives
//  the same bits as the whole sequence, float within rounding of double, quantised and bfloat16 weights, no tokens,
//  the bidirectional and block-causal masks, and compile-time use; the backward pass against central differences in
//  double (causal, windowed, with RoPE, bidirectional, block-causal),
//  its accumulation, the skipped input gradient, and that only dense weights train.
//

#include <snicholls/core/ml/self_attention.hpp>

#include <snicholls/core/ml/minifloat.hpp>
#include <snicholls/core/ml/quantize.hpp>
#include <snicholls/core/ml/rope.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace {

namespace ml = snicholls::ml;
namespace m = snicholls::math;

constexpr std::size_t dm = 6, dh = 4, n = 9;

std::vector<double> wave(std::size_t size, double a, double b)
{
    std::vector<double> v(size);
    for (std::size_t i = 0; i < size; ++i)
        v[i] = a * std::sin(0.37 * static_cast<double>(i) + b);
    return v;
}

// A head's weights, owned, with views for the function
template <typename T>
struct head
{
    std::vector<T> wq, wk, wv, bq, bk, bv;
    head(bool biases)
    {
        auto cast = [](const std::vector<double>& d) {
            std::vector<T> out(d.size());
            std::transform(d.begin(), d.end(), out.begin(), [](double x) { return static_cast<T>(x); });
            return out;
        };
        wq = cast(wave(dh * dm, 0.9, 0.1));
        wk = cast(wave(dh * dm, 0.8, 1.3));
        wv = cast(wave(dh * dm, 1.1, 2.9));
        if (biases) {
            bq = cast(wave(dh, 0.2, 0.4));
            bk = cast(wave(dh, 0.3, 1.7));
            bv = cast(wave(dh, 0.5, 2.2));
        }
    }
    [[nodiscard]] ml::self_attention_weights<ml::dense_weights<T>, T> weights() const
    {
        return {{wq, dh, dm}, {wk, dh, dm}, {wv, dh, dm}, bq, bk, bv};
    }
};

// Buffers for n tokens
template <typename T>
struct run
{
    std::vector<T> q, k, v, lse, out;
    explicit run(std::size_t tokens) : q(tokens * dh), k(tokens * dh), v(tokens * dh), lse(tokens), out(tokens * dh) {}
    [[nodiscard]] ml::self_attention_activations<T> act() { return {q, k, v, lse}; }
};

// The reference: project, every score with the mask, softmax, the weighted sum
struct expected
{
    std::vector<double> out, lse;
};

expected reference(const head<double>& h, const std::vector<double>& x, std::size_t window,
                   const std::vector<double>* cos = nullptr, const std::vector<double>* sin = nullptr,
                   ml::rope_layout layout = ml::rope_layout::half, ml::attention_mask mask = ml::attention_mask::causal,
                   std::size_t block = 1)
{
    const std::size_t tokens = x.size() / dm;
    auto project = [&](const std::vector<double>& w, const std::vector<double>& b, std::size_t t) {
        std::vector<double> p(dh);
        for (std::size_t i = 0; i < dh; ++i) {
            p[i] = b.empty() ? 0 : b[i];
            for (std::size_t j = 0; j < dm; ++j)
                p[i] += w[i * dm + j] * x[t * dm + j];
        }
        return p;
    };
    // RoPE written out: pair (a, b) rotated by the angle of position t
    auto rotate = [&](std::vector<double>& p, std::size_t t) {
        if (cos == nullptr)
            return;
        for (std::size_t i = 0; i < dh / 2; ++i) {
            const std::size_t a = layout == ml::rope_layout::half ? i : 2 * i;
            const std::size_t b = layout == ml::rope_layout::half ? i + dh / 2 : 2 * i + 1;
            const double c = (*cos)[t * dh / 2 + i], s = (*sin)[t * dh / 2 + i], x1 = p[a], x2 = p[b];
            p[a] = x1 * c - x2 * s;
            p[b] = x1 * s + x2 * c;
        }
    };
    std::vector<std::vector<double>> q, k, v;
    for (std::size_t t = 0; t < tokens; ++t) {
        q.push_back(project(h.wq, h.bq, t));
        k.push_back(project(h.wk, h.bk, t));
        v.push_back(project(h.wv, h.bv, t));
        rotate(q.back(), t);
        rotate(k.back(), t);
    }
    expected e{std::vector<double>(tokens * dh, 0.0), std::vector<double>(tokens)};
    for (std::size_t t = 0; t < tokens; ++t) {
        std::vector<double> s(tokens, -std::numeric_limits<double>::infinity());
        double max = s[0];
        for (std::size_t j = 0; j < tokens; ++j)
            if (mask == ml::attention_mask::bidirectional || (mask == ml::attention_mask::block_causal && j < (t / block + 1) * block)
                || (mask == ml::attention_mask::causal && j <= t && (window == 0 || t - j < window))) {
                s[j] = 0;
                for (std::size_t i = 0; i < dh; ++i)
                    s[j] += q[t][i] * k[j][i] / std::sqrt(double(dh));
                max = std::max(max, s[j]);
            }
        double sum = 0;
        for (double sj : s)
            sum += std::exp(sj - max);
        e.lse[t] = max + std::log(sum);
        for (std::size_t j = 0; j < tokens; ++j)
            for (std::size_t i = 0; i < dh; ++i)
                e.out[t * dh + i] += std::exp(s[j] - e.lse[t]) * v[j][i];
    }
    return e;
}

void expect_near(const std::vector<double>& got, const std::vector<double>& want, double tol)
{
    ASSERT_EQ(got.size(), want.size());
    for (std::size_t i = 0; i < got.size(); ++i)
        EXPECT_NEAR(got[i], want[i], tol) << "at " << i;
}

constexpr bool compile_time()
{
    // Two tokens of 2 channels, identity projections, head of 2: position 1 sees keys (1, 0) and (0, 1)
    constexpr std::array<double, 4> eye{1, 0, 0, 1};
    const ml::dense_weights<double> id{eye, 2, 2};
    const ml::self_attention_weights<ml::dense_weights<double>, double> w{id, id, id};
    const std::array<double, 4> x{1, 0, 0, 1};
    std::array<double, 4> q{}, k{}, v{}, out{};
    std::array<double, 2> lse{};
    ml::self_attention_head<2, 2>(w, x, ml::self_attention_activations<double>{q, k, v, lse}, out);
    // Position 0 sees only itself, score 1/sqrt 2; position 1 weighs its own key e^(1/sqrt 2) against e^0
    const double s = 1 / m::sqrt(2.0), a = m::exp(s);
    return out[0] == 1 && out[1] == 0 && m::near(out[2], 1 / (1 + a), 1e-12) && m::near(out[3], a / (1 + a), 1e-12)
           && m::near(lse[0], s, 1e-12) && m::near(lse[1], m::log(1 + a), 1e-12);
}
static_assert(compile_time());

TEST(MlSelfAttention, HeadIsSoftmaxOfTheMaskedScoresTimesTheValues)
{
    const head<double> h(false);
    const auto x = wave(n * dm, 1.0, 0.3);
    run<double> r(n);
    ml::self_attention_head<dm, dh>(h.weights(), x, r.act(), r.out);
    const auto want = reference(h, x, 0);
    expect_near(r.out, want.out, 1e-12);
    expect_near(r.lse, want.lse, 1e-12);
}

TEST(MlSelfAttention, BiasesAreAddedToEachProjection)
{
    const head<double> h(true);
    const auto x = wave(n * dm, 1.0, 0.3);
    run<double> r(n);
    ml::self_attention_head<dm, dh>(h.weights(), x, r.act(), r.out);
    expect_near(r.out, reference(h, x, 0).out, 1e-12);
}

TEST(MlSelfAttention, TheFirstPositionAttendsOnlyToItsOwnValue)
{
    const head<double> h(true);
    const auto x = wave(n * dm, 1.0, 0.3);
    run<double> r(n);
    ml::self_attention_head<dm, dh>(h.weights(), x, r.act(), r.out);
    for (std::size_t i = 0; i < dh; ++i)
        EXPECT_EQ(r.out[i], r.v[i]);
}

TEST(MlSelfAttention, ASlidingWindowSeesOnlyTheLastPositions)
{
    const head<double> h(true);
    const auto x = wave(n * dm, 1.0, 0.3);
    for (std::size_t window : {1u, 2u, 3u, 20u}) {
        run<double> r(n);
        ml::self_attention_head<dm, dh>(h.weights(), x, r.act(), r.out, window);
        const auto want = reference(h, x, window);
        expect_near(r.out, want.out, 1e-12);
        expect_near(r.lse, want.lse, 1e-12);
    }
    // A window of one is each position's own value
    run<double> r(n);
    ml::self_attention_head<dm, dh>(h.weights(), x, r.act(), r.out, 1);
    expect_near(r.out, r.v, 0);
}

TEST(MlSelfAttention, RopeRotatesQueriesAndKeysByTheirPositionsInEitherLayout)
{
    static const ml::rope_table<double, 16, dh> table{};
    std::vector<double> cos, sin;
    for (std::size_t p = 0; p < 16; ++p)
        for (std::size_t i = 0; i < dh / 2; ++i) {
            cos.push_back(table.cos[p][i]);
            sin.push_back(table.sin[p][i]);
        }
    const head<double> h(true);
    const auto x = wave(n * dm, 1.0, 0.3);
    for (auto layout : {ml::rope_layout::half, ml::rope_layout::interleaved}) {
        run<double> r(n);
        ml::self_attention_head<dm, dh>(h.weights(), x, r.act(), r.out, 0, table, layout);
        expect_near(r.out, reference(h, x, 0, &cos, &sin, layout).out, 1e-12);
    }
}

TEST(MlSelfAttention, RopeChangesTheResultAndLeavesTheValuesUnrotated)
{
    static const ml::rope_table<double, 16, dh> table{};
    const head<double> h(true);
    const auto x = wave(n * dm, 1.0, 0.3);
    run<double> plain(n), rotated(n);
    ml::self_attention_head<dm, dh>(h.weights(), x, plain.act(), plain.out);
    ml::self_attention_head<dm, dh>(h.weights(), x, rotated.act(), rotated.out, 0, table);
    EXPECT_EQ(plain.v, rotated.v);
    EXPECT_NE(plain.out, rotated.out);
    // Position 0 is the identity rotation
    for (std::size_t i = 0; i < dh; ++i)
        EXPECT_EQ(plain.q[i], rotated.q[i]);
}

TEST(MlSelfAttention, APrefixGivesTheSameBitsAsTheWholeSequence)
{
    // Causal: position t depends on tokens 0..t only, so decoding one token at a time is this run's prefix
    const head<float> h(true);
    std::vector<float> x;
    for (double d : wave(n * dm, 1.0, 0.3))
        x.push_back(static_cast<float>(d));
    run<float> whole(n);
    ml::self_attention_head<dm, dh>(h.weights(), x, whole.act(), whole.out);
    for (std::size_t len = 0; len <= n; ++len) {
        run<float> part(len);
        ml::self_attention_head<dm, dh>(h.weights(), std::span<const float>(x).first(len * dm), part.act(), part.out);
        EXPECT_TRUE(std::equal(part.out.begin(), part.out.end(), whole.out.begin())) << "prefix " << len;
        EXPECT_TRUE(std::equal(part.lse.begin(), part.lse.end(), whole.lse.begin())) << "prefix " << len;
    }
}

TEST(MlSelfAttention, FloatIsWithinRoundingOfDouble)
{
    const head<double> hd(true);
    const head<float> hf(true);
    const auto xd = wave(n * dm, 1.0, 0.3);
    std::vector<float> xf(xd.size());
    std::transform(xd.begin(), xd.end(), xf.begin(), [](double x) { return static_cast<float>(x); });
    run<float> r(n);
    ml::self_attention_head<dm, dh>(hf.weights(), xf, r.act(), r.out);
    const auto want = reference(hd, xd, 0);
    expect_near(std::vector<double>(r.out.begin(), r.out.end()), want.out, 1e-5);
}

TEST(MlSelfAttention, NoLseBufferSkipsTheLogSumExps)
{
    const head<double> h(true);
    const auto x = wave(n * dm, 1.0, 0.3);
    run<double> with(n), without(n);
    ml::self_attention_head<dm, dh>(h.weights(), x, with.act(), with.out);
    ml::self_attention_head<dm, dh>(h.weights(), x, ml::self_attention_activations<double>{without.q, without.k, without.v, {}},
                                    without.out);
    EXPECT_EQ(with.out, without.out);
}

TEST(MlSelfAttention, NoTokensWritesNothing)
{
    const head<double> h(true);
    run<double> r(0);
    ml::self_attention_head<dm, dh>(h.weights(), std::span<const double>{}, r.act(), r.out);
    EXPECT_TRUE(r.out.empty());
}

TEST(MlSelfAttention, BFloat16WeightsAreWidenedAsLinearWidensThem)
{
    const head<float> hf(true);
    auto narrow = [](const std::vector<float>& f) {
        std::vector<ml::bfloat16> b;
        for (float v : f)
            b.push_back(ml::to_minifloat<ml::bfloat16>(v));
        return b;
    };
    const auto wq = narrow(hf.wq), wk = narrow(hf.wk), wv = narrow(hf.wv);
    const ml::self_attention_weights<ml::dense_weights<ml::bfloat16>, float> w{
        {wq, dh, dm}, {wk, dh, dm}, {wv, dh, dm}, hf.bq, hf.bk, hf.bv};
    std::vector<float> x;
    for (double d : wave(n * dm, 1.0, 0.3))
        x.push_back(static_cast<float>(d));
    run<float> b(n), f(n);
    ml::self_attention_head<dm, dh>(w, x, b.act(), b.out);
    ml::self_attention_head<dm, dh>(hf.weights(), x, f.act(), f.out);
    for (std::size_t i = 0; i < b.out.size(); ++i)
        EXPECT_NEAR(b.out[i], f.out[i], 2e-2);
}

TEST(MlSelfAttention, QuantisedWeightsProjectThroughQ8)
{
    // DModel a multiple of 32: one Q8_0 block per weight row; compared with the same weights dequantised
    constexpr std::size_t qm = 32, tokens = 5;
    std::vector<float> f;
    for (double d : wave(3 * dh * qm, 0.7, 0.9))
        f.push_back(static_cast<float>(d));
    std::vector<ml::block_q8_0> blocks(3 * dh);
    for (std::size_t r = 0; r < 3 * dh; ++r)
        ml::quantize(std::span<const float>(f).subspan(r * qm, qm), std::span<ml::block_q8_0, 1>(blocks.data() + r, 1));
    std::vector<float> back(f.size());
    for (std::size_t r = 0; r < 3 * dh; ++r)
        ml::dequantize(std::span<const ml::block_q8_0, 1>(blocks.data() + r, 1), std::span<float>(back).subspan(r * qm, qm));
    const std::span<const ml::block_q8_0> bs(blocks);
    const ml::self_attention_weights<ml::quantized_weights<ml::block_q8_0>, float> wq{
        {bs.subspan(0, dh), dh, qm}, {bs.subspan(dh, dh), dh, qm}, {bs.subspan(2 * dh, dh), dh, qm}};
    const std::span<const float> bk(back);
    const ml::self_attention_weights<ml::dense_weights<float>, float> wd{
        {bk.subspan(0, dh * qm), dh, qm}, {bk.subspan(dh * qm, dh * qm), dh, qm}, {bk.subspan(2 * dh * qm, dh * qm), dh, qm}};
    std::vector<float> x;
    for (double d : wave(tokens * qm, 1.0, 0.3))
        x.push_back(static_cast<float>(d));
    std::vector<float> q(tokens * dh), k(q.size()), v(q.size()), out(q.size()), q2(q.size()), k2(q.size()), v2(q.size()), out2(q.size());
    ml::self_attention_head<qm, dh>(wq, x, ml::self_attention_activations<float>{q, k, v, {}}, out);
    ml::self_attention_head<qm, dh>(wd, x, ml::self_attention_activations<float>{q2, k2, v2, {}}, out2);
    for (std::size_t i = 0; i < out.size(); ++i)
        EXPECT_NEAR(out[i], out2[i], 5e-2);
}

// The backward pass: L = sum(out . G) for a fixed G, so dL/dout = G
template <typename W>
concept trainable = requires(const ml::self_attention_weights<W, float>& w, std::span<const float> x,
                             const ml::self_attention_activations<float>& a, const ml::self_attention_gradients<float>& g,
                             std::span<float> dx) {
    ml::self_attention_head_backward<32, dh>(w, x, a, x, x, dx, g);
};
static_assert(trainable<ml::dense_weights<float>> && trainable<ml::dense_weights<ml::bfloat16>>);
static_assert(!trainable<ml::quantized_weights<ml::block_q8_0>>, "quantised weights are inference only");

struct grads
{
    std::vector<double> q, k, v, wq, wk, wv, bq, bk, bv, x;
    explicit grads(std::size_t tokens)
        : q(tokens * dh), k(tokens * dh), v(tokens * dh), wq(dh * dm), wk(dh * dm), wv(dh * dm), bq(dh), bk(dh), bv(dh),
          x(tokens * dm)
    {
    }
    [[nodiscard]] ml::self_attention_gradients<double> view() { return {q, k, v, wq, wk, wv, bq, bk, bv}; }
};

template <typename R = ml::no_rope_t>
void expect_backward_is_central_differences(std::size_t window, const R& rope_src = ml::no_rope,
                                            ml::attention_mask mask = ml::attention_mask::causal, std::size_t block = 1)
{
    head<double> h(true);
    auto x = wave(n * dm, 1.0, 0.3);
    const auto gout = wave(n * dh, 0.8, 1.9);
    auto loss = [&] {
        run<double> r(n);
        ml::self_attention_head<dm, dh>(h.weights(), x, r.act(), r.out, window, rope_src, ml::rope_layout::half, mask, block);
        double l = 0;
        for (std::size_t i = 0; i < r.out.size(); ++i)
            l += gout[i] * r.out[i];
        return l;
    };
    run<double> r(n);
    ml::self_attention_head<dm, dh>(h.weights(), x, r.act(), r.out, window, rope_src, ml::rope_layout::half, mask, block);
    grads g(n);
    ml::self_attention_head_backward<dm, dh>(h.weights(), x, r.act(), r.out, gout, g.x, g.view(), window, rope_src, ml::rope_layout::half,
                                             mask, block);
    auto check = [&](std::vector<double>& p, const std::vector<double>& grad, const char* what) {
        for (std::size_t i = 0; i < p.size(); ++i) {
            const double keep = p[i], e = 1e-6;
            p[i] = keep + e;
            const double up = loss();
            p[i] = keep - e;
            const double down = loss();
            p[i] = keep;
            EXPECT_NEAR(grad[i], (up - down) / (2 * e), 1e-7) << what << " " << i;
        }
    };
    check(x, g.x, "x");
    check(h.wq, g.wq, "wq");
    check(h.wk, g.wk, "wk");
    check(h.wv, g.wv, "wv");
    check(h.bq, g.bq, "bq");
    check(h.bk, g.bk, "bk");
    check(h.bv, g.bv, "bv");
}

TEST(MlSelfAttention, BackwardMatchesCentralDifferences)
{
    expect_backward_is_central_differences(0);
}

TEST(MlSelfAttention, BackwardWithASlidingWindowMatchesCentralDifferences)
{
    expect_backward_is_central_differences(2);
    expect_backward_is_central_differences(3);
}

TEST(MlSelfAttention, BackwardWithRopeMatchesCentralDifferences)
{
    static const ml::rope_table<double, 16, dh> table{};
    expect_backward_is_central_differences(0, table);
}

TEST(MlSelfAttention, BackwardWithBidirectionalAndBlockCausalMasksMatchesCentralDifferences)
{
    static const ml::rope_table<double, 16, dh> table{};
    expect_backward_is_central_differences(0, ml::no_rope, ml::attention_mask::bidirectional);
    expect_backward_is_central_differences(0, table, ml::attention_mask::bidirectional);
    expect_backward_is_central_differences(0, ml::no_rope, ml::attention_mask::block_causal, 2);
}

TEST(MlSelfAttention, BidirectionalAndBlockCausalHeadsAreSoftmaxOverTheirKeys)
{
    const head<double> h(true);
    const auto x = wave(n * dm, 1.0, 0.3);
    for (const auto& [mask, block] : {std::pair{ml::attention_mask::bidirectional, std::size_t(1)},
                                     std::pair{ml::attention_mask::block_causal, std::size_t(2)},
                                     std::pair{ml::attention_mask::block_causal, std::size_t(3)}}) {
        run<double> r(n);
        ml::self_attention_head<dm, dh>(h.weights(), x, r.act(), r.out, 0, ml::no_rope, ml::rope_layout::half, mask, block);
        const auto want = reference(h, x, 0, nullptr, nullptr, ml::rope_layout::half, mask, block);
        expect_near(r.out, want.out, 1e-12);
        expect_near(r.lse, want.lse, 1e-12);
    }
    // One block of every token is bidirectional, to the bit; a block of one is causal, to the bit
    run<double> a(n), b(n), c(n), d(n);
    ml::self_attention_head<dm, dh>(h.weights(), x, a.act(), a.out, 0, ml::no_rope, ml::rope_layout::half, ml::attention_mask::block_causal, n);
    ml::self_attention_head<dm, dh>(h.weights(), x, b.act(), b.out, 0, ml::no_rope, ml::rope_layout::half, ml::attention_mask::bidirectional);
    ml::self_attention_head<dm, dh>(h.weights(), x, c.act(), c.out, 0, ml::no_rope, ml::rope_layout::half, ml::attention_mask::block_causal, 1);
    ml::self_attention_head<dm, dh>(h.weights(), x, d.act(), d.out);
    EXPECT_EQ(a.out, b.out);
    EXPECT_EQ(c.out, d.out);
}

TEST(MlSelfAttention, BackwardAccumulatesTheWeightGradientsAndCanSkipTheInputs)
{
    const head<double> h(true);
    const auto x = wave(n * dm, 1.0, 0.3);
    const auto gout = wave(n * dh, 0.8, 1.9);
    run<double> r(n);
    ml::self_attention_head<dm, dh>(h.weights(), x, r.act(), r.out);
    grads once(n), twice(n);
    ml::self_attention_head_backward<dm, dh>(h.weights(), x, r.act(), r.out, gout, once.x, once.view());
    ml::self_attention_head_backward<dm, dh>(h.weights(), x, r.act(), r.out, gout, twice.x, twice.view());
    ml::self_attention_head_backward<dm, dh>(h.weights(), x, r.act(), r.out, gout, std::span<double>{}, twice.view());
    // The weights' and biases' gradients add; the queries', keys' and values' are written each time
    for (std::size_t i = 0; i < once.wq.size(); ++i)
        EXPECT_NEAR(twice.wq[i], 2 * once.wq[i], 1e-15);
    for (std::size_t i = 0; i < dh; ++i)
        EXPECT_NEAR(twice.bv[i], 2 * once.bv[i], 1e-15);
    EXPECT_EQ(twice.q, once.q);
    EXPECT_EQ(twice.k, once.k);
    EXPECT_EQ(twice.v, once.v);
    EXPECT_EQ(twice.x, once.x);   // the skipped input gradient was not touched
}

TEST(MlSelfAttention, BackwardOfNoTokensAddsNothing)
{
    const head<double> h(true);
    run<double> r(0);
    grads g(0);
    ml::self_attention_head_backward<dm, dh>(h.weights(), std::span<const double>{}, r.act(), r.out, r.out, g.x, g.view());
    EXPECT_TRUE(std::all_of(g.wq.begin(), g.wq.end(), [](double v) { return v == 0; }));
}

} // namespace
