//
//  attention_sink_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/attention_sink.hpp>: a sink of -infinity is attend and attend_backward to the bit;
//  the forward against the softmax over the scores and the sink written out; keys and values of different widths; no
//  keys with a sink; the weights summing to one minus the sink's; the backward pass, the sink's gradient included,
//  against central differences; compile-time use.
//

#include <snicholls/core/ml/attention.hpp>
#include <snicholls/core/ml/attention_sink.hpp>

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
namespace m = snicholls::math;

constexpr double none = -std::numeric_limits<double>::infinity();

constexpr bool compile_time()
{
    // Two equal keys and a sink of the same logit: each key and the sink a third
    const std::array<double, 1> q{0};
    const std::array<double, 2> k{1, 2};
    const std::array<double, 4> v{3, 0, 0, 3};
    std::array<double, 2> out{};
    const double lse = ml::attend_with_sink(q, k, v, out, 1.0, 0.0);
    return m::near(out[0], 1.0, 1e-12) && m::near(out[1], 1.0, 1e-12) && m::near(lse, m::log(3.0), 1e-12)
           && m::near(ml::attention_sink_weight(lse, 0.0), 1.0 / 3, 1e-12);
}
static_assert(compile_time());

std::vector<double> wave(std::size_t n, double a, double b)
{
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = a * std::sin(0.41 * static_cast<double>(i) + b);
    return v;
}

TEST(MlAttentionSink, MinusInfinityIsAttendToTheBit)
{
    constexpr std::size_t d = 8, n = 11;
    std::vector<float> q(d), k(n * d), v(n * d), dout(d);
    for (std::size_t i = 0; i < k.size(); ++i) {
        k[i] = std::sin(0.3f * static_cast<float>(i));
        v[i] = std::cos(0.7f * static_cast<float>(i));
    }
    for (std::size_t i = 0; i < d; ++i) {
        q[i] = 0.5f - 0.1f * static_cast<float>(i);
        dout[i] = 0.2f * static_cast<float>(i) - 0.3f;
    }
    std::vector<float> a(d), b(d);
    const float l1 = ml::attend(q, k, v, a, 0.35f);
    const float l2 = ml::attend_with_sink(q, k, v, b, 0.35f, -std::numeric_limits<float>::infinity());
    EXPECT_EQ(std::bit_cast<std::uint32_t>(l1), std::bit_cast<std::uint32_t>(l2));
    for (std::size_t i = 0; i < d; ++i)
        EXPECT_EQ(std::bit_cast<std::uint32_t>(a[i]), std::bit_cast<std::uint32_t>(b[i]));
    std::vector<float> dq1(d), dq2(d), dk1(n * d), dk2(n * d), dv1(n * d), dv2(n * d);
    ml::attend_backward(q, k, v, a, dout, l1, 0.35f, dq1, dk1, dv1);
    const float ds = ml::attend_with_sink_backward(q, k, v, b, dout, l2, 0.35f, -std::numeric_limits<float>::infinity(), dq2, dk2, dv2);
    EXPECT_EQ(ds, 0.0f);
    EXPECT_EQ(dq1, dq2);
    EXPECT_EQ(dk1, dk2);
    EXPECT_EQ(dv1, dv2);
}

TEST(MlAttentionSink, ForwardIsTheSoftmaxWithTheSinkColumnDropped)
{
    constexpr std::size_t dk = 5, dv = 3, n = 7;
    const auto q = wave(dk, 1.0, 0.2), k = wave(n * dk, 0.9, 1.1), v = wave(n * dv, 1.3, -0.4);
    for (const double sink : {-2.0, 0.5, 4.0}) {
        std::vector<double> out(dv);
        const double lse = ml::attend_with_sink(q, k, v, out, 0.7, sink);
        std::vector<double> s(n + 1);
        double mx = sink;
        for (std::size_t j = 0; j < n; ++j) {
            double acc = 0;
            for (std::size_t i = 0; i < dk; ++i)
                acc += q[i] * k[j * dk + i];
            s[j] = 0.7 * acc;
            mx = std::max(mx, s[j]);
        }
        s[n] = sink;
        double z = 0;
        for (const double x : s)
            z += std::exp(x - mx);
        EXPECT_NEAR(lse, mx + std::log(z), 1e-12);
        for (std::size_t i = 0; i < dv; ++i) {
            double want = 0;
            for (std::size_t j = 0; j < n; ++j)
                want += std::exp(s[j] - mx) / z * v[j * dv + i];
            EXPECT_NEAR(out[i], want, 1e-12);
        }
        double mass = ml::attention_sink_weight(lse, sink);
        for (std::size_t j = 0; j < n; ++j)
            mass += std::exp(s[j] - lse);
        EXPECT_NEAR(mass, 1.0, 1e-12);
    }
}

TEST(MlAttentionSink, NoKeysWithASinkIsZeroAndTheSinkTakesEverything)
{
    const std::array<double, 2> q{1, 2};
    std::array<double, 3> out{9, 9, 9};
    const double lse = ml::attend_with_sink(q, std::span<const double>{}, std::span<const double>{}, out, 1.0, 0.25);
    EXPECT_EQ(lse, 0.25);
    EXPECT_EQ(out, (std::array<double, 3>{0, 0, 0}));
    EXPECT_EQ(ml::attention_sink_weight(lse, 0.25), 1.0);
    std::array<double, 3> none_out{};
    EXPECT_EQ(ml::attend_with_sink(q, std::span<const double>{}, std::span<const double>{}, none_out, 1.0), none);
    EXPECT_EQ(ml::attention_sink_weight(none, none), 0.0);
}

TEST(MlAttentionSink, BackwardMatchesCentralDifferences)
{
    constexpr std::size_t dk = 4, dv = 3, n = 6;
    auto q = wave(dk, 0.8, 0.1), k = wave(n * dk, 1.1, 0.9), v = wave(n * dv, 0.7, 2.0);
    const auto w = wave(dv, 1.0, 0.3);   // the loss: w · out
    double sink = 0.4;
    const double scale = 0.6;
    const auto loss = [&] {
        std::vector<double> out(dv);
        ml::attend_with_sink(q, k, v, out, scale, sink);
        double l = 0;
        for (std::size_t i = 0; i < dv; ++i)
            l += w[i] * out[i];
        return l;
    };
    std::vector<double> out(dv), dq(dk), dkeys(n * dk), dvalues(n * dv);
    const double lse = ml::attend_with_sink(q, k, v, out, scale, sink);
    const double dsink = ml::attend_with_sink_backward(q, k, v, out, w, lse, scale, sink, dq, dkeys, dvalues);
    const double h = 1e-6;
    const auto check = [&](double& x, double analytic) {
        const double x0 = x;
        x = x0 + h;
        const double up = loss();
        x = x0 - h;
        const double down = loss();
        x = x0;
        EXPECT_NEAR(analytic, (up - down) / (2 * h), 1e-8);
    };
    for (std::size_t i = 0; i < dk; ++i)
        check(q[i], dq[i]);
    for (std::size_t i = 0; i < n * dk; ++i)
        check(k[i], dkeys[i]);
    for (std::size_t i = 0; i < n * dv; ++i)
        check(v[i], dvalues[i]);
    check(sink, dsink);
}

} // namespace
