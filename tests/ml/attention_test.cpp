//
//  attention_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/attention.hpp>: attend against softmax of the scores times the values, the
//  logsumexp it returns, no keys, masked keys, split-and-merge partials equal to one pass in any order, the
//  backward pass against central differences, the windows, and compile-time use.
//

#include <snicholls/core/ml/attention.hpp>
#include <snicholls/core/ml/softmax.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

namespace {

namespace ml = snicholls::ml;
namespace m = snicholls::math;

static_assert(ml::attention_window(0).first == 0 && ml::attention_window(0).last == 1);
static_assert(ml::attention_window(9, 4).first == 6 && ml::attention_window(9, 4).size() == 4);
static_assert(ml::attention_window(2, 4).first == 0 && ml::attention_window(9, 0).size() == 10);
static_assert(std::is_trivially_copyable_v<ml::attention_accumulator<float, 64>>);
static_assert(sizeof(ml::attention_accumulator<float, 64>) == (64 + 2) * sizeof(float));

constexpr bool compile_time()
{
    // Two keys, the second scoring ln 3 higher: weights 1/4 and 3/4
    const std::array<double, 2> q{1, 0};
    const std::array<double, 4> k{0, 0, m::log(3.0), 0};
    const std::array<double, 4> v{4, 0, 0, 8};
    std::array<double, 2> out{};
    const double lse = ml::attend(q, k, v, out, 1.0);
    return m::near(out[0], 1.0, 1e-12) && m::near(out[1], 6.0, 1e-12) && m::near(lse, m::log(4.0), 1e-12);
}
static_assert(compile_time());

std::vector<double> wave(std::size_t n, double a, double b)
{
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = a * std::sin(0.37 * static_cast<double>(i) + b);
    return v;
}

constexpr std::size_t d = 8, n = 13;
const double scale = 1 / std::sqrt(static_cast<double>(d));

// The reference: every score, softmax, the weighted sum
std::vector<double> reference(const std::vector<double>& q, const std::vector<double>& k, const std::vector<double>& v, double& lse)
{
    const std::size_t keys = k.size() / q.size();
    std::vector<double> s(keys), p(keys), out(q.size(), 0.0);
    for (std::size_t j = 0; j < keys; ++j)
        for (std::size_t i = 0; i < q.size(); ++i)
            s[j] += scale * q[i] * k[j * q.size() + i];
    lse = ml::logsumexp(s);
    ml::softmax(s, p);
    for (std::size_t j = 0; j < keys; ++j)
        for (std::size_t i = 0; i < q.size(); ++i)
            out[i] += p[j] * v[j * q.size() + i];
    return out;
}

TEST(MlAttention, AttendIsSoftmaxOfTheScoresTimesTheValues)
{
    const auto q = wave(d, 2.0, 0.1), k = wave(n * d, 1.5, 0.7), v = wave(n * d, 3.0, 2.2);
    double want_lse = 0;
    const auto want = reference(q, k, v, want_lse);
    std::vector<double> out(d);
    const double lse = ml::attend(q, k, v, out, scale);
    EXPECT_NEAR(lse, want_lse, 1e-14);
    for (std::size_t i = 0; i < d; ++i)
        EXPECT_NEAR(out[i], want[i], 1e-14);
}

TEST(MlAttention, NoKeysGiveZeroAndMinusInfinity)
{
    const auto q = wave(d, 1.0, 0.0);
    std::vector<double> out(d, 5.0);
    const std::span<const double> none{};
    EXPECT_EQ(ml::attend(q, none, none, out, scale), -std::numeric_limits<double>::infinity());
    for (const double x : out)
        EXPECT_EQ(x, 0.0);
    ml::attention_accumulator<double, d> acc;
    acc.result(out);
    EXPECT_EQ(out[0], 0.0);
    EXPECT_EQ(acc.log_sum_exp(), -std::numeric_limits<double>::infinity());
}

TEST(MlAttention, AMaskedKeyHasNoWeight)
{
    const auto v = wave(2 * d, 1.0, 0.4);
    ml::attention_accumulator<double, d> acc;
    acc.add(0.5, std::span<const double>(v).first(d));
    acc.add(-std::numeric_limits<double>::infinity(), std::span<const double>(v).subspan(d));
    std::vector<double> out(d);
    acc.result(out);
    for (std::size_t i = 0; i < d; ++i)
        EXPECT_DOUBLE_EQ(out[i], v[i]);
}

TEST(MlAttention, PartialsMergedInAnyOrderEqualOnePass)
{
    const auto q = wave(d, 2.0, 0.3), k = wave(n * d, 2.5, 1.1), v = wave(n * d, 1.0, 0.9);
    std::vector<double> one(d);
    const double lse = ml::attend(q, k, v, one, scale);

    // Three workers take keys [0,4), [4,9), [9,13); merged left to right and right to left
    const std::array<std::size_t, 4> cut{0, 4, 9, n};
    std::array<ml::attention_accumulator<double, d>, 3> part{};
    for (std::size_t w = 0; w < 3; ++w)
        for (std::size_t j = cut[w]; j < cut[w + 1]; ++j) {
            double s = 0;
            for (std::size_t i = 0; i < d; ++i)
                s += q[i] * k[j * d + i];
            part[w].add(scale * s, std::span<const double>(v).subspan(j * d, d));
        }
    auto ltr = part[0];
    ltr.merge(part[1]);
    ltr.merge(part[2]);
    auto rtl = part[2];
    rtl.merge(part[1]);
    rtl.merge(part[0]);
    ml::attention_accumulator<double, d> empty;
    rtl.merge(empty);
    empty.merge(rtl);
    std::vector<double> a(d), b(d), c(d);
    ltr.result(a);
    rtl.result(b);
    empty.result(c);
    EXPECT_NEAR(ltr.log_sum_exp(), lse, 1e-14);
    EXPECT_NEAR(rtl.log_sum_exp(), lse, 1e-14);
    for (std::size_t i = 0; i < d; ++i) {
        EXPECT_NEAR(a[i], one[i], 1e-14);
        EXPECT_NEAR(b[i], one[i], 1e-14);
        EXPECT_EQ(c[i], b[i]);
    }
}

TEST(MlAttention, BackwardMatchesCentralDifferences)
{
    // L = g · attend(q, K, V); compare dq, dK, dV with differences in every input
    auto q = wave(d, 1.5, 0.2), k = wave(n * d, 1.2, 1.3), v = wave(n * d, 0.8, 2.9);
    const auto g = wave(d, 1.0, 0.6);
    std::vector<double> out(d), dq(d), dk(n * d, 0.0), dv(n * d, 0.0);
    const double lse = ml::attend(q, k, v, out, scale);
    ml::attend_backward(q, k, v, out, g, lse, scale, dq, dk, dv);
    const auto loss = [&] {
        std::vector<double> y(d);
        ml::attend(q, k, v, y, scale);
        double l = 0;
        for (std::size_t i = 0; i < d; ++i)
            l += g[i] * y[i];
        return l;
    };
    const auto numeric = [&](std::vector<double>& x, std::size_t i) {
        const double h = 1e-6, x0 = x[i];
        x[i] = x0 + h;
        const double up = loss();
        x[i] = x0 - h;
        const double down = loss();
        x[i] = x0;
        return (up - down) / (2 * h);
    };
    for (std::size_t i = 0; i < d; ++i)
        EXPECT_NEAR(dq[i], numeric(q, i), 1e-8) << "dq " << i;
    for (std::size_t i = 0; i < n * d; ++i) {
        EXPECT_NEAR(dk[i], numeric(k, i), 1e-8) << "dk " << i;
        EXPECT_NEAR(dv[i], numeric(v, i), 1e-8) << "dv " << i;
    }
    // dkeys and dvalues accumulate: a second query adds to them
    const auto dk1 = dk;
    ml::attend_backward(q, k, v, out, g, lse, scale, dq, dk, dv);
    EXPECT_NEAR(dk[5], 2 * dk1[5], 1e-15);
}

TEST(MlAttention, ACausalWindowIsASliceOfTheCache)
{
    // A query at position 9 with a window of 4 attends to keys 6..9, the same as attending to that slice
    const auto q = wave(d, 1.0, 0.5), k = wave(n * d, 1.0, 0.1), v = wave(n * d, 1.0, 0.8);
    const auto w = ml::attention_window(9, 4);
    std::vector<double> sliced(d), manual(d);
    ml::attend(q, std::span<const double>(k).subspan(w.first * d, w.size() * d), std::span<const double>(v).subspan(w.first * d, w.size() * d), sliced, scale);
    ml::attention_accumulator<double, d> acc;
    for (std::size_t j = 6; j <= 9; ++j) {
        double s = 0;
        for (std::size_t i = 0; i < d; ++i)
            s += q[i] * k[j * d + i];
        acc.add(scale * s, std::span<const double>(v).subspan(j * d, d));
    }
    acc.result(manual);
    for (std::size_t i = 0; i < d; ++i)
        EXPECT_NEAR(sliced[i], manual[i], 1e-14);
}

} // namespace
