//
//  accumulators_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/stats/accumulators.hpp>: each accumulator against its formula, at
//  compile time where possible; merging split data equals one pass; Neumaier's compensation.
//

#include <snicholls/core/stats/accumulators.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace stats = snicholls::stats;

template <typename A, typename T>
constexpr A over(std::initializer_list<T> xs, A a = A{})
{
    for (const T x : xs)
        a.add(x);
    return a;
}

static_assert(over<stats::count>({1.0, 2.0, 3.0}).value() == 3);
static_assert(over<stats::sum<int>>({1, 2, 3}).value() == 6);
static_assert(over<stats::minimum<int>>({4, -2, 7}).value() == -2 && !stats::minimum<int>{}.value());
static_assert(over<stats::maximum<int>>({4, -2, 7}).value() == 7);
static_assert(over<stats::mean<double>>({1.0, 2.0, 6.0}).value() == 3.0);
static_assert(over<stats::variance<double>>({1.0, 2.0, 6.0}).value() == 7.0);   // ((1-3)^2+(2-3)^2+(6-3)^2)/2
static_assert(over<stats::variance<double>>({1.0, 2.0, 6.0}).mean() == 3.0);
static_assert(stats::accumulator<stats::variance<double>, double> && stats::mergeable<stats::variance<double>>);
static_assert(!stats::mergeable<stats::ewma<double>>);
static_assert(std::is_trivially_copyable_v<stats::variance<double>> && sizeof(stats::variance<double>) == 3 * 8);

TEST(Accumulators, EmptyMomentsAreNaN)
{
    EXPECT_TRUE(std::isnan(stats::mean<double>{}.value()));
    EXPECT_TRUE(std::isnan(over<stats::variance<double>>({1.0}).value()));
    EXPECT_TRUE(std::isnan(stats::covariance<double>{}.value()));
}

TEST(Accumulators, MergingSplitDataEqualsOnePass)
{
    std::mt19937_64 g(1);
    std::normal_distribution<double> normal(0.001, 0.02);
    std::vector<double> x(10001), y(10001);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = normal(g);
        y[i] = 0.5 * x[i] + normal(g);
    }
    stats::variance<double> whole, left, right;
    stats::covariance<double> cwhole, cleft, cright;
    for (std::size_t i = 0; i < x.size(); ++i) {
        whole.add(x[i]);
        cwhole.add(x[i], y[i]);
        (i < 3333 ? left : right).add(x[i]);
        (i < 3333 ? cleft : cright).add(std::pair{x[i], y[i]});
    }
    left.merge(right);
    cleft.merge(cright);
    EXPECT_EQ(left.n, whole.n);
    EXPECT_NEAR(left.mean(), whole.mean(), 1e-15);
    EXPECT_NEAR(left.value(), whole.value(), 1e-15);
    EXPECT_NEAR(cleft.value(), cwhole.value(), 1e-15);
    EXPECT_NEAR(cleft.beta(), 0.5, 0.02);
    EXPECT_NEAR(cwhole.correlation(), cwhole.value() / std::sqrt(whole.value() * cwhole.m2y / static_cast<double>(cwhole.n - 1)), 1e-12);
    // The variance agrees with the two-pass formula
    const double m = std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
    double ss = 0;
    for (const double v : x)
        ss += (v - m) * (v - m);
    EXPECT_NEAR(whole.value(), ss / static_cast<double>(x.size() - 1), 1e-15);
}

TEST(Accumulators, NeumaierKeepsTheSmallTermsALongSumLoses)
{
    stats::sum<double> naive;
    stats::compensated_sum<double> careful;
    naive.add(1e16);
    careful.add(1e16);
    for (int i = 0; i < 10000; ++i) {
        naive.add(1.0);
        careful.add(1.0);
    }
    naive.add(-1e16);
    careful.add(-1e16);
    EXPECT_EQ(careful.value(), 10000.0);
    EXPECT_NE(naive.value(), 10000.0);
}

TEST(Accumulators, EwmaFollowsItsRecurrence)
{
    stats::ewma<double> e{.alpha = 0.5};
    e.add(1.0);
    EXPECT_EQ(e.value(), 1.0);
    EXPECT_EQ(e.variance(), 0.0);
    e.add(3.0);   // delta 2: mean 2, variance (1 - .5)(0 + .5 * 4) = 1
    EXPECT_DOUBLE_EQ(e.value(), 2.0);
    EXPECT_DOUBLE_EQ(e.variance(), 1.0);
    EXPECT_DOUBLE_EQ(e.volatility(), 1.0);
}

TEST(Accumulators, HistogramCountsAndMerges)
{
    stats::histogram<double, 4> h{.lo = 0, .hi = 1};
    for (const double x : {-0.1, 0.0, 0.24, 0.25, 0.99, 1.0, std::nan("")})
        h.add(x);
    EXPECT_EQ(h.value(), (std::array<std::uint64_t, 4>{2, 1, 0, 1}));
    EXPECT_EQ(h.below, 2u);   // -0.1 and NaN
    EXPECT_EQ(h.above, 1u);
    auto other = h;
    h.merge(other);
    EXPECT_EQ(h.value()[0], 4u);
    EXPECT_DOUBLE_EQ(h.edge(1), 0.25);
}

} // namespace
