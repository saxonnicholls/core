//
//  normal_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/stats/normal.hpp>: the concept (core's samplers and one of the test's own), the inverse
//  CDF's draw as documented to the bit and monotone in the uniform, Box-Muller's pairing and generator calls, the
//  moments of both, finite draws at the generator's extremes, and draws in a constant expression.
//

#include <snicholls/core/stats/normal.hpp>

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace {

namespace st = snicholls::stats;

// A sampler of the test's own: antithetic pairs over the inverse CDF, z then -z
struct antithetic {
    st::normal_inverse_cdf base{};
    double last = 0;
    bool second = false;
    template <st::full_range_generator G>
    double operator()(G& g)
    {
        second = !second;
        return second ? (last = base(g)) : -last;
    }
};
static_assert(st::normal_sampler<st::normal_inverse_cdf, st::wyrand> && st::normal_sampler<st::normal_box_muller, st::wyrand>
              && st::normal_sampler<antithetic, st::wyrand>);
static_assert(!st::normal_sampler<int, st::wyrand>);

// A generator that returns given values and counts its calls
struct scripted {
    using result_type = std::uint64_t;
    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return std::numeric_limits<std::uint64_t>::max(); }
    std::vector<std::uint64_t> values;
    std::size_t calls = 0;
    result_type operator()() { return values[calls++ % values.size()]; }
};

TEST(StatsNormal, TheInverseCdfIsTheNormalQuantileOfTheDocumentedUniform)
{
    st::wyrand g(42), h(42);
    for (int i = 0; i < 100; ++i) {
        const double u = static_cast<double>((st::random_u64(h) >> 12) * 2 + 1) * 0x1.0p-53;   // (2i + 1) 2^-53
        EXPECT_EQ(st::normal_inverse_cdf{}(g), snicholls::math::normal_quantile(u)) << i;
    }
}

TEST(StatsNormal, TheInverseCdfIsMonotoneInTheUniformAndFiniteAtTheExtremes)
{
    scripted g{{0, 1ull << 40, 1ull << 63, ~0ull}};
    double previous = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < 4; ++i) {
        const double z = st::normal_inverse_cdf{}(g);
        EXPECT_TRUE(std::isfinite(z)) << i;
        EXPECT_GT(z, previous) << i;
        previous = z;
    }
    // The extremes are symmetric: 2^-53 and 1 - 2^-53, about 8.21 sigma out, never infinite (with 53 bits the top
    // uniform rounded to 1)
    const double low = st::normal_inverse_cdf{}(g);
    scripted top{{~0ull}};
    const double high = st::normal_inverse_cdf{}(top);
    EXPECT_LT(low, -8.2);
    EXPECT_NEAR(low, -high, 1e-9 * high);
}

TEST(StatsNormal, BoxMullerGivesPairsFromTwoCallsAndIsFiniteAtTheExtremes)
{
    scripted g{{0, 0, ~0ull, ~0ull}};
    st::normal_box_muller bm;
    const double a = bm(g), b = bm(g);
    EXPECT_EQ(g.calls, 2u);   // the second draw is the pair's other half
    EXPECT_TRUE(std::isfinite(a) && std::isfinite(b));
    const double c = bm(g);
    EXPECT_EQ(g.calls, 4u);
    EXPECT_TRUE(std::isfinite(c));
}

template <typename S>
void moments(S sampler)
{
    st::wyrand g(2026);
    constexpr int n = 400000;
    double m1 = 0, m2 = 0, m4 = 0;
    for (int i = 0; i < n; ++i) {
        const double z = sampler(g);
        m1 += z;
        m2 += z * z;
        m4 += z * z * z * z;
    }
    m1 /= n, m2 /= n, m4 /= n;
    EXPECT_NEAR(m1, 0.0, 0.01);   // standard errors 0.0016, 0.0022 and 0.02
    EXPECT_NEAR(m2, 1.0, 0.015);
    EXPECT_NEAR(m4, 3.0, 0.1);
}

TEST(StatsNormal, BothSamplersAndACustomOneHaveTheNormalsMoments)
{
    moments(st::normal_inverse_cdf{});
    moments(st::normal_box_muller{});
    moments(antithetic{});
}

constexpr bool compile_time()
{
    st::wyrand g(1);
    st::normal_box_muller bm;
    const double a = st::normal_inverse_cdf{}(g), b = bm(g), c = bm(g);
    return a > -9 && a < 9 && b > -9 && b < 9 && c > -9 && c < 9 && b != c;
}
static_assert(compile_time());
static_assert(noexcept(st::normal_inverse_cdf{}(std::declval<st::wyrand&>())));

} // namespace
