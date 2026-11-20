//
//  window_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/signal/window.hpp>: every window, periodic and symmetric, against scipy.signal.get_window;
//  symmetry to the bit; windows of one point; compile-time tables; COLA and NOLA for the usual pairs.
//

#include <snicholls/core/signal/window.hpp>

#include "../support/signal_fixture.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

namespace {

namespace s = snicholls::signal;
namespace m = snicholls::math;

// Compile-time tables and values.
constexpr auto hann8 = s::make_window<double, 8>(s::hann{});
static_assert(hann8[0] == 0.0 && hann8[4] == 1.0 && m::near(hann8[2], 0.5, 1e-16));
constexpr auto hann9 = s::make_window<double, 9>(s::hann{}, s::window_symmetry::symmetric);
static_assert(hann9[0] == 0.0 && hann9[8] == 0.0 && hann9[4] == 1.0 && hann9[1] == hann9[7]);
static_assert(s::make_window<float, 1>(s::kaiser{8.0})[0] == 1.0f && s::make_window<double, 4>(s::rectangular{})[3] == 1.0);
static_assert(m::near(s::window_value<double>(s::kaiser{0.0}, 3, 7), 1.0, 1e-15));          // beta 0 is rectangular
static_assert(m::near(s::window_value<double>(s::tukey{1.0}, 2, 8), hann8[2], 1e-15));       // alpha 1 is Hann
static_assert(s::window_value<double>(s::tukey{0.0}, 0, 8) == 1.0);                          // alpha 0 is rectangular
static_assert(std::is_trivially_copyable_v<s::kaiser> && s::window_denominator(5, s::window_symmetry::symmetric) == 4);
constexpr bool hann_is_cola_at_half()
{
    const auto w = s::make_window<double, 16>(s::hann{});
    return s::is_cola<double>(w, 8) && s::is_nola<double>(w, 8) && !s::is_cola<double>(w, 5) && !s::is_nola<double>(w, 0)
           && !s::is_nola<double>(w, 17);
}
static_assert(hann_is_cola_at_half());

template <typename W>
void expect_scipy(const std::string& name, const W& w)
{
    for (const std::size_t n : {std::size_t{8}, std::size_t{9}, std::size_t{1}})
        for (const auto sym : {s::window_symmetry::periodic, s::window_symmetry::symmetric}) {
            const std::string key = "window_" + name + "_" + std::to_string(n) + (sym == s::window_symmetry::periodic ? "_periodic" : "_symmetric");
            const auto& ref = snicholls::test::signal_record(key);
            std::vector<double> got(n);
            s::fill_window(std::span<double>(got), w, sym);
            ASSERT_EQ(got.size(), ref.size()) << key;
            for (std::size_t i = 0; i < n; ++i)
                EXPECT_NEAR(got[i], ref[i], 1e-15) << key << " " << i;   // a few ulp: scipy sums the cosines in another order
            std::vector<float> f(n);
            s::fill_window(std::span<float>(f), w, sym);
            for (std::size_t i = 0; i < n; ++i)
                EXPECT_EQ(f[i], static_cast<float>(got[i])) << key;   // float computed in double and rounded once
        }
}

TEST(Windows, EveryWindowIsScipysPeriodicAndSymmetric)
{
    expect_scipy("rectangular", s::rectangular{});
    expect_scipy("hann", s::hann{});
    expect_scipy("hamming", s::hamming{});
    expect_scipy("blackman", s::blackman{});
    expect_scipy("blackman_harris", s::blackman_harris{});
    expect_scipy("kaiser", s::kaiser{8.6});
    expect_scipy("gaussian", s::gaussian{1.7});
    expect_scipy("tukey", s::tukey{0.4});
}

TEST(Windows, SymmetricWindowsAreSymmetricToTheBit)
{
    for (const std::size_t n : {std::size_t{7}, std::size_t{64}, std::size_t{401}}) {
        std::vector<double> a(n), b(n), c(n), d(n);
        s::fill_window(std::span<double>(a), s::blackman_harris{}, s::window_symmetry::symmetric);
        s::fill_window(std::span<double>(b), s::kaiser{14.0}, s::window_symmetry::symmetric);
        s::fill_window(std::span<double>(c), s::gaussian{0.2 * static_cast<double>(n)}, s::window_symmetry::symmetric);
        s::fill_window(std::span<double>(d), s::hamming{}, s::window_symmetry::symmetric);
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_EQ(a[i], a[n - 1 - i]);
            EXPECT_EQ(b[i], b[n - 1 - i]);
            EXPECT_EQ(c[i], c[n - 1 - i]);
            EXPECT_EQ(d[i], d[n - 1 - i]);
        }
    }
}

TEST(Windows, PeriodicIsTheSymmetricWindowOfOneMorePointWithoutItsLast)
{
    std::vector<double> p(10), q(11);
    s::fill_window(std::span<double>(p), s::blackman{}, s::window_symmetry::periodic);
    s::fill_window(std::span<double>(q), s::blackman{}, s::window_symmetry::symmetric);
    for (std::size_t i = 0; i < 10; ++i)
        EXPECT_EQ(p[i], q[i]);
}

TEST(Windows, ColaAndNolaAreScipysForTheUsualPairs)
{
    std::vector<double> hann(400), hamming(400), rect(400), blackman(512), kaiser(64);
    s::fill_window(std::span<double>(hann), s::hann{});
    s::fill_window(std::span<double>(hamming), s::hamming{});
    s::fill_window(std::span<double>(rect), s::rectangular{});
    s::fill_window(std::span<double>(blackman), s::blackman{});
    s::fill_window(std::span<double>(kaiser), s::kaiser{12.0});
    EXPECT_TRUE(s::is_cola<double>(hann, 200));
    EXPECT_TRUE(s::is_cola<double>(hann, 100));
    EXPECT_TRUE(s::is_cola<double>(hamming, 200));
    EXPECT_TRUE(s::is_cola<double>(rect, 400));
    EXPECT_TRUE(s::is_cola<double>(blackman, 512 / 3 + 1) == false);
    EXPECT_FALSE(s::is_cola<double>(hann, 160));    // Whisper's hop is not COLA for Hann ...
    EXPECT_TRUE(s::is_nola<double>(hann, 160));     // ... but NOLA, so istft still inverts it
    EXPECT_FALSE(s::is_nola<double>(hann, 400));    // a periodic Hann's first sample is 0: one frame per sample leaves a hole
    EXPECT_TRUE(s::is_nola<double>(kaiser, 64));    // a Kaiser window has no zero
}

} // namespace
