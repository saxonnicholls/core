//
//  resample_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/signal/resample.hpp>: scipy.signal.resample_poly's outputs for five factors (44.1 kHz to
//  48 kHz among them), a sinusoid resampled to the new rate, the identity, output lengths, and kernels built at
//  compile time.
//

#include <snicholls/core/signal/resample.hpp>

#include "../support/signal_fixture.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace {

namespace s = snicholls::signal;
namespace m = snicholls::math;

static_assert(s::resample_size(50, 3, 2) == 75 && s::resample_size(50, 2, 3) == 34 && s::resample_size(50, 6, 4) == 75);
static_assert(s::resample_kernel_size(3, 2) == 61 && s::resample_kernel_size(6, 4) == 61 && s::resample_kernel_size(7, 7) == 1);
static_assert(s::resample_ratio::of(160, 147).up == 160 && s::resample_ratio::of(48000, 44100).down == 147);

// The 2:3 kernel at compile time: its DC gain (the sum of its taps) is up, and it is symmetric.
constexpr s::polyphase_resampler<double, 2, 3> two_thirds{};
constexpr bool kernel_at_compile_time()
{
    double sum = 0;
    for (const double v : two_thirds.kernel())
        sum += v;
    return m::near(sum, 2.0, 1e-13) && two_thirds.kernel()[0] == two_thirds.kernel()[60] && two_thirds.taps == 61;
}
static_assert(kernel_at_compile_time());

TEST(ResamplePoly, IsScipysForFiveFactors)
{
    const auto& x = snicholls::test::signal_record("resample_input");
    for (const auto& [up, down] : std::array<std::pair<std::size_t, std::size_t>, 5>{{{3, 2}, {2, 3}, {1, 4}, {5, 1}, {160, 147}}}) {
        const auto& ref = snicholls::test::signal_record("resample_" + std::to_string(up) + "_" + std::to_string(down));
        std::vector<double> h(s::resample_kernel_size(up, down)), y(s::resample_size(x.size(), up, down));
        ASSERT_EQ(y.size(), ref.size()) << up << "/" << down;
        s::resample_kernel(up, down, std::span<double>(h));
        s::resample_poly<double>(x, up, down, h, y);
        for (std::size_t i = 0; i < y.size(); ++i)
            EXPECT_NEAR(y[i], ref[i], 1e-13) << up << "/" << down << " " << i;
    }
}

TEST(ResamplePoly, ASinusoidKeepsItsFrequencyAtTheNewRate)
{
    // 1 kHz at 44.1 kHz to 48 kHz, through the compile-time kernel; away from the ends it is the same tone sampled at 48 kHz.
    constexpr s::polyphase_resampler<float, 160, 147> cd_to_dat{};
    std::vector<float> x(4410), y(cd_to_dat.output_size(x.size()));
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = static_cast<float>(std::sin(2 * m::pi * 1000.0 * static_cast<double>(i) / 44100.0));
    cd_to_dat.process(x, y);
    ASSERT_EQ(y.size(), 4800u);
    for (std::size_t i = 400; i < 4400; ++i)
        EXPECT_NEAR(y[i], std::sin(2 * m::pi * 1000.0 * static_cast<double>(i) / 48000.0), 2e-3) << i;
}

TEST(ResamplePoly, TheIdentityCopies)
{
    const std::vector<double> x{1, -2, 3.5};
    std::vector<double> h(1), y(3);
    s::resample_kernel(4, 4, std::span<double>(h));
    s::resample_poly<double>(x, 4, 4, h, y);
    EXPECT_EQ(y, x);
}

} // namespace
