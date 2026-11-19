//
//  fir_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/signal/fir.hpp>: low-, high-, band-pass and band-stop taps against scipy.signal.firwin
//  (Hamming and Kaiser windows), linear phase, the responses at the band centres and in the stop bands, direct
//  filtering, and design at compile time.
//

#include <snicholls/core/signal/fir.hpp>

#include "../support/signal_fixture.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

namespace s = snicholls::signal;
namespace m = snicholls::math;

// A 63-tap low-pass designed by the compiler: unit gain at DC, symmetric, small at Nyquist.
constexpr auto lowpass63 = s::make_fir<double, 63>(s::fir_band::lowpass, 0.25);
constexpr bool lowpass_at_compile_time()
{
    double dc = 0;
    for (const double v : lowpass63)
        dc += v;
    for (std::size_t i = 0; i < 63; ++i)
        if (lowpass63[i] != lowpass63[62 - i])
            return false;
    const auto nyquist = s::fir_response<double>(lowpass63, m::pi);
    return m::near(dc, 1.0, 1e-14) && s::abs(nyquist) < 1e-3;
}
static_assert(lowpass_at_compile_time());
static_assert(s::sinc(0.0) == 1.0 && m::near(s::sinc(0.5), 2 / m::pi, 1e-15) && m::near(s::sinc(1.0), 0.0, 1e-15));
static_assert(m::near(s::abs(s::fir_response<double>(s::make_fir<double, 31>(s::fir_band::highpass, 0.3), m::pi)), 1.0, 1e-14));

TEST(FirDesign, TapsAreScipysFirwin)
{
    const std::array<std::pair<const char*, std::vector<double>>, 5> cases{{
        {"firwin_lowpass", [] { std::vector<double> h(31); s::fir_design(std::span<double>(h), s::fir_band::lowpass, 0.3); return h; }()},
        {"firwin_highpass", [] { std::vector<double> h(31); s::fir_design(std::span<double>(h), s::fir_band::highpass, 0.3); return h; }()},
        {"firwin_bandpass", [] { std::vector<double> h(31); s::fir_design(std::span<double>(h), s::fir_band::bandpass, 0.2, 0.5); return h; }()},
        {"firwin_bandstop", [] { std::vector<double> h(31); s::fir_design(std::span<double>(h), s::fir_band::bandstop, 0.2, 0.5); return h; }()},
        {"firwin_kaiser", [] { std::vector<double> h(20); s::fir_design(std::span<double>(h), s::fir_band::lowpass, 0.25, 0.0, s::kaiser{8.0}); return h; }()},
    }};
    for (const auto& [name, h] : cases) {
        const auto& ref = snicholls::test::signal_record(name);
        ASSERT_EQ(h.size(), ref.size()) << name;
        for (std::size_t i = 0; i < h.size(); ++i)
            EXPECT_NEAR(h[i], ref[i], 1e-15) << name << " " << i;
    }
}

TEST(FirDesign, ResponsesAreUnitInThePassBandAndSmallInTheStopBand)
{
    std::vector<double> bp(101), bs(101);
    s::fir_design(std::span<double>(bp), s::fir_band::bandpass, 0.3, 0.6, s::blackman{});
    s::fir_design(std::span<double>(bs), s::fir_band::bandstop, 0.3, 0.6, s::blackman{});
    EXPECT_NEAR(s::abs(s::fir_response<double>(bp, 0.45 * m::pi)), 1.0, 1e-12);   // scaled at the band's centre
    EXPECT_LT(s::abs(s::fir_response<double>(bp, 0.05 * m::pi)), 1e-3);
    EXPECT_LT(s::abs(s::fir_response<double>(bp, 0.9 * m::pi)), 1e-3);
    EXPECT_NEAR(s::abs(s::fir_response<double>(bs, 0.0)), 1.0, 1e-12);
    EXPECT_LT(s::abs(s::fir_response<double>(bs, 0.45 * m::pi)), 1e-3);
    EXPECT_NEAR(s::abs(s::fir_response<double>(bs, m::pi)), 1.0, 1e-3);
    // Linear phase: the response at any frequency is real after removing the delay of (taps - 1) / 2 samples.
    const auto r = s::fir_response<double>(bp, 0.4) * s::expi(0.4 * 50.0);
    EXPECT_NEAR(r.im, 0.0, 1e-13);
}

TEST(FirFilter, IsTheCausalConvolutionFromRest)
{
    const std::array<double, 3> h{0.5, 0.25, -1};
    const std::vector<double> x{1, 0, 0, 2, 1};
    std::vector<double> y(5);
    s::fir_filter<double>(h, x, y);
    EXPECT_EQ(y, (std::vector<double>{0.5, 0.25, -1, 1, 1}));
}

} // namespace
