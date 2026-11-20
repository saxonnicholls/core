//
//  iir_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/signal/iir.hpp>: the Audio EQ Cookbook's eight biquads at their defining frequencies;
//  Butterworth low-, high-, band-pass and band-stop against scipy.signal.butter's sosfreqz and the closed form;
//  sosfilt against scipy's, in one call and in pieces, in place; design at compile time.
//

#include <snicholls/core/signal/iir.hpp>

#include "../support/signal_fixture.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

namespace {

namespace s = snicholls::signal;
namespace m = snicholls::math;
using bq = s::biquad<double>;

static_assert(std::is_trivially_copyable_v<bq> && std::is_trivially_copyable_v<s::biquad_state<float>>);
static_assert(s::butterworth_sections(5, s::iir_band::lowpass) == 3 && s::butterworth_sections(3, s::iir_band::bandpass) == 3);

// A fourth-order Butterworth low-pass designed by the compiler: unit gain at DC, 1/sqrt(2) at the edge.
constexpr auto lp4 = s::make_butterworth<double, 4, s::iir_band::lowpass>(0.2);
static_assert(m::near(s::abs(s::sos_response<double>(lp4, 0.0)), 1.0, 1e-14));
static_assert(m::near(s::abs(s::sos_response<double>(lp4, 0.2 * m::pi)), m::math_limits<double>::inv_sqrt2, 1e-13));
static_assert(m::near(s::abs(s::rbj_biquad(s::biquad_kind::notch, 1000.0, 48000.0, 2.0).response(2 * m::pi * 1000.0 / 48000.0)), 0.0, 1e-12));

double mag(const bq& b, double w) { return s::abs(b.response(w)); }

TEST(Biquad, EveryCookbookFilterHasItsDefiningGains)
{
    const double fs = 48000, f0 = 3000, q = 0.9, w0 = 2 * m::pi * f0 / fs, g = 6.0, a2 = std::pow(10.0, g / 20);
    const auto lp = s::rbj_biquad(s::biquad_kind::lowpass, f0, fs, q);
    EXPECT_NEAR(mag(lp, 0), 1, 1e-14);
    EXPECT_NEAR(mag(lp, w0), q, 1e-13);   // the prototype's gain at the corner is Q
    EXPECT_NEAR(mag(lp, m::pi), 0, 1e-14);
    const auto hp = s::rbj_biquad(s::biquad_kind::highpass, f0, fs, q);
    EXPECT_NEAR(mag(hp, 0), 0, 1e-14);
    EXPECT_NEAR(mag(hp, w0), q, 1e-13);
    EXPECT_NEAR(mag(hp, m::pi), 1, 1e-14);
    const auto bp = s::rbj_biquad(s::biquad_kind::bandpass, f0, fs, q);
    EXPECT_NEAR(mag(bp, w0), 1, 1e-13);
    EXPECT_NEAR(mag(bp, 0), 0, 1e-14);
    const auto notch = s::rbj_biquad(s::biquad_kind::notch, f0, fs, q);
    EXPECT_NEAR(mag(notch, w0), 0, 1e-13);
    EXPECT_NEAR(mag(notch, 0), 1, 1e-14);
    const auto ap = s::rbj_biquad(s::biquad_kind::allpass, f0, fs, q);
    for (const double w : {0.0, 0.3, w0, 2.0, 3.1})
        EXPECT_NEAR(mag(ap, w), 1, 1e-13);
    EXPECT_NEAR(std::abs(s::arg(ap.response(w0))), m::pi, 1e-12);
    const auto peak = s::rbj_biquad(s::biquad_kind::peaking, f0, fs, q, g);
    EXPECT_NEAR(mag(peak, w0), a2, 1e-13);
    EXPECT_NEAR(mag(peak, 0), 1, 1e-14);
    const auto low = s::rbj_biquad(s::biquad_kind::lowshelf, f0, fs, q, g);
    EXPECT_NEAR(mag(low, 0), a2, 1e-13);
    EXPECT_NEAR(mag(low, m::pi), 1, 1e-13);
    const auto high = s::rbj_biquad(s::biquad_kind::highshelf, f0, fs, q, g);
    EXPECT_NEAR(mag(high, 0), 1, 1e-13);
    EXPECT_NEAR(mag(high, m::pi), a2, 1e-13);
}

std::vector<bq> design(std::size_t order, s::iir_band band, double w1, double w2 = 0)
{
    std::vector<bq> sos(s::butterworth_sections(order, band));
    s::butterworth(order, band, w1, w2, std::span<bq>(sos));
    return sos;
}

TEST(Butterworth, ResponsesAreScipysButter)
{
    const auto& w = snicholls::test::signal_record("butter_w");
    const std::array<std::pair<const char*, std::vector<bq>>, 5> cases{{
        {"lowpass4", design(4, s::iir_band::lowpass, 0.2)},
        {"lowpass5", design(5, s::iir_band::lowpass, 0.35)},
        {"highpass3", design(3, s::iir_band::highpass, 0.4)},
        {"bandpass3", design(3, s::iir_band::bandpass, 0.2, 0.45)},
        {"bandstop2", design(2, s::iir_band::bandstop, 0.1, 0.3)},
    }};
    for (const auto& [name, sos] : cases) {
        const auto ref = snicholls::test::signal_complex(std::string("butter_") + name);
        for (std::size_t i = 0; i < w.size(); ++i) {
            const auto h = s::sos_response<double>(sos, w[i]);
            EXPECT_NEAR(h.re, ref[i].re, 1e-12) << name << " " << w[i];
            EXPECT_NEAR(h.im, ref[i].im, 1e-12) << name << " " << w[i];
        }
    }
}

TEST(Butterworth, MagnitudesAreTheClosedFormAtSampleFrequencies)
{
    for (std::size_t order = 1; order <= 8; ++order)
        for (const double w : {0.05, 0.2, 0.37, 0.5, 0.71, 0.95}) {
            const double om = std::tan(m::pi * w / 2);
            const double c = std::tan(m::pi * 0.3 / 2), lo = std::tan(m::pi * 0.25 / 2), hi = std::tan(m::pi * 0.6 / 2);
            const double n2 = 2.0 * static_cast<double>(order);
            const double lp = 1 / std::sqrt(1 + std::pow(om / c, n2));
            const double hp = 1 / std::sqrt(1 + std::pow(c / om, n2));
            const double r = (om * om - lo * hi) / (om * (hi - lo));
            const double bp = 1 / std::sqrt(1 + std::pow(r, n2)), bs = 1 / std::sqrt(1 + std::pow(1 / r, n2));
            EXPECT_NEAR(s::abs(s::sos_response<double>(design(order, s::iir_band::lowpass, 0.3), m::pi * w)), lp, 1e-12) << order;
            EXPECT_NEAR(s::abs(s::sos_response<double>(design(order, s::iir_band::highpass, 0.3), m::pi * w)), hp, 1e-12) << order;
            EXPECT_NEAR(s::abs(s::sos_response<double>(design(order, s::iir_band::bandpass, 0.25, 0.6), m::pi * w)), bp, 1e-11) << order;
            EXPECT_NEAR(s::abs(s::sos_response<double>(design(order, s::iir_band::bandstop, 0.25, 0.6), m::pi * w)), bs, 1e-11) << order;
        }
}

TEST(Sosfilt, IsScipysInOneCallInPiecesAndInPlace)
{
    const auto& x = snicholls::test::signal_record("sosfilt_input");
    const auto& ref = snicholls::test::signal_record("sosfilt_lowpass4");
    const auto sos = design(4, s::iir_band::lowpass, 0.2);
    std::vector<s::biquad_state<double>> state(sos.size());
    std::vector<double> y(x.size());
    s::sosfilt<double>(sos, state, x, y);
    for (std::size_t i = 0; i < x.size(); ++i)
        EXPECT_NEAR(y[i], ref[i], 1e-13);
    std::vector<s::biquad_state<double>> st2(sos.size());
    std::vector<double> z(x.begin(), x.end());
    const std::span<double> all(z);
    s::sosfilt<double>(sos, st2, all.first(20), all.first(20));   // in place, and the state carries over
    s::sosfilt<double>(sos, st2, all.subspan(20), all.subspan(20));
    for (std::size_t i = 0; i < x.size(); ++i)
        EXPECT_EQ(z[i], y[i]);
}

} // namespace
