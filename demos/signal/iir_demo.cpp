//
//  iir_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/signal/iir.hpp>: a Butterworth low-pass designed by the compiler filtering a stream in
//  blocks, the band forms' responses, and the Audio EQ Cookbook's biquads (a 50 Hz notch, a peaking boost, the
//  shelves). Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/signal/iir.hpp>

#include <array>
#include <cstddef>
#include <cstdio>
#include <vector>

int main()
{
    std::printf("snicholls-core signal iir demo\n");
    namespace s = snicholls::signal;
    namespace m = snicholls::math;
    bool ok = true;

    //! [iir]
    // A sixth-order Butterworth low-pass at 0.1 of Nyquist: three sections, computed by the compiler.
    constexpr auto lowpass = s::make_butterworth<double, 6, s::iir_band::lowpass>(0.1);
    static_assert(lowpass.size() == 3);
    // Filter a stream in blocks of 64: the state carries from block to block, as scipy's sosfilt with zi.
    std::array<s::biquad_state<double>, 3> state{};
    std::vector<double> x(1024), y(1024);
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = m::sin(2 * m::pi * 0.01 * static_cast<double>(i)) + 0.5 * m::sin(2 * m::pi * 0.3 * static_cast<double>(i));
    for (std::size_t b = 0; b < x.size(); b += 64)
        s::sosfilt<double>(lowpass, state, std::span<const double>(x).subspan(b, 64), std::span<double>(y).subspan(b, 64));
    // Remove mains hum: a notch at 50 Hz at 48 kHz, Q 30.
    const auto notch = s::rbj_biquad(s::biquad_kind::notch, 50.0, 48000.0, 30.0);
    //! [iir]

    // The 0.3 tone (0.6 of Nyquist) is gone; the 0.01 tone passes with the filter's group delay.
    double high = 0;
    for (std::size_t i = 512; i < 1024; ++i)
        high = m::abs(y[i]) > high ? m::abs(y[i]) : high;
    std::printf("  6th-order low-pass at 0.1: |H(0.02)| %.6f, |H(0.1)| %.6f, |H(0.6)| %.2e; filtered peak %.4f\n",
                s::abs(s::sos_response<double>(lowpass, 0.02 * m::pi)), s::abs(s::sos_response<double>(lowpass, 0.1 * m::pi)),
                s::abs(s::sos_response<double>(lowpass, 0.6 * m::pi)), high);
    ok &= m::near(s::abs(s::sos_response<double>(lowpass, 0.1 * m::pi)), m::math_limits<double>::inv_sqrt2, 1e-12) && high < 1.01 && high > 0.99;

    // The band forms, at run time into a span.
    std::vector<s::biquad<double>> bp(s::butterworth_sections(4, s::iir_band::bandpass)), bs(4), hp(2);
    s::butterworth(4, s::iir_band::bandpass, 0.2, 0.3, std::span<s::biquad<double>>(bp));
    s::butterworth(4, s::iir_band::bandstop, 0.2, 0.3, std::span<s::biquad<double>>(bs));
    s::butterworth(4, s::iir_band::highpass, 0.5, 0.0, std::span<s::biquad<double>>(hp));
    const double centre = 2 * m::atan(m::sqrt(m::tan(m::pi * 0.1) * m::tan(m::pi * 0.15)));   // the prewarped geometric centre
    std::printf("  band-pass 0.2-0.3: |H| at its centre %.12f, at the edges %.6f; band-stop at the centre %.2e; high-pass at Nyquist %.6f\n",
                s::abs(s::sos_response<double>(bp, centre)), s::abs(s::sos_response<double>(bp, 0.2 * m::pi)),
                s::abs(s::sos_response<double>(bs, centre)), s::abs(s::sos_response<double>(hp, m::pi)));
    ok &= m::near(s::abs(s::sos_response<double>(bp, centre)), 1.0, 1e-12) && m::near(s::abs(s::sos_response<double>(bp, 0.2 * m::pi)), m::math_limits<double>::inv_sqrt2, 1e-12);
    ok &= s::abs(s::sos_response<double>(bs, centre)) < 1e-12 && m::near(s::abs(s::sos_response<double>(hp, m::pi)), 1.0, 1e-14);

    // The cookbook: the notch's depth, a 6 dB peak at 1 kHz, shelves, an all-pass's phase, low- and high-pass and band-pass gains.
    const double w50 = 2 * m::pi * 50 / 48000, w1k = 2 * m::pi * 1000 / 48000;
    const auto peak = s::rbj_biquad(s::biquad_kind::peaking, 1000.0, 48000.0, 1.0, 6.0);
    const auto low = s::rbj_biquad(s::biquad_kind::lowshelf, 200.0, 48000.0, 0.707, -12.0);
    const auto shelf = s::rbj_biquad(s::biquad_kind::highshelf, 8000.0, 48000.0, 0.707, 3.0);
    const auto ap = s::rbj_biquad(s::biquad_kind::allpass, 1000.0, 48000.0, 0.707);
    const auto lp = s::rbj_biquad(s::biquad_kind::lowpass, 1000.0, 48000.0, 0.70710678118654752);
    const auto hpb = s::rbj_biquad(s::biquad_kind::highpass, 1000.0, 48000.0, 0.70710678118654752);
    const auto bpb = s::rbj_biquad(s::biquad_kind::bandpass, 1000.0, 48000.0, 2.0);
    const auto db = [](double g) { return 20 * m::log10(g); };
    std::printf("  notch at 50 Hz: %.2e; peaking +6 dB at 1 kHz: %.4f dB; low shelf -12 dB at DC: %.4f dB; high shelf at Nyquist %.4f dB\n",
                s::abs(notch.response(w50)), db(s::abs(peak.response(w1k))), db(s::abs(low.response(0))), db(s::abs(shelf.response(m::pi))));
    std::printf("  all-pass phase at 1 kHz %.6f; low- and high-pass -3 dB at 1 kHz: %.4f, %.4f; band-pass peak %.6f\n", s::arg(ap.response(w1k)),
                db(s::abs(lp.response(w1k))), db(s::abs(hpb.response(w1k))), s::abs(bpb.response(w1k)));
    ok &= s::abs(notch.response(w50)) < 1e-10 && m::near(db(s::abs(peak.response(w1k))), 6.0, 1e-9) && m::near(db(s::abs(low.response(0))), -12.0, 1e-9);
    ok &= m::near(db(s::abs(shelf.response(m::pi))), 3.0, 1e-9) && m::near(m::abs(s::arg(ap.response(w1k))), m::pi, 1e-9);
    ok &= m::near(db(s::abs(lp.response(w1k))), db(m::math_limits<double>::inv_sqrt2), 1e-9) && m::near(s::abs(bpb.response(w1k)), 1.0, 1e-12);
    ok &= m::near(s::abs(hpb.response(w1k)), m::math_limits<double>::inv_sqrt2, 1e-12);
    s::biquad_state<double> st{};
    ok &= st.process(notch, 1.0) == notch.b0;   // an impulse's first output is b0

    std::printf(ok ? "  ok\n" : "  FAILED\n");
    return ok ? 0 : 1;
}
