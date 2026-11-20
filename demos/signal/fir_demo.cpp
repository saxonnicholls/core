//
//  fir_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/signal/fir.hpp>: a low-pass filter designed by the compiler removing a high tone, and the
//  high-pass, band-pass and band-stop designs with their responses. Exits non-zero if a result is not what it
//  documents.
//

#include <snicholls/core/signal/fir.hpp>

#include <cstddef>
#include <cstdio>
#include <vector>

int main()
{
    std::printf("snicholls-core signal fir demo\n");
    namespace s = snicholls::signal;
    namespace m = snicholls::math;
    bool ok = true;

    //! [fir]
    // 63 taps, cut-off at a quarter of Nyquist, Hamming window: designed by the compiler, scipy's firwin(63, 0.25).
    constexpr auto lowpass = s::make_fir<float, 63>(s::fir_band::lowpass, 0.25f);
    // A low tone (0.05 of the sample rate) plus a high one (0.4): the filter keeps the first.
    std::vector<float> x(400), y(400);
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = static_cast<float>(m::sin(2 * m::pi * 0.05 * static_cast<double>(i)) + m::sin(2 * m::pi * 0.4 * static_cast<double>(i)));
    s::fir_filter<float>(lowpass, x, y);
    // The other bands, at run time into a span, with a Kaiser window.
    std::vector<double> band(101), stop(101), high(101);
    s::fir_design(std::span<double>(band), s::fir_band::bandpass, 0.2, 0.4, s::kaiser{8.0});
    s::fir_design(std::span<double>(stop), s::fir_band::bandstop, 0.2, 0.4, s::kaiser{8.0});
    s::fir_design(std::span<double>(high), s::fir_band::highpass, 0.5);
    //! [fir]

    // After the filter's delay of 31 samples, y is the low tone alone.
    double worst = 0;
    for (std::size_t i = 100; i < 400; ++i)
        worst = m::abs(static_cast<double>(y[i]) - m::sin(2 * m::pi * 0.05 * static_cast<double>(i - 31))) > worst
                    ? m::abs(static_cast<double>(y[i]) - m::sin(2 * m::pi * 0.05 * static_cast<double>(i - 31)))
                    : worst;
    const auto at = [](const std::vector<double>& h, double f) { return s::abs(s::fir_response<double>(h, m::pi * f)); };
    std::printf("  low-pass: the high tone removed, within %.3g of the delayed low tone; |H| at 0.1 %.4f, at 0.8 %.2e\n", worst,
                static_cast<double>(s::abs(s::fir_response<float>(lowpass, static_cast<float>(m::pi * 0.1)))),
                static_cast<double>(s::abs(s::fir_response<float>(lowpass, static_cast<float>(m::pi * 0.8)))));
    std::printf("  band-pass 0.2-0.4: |H| at 0.3 %.6f, at 0.1 %.2e; band-stop: at 0.3 %.2e, at 0 %.6f; high-pass at 0.9 %.6f\n", at(band, 0.3),
                at(band, 0.1), at(stop, 0.3), at(stop, 0.0), at(high, 0.9));
    ok &= worst < 1e-2 && at(band, 0.3) > 0.999 && at(band, 0.1) < 1e-3 && at(stop, 0.3) < 1e-3 && m::near(at(stop, 0.0), 1.0, 1e-9);
    ok &= m::abs(at(high, 0.9) - 1.0) < 1e-2 && m::near(s::abs(s::fir_response<double>(high, m::pi)), 1.0, 1e-12) && s::sinc(0.0) == 1.0;

    std::printf(ok ? "  ok\n" : "  FAILED\n");
    return ok ? 0 : 1;
}
