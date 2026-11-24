//
//  goertzel_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/signal/goertzel.hpp>: a compile-time detector whose coefficient is a constant, a tone
//  detector chosen at run time, streaming and the complex bin, a window, the raw coefficients and state, and a DTMF
//  decoder over a bank of eight detectors. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/signal/goertzel.hpp>

#include <array>
#include <cstdio>
#include <string>
#include <vector>

namespace {

namespace s = snicholls::signal;
namespace m = snicholls::math;
constexpr double pi = m::math_limits<double>::pi;

std::vector<float> tones(double f1, double f2, std::size_t n, double fs)
{
    std::vector<float> x(n);
    for (std::size_t j = 0; j < n; ++j) {
        const double t = static_cast<double>(j) / fs;
        x[j] = static_cast<float>(0.5 * m::sin(2 * pi * f1 * t) + (f2 > 0 ? 0.4 * m::sin(2 * pi * f2 * t + 1.0) : 0.0));
    }
    return x;
}

} // namespace

int main()
{
    std::printf("snicholls-core signal goertzel demo\n");
    bool ok = true;

    //! [compile_time]
    // A 1 kHz pilot tone at 8 kHz over blocks of 64: bin 8, the coefficient 2 cos(pi / 4) computed by the compiler.
    using pilot = s::static_goertzel<float, 1000.0, 8000.0, 64>;
    static_assert(pilot::bin_index == 8.0f);
    constexpr float coeff = pilot::coeff;   // a constant: no trigonometry at run time
    static_assert(sizeof(pilot) == sizeof(s::goertzel_state<float>));   // the object is the state alone
    pilot d;
    for (const float x : tones(1000.0, 0.0, 64, 8000.0))
        d.push(x);
    const bool present = d.complete() && d.magnitude() > 0.25f * 64 / 2;   // A n / 2 = 16 for A = 0.5
    //! [compile_time]
    std::printf("  pilot: coeff %.9g (2 cos(pi/4) = %.9g), |X| %.4f, phase %.4f, present %d\n", static_cast<double>(coeff),
                2 * m::cos(pi / 4), static_cast<double>(d.magnitude()), static_cast<double>(d.phase()), present);
    ok &= present && coeff == static_cast<float>(2 * m::cos(pi / 4)) && m::near(static_cast<double>(d.magnitude()), 16.0, 1e-4);
    ok &= m::near(static_cast<double>(d.phase()), -pi / 2, 1e-4) && d.count() == 64;   // a sine is a cosine a quarter turn late
    d.reset();
    ok &= d.count() == 0 && d.power() == 0.0f && d.bin().re == 0.0f;

    //! [run_time]
    // A frequency chosen at run time, between bins (60 Hz mains hum at 1 kHz over 250 samples: bin 15), streamed in
    // two halves, with a Hann window from window.hpp.
    const double hum = 60.0, fs = 1000.0;
    s::goertzel<double> mains(hum, fs, 250);
    std::vector<double> x(250);
    for (std::size_t j = 0; j < x.size(); ++j)
        x[j] = 0.2 * m::cos(2 * pi * hum * static_cast<double>(j) / fs) + 0.01 * m::cos(2 * pi * 213.0 * static_cast<double>(j) / fs);
    const auto window = s::make_window<double, 250>(s::hann{});
    mains.update(std::span<const double>(x).first(100), window);
    mains.update(std::span<const double>(x).subspan(100), window);
    const s::complex<double> X = mains.bin();   // the generalised Goertzel's final step: fft(x * w)[15]
    const double amplitude = 2 * mains.magnitude() / 125.0;   // sum(hann) = n / 2
    //! [run_time]
    std::printf("  mains hum: X = %.6f%+.6fi, amplitude %.6f, bin %.3f\n", X.re, X.im, amplitude, mains.coefficients().bin);
    ok &= m::near(amplitude, 0.2, 1e-5) && m::near(mains.phase(), 0.0, 1e-5) && mains.complete() && mains.count() == 250;
    ok &= m::near(mains.power(), s::norm(X), 1e-12) && mains.coefficients().bin == 15.0;

    // Between bins: the generalised form, and the raw coefficients and state for a layout of the caller's.
    const auto c = s::goertzel_coefficients<double>::at_frequency(61.3, fs, 250);   // bin 15.325
    s::goertzel_state<double> st;
    for (const double v : x)
        st.push(c.coeff, v);
    const auto c2 = s::goertzel_coefficients<double>::at_fractional_bin(15.325, 250);
    std::printf("  61.3 Hz: bin %.4f, |X| %.4f (a tone 1.3 Hz away leaks), same as fractional bin %d\n", c.bin, m::sqrt(st.power(c)),
                c2.coeff == c.coeff);
    ok &= m::near(c.bin, 15.325, 1e-12) && m::sqrt(st.power(c)) < 25.0 && m::near(s::abs(st.bin(c)), m::sqrt(st.power(c)), 1e-9);
    ok &= m::near(c2.coeff, c.coeff, 1e-15) && s::goertzel_coefficients<double>::at_bin(15, 250).block_phase.re == 1.0;
    st.reset();
    ok &= st.count == 0;
    const s::goertzel<double> on_bin = s::goertzel<double>::at_bin(15, 250);
    ok &= on_bin.coefficients().coeff == mains.coefficients().coeff && mains.state().count == 250;

    //! [dtmf]
    // A DTMF decoder: eight detectors in one pass over 205 samples at 8 kHz, their coefficients computed by the compiler.
    constexpr std::array<double, 8> hz{697, 770, 852, 941, 1209, 1336, 1477, 1633};
    constexpr s::goertzel_bank<float, 8> dtmf(hz, 8000.0, 205);
    constexpr char keys[4][5] = {"123A", "456B", "789C", "*0#D"};
    auto decode = [&](const std::vector<float>& block) {
        auto bank = dtmf;   // a copy: the constants and a zeroed state
        bank.update(block);
        const std::size_t r = bank.strongest(0, 4), col = bank.strongest(4, 8);
        for (std::size_t i = 0; i < bank.size(); ++i)   // every other tone well below both
            if (i != r && i != col && bank.power(i) * 10 > (bank.power(r) < bank.power(col) ? bank.power(r) : bank.power(col)))
                return '?';
        return keys[r][col - 4];
    };
    std::string dialled;
    for (const char key : std::string("0123456789*#ABCD"))
        for (std::size_t r = 0; r < 4; ++r)
            for (std::size_t col = 0; col < 4; ++col)
                if (keys[r][col] == key)
                    dialled += decode(tones(hz[r], hz[4 + col], 205, 8000.0));
    //! [dtmf]
    std::printf("  dtmf: dialled \"%s\"; a lone 852 Hz tone decodes to '%c'\n", dialled.c_str(), decode(tones(852.0, 0.0, 205, 8000.0)));
    ok &= dialled == "0123456789*#ABCD" && decode(tones(852.0, 0.0, 205, 8000.0)) == '?';

    // The bank's other accessors.
    auto bank = dtmf;
    bank.update(tones(941.0, 1336.0, 205, 8000.0));
    const auto p = bank.powers();
    ok &= bank.complete() && bank.count() == 205 && bank.strongest() == 3 && p[5] == bank.power(5);
    ok &= m::near(static_cast<double>(bank.magnitude(3)), static_cast<double>(s::abs(bank.bin(3))), 1e-3);
    ok &= bank.phase(3) == s::arg(bank.bin(3)) && bank.state(3).count == 205 && bank.coefficients(0).n == 205;
    std::printf("  bank: '0' = 941 + 1336 Hz, |X| %.2f and %.2f\n", static_cast<double>(bank.magnitude(3)), static_cast<double>(bank.magnitude(5)));
    bank.reset();
    ok &= bank.count() == 0 && bank.power(0) == 0.0f;

    std::printf(ok ? "  ok\n" : "  FAILED\n");
    return ok ? 0 : 1;
}
