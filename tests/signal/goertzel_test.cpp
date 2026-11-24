//
//  goertzel_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/signal/goertzel.hpp>: compile-time coefficients, layout and noexcept claims; every
//  integer bin of random blocks against fft.hpp's in float, double and long double (bin, power and phase); numpy's
//  fft and its transform between bins, with and without windows (tests/fixtures/signal/goertzel.txt, written by
//  scripts/generate_goertzel_fixtures.py); tones at and off a bin, noise, DC and Nyquist; partial and over-long
//  blocks; the bank against single detectors; the 16 DTMF digits.
//

#include <snicholls/core/signal/goertzel.hpp>

#include <snicholls/core/signal/fft.hpp>
#include <snicholls/core/stats/random.hpp>

#include "../support/fixture_file.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

namespace {

namespace s = snicholls::signal;
namespace m = snicholls::math;
constexpr double pi = m::math_limits<double>::pi;

// --- compile time ---------------------------------------------------------------------------------------------

using tone1k = s::static_goertzel<float, 1000.0, 8000.0, 8>;   // bin 1 of 8: an eighth of a turn
static_assert(tone1k::bin_index == 1.0f && tone1k::coefficients.n == 8);
static_assert(tone1k::coeff == static_cast<float>(2.0 * 0.70710678118654752440));
static_assert(tone1k::coefficients.cos_w == s::twiddle<float>(1, 8).re && tone1k::coefficients.sin_w == -s::twiddle<float>(1, 8).im);
constexpr float compile_time_coeff = tone1k::coeff;   // a constant expression
static_assert(compile_time_coeff > 1.41f && compile_time_coeff < 1.42f);
// The quarter turns are exact: DC 2, a quarter 0, Nyquist -2.
static_assert(s::static_goertzel<double, 0.0, 8.0, 8>::coeff == 2.0 && s::static_goertzel<double, 2.0, 8.0, 8>::coeff == 0.0
              && s::static_goertzel<double, 4.0, 8.0, 8>::coeff == -2.0 && s::static_goertzel<double, 2.0, 8.0, 8>::coefficients.sin_w == 1.0);
// A DTMF tone is between bins (697 * 205 / 8000 = 17.86): the generalised form, with its block phase.
using row1 = s::static_goertzel<double, 697.0, 8000.0, 205>;
static_assert(m::near(row1::bin_index, 697.0 * 205 / 8000, 1e-12) && row1::coefficients.block_phase.im != 0.0);
static_assert(m::near(row1::coeff, 2 * 0.8538689046744508, 1e-15));   // 2 cos(2 pi 697 / 8000)
// A window is a compile-time table.
static_assert(s::static_goertzel<double, 1.0, 8.0, 8, s::hann{}>::window.size() == 8 && s::static_goertzel<double, 1.0, 8.0, 8, s::hann{}>::window[4] == 1.0);
static_assert(tone1k::window.empty());
// The run-time form computed by the compiler, and its other constructors.
constexpr s::goertzel<float> dtmf_row1(697.0, 8000.0, 205);
static_assert(dtmf_row1.coefficients().coeff == s::goertzel_coefficients<float>::at_frequency(697.0, 8000.0, 205).coeff);
static_assert(s::goertzel<double>::at_bin(3, 16).coefficients().bin == 3.0 && s::goertzel<double>().coefficients().n == 1);
// A frequency on a bin snaps to the integer form; bins and frequencies reduce modulo the block and the sample rate.
static_assert(s::goertzel_coefficients<double>::at_frequency(3.0 * 48000 / 7, 48000.0, 7).coeff == s::goertzel_coefficients<double>::at_bin(3, 7).coeff);
static_assert(s::goertzel_coefficients<double>::at_frequency(3.0 * 48000 / 7, 48000.0, 7).block_phase == s::complex<double>{1.0, 0.0});
static_assert(s::goertzel_coefficients<double>::at_bin(10, 7).bin == 3.0);
static_assert(s::goertzel_coefficients<double>::at_frequency(-1000.0, 8000.0, 8).bin == 7.0);
static_assert(s::goertzel_coefficients<double>::at_fractional_bin(2.5, 16).coeff == s::goertzel_coefficients<double>::at_frequency(2.5, 16.0, 16).coeff);

// A whole detection at compile time: a tone on bin 3 of 32.
constexpr double detect_at_compile_time()
{
    s::goertzel<double> d = s::goertzel<double>::at_bin(3, 32);
    for (std::size_t j = 0; j < 32; ++j)
        d.push(m::cos(2 * m::math_limits<double>::pi * 3.0 * static_cast<double>(j) / 32.0));
    return d.magnitude();
}
static_assert(m::near(detect_at_compile_time(), 16.0, 1e-12));   // A n / 2

// Layout: the state is two values and a count; the compile-time detector is the state alone.
static_assert(std::is_trivially_copyable_v<s::goertzel_state<float>> && std::is_standard_layout_v<s::goertzel_state<float>>);
static_assert(std::is_trivially_copyable_v<s::goertzel_coefficients<double>> && std::is_trivially_copyable_v<s::goertzel<float>>);
static_assert(std::is_trivially_copyable_v<s::goertzel_bank<float, 8>> && std::is_standard_layout_v<s::goertzel_bank<float, 8>>);
static_assert(std::is_trivially_copyable_v<tone1k> && sizeof(tone1k) == sizeof(s::goertzel_state<float>));
static_assert(sizeof(s::goertzel_state<float>) == 2 * sizeof(float) + sizeof(std::size_t) && alignof(s::goertzel_state<double>) == alignof(double));
static_assert(sizeof(s::goertzel_state<double>) == 2 * sizeof(double) + sizeof(std::size_t));
// noexcept throughout.
static_assert(noexcept(std::declval<s::goertzel<float>&>().push(1.0f)) && noexcept(std::declval<s::goertzel<float>&>().update(std::span<const float>{})));
static_assert(noexcept(std::declval<const s::goertzel<double>&>().bin()) && noexcept(std::declval<const s::goertzel<double>&>().power()));
static_assert(noexcept(std::declval<tone1k&>().push(1.0f)) && noexcept(std::declval<const tone1k&>().phase()) && noexcept(std::declval<tone1k&>().reset()));
static_assert(noexcept(std::declval<s::goertzel_bank<float, 8>&>().push(1.0f)) && noexcept(std::declval<const s::goertzel_bank<float, 8>&>().powers()));
static_assert(noexcept(s::goertzel_coefficients<float>::at_frequency(1.0, 2.0, 3)) && noexcept(std::declval<s::goertzel_state<float>&>().push(1.0f, 1.0f)));

// --- helpers --------------------------------------------------------------------------------------------------

template <typename T>
std::vector<T> noise(std::size_t n, std::uint64_t seed)
{
    snicholls::stats::wyrand g(seed);
    std::vector<T> x(n);
    for (auto& v : x)
        v = static_cast<T>(2.0 * snicholls::stats::uniform_real(g) - 1.0);
    return x;
}

template <typename T>
T norm2(const std::vector<T>& x)
{
    long double s2 = 0;
    for (const T v : x)
        s2 += static_cast<long double>(v) * v;
    return static_cast<T>(std::sqrt(s2));
}

template <typename T>
constexpr T eps() { return std::numeric_limits<T>::epsilon(); }

// The header's stated bound on a detector's bin: eps n ||x|| / max(|sin w|, 1 / n).
template <typename T>
T goertzel_bound(std::size_t n, long double sin_w, T x_norm)
{
    const long double nn = static_cast<long double>(n);
    return static_cast<T>(static_cast<long double>(eps<T>()) * nn * x_norm / std::max(std::fabs(sin_w), 1.0L / nn));
}

// fft.hpp's stated bound, 2 eps ceil(log2 L) ||x||, L the length the passes run at.
template <typename T>
T fft_bound(const s::fft_plan<T>& plan, T x_norm)
{
    const std::size_t len = plan.uses_bluestein() ? plan.layout().m : plan.size();
    return T(2) * eps<T>() * static_cast<T>(std::max(1.0, std::ceil(std::log2(static_cast<double>(len))))) * x_norm;
}

template <typename T>
T wrapped(T a)
{
    const T pi = static_cast<T>(m::math_limits<s::compute_t<T>>::pi);
    while (a > pi) a -= 2 * pi;
    while (a < -pi) a += 2 * pi;
    return a;
}

// --- every integer bin against the FFT --------------------------------------------------------------------------

template <typename T>
void every_bin_is_the_ffts(std::size_t n, std::uint64_t seed)
{
    const auto x = noise<T>(n, seed);
    std::vector<s::complex<T>> X(n), storage(s::fft_plan<T>::storage_size(n)), work(s::fft_plan<T>::work_size(n));
    for (std::size_t j = 0; j < n; ++j)
        X[j] = {x[j], T(0)};
    const s::fft_plan<T> plan(n, storage);
    s::fft(plan, std::span<s::complex<T>>(X), std::span<s::complex<T>>(work));
    const T xn = norm2(x);
    for (std::size_t k = 0; k < n; ++k) {
        s::goertzel<T> d = s::goertzel<T>::at_bin(k, n);
        d.update(x);
        ASSERT_TRUE(d.complete());
        const long double sin_w = std::sin(2.0L * m::math_limits<long double>::pi * static_cast<long double>(k) / static_cast<long double>(n));
        const T tol = goertzel_bound<T>(n, sin_w, xn) + fft_bound(plan, xn);
        const s::complex<T> b = d.bin();
        EXPECT_LE(s::abs(b - X[k]), tol) << "n " << n << " k " << k;
        const T ref = s::abs(X[k]);
        EXPECT_LE(std::fabs(d.power() - ref * ref), 2 * tol * (ref + tol)) << "n " << n << " k " << k;
        EXPECT_LE(std::fabs(s::norm(b) - d.power()), 4 * tol * (ref + tol)) << "n " << n << " k " << k;
        if (ref > 100 * tol) {   // a phase is only meaningful when the bin is well above the error
            EXPECT_LE(std::fabs(wrapped(d.phase() - s::arg(X[k]))), 2 * tol / ref) << "n " << n << " k " << k;
        }
    }
}

TEST(GoertzelAgainstFft, EveryIntegerBinOfRandomBlocksIsTheFftsBinInDouble)
{
    for (std::size_t n : {1u, 2u, 3u, 16u, 63u, 64u, 205u, 1000u, 1009u, 4096u})
        every_bin_is_the_ffts<double>(n, 1000 + n);
}

TEST(GoertzelAgainstFft, EveryIntegerBinOfRandomBlocksIsTheFftsBinInFloat)
{
    for (std::size_t n : {1u, 2u, 7u, 16u, 63u, 64u, 205u, 1000u, 1009u, 4096u})
        every_bin_is_the_ffts<float>(n, 2000 + n);
}

TEST(GoertzelAgainstFft, EveryIntegerBinOfRandomBlocksIsTheFftsBinInLongDouble)
{
    for (std::size_t n : {1u, 5u, 16u, 63u, 256u, 1000u})
        every_bin_is_the_ffts<long double>(n, 3000 + n);
}

// --- numpy's values -------------------------------------------------------------------------------------------

struct fixture
{
    std::map<std::string, std::vector<double>> signals, ffts;
    struct dtft
    {
        std::string signal, window;
        double f, fs, re, im;
    };
    std::vector<dtft> dtfts;
};

const fixture& numpy_fixture()
{
    static const fixture f = [] {
        fixture out;
        for (const auto& row : snicholls::test::read_fixture("signal/goertzel.txt")) {
            const auto& v = row.fields;
            if (v[0] == "signal") {
                for (std::size_t i = 3; i < v.size(); ++i)
                    out.signals[v[1]].push_back(std::stod(v[i]));
            } else if (v[0] == "fft") {
                for (std::size_t i = 2; i < v.size(); ++i)
                    out.ffts[v[1]].push_back(std::stod(v[i]));
            } else if (v[0] == "dtft") {
                out.dtfts.push_back({v[1], v[2], std::stod(v[3]), std::stod(v[4]), std::stod(v[5]), std::stod(v[6])});
            }
        }
        return out;
    }();
    return f;
}

template <typename T>
std::vector<T> numpy_window(const std::string& name, std::size_t n)
{
    std::vector<T> w(n);
    if (name == "rect")
        s::fill_window(std::span<T>(w), s::rectangular{});
    else if (name == "hann")
        s::fill_window(std::span<T>(w), s::hann{});
    else
        s::fill_window(std::span<T>(w), s::kaiser{8.6});
    return w;
}

template <typename T>
void numpy_agrees()
{
    const fixture& f = numpy_fixture();
    ASSERT_EQ(f.signals.size(), 5u);
    std::size_t checked = 0;
    for (const auto& [name, X] : f.ffts) {
        std::vector<T> x;
        for (const double v : f.signals.at(name))
            x.push_back(static_cast<T>(v));   // float32 values: exact in every type
        const std::size_t n = x.size();
        ASSERT_EQ(X.size(), 2 * n);
        const T xn = norm2(x);
        for (std::size_t k = 0; k < n; ++k) {
            s::goertzel<T> d = s::goertzel<T>::at_bin(k, n);
            d.update(x);
            const long double sin_w = std::sin(2.0L * m::math_limits<long double>::pi * static_cast<long double>(k) / static_cast<long double>(n));
            const T tol = goertzel_bound<T>(n, sin_w, xn) + T(1e-13) * xn;   // numpy's own rounding, in double
            EXPECT_LE(s::abs(d.bin() - s::complex<T>{static_cast<T>(X[2 * k]), static_cast<T>(X[2 * k + 1])}), tol) << name << " k " << k;
            ++checked;
        }
    }
    for (const auto& r : f.dtfts) {
        std::vector<T> x;
        for (const double v : f.signals.at(r.signal))
            x.push_back(static_cast<T>(v));
        const std::size_t n = x.size();
        const auto w = numpy_window<T>(r.window, n);
        s::goertzel<T> d(r.f, r.fs, n);
        d.update(x, w);
        const long double sin_w = std::sin(2.0L * m::math_limits<long double>::pi * static_cast<long double>(r.f) / static_cast<long double>(r.fs));
        // The window table's rounding adds eps ||x||_1 at most, and numpy's sum its own.
        const T tol = goertzel_bound<T>(n, sin_w, norm2(x)) + T(4) * eps<T>() * static_cast<T>(n) * norm2(x) + T(1e-12) * norm2(x);
        EXPECT_LE(s::abs(d.bin() - s::complex<T>{static_cast<T>(r.re), static_cast<T>(r.im)}), tol)
            << r.signal << " " << r.window << " f " << r.f;
        ++checked;
    }
    EXPECT_EQ(checked, 64u + 63u + 101u + 200u + 4 * 15u + 16u);
}

TEST(GoertzelAgainstNumpy, IntegerBinsAndTheTransformBetweenBinsWithWindowsInDouble) { numpy_agrees<double>(); }
TEST(GoertzelAgainstNumpy, IntegerBinsAndTheTransformBetweenBinsWithWindowsInFloat) { numpy_agrees<float>(); }
TEST(GoertzelAgainstNumpy, IntegerBinsAndTheTransformBetweenBinsWithWindowsInLongDouble) { numpy_agrees<long double>(); }

TEST(GoertzelAgainstNumpy, CompileTimeDetectorWithAHannWindowIsNumpys)
{
    const fixture& f = numpy_fixture();
    std::vector<double> x = f.signals.at("noise64");
    s::static_goertzel<double, 3.5, 64.0, 64, s::hann{}> d;
    d.update(x);
    bool found = false;
    for (const auto& r : f.dtfts)
        if (r.signal == "noise64" && r.window == "hann" && r.f == 3.5) {
            EXPECT_NEAR(d.bin().re, r.re, 1e-12);
            EXPECT_NEAR(d.bin().im, r.im, 1e-12);
            EXPECT_NEAR(d.power(), r.re * r.re + r.im * r.im, 1e-11);
            found = true;
        }
    EXPECT_TRUE(found);
}

// --- tones, noise and the edges ---------------------------------------------------------------------------------

template <typename T>
void tone_on_a_bin()
{
    const std::size_t n = 256, k = 37;
    const T a = T(0.75), phi = T(0.6);
    std::vector<T> x(n);
    for (std::size_t j = 0; j < n; ++j)
        x[j] = a * static_cast<T>(std::cos(2.0L * m::math_limits<long double>::pi * k * j / n + phi));
    s::goertzel<T> d = s::goertzel<T>::at_bin(k, n);
    d.update(x);
    const T tol = T(64) * eps<T>() * T(n);
    EXPECT_NEAR(d.magnitude(), a * T(n) / 2, tol);   // A n / 2
    EXPECT_NEAR(d.phase(), phi, tol / T(n));
    // Every other bin sees nothing (leakage is zero on a bin).
    for (std::size_t q : {0u, 1u, 36u, 38u, 128u, 218u}) {
        s::goertzel<T> o = s::goertzel<T>::at_bin(q, n);
        o.update(x);
        EXPECT_LE(o.magnitude(), tol) << q;
    }
    // The mirror bin n - k is the conjugate.
    s::goertzel<T> mirror = s::goertzel<T>::at_bin(n - k, n);
    mirror.update(x);
    EXPECT_NEAR(mirror.phase(), -phi, tol / T(n));
}

TEST(GoertzelTones, ToneOnABinHasMagnitudeANOverTwoAndItsPhase)
{
    tone_on_a_bin<float>();
    tone_on_a_bin<double>();
    tone_on_a_bin<long double>();
}

TEST(GoertzelTones, ToneOffABinIsTheDirectSumAtThatFrequency)
{
    const std::size_t n = 200;
    const double fs = 1000.0, f = 123.4;   // bin 24.68
    std::vector<double> x(n);
    for (std::size_t j = 0; j < n; ++j)
        x[j] = std::sin(2 * pi * f * static_cast<double>(j) / fs);
    s::goertzel<double> d(f, fs, n);
    d.update(x);
    long double re = 0, im = 0;
    for (std::size_t j = 0; j < n; ++j) {
        re += x[j] * std::cos(-2.0L * m::math_limits<long double>::pi * f * j / fs);
        im += x[j] * std::sin(-2.0L * m::math_limits<long double>::pi * f * j / fs);
    }
    EXPECT_NEAR(d.bin().re, static_cast<double>(re), 1e-10);
    EXPECT_NEAR(d.bin().im, static_cast<double>(im), 1e-10);
    EXPECT_NEAR(d.magnitude(), static_cast<double>(n) / 2, 1.0);   // A n / 2, less the leakage of its negative frequency
    // The nearest integer bin (25) leaks: it sees less of the tone.
    s::goertzel<double> near_bin = s::goertzel<double>::at_bin(25, n);
    near_bin.update(x);
    EXPECT_LT(near_bin.magnitude(), d.magnitude() * 0.9);
}

TEST(GoertzelTones, NoisePowerAveragesToNTimesTheVariance)
{
    const std::size_t n = 128;
    double sum = 0;
    const int blocks = 400;
    for (int b = 0; b < blocks; ++b) {
        s::goertzel<double> d = s::goertzel<double>::at_bin(17, n);
        d.update(noise<double>(n, 77 + static_cast<std::uint64_t>(b)));
        sum += d.power();
    }
    // Uniform on [-1, 1): variance 1/3, so E|X_k|^2 = n / 3.
    EXPECT_NEAR(sum / blocks / (static_cast<double>(n) / 3), 1.0, 0.15);
}

TEST(GoertzelEdges, DcIsTheSumAndNyquistTheAlternatingSum)
{
    const std::size_t n = 64;
    std::vector<double> dc(n, 0.5), alt(n);
    for (std::size_t j = 0; j < n; ++j)
        alt[j] = j % 2 == 0 ? 1.0 : -1.0;
    s::goertzel<double> d0 = s::goertzel<double>::at_bin(0, n), d0alt = d0, nyq = s::goertzel<double>::at_bin(n / 2, n), nyqdc = nyq;
    EXPECT_EQ(d0.coefficients().coeff, 2.0);
    EXPECT_EQ(nyq.coefficients().coeff, -2.0);
    EXPECT_EQ(nyq.coefficients().sin_w, 0.0);
    d0.update(dc);
    d0alt.update(alt);
    nyq.update(alt);
    nyqdc.update(dc);
    EXPECT_EQ(d0.bin(), (s::complex<double>{32.0, 0.0}));   // exact: small integers
    EXPECT_EQ(d0.power(), 1024.0);
    EXPECT_EQ(d0.phase(), 0.0);
    EXPECT_EQ(nyq.bin(), (s::complex<double>{64.0, 0.0}));
    EXPECT_EQ(nyq.power(), 4096.0);
    EXPECT_EQ(d0alt.power(), 0.0);
    EXPECT_EQ(nyqdc.power(), 0.0);
    // A negative DC has phase pi.
    s::goertzel<double> neg = s::goertzel<double>::at_bin(0, 4);
    neg.update(std::vector<double>{-1, -1, -1, -1});
    EXPECT_DOUBLE_EQ(neg.phase(), pi);
    // An odd block has no Nyquist bin: its last bins are a conjugate pair.
    const auto x = noise<double>(7, 5);
    s::goertzel<double> b3 = s::goertzel<double>::at_bin(3, 7), b4 = s::goertzel<double>::at_bin(4, 7);
    b3.update(x);
    b4.update(x);
    EXPECT_NEAR(b3.bin().re, b4.bin().re, 1e-14);
    EXPECT_NEAR(b3.bin().im, -b4.bin().im, 1e-14);
}

TEST(GoertzelEdges, FrequenciesAliasIntoTheSampleRate)
{
    const auto a = s::goertzel_coefficients<double>::at_frequency(1100.0, 8000.0, 100);
    const auto b = s::goertzel_coefficients<double>::at_frequency(9100.0, 8000.0, 100);
    const auto c = s::goertzel_coefficients<double>::at_frequency(-6900.0, 8000.0, 100);
    EXPECT_NEAR(a.coeff, b.coeff, 1e-14);
    EXPECT_NEAR(a.coeff, c.coeff, 1e-14);
    EXPECT_NEAR(a.bin, 13.75, 1e-12);
    EXPECT_NEAR(b.bin, 13.75, 1e-12);
    // The block phase of a quarter-bin offset is a quarter turn back.
    EXPECT_NEAR(a.block_phase.re, std::cos(-2 * pi * 0.75), 1e-15);
    EXPECT_NEAR(a.block_phase.im, std::sin(-2 * pi * 0.75), 1e-15);
}

TEST(GoertzelStreaming, PartialAndOverLongBlocksAreTheTransformOfWhatWasPushed)
{
    const std::size_t n = 50;
    const auto x = noise<double>(80, 9);
    for (const double f : {7.0, 7.3}) {
        s::goertzel<double> d(f, 50.0, n);
        for (std::size_t c = 1; c <= x.size(); ++c) {
            d.push(x[c - 1]);
            EXPECT_EQ(d.count(), c);
            EXPECT_EQ(d.complete(), c >= n);
            long double re = 0, im = 0;
            for (std::size_t j = 0; j < c; ++j) {
                re += x[j] * std::cos(-2.0L * m::math_limits<long double>::pi * f * j / 50.0L);
                im += x[j] * std::sin(-2.0L * m::math_limits<long double>::pi * f * j / 50.0L);
            }
            EXPECT_NEAR(d.bin().re, static_cast<double>(re), 1e-11) << "f " << f << " c " << c;
            EXPECT_NEAR(d.bin().im, static_cast<double>(im), 1e-11) << "f " << f << " c " << c;
        }
    }
    s::goertzel<double> empty(7.0, 50.0, n);
    EXPECT_EQ(empty.bin(), (s::complex<double>{0.0, 0.0}));
    EXPECT_EQ(empty.power(), 0.0);
}

TEST(GoertzelStreaming, PushUpdateAndResetAreTheSameRecurrence)
{
    const auto x = noise<float>(300, 11);
    s::goertzel<float> a(1234.5, 8000.0, 300), b = a;
    for (const float v : x)
        a.push(v);
    b.update(std::span<const float>(x).first(100));
    b.update(std::span<const float>(x).subspan(100));
    EXPECT_EQ(a.state().s1, b.state().s1);   // bit-identical
    EXPECT_EQ(a.state().s2, b.state().s2);
    EXPECT_EQ(a.power(), b.power());
    a.reset();
    EXPECT_EQ(a.count(), 0u);
    EXPECT_EQ(a.power(), 0.0f);
    EXPECT_EQ(a.coefficients().coeff, b.coefficients().coeff);   // reset keeps the coefficients
    a.update(x);
    EXPECT_EQ(a.power(), b.power());
    // A copy is an independent detector (trivially copyable state).
    s::goertzel<float> c = a;
    c.push(1.0f);
    EXPECT_NE(c.count(), a.count());
    // The free state is the same arithmetic.
    s::goertzel_state<float> st;
    for (const float v : x)
        st.push(a.coefficients().coeff, v);
    EXPECT_EQ(st.power(a.coefficients()), a.power());
    EXPECT_EQ(st.bin(a.coefficients()), a.bin());
    st.reset();
    EXPECT_EQ(st.count, 0u);
}

TEST(GoertzelStreaming, CompileTimeDetectorIsTheRunTimeOneBitForBit)
{
    const auto x = noise<float>(205, 13);
    using detector = s::static_goertzel<float, 941.0, 8000.0, 205>;
    detector a;
    s::goertzel<float> b(detector::coefficients);
    a.update(x);
    b.update(x);
    EXPECT_EQ(a.power(), b.power());
    EXPECT_EQ(a.bin(), b.bin());
    EXPECT_EQ(a.magnitude(), b.magnitude());
    EXPECT_EQ(a.phase(), b.phase());
    EXPECT_TRUE(a.complete());
    EXPECT_EQ(a.count(), 205u);
    EXPECT_EQ(a.state().s1, b.state().s1);
    a.reset();
    EXPECT_EQ(a.count(), 0u);
    // With a window: the compile-time table is make_window's, the result the run-time windowed update's.
    s::static_goertzel<float, 941.0, 8000.0, 205, s::kaiser{8.6}> aw;
    constexpr auto w = s::make_window<float, 205>(s::kaiser{8.6});
    EXPECT_EQ(decltype(aw)::window, w);
    aw.update(x);
    b.reset();
    b.update(x, w);
    EXPECT_EQ(aw.power(), b.power());
}

// --- the bank and DTMF ------------------------------------------------------------------------------------------

constexpr std::array<double, 8> dtmf_hz{697, 770, 852, 941, 1209, 1336, 1477, 1633};
constexpr char dtmf_keys[4][5] = {"123A", "456B", "789C", "*0#D"};

TEST(GoertzelBank, EachDetectorIsASingleDetectorBitForBit)
{
    constexpr s::goertzel_bank<double, 8> bank0(dtmf_hz, 8000.0, 205);   // computed by the compiler
    static_assert(bank0.size() == 8 && bank0.coefficients(3).coeff == s::goertzel_coefficients<double>::at_frequency(941.0, 8000.0, 205).coeff);
    auto bank = bank0;
    const auto x = noise<double>(205, 17);
    bank.update(x);
    EXPECT_TRUE(bank.complete());
    EXPECT_EQ(bank.count(), 205u);
    const auto p = bank.powers();
    for (std::size_t i = 0; i < 8; ++i) {
        s::goertzel<double> d(bank.coefficients(i));   // the bank's constants (built by the compiler, so not
        d.update(x);                                   // necessarily <cmath>'s to the last bit)
        const auto rt = s::goertzel_coefficients<double>::at_frequency(dtmf_hz[i], 8000.0, 205);
        EXPECT_NEAR(rt.coeff, bank.coefficients(i).coeff, 4e-16);
        EXPECT_EQ(bank.power(i), d.power());
        EXPECT_EQ(p[i], d.power());
        EXPECT_EQ(bank.bin(i), d.bin());
        EXPECT_EQ(bank.magnitude(i), d.magnitude());
        EXPECT_EQ(bank.phase(i), d.phase());
        EXPECT_EQ(bank.state(i).s1, d.state().s1);
    }
    // Windowed, and from precomputed coefficients.
    std::array<s::goertzel_coefficients<double>, 2> c{s::goertzel_coefficients<double>::at_bin(3, 205), s::goertzel_coefficients<double>::at_frequency(1000.0, 8000.0, 205)};
    s::goertzel_bank<double, 2> two(c);
    const auto w = s::make_window<double, 205>(s::hann{});
    two.update(std::span<const double>(x).first(5), w);
    two.update(std::span<const double>(x).subspan(5), w);
    s::goertzel<double> d(c[1]);
    d.update(x, w);
    EXPECT_EQ(two.power(1), d.power());
    two.reset();
    EXPECT_EQ(two.count(), 0u);
    EXPECT_EQ(two.power(0), 0.0);
    s::goertzel_bank<float, 1> dflt;
    EXPECT_EQ(dflt.coefficients(0).n, 1u);
}

template <typename T>
std::vector<T> dtmf_tone(std::size_t row, std::size_t col, std::size_t n, std::uint64_t seed)
{
    auto x = noise<T>(n, seed);
    for (std::size_t j = 0; j < n; ++j) {
        const double t = static_cast<double>(j) / 8000.0;
        x[j] = static_cast<T>(0.05 * static_cast<double>(x[j]) + 0.5 * std::sin(2 * pi * dtmf_hz[row] * t)
                              + 0.4 * std::sin(2 * pi * dtmf_hz[4 + col] * t + 1.0));
    }
    return x;
}

// A decoder: the strongest row and column, each well above every other tone of its group.
template <typename T>
char decode(const s::goertzel_bank<T, 8>& bank)
{
    const std::size_t r = bank.strongest(0, 4), c = bank.strongest(4, 8);
    for (std::size_t i = 0; i < 8; ++i)
        if (i != r && i != c && bank.power(i) * 10 > std::min(bank.power(r), bank.power(c)))
            return '?';
    return dtmf_keys[r][c - 4];
}

template <typename T>
void every_digit()
{
    constexpr s::goertzel_bank<T, 8> proto(dtmf_hz, 8000.0, 205);
    for (std::size_t r = 0; r < 4; ++r)
        for (std::size_t c = 0; c < 4; ++c) {
            auto bank = proto;
            bank.update(dtmf_tone<T>(r, c, 205, 100 + 4 * r + c));
            EXPECT_EQ(decode(bank), dtmf_keys[r][c]) << r << " " << c;
        }
    // Silence (noise alone) and a single tone decode to nothing.
    auto bank = proto;
    bank.update(noise<T>(205, 99));
    EXPECT_EQ(decode(bank), '?');
    bank.reset();
    std::vector<T> one(205);
    for (std::size_t j = 0; j < 205; ++j)
        one[j] = static_cast<T>(std::sin(2 * pi * 852.0 * static_cast<double>(j) / 8000.0));
    bank.update(one);
    EXPECT_EQ(decode(bank), '?');
    EXPECT_EQ(bank.strongest(), 2u);
}

TEST(GoertzelDtmf, TheBankDecodesEachOfTheSixteenDigits)
{
    every_digit<float>();
    every_digit<double>();
    every_digit<long double>();
}

} // namespace
