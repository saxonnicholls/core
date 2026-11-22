//
//  fft_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/signal/fft.hpp>: every length from 1 to 300 against the direct transform in long double
//  within the stated bound, powers of two to 2^16 and primes to 10007 bin by bin, float, numpy's normalisations, round
//  trips and Parseval, the real transforms and their symmetry, two and three dimensions, the in-place power-of-two
//  plan and its bit reversal, every backward pass against central differences, and whole transforms at compile time.
//

#include <snicholls/core/signal/fft.hpp>

#include "../support/signal_fixture.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

namespace s = snicholls::signal;
using cd = s::complex<double>;
using cl = s::complex<long double>;

// --- compile time ---------------------------------------------------------------------------------------------

static_assert(s::fft_layout::of(1000).factor_count == 5 && s::fft_layout::of(1000).factors[0] == 4 && s::fft_layout::of(1000).factors[1] == 2
              && s::fft_layout::of(1000).factors[2] == 5 && !s::fft_layout::of(1000).bluestein());
static_assert(s::fft_layout::of(1009).bluestein() && s::fft_layout::of(1009).m == 2048 && s::fft_layout::of(1009).storage_size() == 1009 + 4096);
static_assert(s::fft_layout::of(31 * 29).factor_count == 2 && !s::fft_layout::of(31 * 29).bluestein() && s::fft_layout::of(37).bluestein());
static_assert(s::fft_layout::of(0).storage_size() == 0 && s::fft_layout::of(1).storage_size() == 1 && s::fft_layout::of(1).factor_count == 0);
static_assert(std::is_trivially_copyable_v<s::fft_plan<float>> && std::is_trivially_copyable_v<s::rfft_plan<double>>);
static_assert(std::is_trivially_copyable_v<s::fft_layout> && sizeof(s::complex<float>) == 8);
static_assert(noexcept(s::fft(std::declval<const s::fft_plan<double>&>(), std::span<cd>{}, std::span<cd>{})));
static_assert(s::bit_reversal_permutation<8>() == std::array<std::uint32_t, 8>{0, 4, 2, 6, 1, 5, 3, 7});
static_assert(s::fft_scale<double>(s::fft_norm::backward, 4, true) == 0.25 && s::fft_scale<double>(s::fft_norm::ortho, 4, false) == 0.5
              && s::fft_scale<double>(s::fft_norm::forward, 4, false) == 0.25 && s::fft_scale<double>(s::fft_norm::none, 4, true) == 1.0);

template <std::size_t N>
constexpr std::array<cd, N> ramp()
{
    std::array<cd, N> x{};
    for (std::size_t i = 0; i < N; ++i)
        x[i] = {static_cast<double>(i % 7) - 3.0, static_cast<double>((i * 5) % 11) / 4.0};
    return x;
}

template <std::size_t N>
constexpr double max_error_against_dft()
{
    const auto x = ramp<N>();
    const auto y = s::fft(x);
    std::array<cd, N> ref{}, table{};
    s::twiddles<double>(table);
    s::dft<double>(x, ref, table);
    double e = 0;
    for (std::size_t k = 0; k < N; ++k)
        e = std::max(e, s::abs(y[k] - ref[k]));
    return e;
}

// Whole transforms in static_asserts: mixed radix, a prime through Bluestein, a power of two.
static_assert(max_error_against_dft<12>() < 1e-13);
static_assert(max_error_against_dft<17>() < 1e-13);
static_assert(max_error_against_dft<64>() < 1e-12);

constexpr bool round_trips_at_compile_time()
{
    const auto x = ramp<30>();
    const auto back = s::ifft(s::fft(x));
    for (std::size_t i = 0; i < x.size(); ++i)
        if (s::abs(back[i] - x[i]) > 1e-14)
            return false;
    return true;
}
static_assert(round_trips_at_compile_time());

constexpr bool in_place_matches_stockham()
{
    auto x = ramp<32>();
    const auto y = s::fft(x);
    s::static_fft_plan_v<double, 32>.forward_in_place(x);
    for (std::size_t k = 0; k < 32; ++k)
        if (s::abs(x[k] - y[k]) > 1e-13)
            return false;
    s::static_fft_plan_v<double, 32>.inverse_in_place(x);
    const auto r = ramp<32>();
    for (std::size_t k = 0; k < 32; ++k)
        if (s::abs(x[k] - r[k]) > 1e-14)
            return false;
    return s::static_fft_plan_v<double, 32>.bit_reversal()[1] == 16;
}
static_assert(in_place_matches_stockham());

// A 1,024-point Parseval at compile time: the plan's tables and the transform both in constant evaluation.
constexpr bool parseval_1024()
{
    const auto x = ramp<1024>();
    const auto y = s::fft(x, s::fft_norm::ortho);
    double ex = 0, ey = 0;
    for (std::size_t i = 0; i < 1024; ++i) {
        ex += s::norm(x[i]);
        ey += s::norm(y[i]);
    }
    return snicholls::math::near(ex, ey, 1e-10 * ex);
}
static_assert(parseval_1024());

// A real transform at compile time: its half spectrum is the complex transform's first half, and it inverts.
constexpr bool real_at_compile_time()
{
    constexpr std::array<double, 10> x{1, -2, 3, 0.5, 0, 4, -1, 2, 2, -3};
    const auto h = s::rfft(x);
    std::array<cd, 10> c{};
    for (std::size_t i = 0; i < 10; ++i)
        c[i] = {x[i], 0};
    const auto f = s::fft(c);
    for (std::size_t k = 0; k < h.size(); ++k)
        if (s::abs(h[k] - f[k]) > 1e-13)
            return false;
    const auto back = s::irfft<10>(h);
    for (std::size_t i = 0; i < 10; ++i)
        if (snicholls::math::abs(back[i] - x[i]) > 1e-14)
            return false;
    constexpr std::array<double, 7> odd{1, 2, 3, 4, 5, 6, 7};
    return snicholls::math::near(s::rfft(odd)[0].re, 28.0, 1e-13) && snicholls::math::near(s::irfft<7>(s::rfft(odd))[6], 7.0, 1e-13);
}
static_assert(real_at_compile_time());

// A run-time plan built in constant evaluation over a std::array.
constexpr double run_time_plan_at_compile_time()
{
    std::array<cd, s::fft_plan<double>::storage_size(6)> st{};
    const s::fft_plan<double> p(6, st);
    std::array<cd, 6> x{{{1, 0}, {1, 0}, {1, 0}, {1, 0}, {1, 0}, {1, 0}}};
    std::array<cd, s::fft_plan<double>::work_size(6)> w{};
    s::fft(p, x, w);
    return x[0].re + s::abs(x[1]) + s::abs(x[5]);
}
static_assert(snicholls::math::near(run_time_plan_at_compile_time(), 6.0, 1e-14));

// --- run time -------------------------------------------------------------------------------------------------

struct lcg
{
    std::uint64_t s;
    double next()
    {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(s >> 11) * 0x1.0p-53 - 0.5;
    }
};

template <typename T>
std::vector<s::complex<T>> random_complex(std::size_t n, std::uint64_t seed)
{
    lcg g{seed};
    std::vector<s::complex<T>> x(n);
    for (auto& v : x) {
        v.re = static_cast<T>(g.next());
        v.im = static_cast<T>(g.next());
    }
    return x;
}

template <typename T>
struct planned
{
    std::vector<s::complex<T>> storage, work;
    s::fft_plan<T> plan;
    explicit planned(std::size_t n)
        : storage(s::fft_plan<T>::storage_size(n)), work(s::fft_plan<T>::work_size(n)), plan(n, std::span<s::complex<T>>(storage))
    {
    }
};

template <typename T>
std::vector<cl> widen(const std::vector<s::complex<T>>& x)
{
    std::vector<cl> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        out[i] = {static_cast<long double>(x[i].re), static_cast<long double>(x[i].im)};
    return out;
}

// One bin of the direct transform in long double, by its definition.
cl direct_bin(const std::vector<cl>& x, std::size_t k)
{
    const std::size_t n = x.size();
    cl acc{};
    std::size_t e = 0;
    for (std::size_t j = 0; j < n; ++j) {
        acc += x[j] * s::twiddle<long double>(e, n);
        e = (e + k) % n;
    }
    return acc;
}

template <typename T>
long double l2(const std::vector<s::complex<T>>& x)
{
    long double a = 0;
    for (const auto& v : x)
        a += static_cast<long double>(v.re) * v.re + static_cast<long double>(v.im) * v.im;
    return std::sqrt(a);
}

// The stated bound: 2 eps ceil(log2 L) ||x||_2, L the length the passes run at (n, or Bluestein's m).
template <typename T>
long double bound(const s::fft_plan<T>& p, long double norm2)
{
    const std::size_t len = p.uses_bluestein() ? p.layout().m : p.size();
    const long double lg = std::max<long double>(1, std::ceil(std::log2(static_cast<long double>(len))));
    return 2 * static_cast<long double>(std::numeric_limits<T>::epsilon()) * lg * norm2;
}

// The direct transform's own error, where long double is no wider than T (MSVC, Apple's ARM64): a recursive sum of n
// complex products, each twiddle within an ulp, is within (n + 3) eps ||x||_1. Where long double is wider, none.
template <typename T>
long double reference_error(const std::vector<s::complex<T>>& x)
{
    if constexpr (std::numeric_limits<long double>::digits > std::numeric_limits<T>::digits)
        return 0;
    long double l1 = 0;
    for (const auto& v : x)
        l1 += std::sqrt(static_cast<long double>(v.re) * v.re + static_cast<long double>(v.im) * v.im);
    return static_cast<long double>(x.size() + 3) * std::numeric_limits<long double>::epsilon() * l1;
}

template <typename T>
long double error_against_bins(std::size_t n, const std::vector<std::size_t>& bins, std::uint64_t seed, long double& limit)
{
    planned<T> p(n);
    auto x = random_complex<T>(n, seed);
    const auto xl = widen(x);
    limit = bound(p.plan, l2(x)) + reference_error<T>(x);
    s::fft(p.plan, std::span<s::complex<T>>(x), std::span<s::complex<T>>(p.work));
    long double e = 0;
    for (const std::size_t k : bins) {
        const cl r = direct_bin(xl, k);
        const long double dr = x[k].re - r.re, di = x[k].im - r.im;
        e = std::max(e, std::sqrt(dr * dr + di * di));
    }
    return e;
}

TEST(FftAccuracy, EveryLengthFromOneTo300IsTheDirectTransformWithinTheBound)
{
    for (std::size_t n = 1; n <= 300; ++n) {
        std::vector<std::size_t> all(n);
        for (std::size_t k = 0; k < n; ++k)
            all[k] = k;
        long double limit = 0;
        const long double e = error_against_bins<double>(n, all, 1000 + n, limit);
        EXPECT_LE(e, limit) << "n = " << n;
    }
}

TEST(FftAccuracy, PowersOfTwoToTwoToTheSixteenAndPrimesTo10007AgreeBinByBin)
{
    if (std::numeric_limits<long double>::digits <= std::numeric_limits<double>::digits)
        GTEST_SKIP() << "long double is double here (Apple ARM64, MSVC): no wider reference to check the last bits against";

    for (const std::size_t n : {std::size_t{1} << 10, std::size_t{1} << 12, std::size_t{1} << 16, std::size_t{1009}, std::size_t{4093},
                                std::size_t{10007}, std::size_t{3 * 5 * 7 * 11 * 13}, std::size_t{2 * 31 * 31}}) {
        std::vector<std::size_t> bins{0, 1, 2, n / 2, n - 1};
        lcg g{n};
        for (int i = 0; i < 24; ++i)
            bins.push_back(static_cast<std::size_t>((g.next() + 0.5) * static_cast<double>(n)) % n);
        long double limit = 0;
        const long double e = error_against_bins<double>(n, bins, 7 * n, limit);
        EXPECT_LE(e, limit) << "n = " << n;
    }
}

TEST(FftAccuracy, FloatIsWithinItsOwnEpsilonsBound)
{
    for (const std::size_t n : {std::size_t{1}, std::size_t{2}, std::size_t{37}, std::size_t{256}, std::size_t{1000}, std::size_t{1009}, std::size_t{4096}}) {
        std::vector<std::size_t> bins;
        for (std::size_t k = 0; k < n; k += 1 + n / 97)
            bins.push_back(k);
        long double limit = 0;
        const long double e = error_against_bins<float>(n, bins, 3 * n, limit);
        EXPECT_LE(e, limit) << "n = " << n;
    }
}

TEST(FftConventions, NormalisationsAreNumpys)
{
    const auto x = snicholls::test::signal_complex("fft12_input");
    planned<double> p(12);
    const std::array<std::pair<const char*, s::fft_norm>, 3> norms{{{"backward", s::fft_norm::backward},
                                                                     {"forward", s::fft_norm::forward},
                                                                     {"ortho", s::fft_norm::ortho}}};
    for (const auto& [name, norm] : norms) {
        auto y = x;
        s::fft(p.plan, std::span<cd>(y), std::span<cd>(p.work), norm);
        const auto ref = snicholls::test::signal_complex(std::string("fft12_") + name);
        auto z = x;
        s::ifft(p.plan, std::span<cd>(z), std::span<cd>(p.work), norm);
        const auto iref = snicholls::test::signal_complex(std::string("ifft12_") + name);
        for (std::size_t k = 0; k < 12; ++k) {
            EXPECT_NEAR(y[k].re, ref[k].re, 1e-14) << name;
            EXPECT_NEAR(y[k].im, ref[k].im, 1e-14) << name;
            EXPECT_NEAR(z[k].re, iref[k].re, 1e-14) << name;
            EXPECT_NEAR(z[k].im, iref[k].im, 1e-14) << name;
        }
    }
    auto none = x;
    s::fft(p.plan, std::span<cd>(none), std::span<cd>(p.work), s::fft_norm::none);
    s::ifft(p.plan, std::span<cd>(none), std::span<cd>(p.work), s::fft_norm::none);
    EXPECT_NEAR(none[3].re, 12 * x[3].re, 1e-13);   // neither direction scaled
}

TEST(FftIdentities, RoundTripsAndParsevalHoldForEveryNormalisation)
{
    for (const std::size_t n : {std::size_t{5}, std::size_t{64}, std::size_t{97}, std::size_t{360}, std::size_t{1031}}) {
        planned<double> p(n);
        const auto x = random_complex<double>(n, n);
        for (const auto norm : {s::fft_norm::backward, s::fft_norm::forward, s::fft_norm::ortho}) {
            auto y = x;
            s::fft(p.plan, std::span<cd>(y), std::span<cd>(p.work), norm);
            if (norm == s::fft_norm::ortho) {
                EXPECT_NEAR(static_cast<double>(l2(y)), static_cast<double>(l2(x)), 1e-13 * static_cast<double>(l2(x))) << n;
            }
            s::ifft(p.plan, std::span<cd>(y), std::span<cd>(p.work), norm);
            for (std::size_t i = 0; i < n; ++i)
                ASSERT_LE(s::abs(y[i] - x[i]), static_cast<double>(bound(p.plan, l2(x)))) << n;
        }
    }
}

TEST(FftPlans, TablesBuiltAtCompileTimeAgreeWithThoseBuiltAtRunTime)
{
    constexpr s::static_fft_plan<double, 360> ct{};
    planned<double> rt(360);
    for (std::size_t k = 0; k < 360; ++k) {
        EXPECT_NEAR(ct.tables()[k].re, rt.storage[k].re, 4e-16);
        EXPECT_NEAR(ct.tables()[k].im, rt.storage[k].im, 4e-16);
    }
    EXPECT_EQ(rt.storage[90], (cd{0, -1}));   // the quarter turn is exact
    EXPECT_EQ(rt.storage[180], (cd{-1, 0}));
    // The static plan through the free functions gives the same transform as a run-time plan.
    auto a = random_complex<double>(360, 9);
    auto b = a;
    std::array<cd, 360> w{};
    s::fft(ct.plan(), std::span<cd>(a), std::span<cd>(w));
    s::fft(rt.plan, std::span<cd>(b), std::span<cd>(rt.work));
    for (std::size_t k = 0; k < 360; ++k)
        EXPECT_LE(s::abs(a[k] - b[k]), 1e-13);
}

TEST(FftPlans, LengthsZeroAndOneAreHandled)
{
    planned<double> zero(0);
    std::vector<cd> none;
    s::fft(zero.plan, std::span<cd>(none), std::span<cd>(zero.work));
    planned<double> one(1);
    std::vector<cd> x{{3, -2}};
    s::fft(one.plan, std::span<cd>(x), std::span<cd>(one.work));
    EXPECT_EQ(x[0], (cd{3, -2}));
    s::ifft(one.plan, std::span<cd>(x), std::span<cd>(one.work));
    EXPECT_EQ(x[0], (cd{3, -2}));
}

struct real_planned
{
    std::vector<cd> storage, work;
    s::rfft_plan<double> plan;
    explicit real_planned(std::size_t n)
        : storage(s::rfft_plan<double>::storage_size(n)), work(s::rfft_plan<double>::work_size(n)), plan(n, std::span<cd>(storage))
    {
    }
};

TEST(RealFft, HalfSpectrumIsNumpysAndTheComplexTransformsFirstHalf)
{
    for (const std::size_t n : {std::size_t{10}, std::size_t{11}}) {
        const auto& x = snicholls::test::signal_record("rfft" + std::to_string(n) + "_input");
        const auto ref = snicholls::test::signal_complex("rfft" + std::to_string(n));
        real_planned p(n);
        std::vector<cd> out(p.plan.bins());
        s::rfft(p.plan, std::span<const double>(x), std::span<cd>(out), std::span<cd>(p.work));
        for (std::size_t k = 0; k < out.size(); ++k) {
            EXPECT_NEAR(out[k].re, ref[k].re, 1e-14) << n;
            EXPECT_NEAR(out[k].im, ref[k].im, 1e-14) << n;
        }
    }
    for (std::size_t n = 1; n <= 80; ++n) {
        real_planned p(n);
        planned<double> c(n);
        const auto z = random_complex<double>(n, n);
        std::vector<double> x(n);
        std::vector<cd> full(n), half(p.plan.bins());
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = z[i].re;
            full[i] = {x[i], 0};
        }
        s::rfft(p.plan, std::span<const double>(x), std::span<cd>(half), std::span<cd>(p.work));
        s::fft(c.plan, std::span<cd>(full), std::span<cd>(c.work));
        for (std::size_t k = 0; k < half.size(); ++k)
            EXPECT_LE(s::abs(half[k] - full[k]), 1e-14 * static_cast<double>(n)) << n << " " << k;
        // Hermitian symmetry of the full spectrum, which rfft omits; DC (and Nyquist) real
        for (std::size_t k = 1; k < n; ++k)
            EXPECT_LE(s::abs(full[k] - s::conj(full[n - k])), 1e-13) << n;
        EXPECT_EQ(half[0].im, 0.0);
        if (n % 2 == 0) {
            EXPECT_EQ(half[n / 2].im, 0.0);
        }
    }
}

TEST(RealFft, InverseIsNumpysAndIgnoresTheImaginaryPartsOfDcAndNyquist)
{
    const auto h = snicholls::test::signal_complex("irfft_input");
    for (const std::size_t n : {std::size_t{10}, std::size_t{11}}) {
        const auto& ref = snicholls::test::signal_record("irfft" + std::to_string(n));
        real_planned p(n);
        std::vector<double> out(n);
        const std::vector<cd> in(h.begin(), h.begin() + static_cast<std::ptrdiff_t>(p.plan.bins()));
        s::irfft(p.plan, std::span<const cd>(in), std::span<double>(out), std::span<cd>(p.work));
        for (std::size_t i = 0; i < n; ++i)
            EXPECT_NEAR(out[i], ref[i], 1e-14) << n;
    }
    for (std::size_t n = 1; n <= 64; ++n) {
        real_planned p(n);
        std::vector<double> x(n), back(n);
        lcg g{n};
        for (auto& v : x)
            v = g.next();
        std::vector<cd> half(p.plan.bins());
        for (const auto norm : {s::fft_norm::backward, s::fft_norm::ortho, s::fft_norm::forward}) {
            s::rfft(p.plan, std::span<const double>(x), std::span<cd>(half), std::span<cd>(p.work), norm);
            s::irfft(p.plan, std::span<const cd>(half), std::span<double>(back), std::span<cd>(p.work), norm);
            for (std::size_t i = 0; i < n; ++i)
                EXPECT_NEAR(back[i], x[i], 1e-14) << n;
        }
    }
}

TEST(FftSeveralDimensions, Fft2IsNumpysAndFftnIsSeparable)
{
    const auto x = snicholls::test::signal_complex("fft2_input");
    const auto ref = snicholls::test::signal_complex("fft2");
    planned<double> rows(3), cols(4);
    auto y = x;
    std::vector<cd> work(s::fftn_work_size<double>(std::array<s::fft_plan<double>, 2>{rows.plan, cols.plan}));
    s::fft2(rows.plan, cols.plan, std::span<cd>(y), std::span<cd>(work));
    for (std::size_t i = 0; i < y.size(); ++i)
        EXPECT_LE(s::abs(y[i] - ref[i]), 1e-13);
    s::fft2(rows.plan, cols.plan, std::span<cd>(y), std::span<cd>(work), s::fft_norm::backward, true);
    for (std::size_t i = 0; i < y.size(); ++i)
        EXPECT_LE(s::abs(y[i] - x[i]), 1e-14);

    // 2 x 3 x 5 against the triple sum
    planned<double> a(2), b(3), c(5);
    const std::array<s::fft_plan<double>, 3> plans{a.plan, b.plan, c.plan};
    const auto v = random_complex<double>(30, 4);
    auto f = v;
    std::vector<cd> w3(s::fftn_work_size<double>(plans));
    s::fftn<double>(plans, std::span<cd>(f), std::span<cd>(w3), s::fft_norm::ortho);
    for (std::size_t k0 = 0; k0 < 2; ++k0)
        for (std::size_t k1 = 0; k1 < 3; ++k1)
            for (std::size_t k2 = 0; k2 < 5; ++k2) {
                cl acc{};
                for (std::size_t j0 = 0; j0 < 2; ++j0)
                    for (std::size_t j1 = 0; j1 < 3; ++j1)
                        for (std::size_t j2 = 0; j2 < 5; ++j2) {
                            const cd e = v[(j0 * 3 + j1) * 5 + j2];
                            acc += cl{e.re, e.im} * s::twiddle<long double>(j0 * k0, 2) * s::twiddle<long double>(j1 * k1, 3)
                                   * s::twiddle<long double>(j2 * k2, 5);
                        }
                const cd got = f[(k0 * 3 + k1) * 5 + k2];
                const long double sc = 1 / std::sqrt(30.0L);
                EXPECT_NEAR(got.re, static_cast<double>(acc.re * sc), 1e-14);
                EXPECT_NEAR(got.im, static_cast<double>(acc.im * sc), 1e-14);
            }

    // fft_rows is a batch of one-dimensional transforms
    const auto r = random_complex<double>(15, 2);
    auto batched = r;
    s::fft_rows(c.plan, std::span<cd>(batched), std::span<cd>(c.work));
    for (std::size_t row = 0; row < 3; ++row) {
        std::vector<cd> one(r.begin() + static_cast<std::ptrdiff_t>(5 * row), r.begin() + static_cast<std::ptrdiff_t>(5 * row + 5));
        s::fft(c.plan, std::span<cd>(one), std::span<cd>(c.work));
        for (std::size_t k = 0; k < 5; ++k)
            EXPECT_EQ(batched[5 * row + k], one[k]);
    }
}

// --- gradients ------------------------------------------------------------------------------------------------

// L(X) = sum_k w_k |X_k|^2 + Re(c_k X_k): dL/dX = 2 w X + conj(c) in the convention g = dL/dre + i dL/dim.
double complex_loss(const std::vector<cd>& y, std::vector<cd>* grad)
{
    double l = 0;
    for (std::size_t k = 0; k < y.size(); ++k) {
        const double w = 0.3 + 0.1 * static_cast<double>(k % 5);
        const cd c{0.2 * static_cast<double>(k % 3) - 0.1, 0.05 * static_cast<double>(k % 4)};
        l += w * s::norm(y[k]) + (c * y[k]).re;
        if (grad)
            (*grad)[k] = y[k] * (2 * w) + s::conj(c);
    }
    return l;
}

TEST(FftBackward, ComplexTransformsMatchCentralDifferences)
{
    for (const std::size_t n : {std::size_t{6}, std::size_t{8}, std::size_t{13}}) {
        planned<double> p(n);
        const auto x = random_complex<double>(n, 11 * n);
        for (const bool inverse : {false, true})
            for (const auto norm : {s::fft_norm::backward, s::fft_norm::ortho, s::fft_norm::forward}) {
                const auto forward = [&](std::vector<cd> v) {
                    if (inverse)
                        s::ifft(p.plan, std::span<cd>(v), std::span<cd>(p.work), norm);
                    else
                        s::fft(p.plan, std::span<cd>(v), std::span<cd>(p.work), norm);
                    return v;
                };
                std::vector<cd> g(n);
                complex_loss(forward(x), &g);
                if (inverse)
                    s::ifft_backward(p.plan, std::span<cd>(g), std::span<cd>(p.work), norm);
                else
                    s::fft_backward(p.plan, std::span<cd>(g), std::span<cd>(p.work), norm);
                const double h = 1e-6;
                for (std::size_t j = 0; j < n; ++j)
                    for (int part = 0; part < 2; ++part) {
                        auto up = x, down = x;
                        (part == 0 ? up[j].re : up[j].im) += h;
                        (part == 0 ? down[j].re : down[j].im) -= h;
                        const double numeric = (complex_loss(forward(up), nullptr) - complex_loss(forward(down), nullptr)) / (2 * h);
                        EXPECT_NEAR(part == 0 ? g[j].re : g[j].im, numeric, 1e-7) << n << " " << j;
                    }
            }
    }
}

TEST(FftBackward, RealTransformsMatchCentralDifferences)
{
    for (const std::size_t n : {std::size_t{8}, std::size_t{9}, std::size_t{12}}) {
        real_planned p(n);
        std::vector<double> x(n);
        lcg gen{n};
        for (auto& v : x)
            v = gen.next();
        for (const auto norm : {s::fft_norm::backward, s::fft_norm::ortho}) {
            // rfft: x -> X
            const auto forward = [&](const std::vector<double>& v) {
                std::vector<cd> out(p.plan.bins());
                s::rfft(p.plan, std::span<const double>(v), std::span<cd>(out), std::span<cd>(p.work), norm);
                return out;
            };
            std::vector<cd> g(p.plan.bins());
            complex_loss(forward(x), &g);
            std::vector<double> gx(n);
            s::rfft_backward(p.plan, std::span<const cd>(g), std::span<double>(gx), std::span<cd>(p.work), norm);
            const double h = 1e-6;
            for (std::size_t j = 0; j < n; ++j) {
                auto up = x, down = x;
                up[j] += h;
                down[j] -= h;
                EXPECT_NEAR(gx[j], (complex_loss(forward(up), nullptr) - complex_loss(forward(down), nullptr)) / (2 * h), 1e-7) << n;
            }
            // irfft: X -> x, with L = sum_j c_j x_j^2 + d_j x_j
            const auto inverse = [&](const std::vector<cd>& v) {
                std::vector<double> out(n);
                s::irfft(p.plan, std::span<const cd>(v), std::span<double>(out), std::span<cd>(p.work), norm);
                return out;
            };
            const auto real_loss = [](const std::vector<double>& y, std::vector<double>* grad) {
                double l = 0;
                for (std::size_t j = 0; j < y.size(); ++j) {
                    const double c = 0.4 + 0.1 * static_cast<double>(j % 3), d = 0.3 - 0.05 * static_cast<double>(j);
                    l += c * y[j] * y[j] + d * y[j];
                    if (grad)
                        (*grad)[j] = 2 * c * y[j] + d;
                }
                return l;
            };
            const auto spec = forward(x);
            std::vector<double> gy(n);
            real_loss(inverse(spec), &gy);
            std::vector<cd> gs(p.plan.bins());
            s::irfft_backward(p.plan, std::span<const double>(gy), std::span<cd>(gs), std::span<cd>(p.work), norm);
            for (std::size_t k = 0; k < spec.size(); ++k)
                for (int part = 0; part < 2; ++part) {
                    auto up = spec, down = spec;
                    (part == 0 ? up[k].re : up[k].im) += h;
                    (part == 0 ? down[k].re : down[k].im) -= h;
                    const double numeric = (real_loss(inverse(up), nullptr) - real_loss(inverse(down), nullptr)) / (2 * h);
                    EXPECT_NEAR(part == 0 ? gs[k].re : gs[k].im, numeric, 1e-7) << n << " " << k;
                }
        }
    }
}

TEST(FftBackward, SeveralDimensionsMatchCentralDifferences)
{
    planned<double> a(3), b(4);
    const std::array<s::fft_plan<double>, 2> plans{a.plan, b.plan};
    std::vector<cd> work(s::fftn_work_size<double>(plans));
    const auto x = random_complex<double>(12, 5);
    const auto forward = [&](std::vector<cd> v) {
        s::fftn<double>(plans, std::span<cd>(v), std::span<cd>(work), s::fft_norm::ortho);
        return v;
    };
    std::vector<cd> g(12);
    complex_loss(forward(x), &g);
    s::fftn_backward<double>(plans, std::span<cd>(g), std::span<cd>(work), s::fft_norm::ortho);
    const double h = 1e-6;
    for (std::size_t j = 0; j < 12; ++j) {
        auto up = x, down = x;
        up[j].im += h;
        down[j].im -= h;
        EXPECT_NEAR(g[j].im, (complex_loss(forward(up), nullptr) - complex_loss(forward(down), nullptr)) / (2 * h), 1e-7);
    }
}

} // namespace
