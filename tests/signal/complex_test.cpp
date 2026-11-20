//
//  complex_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/signal/complex.hpp>: the arithmetic, conj, norm, abs, arg, polar, expi and sqrt at
//  compile time and at run time; the layout contract; and twiddle's exact quarter turns and accuracy against long
//  double.
//

#include <snicholls/core/signal/complex.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace {

namespace s = snicholls::signal;
namespace m = snicholls::math;
using cd = s::complex<double>;

// Layout: two T, real first, trivially copyable, usable as a template argument.
static_assert(sizeof(s::complex<float>) == 2 * sizeof(float) && alignof(s::complex<double>) == alignof(double));
static_assert(std::is_trivially_copyable_v<cd> && std::is_standard_layout_v<cd> && std::is_aggregate_v<cd>);
template <cd Z>
constexpr double real_of() { return Z.re; }
static_assert(real_of<cd{1.5, 2}>() == 1.5);

// Arithmetic at compile time.
static_assert(cd{1, 2} + cd{3, -1} == cd{4, 1} && cd{1, 2} - cd{3, -1} == cd{-2, 3});
static_assert(cd{1, 2} * cd{3, -1} == cd{5, 5} && cd{1, 2} * 2.0 == cd{2, 4} && 2.0 * cd{1, 2} == cd{2, 4});
static_assert(cd{5, 5} / cd{3, -1} == cd{1, 2} && cd{2, 4} / 2.0 == cd{1, 2} && -cd{1, -2} == cd{-1, 2});
static_assert(s::conj(cd{1, 2}) == cd{1, -2} && s::times_i(cd{1, 2}) == cd{-2, 1} && s::norm(cd{3, 4}) == 25.0);
static_assert(m::near(s::abs(cd{3, 4}), 5.0, 1e-15) && m::near(s::arg(cd{0, 1}), m::pi / 2, 1e-15));
static_assert(m::near(s::polar(2.0, m::pi / 2).im, 2.0, 1e-15) && m::near(s::expi(m::pi).re, -1.0, 1e-15));
static_assert(m::near(s::sqrt(cd{-4, 0}).im, 2.0, 1e-15) && s::sqrt(cd{0, 0}) == cd{0, 0});
static_assert(m::near(s::sqrt(cd{3, -4}).re, 2.0, 1e-15) && m::near(s::sqrt(cd{3, -4}).im, -1.0, 1e-15));
static_assert(m::near(s::sqrt(cd{-3, -4}).re, 1.0, 1e-15) && m::near(s::sqrt(cd{-3, -4}).im, -2.0, 1e-15));
constexpr cd compound()
{
    cd z{1, 1};
    z += cd{1, 0};
    z -= cd{0, 2};
    z *= cd{0, 1};
    z *= 2.0;
    z /= 4.0;
    return z;
}
static_assert(compound() == cd{0.5, 1});

// Twiddles: exact at the quarter turns, symmetric, at compile time.
static_assert(s::twiddle<double>(0, 8) == cd{1, 0} && s::twiddle<double>(2, 8) == cd{0, -1} && s::twiddle<double>(4, 8) == cd{-1, 0});
static_assert(s::twiddle<double>(6, 8) == cd{0, 1} && s::twiddle<double>(11, 8) == s::twiddle<double>(3, 8));
static_assert(m::near(s::twiddle<double>(1, 8).re, m::math_limits<double>::inv_sqrt2, 1e-16) && s::twiddle<float>(3, 12) == s::complex<float>{0, -1});
static_assert(noexcept(s::twiddle<double>(1, 3)) && noexcept(cd{} * cd{}));

TEST(Complex, RunTimeArithmeticIsTheSame)
{
    volatile double a = 1.25, b = -0.5;
    const cd x{a, b}, y{b, a};
    EXPECT_EQ(x * y, (cd{a * b - b * a, a * a + b * b}));
    EXPECT_DOUBLE_EQ(s::abs(x), std::hypot(1.25, -0.5));
    EXPECT_DOUBLE_EQ(s::arg(x), std::atan2(-0.5, 1.25));
    const cd r = s::sqrt(x);
    EXPECT_NEAR((r * r).re, x.re, 1e-15);
    EXPECT_NEAR((r * r).im, x.im, 1e-15);
    const cd q = x / y;
    EXPECT_NEAR((q * y).re, x.re, 1e-15);
    EXPECT_NEAR((q * y).im, x.im, 1e-15);
}

TEST(Twiddle, IsTheRootOfUnityToTheLastBit)
{
    EXPECT_EQ(s::twiddle<double>(250, 1000), (cd{0, -1}));
    EXPECT_EQ(s::twiddle<long double>(3, 4), (s::complex<long double>{0, 1}));
    if (std::numeric_limits<long double>::digits <= std::numeric_limits<double>::digits)
        GTEST_SKIP() << "long double is double here (Apple ARM64, MSVC): no wider reference to check the last bits against";
    // Against long double's cos and sin of 2 pi k / n, for every k of several n: within one ulp of double.
    for (const std::size_t n : {std::size_t{3}, std::size_t{7}, std::size_t{64}, std::size_t{1000}, std::size_t{4099}})
        for (std::size_t k = 0; k < n; ++k) {
            const cd w = s::twiddle<double>(k, n);
            const long double a = 2 * 3.14159265358979323846264338327950288L * static_cast<long double>(k) / static_cast<long double>(n);
            EXPECT_NEAR(w.re, static_cast<double>(std::cos(a)), 2.3e-16) << k << "/" << n;
            EXPECT_NEAR(w.im, static_cast<double>(-std::sin(a)), 2.3e-16) << k << "/" << n;
            EXPECT_EQ(s::twiddle<double>(n - k, n).re, k == 0 ? 1.0 : w.re);   // conjugate symmetry, exactly
        }
}

} // namespace
