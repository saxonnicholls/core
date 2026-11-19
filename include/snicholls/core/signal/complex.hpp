//
//  complex.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::signal::complex<T>`, a trivially copyable complex number for the transforms, usable in constant evaluation, and `twiddle<T>(k, n)`, the root of unity `e^(-2 pi i k / n)` to the last bit the type allows.
/// \ingroup core
///
/// **Overview.** The transforms of `signal/` need a complex number that works in a `static_assert`, is plain data
/// on a GPU or an FPGA, and costs nothing at run time. `std::complex` is `constexpr` in C++20 only in part (no
/// `abs`, no `polar`), its multiplication recovers from infinities with calls that are not `constexpr` in every
/// standard library, and it cannot be specialised for a number type of the caller's. `complex<T>` is two `T`s, `re`
/// and `im`, with the field operations, `conj`, `norm` (`|z|^2`), `abs`, `arg`, `polar` and `expi`, all
/// `constexpr`, in the argument's type (`float`, `double`, `long double` or a `math::math_number` of the caller's).
///
/// `twiddle<T>(k, n)` is `e^(-2 pi i k / n)`. It reduces `k / n` to the first octant with integer arithmetic before
/// any trigonometry, so the quarter turns are exact (`twiddle(n / 4, n)` is `-i` exactly) and every value is the
/// correctly reduced sine and cosine of an angle below pi / 4, computed in `math_limits<T>::compute_type` (`double`
/// for `float`) and rounded once.
///
/// \snippet signal/complex_demo.cpp complex
///
/// **Relationships.** Every header of `signal/`: `signal/fft.hpp` builds its tables with `twiddle`. The
/// functions are `math/constexpr_math.hpp`'s. Group: \ref core.
///
/// **Implementations.** One. There is deliberately no conversion to `std::complex`, so this header does not pull in
/// `<complex>`: the two layouts are the same (an array of two `T`, real part first), as cuFFT's `cufftComplex` and
/// C's `_Complex` are, so data can be exchanged by copying.
///
/// **Guarantees.** Trivially copyable, standard layout, `sizeof(complex<T>) == 2 * sizeof(T)`, structural (usable as
/// a non-type template parameter). Every function is `constexpr` and `noexcept`. Multiplication is the textbook
/// formula, without C's recovery of infinities: an infinite operand may give NaN where Annex G gives an infinity.
///
/// **Errors.** None. Division by zero gives infinities or NaN, as the type's own division does.
///
/// **Performance.** Not benchmarked on its own; `signal/fft.hpp`'s benchmark runs on it. Each operation is the
/// arithmetic written out, inlined.
///
/// **Tests and demos.** `tests/signal/complex_test.cpp`, `demos/signal/complex_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<type_traits>`, `snicholls/core/math/constexpr_math.hpp`.
///
/// **Known issues.** `abs` is `sqrt(re^2 + im^2)` without rescaling, so it overflows when a part exceeds the square
/// root of the type's largest value (about 1e154 for a `double`).
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_signal_complex_hpp
#define snicholls_core_signal_complex_hpp

#include <snicholls/core/math/constexpr_math.hpp>

#include <cstddef>
#include <type_traits>

/// \namespace snicholls::signal
/// \brief Signal processing at compile time and at run time: transforms, windows, spectrograms, filters, resampling.
namespace snicholls::signal
{

/// \brief A complex number of two `T`s, real part first: the transforms' element type.
/// \ingroup core
/// \tparam T A `math::math_number`.
template <math::math_number T>
struct complex
{
    T re{};   ///< The real part.
    T im{};   ///< The imaginary part.

    /// \brief Adds `z`.
    /// \param z The addend.
    /// \return `*this`.
    constexpr complex& operator+=(const complex& z) noexcept { re += z.re; im += z.im; return *this; }
    /// \brief Subtracts `z`.
    /// \param z The subtrahend.
    /// \return `*this`.
    constexpr complex& operator-=(const complex& z) noexcept { re -= z.re; im -= z.im; return *this; }
    /// \brief Multiplies by `z`.
    /// \param z The factor.
    /// \return `*this`.
    constexpr complex& operator*=(const complex& z) noexcept
    {
        const T r = re * z.re - im * z.im;
        im = re * z.im + im * z.re;
        re = r;
        return *this;
    }
    /// \brief Multiplies by a real number.
    /// \param s The factor.
    /// \return `*this`.
    constexpr complex& operator*=(T s) noexcept { re *= s; im *= s; return *this; }
    /// \brief Divides by a real number.
    /// \param s The divisor.
    /// \return `*this`.
    constexpr complex& operator/=(T s) noexcept { re /= s; im /= s; return *this; }

    /// \brief The sum.
    /// \param a The first operand.
    /// \param b The second operand.
    /// \return `a + b`.
    friend constexpr complex operator+(complex a, const complex& b) noexcept { return a += b; }
    /// \brief The difference.
    /// \param a The first operand.
    /// \param b The second operand.
    /// \return `a - b`.
    friend constexpr complex operator-(complex a, const complex& b) noexcept { return a -= b; }
    /// \brief The product.
    /// \param a The first operand.
    /// \param b The second operand.
    /// \return `a b`.
    friend constexpr complex operator*(complex a, const complex& b) noexcept { return a *= b; }
    /// \brief The product with a real number.
    /// \param a The complex operand.
    /// \param s The real operand.
    /// \return `a s`.
    friend constexpr complex operator*(complex a, T s) noexcept { return a *= s; }
    /// \brief The product with a real number.
    /// \param s The real operand.
    /// \param a The complex operand.
    /// \return `s a`.
    friend constexpr complex operator*(T s, complex a) noexcept { return a *= s; }
    /// \brief The quotient by a real number.
    /// \param a The complex operand.
    /// \param s The divisor.
    /// \return `a / s`.
    friend constexpr complex operator/(complex a, T s) noexcept { return a /= s; }
    /// \brief The quotient, `a conj(b) / |b|^2`.
    /// \param a The dividend.
    /// \param b The divisor.
    /// \return `a / b`.
    friend constexpr complex operator/(const complex& a, const complex& b) noexcept
    {
        const T d = b.re * b.re + b.im * b.im;
        return {(a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d};
    }
    /// \brief The negation.
    /// \param a The operand.
    /// \return `-a`.
    friend constexpr complex operator-(const complex& a) noexcept { return {-a.re, -a.im}; }
    /// \brief Equality of both parts.
    /// \param a The first operand.
    /// \param b The second operand.
    /// \return Whether they are equal.
    friend constexpr bool operator==(const complex& a, const complex& b) noexcept = default;
};

/// \brief The complex conjugate.
/// \tparam T A `math::math_number`.
/// \param z The number.
/// \return `re - i im`.
template <math::math_number T>
constexpr complex<T> conj(const complex<T>& z) noexcept { return {z.re, -z.im}; }

/// \brief `z` times `i`: a quarter turn, exact.
/// \tparam T A `math::math_number`.
/// \param z The number.
/// \return `-im + i re`.
template <math::math_number T>
constexpr complex<T> times_i(const complex<T>& z) noexcept { return {-z.im, z.re}; }

/// \brief The squared magnitude, `re^2 + im^2` (`std::norm`'s meaning).
/// \tparam T A `math::math_number`.
/// \param z The number.
/// \return `|z|^2`.
template <math::math_number T>
constexpr T norm(const complex<T>& z) noexcept { return z.re * z.re + z.im * z.im; }

/// \brief The magnitude, `sqrt(re^2 + im^2)`.
/// \tparam T A `math::math_number`.
/// \param z The number.
/// \return `|z|`.
template <math::math_number T>
constexpr T abs(const complex<T>& z) noexcept { return math::sqrt(norm(z)); }

/// \brief The argument, `atan2(im, re)`, in (-pi, pi].
/// \tparam T A `math::math_number`.
/// \param z The number.
/// \return The angle.
template <math::math_number T>
constexpr T arg(const complex<T>& z) noexcept { return math::atan2(z.im, z.re); }

/// \brief `r e^(i theta)`.
/// \tparam T A `math::math_number`.
/// \param r The magnitude.
/// \param theta The angle.
/// \return The number.
template <math::math_number T>
constexpr complex<T> polar(T r, T theta) noexcept { return {r * math::cos(theta), r * math::sin(theta)}; }

/// \brief `e^(i theta)`.
/// \tparam T A `math::math_number`.
/// \param theta The angle.
/// \return `cos theta + i sin theta`.
template <math::math_number T>
constexpr complex<T> expi(T theta) noexcept { return {math::cos(theta), math::sin(theta)}; }

/// \brief The principal square root, with a non-negative real part.
/// \tparam T A `math::math_number`.
/// \param z The number.
/// \return `sqrt(z)`.
template <math::math_number T>
constexpr complex<T> sqrt(const complex<T>& z) noexcept
{
    const T r = abs(z);
    if (r == T(0))
        return {T(0), T(0)};
    const T a = math::sqrt((r + math::abs(z.re)) / T(2));
    if (z.re >= T(0))
        return {a, z.im / (T(2) * a)};
    return {math::abs(z.im) / (T(2) * a), z.im < T(0) ? -a : a};
}

/// \brief The type the tables are computed in for elements of type `T`: `math_limits<T>::compute_type` (`double` for `float`).
/// \ingroup core
/// \tparam T A `math::math_number`.
template <math::math_number T>
using compute_t = typename math::math_limits<T>::compute_type;

namespace detail::complex_
{
    // (cos, sin) of (pi / 2) r / n for 0 <= r <= n / 2: an angle of at most pi / 4.
    template <typename C>
    constexpr complex<C> first_octant(std::size_t r, std::size_t n) noexcept
    {
        const C phi = math::math_limits<C>::pi / C(2) * (C(static_cast<long long>(r)) / C(static_cast<long long>(n)));
        if (2 * r == n) {   // pi / 4: one value for both, so twiddle(n - k) is conj(twiddle(k)) exactly
            const C c = math::cos(phi);
            return {c, c};
        }
        return {math::cos(phi), math::sin(phi)};
    }

    // (cos, sin) of 2 pi k / n with k < n, the octant reduced in integers so the quarter turns are exact.
    template <typename C>
    constexpr complex<C> unit(std::size_t k, std::size_t n) noexcept
    {
        const std::size_t q = (4 * k) / n, r = 4 * k - q * n;   // 2 pi k / n = (pi / 2) (q + r / n)
        complex<C> c = 2 * r <= n ? first_octant<C>(r, n) : [&] {
            const complex<C> s = first_octant<C>(n - r, n);
            return complex<C>{s.im, s.re};
        }();
        for (std::size_t i = 0; i < q; ++i)
            c = {-c.im, c.re};
        return c;
    }
} // namespace detail::complex_

/// \brief The root of unity `e^(-2 pi i k / n)`, the forward transform's twiddle factor.
///
/// `k` is reduced modulo `n`, then to an angle of at most pi / 4 in integers, so the result is the type's correctly
/// rounded value of a sine and a cosine computed in `compute_t<T>`; quarter turns are exact.
/// \tparam T A `math::math_number`.
/// \param k The exponent.
/// \param n The order, at least 1.
/// \return `cos(2 pi k / n) - i sin(2 pi k / n)`.
template <math::math_number T>
constexpr complex<T> twiddle(std::size_t k, std::size_t n) noexcept
{
    const complex<compute_t<T>> c = detail::complex_::unit<compute_t<T>>(k % n, n);
    return {static_cast<T>(c.re), static_cast<T>(-c.im)};
}

} // namespace snicholls::signal

#endif // snicholls_core_signal_complex_hpp
