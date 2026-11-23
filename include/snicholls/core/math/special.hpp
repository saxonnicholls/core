//
//  special.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::math`: special functions for optimisation and statistics, `constexpr` and stable in the tails: Lambert W (both real branches), softplus, the logistic function, log Phi, and the entropy and Huber atoms.
/// \ingroup core
///
/// **Overview.** These are the functions a convex model, a likelihood or a sizing rule is written in, each
/// computed so that no intermediate overflows or cancels where the result itself is representable:
///
/// - `lambert_w0`, `lambert_wm1`: the two real branches of W, the inverse of w e^w (Corless, Gonnet,
///   Hare, Jeffrey and Knuth, *On the Lambert W function*, 1996). W solves x e^x = c, and with it Kelly
///   sizing, the Wright omega, queue and growth equations in closed form.
/// - `softplus` = log(1 + e^x), `logistic` = 1 / (1 + e^-x), `log_logistic` = log logistic(x): the
///   smooth hinge and the sigmoid of a classifier, with neither tail overflowing.
/// - `log_normcdf` = log Phi(x), accurate to the last digits for every x, where log(normal_cdf(x))
///   underflows to -inf below about -38: the probit log-likelihood, and SciPy's `log_ndtr`.
/// - `entr`, `rel_entr`, `kl_div`, `huber`, `xlogy`: the atoms of CVXPY (and SciPy), with their
///   conventions at 0 and infinity.
///
/// Every function is a template on its argument's type, as in `math/constexpr_math.hpp`: `float`,
/// `double`, `long double`, or a type of the caller's with `math_limits`; an integer computes in `double`.
/// The work is done in `compute_type` (float in double, rounded once), with `constexpr_math`'s functions,
/// so a constant evaluation uses its kernels and a run-time call `<cmath>`.
///
/// \snippet math/special_demo.cpp special
///
/// **Relationships.** Built on `math/constexpr_math.hpp` (`exp`, `log`, `log1p`, `sqrt`, `normal_cdf`).
/// `math/roots.hpp` is the general solver when no closed form exists. Group: \ref core.
///
/// **Guarantees.** Every function is `constexpr` and does not throw. NaN propagates. A domain error is
/// NaN: `lambert_w0` below -1/e, `lambert_wm1` outside [-1/e, 0), `huber` with M < 0.
///
/// **Errors.** None: domain errors are NaN, poles are infinities, as in `<cmath>`.
///
/// **Performance.** Not benchmarked. `lambert_w0` and `lambert_wm1` cost a seed and two to four Halley
/// steps, each an `exp` or a `log`; the others an `exp`, a `log` or both.
///
/// **Tests and demos.** `tests/math/special_test.cpp`, `demos/math/special_demo.cpp`.
///
/// **Dependencies.** `<limits>`, `<type_traits>`, and `math/constexpr_math.hpp`.
///
/// **Known issues.** Near the branch point -1/e, W(z) changes by the square root of a change in z, so a
/// z rounded to the nearest double carries an error of about sqrt(eps) into W: inherent, not the method's.
/// `lambert_w0` above 1e300 (a double's; the type's largest value / 1e8 otherwise) returns the
/// asymptotic series, about 1e-9 relative, because w e^w would overflow inside the iteration.
///
/// **Status.** Experimental.

#ifndef snicholls_core_math_special_hpp
#define snicholls_core_math_special_hpp

#include <snicholls/core/math/constexpr_math.hpp>

#include <limits>
#include <type_traits>

namespace snicholls::math
{

namespace detail
{
template <typename T>
inline constexpr T e_t = T(2.71828182845904523536028747135266250L);
template <typename T>
inline constexpr T inv_e_t = T(0.367879441171442321595523770161460867L);
template <typename T>
inline constexpr T nan_t = std::numeric_limits<T>::quiet_NaN();
template <typename T>
inline constexpr T inf_t = std::numeric_limits<T>::infinity();

// The tolerance at -1/e: a rounded -1/e may fall an ulp below it. 1e-12 in double and wider; a float's epsilon for a
// float argument, whose nearest value to -1/e lies 9e-9 below it.
template <typename X>
constexpr double branch_fuzz_() { return std::numeric_limits<X>::epsilon() > 1e-12 ? double(std::numeric_limits<X>::epsilon()) : 1e-12; }

// W0 by Corless et al. (1996): a Puiseux seed at the branch point, a log-asymptotic seed for large z, Halley's iteration.
template <typename T>
constexpr T lambert_w0_c(T z, T fuzz)
{
    if (is_nan(z))
        return z;
    if (is_inf(z))
        return z > T(0) ? inf_t<T> : nan_t<T>;
    if (z <= -inv_e_t<T>) {
        // tolerate the one-ulp fuzz of a rounded -1/E (which is itself just below -1/e); refuse below
        if (z < -inv_e_t<T> - fuzz)
            return nan_t<T>;
        return T(-1);
    }
    if (z == T(0))
        return T(0);

    // ---- seed ---------------------------------------------------------------
    T w = T(0);
    const T p2 = T(2) * (e_t<T> * z + T(1));   // 2(1 + E z) >= 0 on the domain
    if (p2 < T(2.0e-2)) {
        // branch-point Puiseux series in p = sqrt(2(1 + E z))  (Corless et
        // al. 1996, eq. 4.22): W = -1 + p - p^2/3 + 11 p^3/72 - 43 p^4/540
        const T p = sqrt(p2);
        w = T(-1) + p * (T(1) + p * (T(-1) / T(3) + p * (T(11) / T(72) - p * T(43) / T(540))));
        if (p < T(1e-5))   // series already ~1e-25: Halley below would divide by ~0
            return w;
    } else if (z < T(3)) {
        w = z >= T(0) ? log(T(1) + z) : z * (T(1) - z);   // crude, Halley finishes
    } else {
        // asymptotic seed W ~ L1 - L2 + L2/L1 + L2(L2-2)/(2 L1^2)
        // (Corless et al. 1996, eq. 4.19)
        const T l1 = log(z);
        const T l2 = log(l1);
        w = l1 - l2 + l2 / l1 + l2 * (l2 - T(2)) / (T(2) * l1 * l1);
        constexpr T big = std::numeric_limits<T>::max_exponent10 > 308 ? std::numeric_limits<T>::max() / T(1e8) : T(1.0e300);
        if (z > big)
            return w;   // w e^w would overflow inside Halley; the asymptotic series is ~1e-9 relative here
    }

    // ---- Halley (cubic convergence; see Corless et al. 1996, sec. 5) --------
    for (int i = 0; i < 32; ++i) {
        const T ew = exp(w);
        const T f = w * ew - z;
        const T wp1 = w + T(1);
        const T denom = ew * wp1 - (w + T(2)) * f / (T(2) * wp1);
        const T step = f / denom;
        w -= step;
        if (abs(step) <= T(1e-16) * (abs(w) + T(1e-300)))
            break;
    }
    return w;
}

// W-1 on [-1/e, 0): the Puiseux series with p = -sqrt(2(1 + E z)) near the branch point, the log-asymptotic seed
// L1 - L2 + L2/L1 (L1 = log(-z), L2 = log(-L1)) towards 0, then Halley's iteration on g(w) = w + log(-w) - log(-z),
// which is w e^w = z in logarithms and stays finite where e^w underflows.
template <typename T>
constexpr T lambert_wm1_c(T z, T fuzz)
{
    if (is_nan(z))
        return z;
    if (z >= T(0))
        return z == T(0) ? -inf_t<T> : nan_t<T>;
    if (z <= -inv_e_t<T>)
        return z < -inv_e_t<T> - fuzz ? nan_t<T> : T(-1);

    T w = T(0);
    if (z < T(-0.25)) {
        const T p2 = T(2) * (e_t<T> * z + T(1));
        const T p = -sqrt(p2 > T(0) ? p2 : T(0));
        w = T(-1) + p * (T(1) + p * (T(-1) / T(3) + p * (T(11) / T(72) - p * T(43) / T(540))));
        if (p > T(-1e-5))
            return w;
    } else {
        const T l1 = log(-z);
        const T l2 = log(-l1);
        w = l1 - l2 + l2 / l1;
    }
    const T lz = log(-z);
    for (int i = 0; i < 32; ++i) {
        const T g = w + log(-w) - lz;
        const T g1 = T(1) + T(1) / w;     // g'
        const T g2 = T(-1) / (w * w);     // g''
        const T step = g / (g1 - g * g2 / (T(2) * g1));
        const T before = w;
        w -= step;
        if (!(w < T(-1)))                 // never cross onto the principal branch
            w = (before + T(-1)) / T(2);
        if (abs(step) <= T(1e-16) * abs(w))
            break;
    }
    return w;
}

template <typename T>
constexpr T softplus_c(T x)
{
    // Beyond digits * ln 2, e^-|x| is below half an ulp of the result: softplus is x above and e^x below.
    constexpr T cut = T(std::numeric_limits<T>::digits) * T(0.693147180559945309417232121458176568L);
    if (is_nan(x)) return x;
    if (x > cut) return x;
    if (x < -cut) return exp(x);
    return x > T(0) ? x + log1p(exp(-x)) : log1p(exp(x));
}

template <typename T>
constexpr T logistic_c(T x)
{
    if (x >= T(0))
        return T(1) / (T(1) + exp(-x));
    const T ex = exp(x);   // x < 0 or NaN
    return ex / (T(1) + ex);
}

// log Phi(x) as SciPy's log_ndtr: log1p(-Phi(-x)) for x >= 0, log(Phi(x)) down to -20, and below it the asymptotic
// series of Abramowitz and Stegun 26.2.12 for the Mills ratio, Phi(x) = phi(x) / |x| * (1 + sum_n (-1)^n (2n-1)!! / x^2n),
// taken in logarithms. At x = -20 its terms fall by (2n-1) / 400 each, so a double needs about ten.
template <typename T>
constexpr T log_normcdf_c(T x)
{
    if (is_nan(x)) return x;
    if (x >= T(0)) return log1p(-normal_cdf(-x));
    if (x > T(-20)) return log(normal_cdf(x));
    if (is_inf(x)) return x;
    const T x2 = x * x;
    T term = T(1), sum = T(0);
    for (int n = 1; n < 100; ++n) {
        term *= -T(2 * n - 1) / x2;
        sum += term;
        if (abs(term) <= std::numeric_limits<T>::epsilon() / T(16))
            break;
    }
    return -x2 / T(2) - log(-x) - limits<T>::half_log_2pi + log1p(sum);
}

template <typename T>
constexpr T entr_c(T x)
{
    if (is_nan(x)) return x;
    if (x < T(0)) return -inf_t<T>;
    if (x == T(0)) return T(0);
    return -x * log(x);
}

template <typename T>
constexpr T rel_entr_c(T x, T y)
{
    if (is_nan(x) || is_nan(y)) return nan_t<T>;
    if (x > T(0) && y > T(0)) {
        // log(x / y) is accurate when x is near y; once x / y overflows or underflows, the difference of logarithms
        const T r = x / y;
        if (!is_inf(r) && r >= std::numeric_limits<T>::min())
            return x * log(r);
        return x * (log(x) - log(y));
    }
    if (x == T(0) && y >= T(0)) return T(0);
    return inf_t<T>;
}

template <typename T>
constexpr T kl_div_c(T x, T y)
{
    if (is_nan(x) || is_nan(y)) return nan_t<T>;
    if (x > T(0) && y > T(0)) {
        if (is_inf(x) || is_inf(y))
            return is_inf(x) && is_inf(y) ? nan_t<T> : inf_t<T>;   // x log(x/y) - x + y grows without bound
        return rel_entr_c(x, y) - x + y;
    }
    if (x == T(0) && y >= T(0)) return y;
    return inf_t<T>;
}

template <typename T>
constexpr T huber_c(T x, T m)
{
    if (is_nan(x) || is_nan(m) || m < T(0)) return nan_t<T>;
    const T a = abs(x);
    return a <= m ? x * x : T(2) * m * a - m * m;
}

template <typename T>
constexpr T xlogy_c(T x, T y)
{
    if (x == T(0) && !is_nan(y)) return T(0);
    return x * log(y);
}

template <typename X>
using special_t = compute_t<real_t<X>>;
} // namespace detail

/// \brief The principal branch W0 of the Lambert W function: the w >= -1 with w e^w = z.
///
/// A Puiseux series in p = sqrt(2(1 + e z)) seeds it near the branch point -1/e (Corless et al. 1996,
/// eq. 4.22), log(1 + z) or z(1 - z) for moderate z, and the asymptotic series L1 - L2 + L2/L1 + ... for
/// z >= 3 (eq. 4.19); Halley's iteration finishes (section 5).
/// \ingroup core
/// \tparam X An integer or a `math_number`.
/// \param z The argument, at least -1/e.
/// \return W0(z): NaN below -1/e (beyond a tolerance of 1e-12, or a float's epsilon, for a rounded -1/e), exactly -1 at the branch point, 0 at 0, +inf at +inf.
template <math_argument X>
constexpr real_t<X> lambert_w0(X z)
{
    using T = detail::special_t<X>;
    return static_cast<real_t<X>>(detail::lambert_w0_c(static_cast<T>(z), T(detail::branch_fuzz_<real_t<X>>())));
}

/// \brief The lower branch W-1 of the Lambert W function: the w <= -1 with w e^w = z, for z in [-1/e, 0).
///
/// Seeded by the Puiseux series in p = -sqrt(2(1 + e z)) for z < -1/4 and by L1 - L2 + L2/L1, with
/// L1 = log(-z) and L2 = log(-L1), towards 0 (Corless et al. 1996, eqs. 4.19 and 4.22); finished by
/// Halley's iteration on w + log(-w) = log(-z), which needs no e^w and so is exact down to the smallest z.
/// \ingroup core
/// \tparam X An integer or a `math_number`.
/// \param z The argument, in [-1/e, 0).
/// \return W-1(z): exactly -1 at the branch point, -inf at 0, NaN below -1/e (with `lambert_w0`'s tolerance) and above 0.
template <math_argument X>
constexpr real_t<X> lambert_wm1(X z)
{
    using T = detail::special_t<X>;
    return static_cast<real_t<X>>(detail::lambert_wm1_c(static_cast<T>(z), T(detail::branch_fuzz_<real_t<X>>())));
}

/// \brief softplus(x) = log(1 + e^x), the smooth hinge: x for large x, e^x for very negative x, never overflowing.
/// \ingroup core
/// \tparam X An integer or a `math_number`.
/// \param x The argument.
/// \return log(1 + e^x), positive for every finite x; +inf at +inf, 0 at -inf.
template <math_argument X>
constexpr real_t<X> softplus(X x)
{
    return static_cast<real_t<X>>(detail::softplus_c(static_cast<detail::special_t<X>>(x)));
}

/// \brief The logistic function 1 / (1 + e^-x), computed without overflow in either tail.
/// \ingroup core
/// \tparam X An integer or a `math_number`.
/// \param x The argument.
/// \return A value in [0, 1]; e^x (to rounding) far left, 1 far right.
template <math_argument X>
constexpr real_t<X> logistic(X x)
{
    return static_cast<real_t<X>>(detail::logistic_c(static_cast<detail::special_t<X>>(x)));
}

/// \brief log(logistic(x)) = -softplus(-x), finite where logistic(x) underflows.
/// \ingroup core
/// \tparam X An integer or a `math_number`.
/// \param x The argument.
/// \return A value at most 0; x far left, -e^-x far right.
template <math_argument X>
constexpr real_t<X> log_logistic(X x)
{
    return static_cast<real_t<X>>(-detail::softplus_c(-static_cast<detail::special_t<X>>(x)));
}

/// \brief log Phi(x), the logarithm of the standard normal distribution function, accurate for every x.
///
/// As SciPy's `log_ndtr`: log1p(-Phi(-x)) for x >= 0, log(Phi(x)) through `erfc` for -20 < x < 0, and below
/// -20 the asymptotic expansion of the Mills ratio (Abramowitz and Stegun 26.2.12),
/// log Phi(x) = -x^2/2 - log(-x) - log(2 pi)/2 + log(1 + sum_n (-1)^n (2n-1)!! / x^(2n)), summed to the type's
/// precision.
/// \ingroup core
/// \tparam X An integer or a `math_number`.
/// \param x The argument.
/// \return log Phi(x): 0 at +inf, -inf at -inf, about -x^2/2 far left where Phi itself underflows.
template <math_argument X>
constexpr real_t<X> log_normcdf(X x)
{
    return static_cast<real_t<X>>(detail::log_normcdf_c(static_cast<detail::special_t<X>>(x)));
}

/// \brief The entropy atom -x log x, as CVXPY's `entr`.
/// \ingroup core
/// \tparam X An integer or a `math_number`.
/// \param x The argument.
/// \return -x log x for x > 0, 0 at 0, -inf for x < 0 and at +inf.
template <math_argument X>
constexpr real_t<X> entr(X x)
{
    return static_cast<real_t<X>>(detail::entr_c(static_cast<detail::special_t<X>>(x)));
}

/// \brief The relative entropy atom x log(x / y), as CVXPY's and SciPy's `rel_entr`.
/// \ingroup core
/// \tparam X An integer or a `math_number`.
/// \tparam Y An integer or a `math_number`.
/// \param x The first argument.
/// \param y The second argument.
/// \return x log(x / y) for x, y > 0 (without overflow when x / y does); 0 when x = 0 and y >= 0; +inf otherwise; NaN if either is NaN.
template <math_argument X, math_argument Y>
constexpr real_t<std::common_type_t<X, Y>> rel_entr(X x, Y y)
{
    using R = real_t<std::common_type_t<X, Y>>;
    using T = detail::special_t<R>;
    return static_cast<R>(detail::rel_entr_c(static_cast<T>(x), static_cast<T>(y)));
}

/// \brief The Kullback-Leibler atom x log(x / y) - x + y, as CVXPY's and SciPy's `kl_div`: at least 0.
/// \ingroup core
/// \tparam X An integer or a `math_number`.
/// \tparam Y An integer or a `math_number`.
/// \param x The first argument.
/// \param y The second argument.
/// \return x log(x / y) - x + y for x, y > 0 (+inf when exactly one is infinite); y when x = 0 and y >= 0; +inf otherwise; NaN if either is NaN.
template <math_argument X, math_argument Y>
constexpr real_t<std::common_type_t<X, Y>> kl_div(X x, Y y)
{
    using R = real_t<std::common_type_t<X, Y>>;
    using T = detail::special_t<R>;
    return static_cast<R>(detail::kl_div_c(static_cast<T>(x), static_cast<T>(y)));
}

/// \brief The Huber function as CVXPY defines it: x^2 for |x| <= M, 2 M |x| - M^2 beyond.
///
/// CVXPY's scaling, twice SciPy's `huber(M, x)`; continuous with a continuous derivative at |x| = M.
/// \ingroup core
/// \tparam X An integer or a `math_number`.
/// \tparam M An integer or a `math_number`.
/// \param x The argument.
/// \param m The threshold M, at least 0.
/// \return The Huber value; NaN when M < 0 or either argument is NaN.
template <math_argument X, math_argument M = X>
constexpr real_t<std::common_type_t<X, M>> huber(X x, M m = M(1))
{
    using R = real_t<std::common_type_t<X, M>>;
    using T = detail::special_t<R>;
    return static_cast<R>(detail::huber_c(static_cast<T>(x), static_cast<T>(m)));
}

/// \brief x log y, with 0 log y = 0 for every y that is not NaN, as SciPy's `xlogy`.
/// \ingroup core
/// \tparam X An integer or a `math_number`.
/// \tparam Y An integer or a `math_number`.
/// \param x The first argument.
/// \param y The second argument.
/// \return x log y; 0 when x = 0 (even for y = 0 or y < 0), unless y is NaN.
template <math_argument X, math_argument Y>
constexpr real_t<std::common_type_t<X, Y>> xlogy(X x, Y y)
{
    using R = real_t<std::common_type_t<X, Y>>;
    using T = detail::special_t<R>;
    return static_cast<R>(detail::xlogy_c(static_cast<T>(x), static_cast<T>(y)));
}

} // namespace snicholls::math

#endif /* snicholls_core_math_special_hpp */
