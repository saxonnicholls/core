//
//  constexpr_math.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::math`: exp, log, expm1, log1p, sqrt, pow, the trigonometric and hyperbolic functions, erf, gamma, and the normal and chi-square distributions, all `constexpr`, for any floating-point type.
/// \ingroup core
///
/// **Overview.** `<cmath>` is not `constexpr` in C++20, so a table built at compile time (a normal
/// quantile grid, a discount curve, a Sobol point mapped to a normal draw) cannot call it. These
/// functions can: during constant evaluation they use series and argument-reduction kernels, and at run
/// time they defer to the correctly rounded `<cmath>` (or, for a type of the caller's, to its own functions,
/// found by argument-dependent lookup), so run-time results are the library's to the bit and compile-time
/// ones agree with them to a few units in the last place.
///
/// Every function is a template on its argument's type: `float`, `double`, `long double`, or a type of the
/// caller's (a multiprecision or fixed-point number) for which `math_limits` is specialised. An integer
/// argument computes in `double`, as `<cmath>` does. What a kernel needs to know about a type (its overflow
/// bounds, how far a series runs, its constants) is in `math_limits<T>`, one specialisation per type, as
/// `std::numeric_limits` is; `compute_type` says which type the kernels run in (`float` runs in `double` and is
/// rounded once, so it is correctly rounded nearly always).
///
/// \snippet math/constexpr_math_demo.cpp math
///
/// **Relationships.** `stats/bootstrap.hpp` (the normal quantile for BCa intervals),
/// `stats/low_discrepancy.hpp` (Sobol points to normal draws), `finance/attribution.hpp`. Group: \ref core.
///
/// **Guarantees.** Every function is `constexpr`. NaN and infinities propagate as in `<cmath>`. `double` results
/// are bit-identical to those before the functions were generic.
///
/// **Errors.** None: a domain error is NaN, as in `<cmath>`.
///
/// **Performance.** At run time, `<cmath>`'s. `normal_quantile` costs a logarithm, a square root and an
/// `erfc`.
///
/// **Tests and demos.** `tests/math/constexpr_math_test.cpp`, `demos/math/constexpr_math_demo.cpp`.
///
/// **Dependencies.** `<bit>`, `<cmath>`, `<concepts>`, `<cstdint>`, `<limits>`, `<type_traits>`.
///
/// **Known issues.** At compile time `sin` and `cos` reduce arguments with three parts of pi/2, accurate
/// for |x| up to about 1e9; beyond that compile-time results drift. `lgamma`, `gamma_p` and `chi_square_cdf` use
/// Lanczos' coefficients and are good to about 15 digits in every type. `long double` is `double` on Apple ARM and
/// MSVC; its limits follow `std::numeric_limits`, so it is right either way. A type of the caller's needs
/// arithmetic, comparisons, conversion to and from `long long` and `double`, `std::numeric_limits`, and, for run
/// time, `exp`, `log`, ... in its own namespace.
///
/// **Status.** New in 0.5.0.

#ifndef snicholls_core_math_constexpr_math_hpp
#define snicholls_core_math_constexpr_math_hpp

#include <bit>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace snicholls::math
{

/// \brief pi.
inline constexpr double pi  = 3.14159265358979323846264338327950288;
/// \brief e.
inline constexpr double e_v = 2.71828182845904523536028747135266250;
/// \brief 2 / sqrt(pi).
inline constexpr double two_over_sqrt_pi = 1.12837916709551257390;
/// \brief 1 / sqrt(pi).
inline constexpr double inv_sqrt_pi = 5.64189583547756286948e-01;

/// \brief Positive infinity.
inline constexpr double inf = std::numeric_limits<double>::infinity();
/// \brief A quiet NaN.
inline constexpr double nan = std::numeric_limits<double>::quiet_NaN();

/// \brief What the kernels need to know about a number type: specialise it for a type of your own, as `std::numeric_limits` is.
///
/// A specialisation has `compute_type` (the type the kernels run in) and, when `compute_type` is the type itself,
/// the constants of `math_limits<double>` below, in that type. Core specialises `float` (computed in `double`),
/// `double` and `long double`.
/// \ingroup core
/// \tparam T The number type.
template <typename T>
struct math_limits;

/// \brief `double`'s limits: the constants the kernels were written with.
template <>
struct math_limits<double>
{
    /// \brief The type the kernels run in: `double`.
    using compute_type = double;
    static constexpr double pi = 3.14159265358979323846264338327950288;   ///< pi.
    static constexpr double two_over_sqrt_pi = 1.12837916709551257390;    ///< 2 / sqrt(pi).
    static constexpr double inv_sqrt_pi = 5.64189583547756286948e-01;     ///< 1 / sqrt(pi).
    static constexpr double sqrt2 = 1.41421356237309504880;               ///< sqrt(2).
    static constexpr double inv_sqrt2 = 0.70710678118654752440;           ///< 1 / sqrt(2).
    static constexpr double sqrt_2pi = 2.50662827463100050242;            ///< sqrt(2 pi).
    static constexpr double half_log_2pi = 0.91893853320467274178;        ///< log(2 pi) / 2.
    static constexpr double ln2_hi = 6.93147180369123816490e-01;          ///< ln 2's leading bits: k * ln2_hi is exact for |k| < 2^20.
    static constexpr double ln2_lo = 1.90821492927058770002e-10;          ///< ln 2 - ln2_hi.
    static constexpr double ln2_lo2 = 0.0;                                ///< ln 2 - ln2_hi - ln2_lo, for a type wider than double.
    static constexpr double inv_ln2 = 1.44269504088896338700e+00;         ///< 1 / ln 2.
    static constexpr double pio2_1 = 1.57079632673412561417e+00;          ///< pi/2 in three parts (Cody-Waite).
    static constexpr double pio2_2 = 6.07710050630396597660e-11;          ///< The second part.
    static constexpr double pio2_3 = 2.02226624871116645580e-21;          ///< The third part.
    static constexpr double pio2_4 = 0.0;                                 ///< A fourth part, for a type wider than 64 bits.
    static constexpr double inv_pio2 = 6.36619772367581382433e-01;        ///< 2 / pi.
    static constexpr double exp_overflow = 709.79;                        ///< exp is infinite above this.
    static constexpr double exp_underflow = -745.2;                       ///< exp is zero below this.
    static constexpr double whole_from = 9007199254740992.0;              ///< 2^digits: every value this large is a whole number.
    static constexpr double integral_below = 9.0e15;                      ///< The largest whole number pow raises by repeated squaring.
    static constexpr double series_tolerance = 1e-20;                     ///< A series stops when a term is this small relative to the sum.
    static constexpr double trig_tolerance = 1e-22;                       ///< The sine and cosine series stop below this term.
    static constexpr double gamma_tolerance = 1e-17;                      ///< The incomplete gamma series and fraction stop here.
    static constexpr double tiny = 1e-300;                                ///< Lentz's floor.
    static constexpr double tanh_one = 20.0;                              ///< tanh is 1 above this.
    static constexpr double erf_one = 6.5;                                ///< erf is 1 above this.
    static constexpr double erfc_zero = 27.0;                             ///< erfc is 0 above this.
    static constexpr int erfc_terms = 60;                                 ///< Terms of erfc's continued fraction, enough at x = 2.

    /// \brief A first guess at sqrt(x) from its bits, which Newton's steps then refine.
    /// \param x Positive and finite.
    /// \return The guess.
    static constexpr double sqrt_seed(double x) noexcept
    {
        return std::bit_cast<double>(0x1FF7A3BEA91D9B1BULL + (std::bit_cast<std::uint64_t>(x) >> 1));
    }
    static constexpr int sqrt_steps = 8;   ///< Newton's steps from the seed.
};

/// \brief `float`'s limits: the kernels run in `double` and the result is rounded once.
template <>
struct math_limits<float>
{
    /// \brief The type the kernels run in: `double`, the result rounded once.
    using compute_type = double;
};

namespace detail
{
    // A standard floating-point type's limits, from std::numeric_limits: what long double uses, whatever it is.
    // The splits of ln 2 and pi/2 are double's, whose parts together carry about 85 and 119 bits.
    template <typename T>
    struct ieee_limits
    {
        using compute_type = T;
        using L = std::numeric_limits<T>;
        // Long double literals, not std::numbers: libc++ gives those only double's precision.
        static constexpr T ln2_full = T(0.693147180559945309417232121458176568L);
        static constexpr T pi = T(3.14159265358979323846264338327950288L);
        static constexpr T inv_sqrt_pi = T(0.564189583547756286948079451560772586L);
        static constexpr T two_over_sqrt_pi = 2 * inv_sqrt_pi;
        static constexpr T sqrt2 = T(1.41421356237309504880168872420969808L);
        static constexpr T inv_sqrt2 = T(0.707106781186547524400844362104849039L);
        static constexpr T sqrt_2pi = T(2.50662827463100050241576528481104525L);
        static constexpr T half_log_2pi = T(0.918938533204672741780329736405617640L);
        static constexpr T ln2_hi = math_limits<double>::ln2_hi;
        static constexpr T ln2_lo = math_limits<double>::ln2_lo;
        static constexpr T ln2_lo2 = T(1.16122272293625324218337828526629425e-26L);   // ln 2 - ln2_hi - ln2_lo
        static constexpr T inv_ln2 = T(1.44269504088896340735992468100189214L);
        // Up to 64 bits, double's three Cody-Waite parts (about 100 bits of pi/2) suffice. Wider (binary128) takes pi/2
        // as four non-overlapping doubles, about 212 bits: each k * part is exact in 113 bits.
        static constexpr bool wide = L::digits > 64;
        static constexpr T pio2_1 = wide ? T(0x1.921fb54442d18p+0) : T(math_limits<double>::pio2_1);
        static constexpr T pio2_2 = wide ? T(0x1.1a62633145c07p-54) : T(math_limits<double>::pio2_2);
        static constexpr T pio2_3 = wide ? T(-0x1.f1976b7ed8fbcp-110) : T(math_limits<double>::pio2_3);
        static constexpr T pio2_4 = wide ? T(0x1.4cf98e804177dp-164) : T(0);
        static constexpr T inv_pio2 = T(0.636619772367581343075535053490057448L);
        static constexpr T exp_overflow = T(L::max_exponent) * ln2_full;
        static constexpr T exp_underflow = T(L::min_exponent - L::digits - 1) * ln2_full;
        static constexpr T whole_from = [] {
            T p = 1;
            for (int i = 0; i < L::digits; ++i)
                p *= 2;
            return p;
        }();
        static constexpr T integral_below = T(4.0e18);
        static constexpr T series_tolerance = L::epsilon() * T(1e-4);
        static constexpr T trig_tolerance = L::epsilon() * T(1e-6);
        static constexpr T gamma_tolerance = L::epsilon() / 20;
        static constexpr T tiny = L::min() / L::epsilon();
        static constexpr T tanh_one = T(L::digits) * ln2_full / 2 + 1;
        static constexpr T erf_one = L::digits <= 53 ? T(6.5) : (L::digits <= 64 ? T(6.7) : T(9.0));
        static constexpr T erfc_zero = L::digits <= 53 ? T(27) : T(107);
        static constexpr int erfc_terms = L::digits <= 53 ? 60 : (L::digits <= 64 ? 100 : 400);

        // Halve the exponent by powers of four, then (1 + m) / 2 is within 25% of sqrt(m) for m in [1, 4).
        static constexpr T sqrt_seed(T x) noexcept
        {
            T y = 1;
            constexpr T big = T(18446744073709551616.0L), half_big = T(4294967296.0L);   // 2^64, 2^32
            while (x >= big) { x /= big; y *= half_big; }
            while (x < 1 / big) { x *= big; y /= half_big; }
            while (x >= 4) { x /= 4; y *= 2; }
            while (x < 1) { x *= 4; y /= 2; }
            return y * (1 + x) / 2;
        }
        static constexpr int sqrt_steps = 8;
    };
} // namespace detail

/// \brief `long double`'s limits, from `std::numeric_limits`: right whether it is 53, 64 or 113 bits.
template <>
struct math_limits<long double> : detail::ieee_limits<long double>
{
};

/// \brief A number type the functions accept as it is: one with `math_limits`.
/// \ingroup core
template <typename T>
concept math_number = requires { typename math_limits<std::remove_cvref_t<T>>::compute_type; };

/// \brief The type a function computes in for an argument of type `T`: `double` for an integer, else `T`.
/// \ingroup core
template <typename T>
using real_t = std::conditional_t<std::is_integral_v<T>, double, T>;

/// \brief An argument the functions accept: an integer, or a `math_number`.
/// \ingroup core
template <typename T>
concept math_argument = std::is_integral_v<T> || math_number<T>;

namespace detail
{
    template <typename T>
    using compute_t = typename math_limits<T>::compute_type;
    template <typename T>
    using limits = math_limits<compute_t<T>>;

    template <typename T>
    constexpr bool nan_(T x) { return x != x; }
    template <typename T>
    constexpr bool inf_(T x) { return x == std::numeric_limits<T>::infinity() || x == -std::numeric_limits<T>::infinity(); }
    template <typename T>
    constexpr T abs_(T x) { return x < 0 ? -x : x; }

    template <typename T>
    constexpr T trunc_(T x)
    {
        if (nan_(x) || inf_(x))
            return x;
        T a = abs_(x);
        if (a >= limits<T>::whole_from) // already integral
            return x;
        // Wider than long long: take off the whole multiple of 2^32 first, then the rest fits
        const T high = a >= T(4611686018427387904.0L) ? T(static_cast<long long>(a / T(4294967296.0L))) * T(4294967296.0L) : T(0);
        T r = high + static_cast<T>(static_cast<long long>(a - high));
        return x < 0 ? -r : r;
    }

    template <typename T>
    constexpr T round_(T x)
    {
        return x < 0 ? trunc_(x - T(0.5)) : trunc_(x + T(0.5));
    }

    template <typename T>
    constexpr bool integral_(T x)
    {
        return !nan_(x) && !inf_(x) && x == trunc_(x) && abs_(x) < limits<T>::integral_below;
    }

// x * 2^k without <cmath>.  k is at most ~2100 in magnitude for finite results.
template <typename T>
constexpr T ldexp2(T x, long long k)
{
    while (k > 0)
    {
        x *= T(2.0);
        --k;
        if (inf_(x))
            return x;
    }
    while (k < 0)
    {
        x *= T(0.5);
        ++k;
        if (x == T(0.0))
            return x;
    }
    return x;
}

template <typename T>
constexpr T exp_ct(T x)
{
    using L = limits<T>;
    if (nan_(x)) return x;
    if (x > L::exp_overflow) return std::numeric_limits<T>::infinity();
    if (x < L::exp_underflow) return T(0.0);

    const long long k = static_cast<long long>(round_(x * L::inv_ln2));
    const T kd = static_cast<T>(k);
    const T r = ((x - kd * L::ln2_hi) - kd * L::ln2_lo) - kd * L::ln2_lo2; // |r| <= ~0.347

    T term = T(1.0);
    T sum = T(1.0);
    for (int n = 1; n < 40; ++n)
    {
        term *= r / T(n);
        sum += term;
        if (abs_(term) < L::series_tolerance * abs_(sum))
            break;
    }
    return ldexp2(sum, k);
}

template <typename T>
constexpr T log_ct(T x)
{
    using L = limits<T>;
    if (nan_(x)) return x;
    if (x < T(0.0)) return std::numeric_limits<T>::quiet_NaN();
    if (x == T(0.0)) return -std::numeric_limits<T>::infinity();
    if (inf_(x)) return x;

    // Normalize x = m * 2^e with m in [1, 2), then center on 1.
    T m = x;
    long long e = 0;
    while (m >= T(2.0)) { m *= T(0.5); ++e; }
    while (m < T(1.0))  { m *= T(2.0); --e; }
    if (m > L::sqrt2)
    {
        m *= T(0.5);
        ++e;
    }

    // ln m = 2 atanh((m-1)/(m+1))
    const T t = (m - T(1.0)) / (m + T(1.0));
    const T t2 = t * t;
    T term = t;
    T sum = t;
    for (int n = 3; n < 60; n += 2)
    {
        term *= t2;
        const T contrib = term / T(n);
        sum += contrib;
        if (abs_(contrib) < L::series_tolerance * abs_(sum))
            break;
    }
    const T lnm = T(2.0) * sum;
    const T ed = static_cast<T>(e);
    return ed * L::ln2_hi + ((lnm + ed * L::ln2_lo) + ed * L::ln2_lo2);
}

template <typename T>
constexpr T sin_series(T r)
{
    const T r2 = r * r;
    T term = r;
    T sum = r;
    for (int n = 1; n < 20; ++n)
    {
        term *= -r2 / ((T(2.0) * T(n)) * (T(2.0) * T(n) + T(1.0)));
        sum += term;
        if (abs_(term) < limits<T>::trig_tolerance)
            break;
    }
    return sum;
}

template <typename T>
constexpr T cos_series(T r)
{
    const T r2 = r * r;
    T term = T(1.0);
    T sum = T(1.0);
    for (int n = 1; n < 20; ++n)
    {
        term *= -r2 / ((T(2.0) * T(n) - T(1.0)) * (T(2.0) * T(n)));
        sum += term;
        if (abs_(term) < limits<T>::trig_tolerance)
            break;
    }
    return sum;
}

// Reduce x to r with x = k*(pi/2) + r, |r| <= pi/4; writes quadrant k mod 4.
template <typename T>
struct reduced { T r; int quadrant; };

template <typename T>
constexpr reduced<T> reduce_pio2(T x)
{
    using L = limits<T>;
    const T kd = round_(x * L::inv_pio2);
    const long long k = static_cast<long long>(kd);
    const T r = (((x - kd * L::pio2_1) - kd * L::pio2_2) - kd * L::pio2_3) - kd * L::pio2_4;
    int q = static_cast<int>(k % 4);
    if (q < 0)
        q += 4;
    return {r, q};
}

template <typename T>
constexpr T sin_ct(T x)
{
    if (nan_(x) || inf_(x)) return std::numeric_limits<T>::quiet_NaN();
    const auto [r, q] = reduce_pio2(x);
    switch (q)
    {
        case 0: return sin_series(r);
        case 1: return cos_series(r);
        case 2: return -sin_series(r);
        default: return -cos_series(r);
    }
}

template <typename T>
constexpr T cos_ct(T x)
{
    if (nan_(x) || inf_(x)) return std::numeric_limits<T>::quiet_NaN();
    const auto [r, q] = reduce_pio2(x);
    switch (q)
    {
        case 0: return cos_series(r);
        case 1: return -sin_series(r);
        case 2: return -cos_series(r);
        default: return sin_series(r);
    }
}

template <typename T>
constexpr T sqrt_ct(T x)
{
    if (nan_(x)) return x;
    if (x < T(0.0)) return std::numeric_limits<T>::quiet_NaN();
    if (x == T(0.0) || inf_(x)) return x;

    // An estimate from the type's limits, then Newton to full precision.
    T y = limits<T>::sqrt_seed(x);
    for (int i = 0; i < limits<T>::sqrt_steps; ++i)
        y = T(0.5) * (y + x / y);
    return y;
}

// pow_int is TOTAL: it never divides by zero.  `1.0 / pow_int(0.0, n)` is a
// division by zero, which is UB and therefore a hard error in a constant
// expression.  The
// result of 0^(-n) is a pole, so the honest kernel answer is +inf, which is
// also exactly what the run-time std::pow returns. Keeping the compile-time
// and run-time paths in agreement is this header's contract.
template <typename T>
constexpr T pow_int(T a, long long n)
{
    if (n < 0)
    {
        const T d = pow_int(a, -n);
        return d == T(0.0) ? std::numeric_limits<T>::infinity() : T(1.0) / d;
    }
    T result = T(1.0);
    T base = a;
    while (n > 0)
    {
        if (n & 1)
            result *= base;
        base *= base;
        n >>= 1;
    }
    return result;
}

template <typename T>
constexpr T pow_ct(T a, T b)
{
    if (b == T(0.0)) return T(1.0);
    if (a == T(1.0)) return T(1.0);
    if (nan_(a) || nan_(b)) return std::numeric_limits<T>::quiet_NaN();
    if (integral_(b))
        return pow_int(a, static_cast<long long>(b));
    if (a < T(0.0)) return std::numeric_limits<T>::quiet_NaN();             // non-integral power of a negative base
    if (a == T(0.0)) return b > T(0.0) ? T(0.0) : std::numeric_limits<T>::infinity();
    return exp_ct(b * log_ct(a));
}

template <typename T>
constexpr T sinh_ct(T x)
{
    if (nan_(x)) return x;
    return T(0.5) * (exp_ct(x) - exp_ct(-x));
}

template <typename T>
constexpr T cosh_ct(T x)
{
    if (nan_(x)) return x;
    return T(0.5) * (exp_ct(x) + exp_ct(-x));
}

template <typename T>
constexpr T tanh_ct(T x)
{
    if (nan_(x)) return x;
    if (x < T(0.0)) return -tanh_ct(-x);
    if (x > limits<T>::tanh_one) return T(1.0);
    const T e2 = exp_ct(T(-2.0) * x);
    return (T(1.0) - e2) / (T(1.0) + e2);
}

// e^x - 1 without the cancellation near 0: the series below |x| = 1, exp - 1 above. A negative x goes
// through m = e^-x - 1, whose series has no alternating terms: e^x - 1 = -m / (1 + m).
template <typename T>
constexpr T expm1_ct(T x)
{
    if (nan_(x)) return x;
    if (abs_(x) >= T(1.0)) return exp_ct(x) - T(1.0);
    if (x < T(0.0))
    {
        const T m = expm1_ct(-x);
        return -m / (T(1.0) + m);
    }
    T term = x;
    T sum = x;
    for (int n = 2; n < 60; ++n)
    {
        term *= x / T(n);
        sum += term;
        if (abs_(term) < limits<T>::series_tolerance * abs_(sum))
            break;
    }
    return sum;
}

// log(1 + x) without the cancellation near 0: log(u) * x / (u - 1), with u = 1 + x rounded (Goldberg's
// correction), which cancels the rounding error of u.
template <typename T>
constexpr T log1p_ct(T x)
{
    if (nan_(x)) return x;
    if (x < T(-1.0)) return std::numeric_limits<T>::quiet_NaN();
    if (x == T(-1.0)) return -std::numeric_limits<T>::infinity();
    if (inf_(x)) return x;
    const T u = T(1.0) + x;
    if (u == T(1.0)) return x;
    return log_ct(u) * (x / (u - T(1.0)));
}

// atan via two argument-halvings atan(y) = 2 atan(y/(1+sqrt(1+y^2))) followed
// by the Maclaurin series, with the range reduction atan(x)=pi/2-atan(1/x).
template <typename T>
constexpr T atan_ct(T x)
{
    using L = limits<T>;
    if (nan_(x)) return x;
    if (x < T(0.0)) return -atan_ct(-x);
    if (inf_(x)) return L::pi / T(2.0);
    if (x > T(1.0)) return L::pi / T(2.0) - atan_ct(T(1.0) / x);

    T y = x;
    for (int i = 0; i < 2; ++i)
        y = y / (T(1.0) + sqrt_ct(T(1.0) + y * y));
    const T y2 = y * y;
    T term = y, sum = y;
    for (int n = 1; n < 60; ++n)
    {
        term *= -y2;
        const T contrib = term / (T(2.0) * T(n) + T(1.0));
        sum += contrib;
        if (abs_(contrib) < L::series_tolerance * abs_(sum))
            break;
    }
    return T(4.0) * sum; // 2^2 from the two halvings
}

template <typename T>
constexpr T asin_ct(T x)
{
    using L = limits<T>;
    if (nan_(x) || x < T(-1.0) || x > T(1.0)) return std::numeric_limits<T>::quiet_NaN();
    if (x == T(1.0)) return L::pi / T(2.0);
    if (x == T(-1.0)) return -L::pi / T(2.0);
    return atan_ct(x / sqrt_ct(T(1.0) - x * x));
}

template <typename T>
constexpr T acos_ct(T x)
{
    if (nan_(x) || x < T(-1.0) || x > T(1.0)) return std::numeric_limits<T>::quiet_NaN();
    return limits<T>::pi / T(2.0) - asin_ct(x);
}

// erf: Maclaurin series on |x| <= 2 (alternating, mild cancellation there),
// Laplace continued fraction for erfc beyond, evaluated backward.
template <typename T>
constexpr T erf_ct(T x)
{
    using L = limits<T>;
    if (nan_(x)) return x;
    if (x < T(0.0)) return -erf_ct(-x);
    if (x > L::erf_one) return T(1.0); // erfc below half an ulp of 1

    if (x <= T(2.0))
    {
        // 2/sqrt(pi) * sum (-1)^n x^(2n+1) / (n! (2n+1))
        const T x2 = x * x;
        T term = x; // (-1)^n x^(2n+1) / n!
        T sum = x;
        for (int n = 1; n < 80; ++n)
        {
            term *= -x2 / T(n);
            const T contrib = term / (T(2.0) * T(n) + T(1.0));
            sum += contrib;
            if (abs_(contrib) < L::series_tolerance * abs_(sum))
                break;
        }
        return L::two_over_sqrt_pi * sum;
    }

    // erfc(x) = e^{-x^2}/sqrt(pi) * 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + ...))))
    T k = x;
    for (int n = L::erfc_terms; n >= 1; --n)
        k = x + (T(n) / T(2.0)) / k;
    const T erfc = exp_ct(-x * x) * L::inv_sqrt_pi / k;
    return T(1.0) - erfc;
}

// erfc without cancellation in the tail: the continued fraction beyond 2, 1 - erf below, 2 - erfc(-x) for x < 0.
template <typename T>
constexpr T erfc_ct(T x)
{
    using L = limits<T>;
    if (nan_(x)) return x;
    if (x < T(0.0)) return T(2.0) - erfc_ct(-x);
    if (x <= T(2.0)) return T(1.0) - erf_ct(x);
    if (x > L::erfc_zero) return T(0.0);
    T k = x;
    for (int n = L::erfc_terms; n >= 1; --n)
        k = x + (T(n) / T(2.0)) / k;
    return exp_ct(-x * x) * L::inv_sqrt_pi / k;
}

// The kernel in the compute type during constant evaluation, the library's function at run time.
template <typename T, typename Kernel, typename RunTime>
constexpr T dispatch(T x, Kernel kernel, RunTime run_time)
{
    if (std::is_constant_evaluated())
        return static_cast<T>(kernel(static_cast<compute_t<T>>(x)));
    return run_time(x);
}

} // namespace detail

/// \brief Whether `x` is NaN.
/// \tparam X An integer or a `math_number`.
/// \param x The argument.
/// \return The result.
template <math_argument X>
constexpr bool is_nan(X x) { return detail::nan_(static_cast<real_t<X>>(x)); }
/// \brief Whether `x` is infinite.
/// \tparam X An integer or a `math_number`.
/// \param x The argument.
/// \return The result.
template <math_argument X>
constexpr bool is_inf(X x) { return detail::inf_(static_cast<real_t<X>>(x)); }
/// \brief `|x|`.
/// \tparam X An integer or a `math_number`.
/// \param x The argument.
/// \return The result.
template <math_argument X>
constexpr real_t<X> abs(X x) { return detail::abs_(static_cast<real_t<X>>(x)); }
/// \brief `x` rounded towards zero.
/// \tparam X An integer or a `math_number`.
/// \param x The argument.
/// \return The result.
template <math_argument X>
constexpr real_t<X> trunc(X x) { return detail::trunc_(static_cast<real_t<X>>(x)); }
/// \brief `x` rounded down.
/// \tparam X An integer or a `math_number`.
/// \param x The argument.
/// \return The result.
template <math_argument X>
constexpr real_t<X> floor(X x)
{
    using T = real_t<X>;
    const T t = detail::trunc_(static_cast<T>(x));
    return static_cast<T>(x) < t ? t - T(1.0) : t;
}
/// \brief `x` rounded to nearest, halves away from zero.
/// \tparam X An integer or a `math_number`.
/// \param x The argument.
/// \return The result.
template <math_argument X>
constexpr real_t<X> round(X x) { return detail::round_(static_cast<real_t<X>>(x)); }
/// \brief Whether `x` is a whole number small enough for `pow` to raise by repeated squaring (below 9e15 for a double).
/// \tparam X An integer or a `math_number`.
/// \param x The argument.
/// \return The result.
template <math_argument X>
constexpr bool is_integral(X x) { return detail::integral_(static_cast<real_t<X>>(x)); }

// Public kernels: constexpr path during constant evaluation, the library's own at run time.

/// \brief `e^x`.
/// \tparam X An integer or a `math_number`.
/// \param x The argument.
/// \return The result.
template <math_argument X>
constexpr real_t<X> exp(X x)
{
    return detail::dispatch<real_t<X>>(x, [](auto v) { return detail::exp_ct(v); }, [](auto v) { using std::exp; return exp(v); });
}

/// \brief The natural logarithm.
/// \tparam X An integer or a `math_number`.
/// \param x The argument.
/// \return The result.
template <math_argument X>
constexpr real_t<X> log(X x)
{
    return detail::dispatch<real_t<X>>(x, [](auto v) { return detail::log_ct(v); }, [](auto v) { using std::log; return log(v); });
}

/// \brief The sine.
/// \tparam X An integer or a `math_number`.
/// \param x The argument.
/// \return The result.
template <math_argument X>
constexpr real_t<X> sin(X x)
{
    return detail::dispatch<real_t<X>>(x, [](auto v) { return detail::sin_ct(v); }, [](auto v) { using std::sin; return sin(v); });
}

/// \brief The cosine.
/// \tparam X An integer or a `math_number`.
/// \param x The argument.
/// \return The result.
template <math_argument X>
constexpr real_t<X> cos(X x)
{
    return detail::dispatch<real_t<X>>(x, [](auto v) { return detail::cos_ct(v); }, [](auto v) { using std::cos; return cos(v); });
}

/// \brief The tangent.
/// \tparam X An integer or a `math_number`.
/// \param x The argument.
/// \return The result.
template <math_argument X>
constexpr real_t<X> tan(X x) { return sin(x) / cos(x); }
/// \brief The cotangent.
/// \tparam X An integer or a `math_number`.
/// \param x The argument.
/// \return The result.
template <math_argument X>
constexpr real_t<X> cot(X x) { return cos(x) / sin(x); }

/// \brief The square root.
/// \tparam X An integer or a `math_number`.
/// \param x The argument.
/// \return The result.
template <math_argument X>
constexpr real_t<X> sqrt(X x)
{
    return detail::dispatch<real_t<X>>(x, [](auto v) { return detail::sqrt_ct(v); }, [](auto v) { using std::sqrt; return sqrt(v); });
}

/// \brief `a^b`, in the common type of `a` and `b` (`double` for integers).
/// \tparam A An integer or a `math_number`.
/// \tparam B An integer or a `math_number`.
/// \param a The base.
/// \param b The exponent.
/// \return The result.
template <math_argument A, math_argument B>
constexpr real_t<std::common_type_t<A, B>> pow(A a, B b)
{
    using T = real_t<std::common_type_t<A, B>>;
    if (std::is_constant_evaluated())
        return static_cast<T>(detail::pow_ct(static_cast<detail::compute_t<T>>(a), static_cast<detail::compute_t<T>>(b)));
    using std::pow;
    return pow(static_cast<T>(a), static_cast<T>(b));
}

/// \brief The error function.
/// \tparam X An integer or a `math_number`.
/// \param x The argument.
/// \return The result.
template <math_argument X>
constexpr real_t<X> erf(X x)
{
    return detail::dispatch<real_t<X>>(x, [](auto v) { return detail::erf_ct(v); }, [](auto v) { using std::erf; return erf(v); });
}

/// \brief The complementary error function, `1 - erf(x)`, accurate in the tail.
/// \tparam X An integer or a `math_number`.
/// \param x The argument.
/// \return The result.
template <math_argument X>
constexpr real_t<X> erfc(X x)
{
    return detail::dispatch<real_t<X>>(x, [](auto v) { return detail::erfc_ct(v); }, [](auto v) { using std::erfc; return erfc(v); });
}

/// \brief The hyperbolic sine.
/// \tparam X An integer or a `math_number`.
/// \param x The argument.
/// \return The result.
template <math_argument X>
constexpr real_t<X> sinh(X x)
{
    return detail::dispatch<real_t<X>>(x, [](auto v) { return detail::sinh_ct(v); }, [](auto v) { using std::sinh; return sinh(v); });
}

/// \brief The hyperbolic cosine.
/// \tparam X An integer or a `math_number`.
/// \param x The argument.
/// \return The result.
template <math_argument X>
constexpr real_t<X> cosh(X x)
{
    return detail::dispatch<real_t<X>>(x, [](auto v) { return detail::cosh_ct(v); }, [](auto v) { using std::cosh; return cosh(v); });
}

/// \brief The hyperbolic tangent.
/// \tparam X An integer or a `math_number`.
/// \param x The argument.
/// \return The result.
template <math_argument X>
constexpr real_t<X> tanh(X x)
{
    return detail::dispatch<real_t<X>>(x, [](auto v) { return detail::tanh_ct(v); }, [](auto v) { using std::tanh; return tanh(v); });
}

/// \brief `e^x - 1`, accurate for `x` near 0, where `exp(x) - 1` cancels.
/// \tparam X An integer or a `math_number`.
/// \param x The argument.
/// \return The result.
template <math_argument X>
constexpr real_t<X> expm1(X x)
{
    return detail::dispatch<real_t<X>>(x, [](auto v) { return detail::expm1_ct(v); }, [](auto v) { using std::expm1; return expm1(v); });
}

/// \brief `log(1 + x)`, accurate for `x` near 0, where `log(1 + x)` loses the digits of `x`.
/// \tparam X An integer or a `math_number`.
/// \param x The argument: NaN below -1, -infinity at -1.
/// \return The result.
template <math_argument X>
constexpr real_t<X> log1p(X x)
{
    return detail::dispatch<real_t<X>>(x, [](auto v) { return detail::log1p_ct(v); }, [](auto v) { using std::log1p; return log1p(v); });
}

/// \brief The arc tangent.
/// \tparam X An integer or a `math_number`.
/// \param x The argument.
/// \return The result.
template <math_argument X>
constexpr real_t<X> atan(X x)
{
    return detail::dispatch<real_t<X>>(x, [](auto v) { return detail::atan_ct(v); }, [](auto v) { using std::atan; return atan(v); });
}

// The QUADRANT-AWARE arc tangent, in C's argument order: atan2(y, x) is the
// angle of the point (x, y), in (-pi, pi].
//
// Note the order, because three conventions collide here and getting it
// backwards is a silent wrong answer rather than an error:
//
//     C / C++ / SymPy / numpy    atan2(y, x)      y FIRST
//     Mathematica               ArcTan[x, y]      x FIRST
//
// This kernel follows C, because it is a C++ function called atan2 and any
// other choice would ambush every caller. The SYMBOLIC layer in functions.hpp
// stores the node in Mathematica's order so that `wolfram()` prints correctly,
// and provides both spellings with the relationship pinned by a test.
/// \brief The angle of the point (x, y), in (-pi, pi]: C's argument order, y first.
/// \tparam Y An integer or a `math_number`.
/// \tparam X An integer or a `math_number`.
/// \param y The ordinate.
/// \param x The abscissa.
/// \return The result.
template <math_argument Y, math_argument X>
constexpr real_t<std::common_type_t<Y, X>> atan2(Y y, X x)
{
    using T = real_t<std::common_type_t<Y, X>>;
    const T pi_t = static_cast<T>(detail::limits<T>::pi);
    const T yt = static_cast<T>(y), xt = static_cast<T>(x);
    if (xt > T(0.0))
        return atan(yt / xt);
    if (xt < T(0.0))
        return yt >= T(0.0) ? atan(yt / xt) + pi_t : atan(yt / xt) - pi_t;
    if (yt > T(0.0))
        return pi_t / T(2.0);
    if (yt < T(0.0))
        return -pi_t / T(2.0);
    return T(0.0); // atan2(0, 0): 0 by convention, and the one value with no angle
}

/// \brief The arc sine.
/// \tparam X An integer or a `math_number`.
/// \param x The argument.
/// \return The result.
template <math_argument X>
constexpr real_t<X> asin(X x)
{
    return detail::dispatch<real_t<X>>(x, [](auto v) { return detail::asin_ct(v); }, [](auto v) { using std::asin; return asin(v); });
}

/// \brief The arc cosine.
/// \tparam X An integer or a `math_number`.
/// \param x The argument.
/// \return The result.
template <math_argument X>
constexpr real_t<X> acos(X x)
{
    return detail::dispatch<real_t<X>>(x, [](auto v) { return detail::acos_ct(v); }, [](auto v) { using std::acos; return acos(v); });
}

/// \brief Whether `a` and `b` agree to `tol`, relative to their size.
/// \tparam T A `math_number`; integers and mixed arguments are `double`.
/// \param a The first value.
/// \param b The second value.
/// \param tol The relative tolerance.
/// \return The result.
template <typename T = double>
    requires math_number<T>
constexpr bool near(std::type_identity_t<T> a, std::type_identity_t<T> b, std::type_identity_t<T> tol = T(1e-9))
{
    const T scale = T(1.0) + (abs(a) > abs(b) ? abs(a) : abs(b));
    return abs(a - b) <= tol * scale;
}

namespace detail
{
template <typename T>
constexpr T normal_pdf_c(T x)
{
    using L = detail::limits<T>;
    const T xt = static_cast<T>(x);
    return exp(T(-0.5) * xt * xt) * static_cast<T>(L::inv_sqrt_pi / L::sqrt2);
}
} // namespace detail

/// \brief The standard normal density.
/// \tparam X An integer or a `math_number`.
/// \param x The argument.
/// \return The result.
template <math_argument X>
constexpr real_t<X> normal_pdf(X x)
{
    return static_cast<real_t<X>>(detail::normal_pdf_c(static_cast<detail::compute_t<real_t<X>>>(x)));
}

namespace detail
{
template <typename T>
constexpr T normal_cdf_c(T x)
{
    return T(0.5) * erfc(-static_cast<T>(x) * static_cast<T>(detail::limits<T>::inv_sqrt2));
}
} // namespace detail

/// \brief The standard normal distribution function, `P(Z <= x)`, accurate in both tails.
/// \tparam X An integer or a `math_number`.
/// \param x The argument.
/// \return The result.
template <math_argument X>
constexpr real_t<X> normal_cdf(X x)
{
    return static_cast<real_t<X>>(detail::normal_cdf_c(static_cast<detail::compute_t<real_t<X>>>(x)));
}

namespace detail
{
    // Acklam's rational approximation to the normal quantile, relative error below 1.15e-9.
    inline constexpr double acklam_a[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                                          1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
    inline constexpr double acklam_b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                                          6.680131188771972e+01, -1.328068155288572e+01};
    inline constexpr double acklam_c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                                          -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
    inline constexpr double acklam_d[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00};
} // namespace detail

namespace detail
{
template <typename T>
constexpr T normal_quantile_c(T p_in)
{
    const T p = static_cast<T>(p_in);
    if (detail::nan_(p) || p < T(0.0) || p > T(1.0)) return std::numeric_limits<T>::quiet_NaN();
    if (p == T(0.0)) return -std::numeric_limits<T>::infinity();
    if (p == T(1.0)) return std::numeric_limits<T>::infinity();
    const auto A = [](int i) { return T(detail::acklam_a[i]); };
    const auto B = [](int i) { return T(detail::acklam_b[i]); };
    const auto C = [](int i) { return T(detail::acklam_c[i]); };
    const auto D = [](int i) { return T(detail::acklam_d[i]); };
    const T low = T(0.02425);
    T x = T(0);
    if (p < low) {
        const T q = sqrt(T(-2.0) * log(p));
        x = (((((C(0) * q + C(1)) * q + C(2)) * q + C(3)) * q + C(4)) * q + C(5)) / ((((D(0) * q + D(1)) * q + D(2)) * q + D(3)) * q + T(1.0));
    } else if (p <= T(1.0) - low) {
        const T q = p - T(0.5), r = q * q;
        x = (((((A(0) * r + A(1)) * r + A(2)) * r + A(3)) * r + A(4)) * r + A(5)) * q /
            (((((B(0) * r + B(1)) * r + B(2)) * r + B(3)) * r + B(4)) * r + T(1.0));
    } else {
        const T q = sqrt(T(-2.0) * log(T(1.0) - p));
        x = -(((((C(0) * q + C(1)) * q + C(2)) * q + C(3)) * q + C(4)) * q + C(5)) / ((((D(0) * q + D(1)) * q + D(2)) * q + D(3)) * q + T(1.0));
    }
    // One Halley step on F(x) - p, using the accurate tail of erfc
    const T e = normal_cdf(x) - p;
    const T u = e * static_cast<T>(detail::limits<T>::sqrt_2pi) * exp(T(0.5) * x * x);
    return x - u / (T(1.0) + T(0.5) * x * u);
}
} // namespace detail

/// \brief The standard normal quantile, the inverse of `normal_cdf`: Acklam's approximation, then one Halley step.
///
/// 0 and 1 give -inf and +inf; anything outside [0, 1] is NaN.
/// \tparam X An integer or a `math_number`.
/// \param p The probability.
/// \return The result.
template <math_argument X>
constexpr real_t<X> normal_quantile(X p)
{
    return static_cast<real_t<X>>(detail::normal_quantile_c(static_cast<detail::compute_t<real_t<X>>>(p)));
}

namespace detail
{
template <typename T>
constexpr T lgamma_c(T x_in)
{
    const T pi_t = static_cast<T>(detail::limits<T>::pi);
    T x = static_cast<T>(x_in);
    if (detail::nan_(x)) return x;
    if (x < T(0.5))
        return log(pi_t / abs(sin(pi_t * x))) - lgamma_c(T(1.0) - x);
    constexpr double c[] = {0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
                            12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7};
    x -= T(1.0);
    T a = T(c[0]);
    for (int i = 1; i < 9; ++i)
        a += T(c[i]) / (x + T(i));
    const T t = x + T(7.5);
    return static_cast<T>(detail::limits<T>::half_log_2pi) + (x + T(0.5)) * log(t) - t + log(a);
}
} // namespace detail

/// \brief The logarithm of the gamma function, by Lanczos' approximation (g = 7, 9 terms): about 15 digits.
///
/// Its own implementation at run time too, because `std::lgamma` writes the global `signgam` and so is not
/// thread-safe on every platform.
/// \tparam X An integer or a `math_number`.
/// \param x The argument; not a non-positive integer.
/// \return `log |Gamma(x)|`.
template <math_argument X>
constexpr real_t<X> lgamma(X x)
{
    return static_cast<real_t<X>>(detail::lgamma_c(static_cast<detail::compute_t<real_t<X>>>(x)));
}

namespace detail
{
template <typename T>
constexpr T gamma_p_c(T a_in, T x_in)
{
    using L = detail::limits<T>;
    const T a = static_cast<T>(a_in), x = static_cast<T>(x_in);
    if (detail::nan_(a) || detail::nan_(x) || a <= T(0.0) || x < T(0.0)) return std::numeric_limits<T>::quiet_NaN();
    if (x == T(0.0)) return T(0.0);
    const T front = exp(-x + a * log(x) - lgamma(a));
    if (x < a + T(1.0)) {
        T term = T(1.0) / a, sum = term;
        for (int n = 1; n < 1000; ++n) {
            term *= x / (a + T(n));
            sum += term;
            if (abs(term) < abs(sum) * static_cast<T>(L::gamma_tolerance))
                break;
        }
        return sum * front;
    }
    const T tiny = static_cast<T>(L::tiny);
    T b = x + T(1.0) - a, c = T(1.0) / tiny, d = T(1.0) / b, h = d;
    for (int i = 1; i < 1000; ++i) {
        const T an = -T(i) * (T(i) - a);
        b += T(2.0);
        d = an * d + b;
        d = abs(d) < tiny ? tiny : d;
        c = b + an / c;
        c = abs(c) < tiny ? tiny : c;
        d = T(1.0) / d;
        const T delta = d * c;
        h *= delta;
        if (abs(delta - T(1.0)) < static_cast<T>(L::gamma_tolerance))
            break;
    }
    return T(1.0) - front * h;
}
} // namespace detail

/// \brief The regularised lower incomplete gamma function `P(a, x)`: a series below `a + 1`, Lentz's continued fraction above.
/// \tparam A An integer or a `math_number`.
/// \tparam X An integer or a `math_number`.
/// \param a_in The shape, above zero.
/// \param x_in The argument, at least zero.
/// \return `P(a, x)`, in [0, 1].
template <math_argument A, math_argument X>
constexpr real_t<std::common_type_t<A, X>> gamma_p(A a_in, X x_in)
{
    using C = detail::compute_t<real_t<std::common_type_t<A, X>>>;
    return static_cast<real_t<std::common_type_t<A, X>>>(detail::gamma_p_c(static_cast<C>(a_in), static_cast<C>(x_in)));
}

namespace detail
{
template <typename T>
constexpr T chi_square_cdf_c(T x, T k)
{
    return static_cast<T>(x) <= T(0.0) ? T(0.0) : gamma_p_c(static_cast<T>(k) / T(2.0), static_cast<T>(x) / T(2.0));
}
} // namespace detail

/// \brief The chi-square distribution function with `k` degrees of freedom.
/// \tparam X An integer or a `math_number`.
/// \tparam K An integer or a `math_number`.
/// \param x The value.
/// \param k The degrees of freedom, above zero.
/// \return `P(X <= x)`.
template <math_argument X, math_argument K>
constexpr real_t<std::common_type_t<X, K>> chi_square_cdf(X x, K k)
{
    using C = detail::compute_t<real_t<std::common_type_t<X, K>>>;
    return static_cast<real_t<std::common_type_t<X, K>>>(detail::chi_square_cdf_c(static_cast<C>(x), static_cast<C>(k)));
}

} // namespace snicholls::math

#endif /* snicholls_core_math_constexpr_math_hpp */
