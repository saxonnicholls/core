//
//  roots.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::math`: one-dimensional root finding, `constexpr`: bisection, Brent's method, a safeguarded Newton iteration, and outward bracketing.
/// \ingroup core
///
/// **Overview.** Solving f(x) = 0 for one unknown is the inner step of an implied volatility, a yield, a
/// Kepler anomaly or a break-even. Every routine here is a `constexpr` function template on the number
/// type `T` (`float`, `double`, `long double`, or a type of the caller's with `std::numeric_limits`) and on
/// the callable, so a root can be found by the compiler for a constant table, or at run time on the hot
/// path with no allocation.
///
/// - `bisect` halves a bracket: one bit per evaluation, guaranteed. Use it when f is merely continuous
///   or noisy.
/// - `brent` is Brent's 1973 method (inverse quadratic interpolation, the secant step and bisection,
///   with his safeguards): superlinear on smooth functions, never much worse than bisection. The default
///   choice.
/// - `newton_safeguarded` takes f and f' together and steps by Newton inside a bracket, falling back to
///   bisection whenever the step leaves the bracket or does not halve the previous one (a stall).
///   Use it when the derivative is cheap (an option's vega comes with its price).
/// - `bracket` widens an interval around a guess until f changes sign, for the three above.
///
/// Each returns a `root_result`: the point, f there, the number of evaluations of f after the two
/// endpoints, and a `root_status`.
///
/// \snippet math/roots_demo.cpp roots
///
/// **Relationships.** Uses `math/constexpr_math.hpp` for `abs` and `is_nan`; `math/special.hpp` uses
/// nothing from here. Group: \ref core.
///
/// **Guarantees.** `constexpr`, no allocation, no exceptions of their own (an exception from f
/// propagates). Convergence is declared when the bracket's half-width is at most
/// `tolerance / 2 + 2 eps |x|` (Brent's test; `tolerance = 0` asks for the type's full precision), or f is
/// exactly zero. A bracketing method keeps a sign change inside its interval throughout; the result is
/// the evaluated point with the smaller |f|.
///
/// **Errors.** None thrown: `root_status::not_bracketed` when f(lo) and f(hi) do not differ in sign (or
/// one is NaN), `root_status::max_iterations` when the budget runs out; the best point so far is still
/// returned.
///
/// **Performance.** Not benchmarked; the cost is the evaluations of f, counted in `iterations`.
///
/// **Tests and demos.** `tests/math/roots_test.cpp`, `demos/math/roots_demo.cpp`.
///
/// **Dependencies.** `<concepts>`, `<limits>`, `<type_traits>`, `<utility>`, and
/// `math/constexpr_math.hpp`.
///
/// **Known issues.** A root of even multiplicity has no sign change, so no bracketing method finds it;
/// minimise |f| instead. Near a multiple root f is flat and dominated by rounding, so the bracket closes
/// on a sign change of the rounding noise: the answer is as good as f's evaluation allows, no better.
///
/// **Status.** Experimental.

#ifndef snicholls_core_math_roots_hpp
#define snicholls_core_math_roots_hpp

#include <snicholls/core/math/constexpr_math.hpp>

#include <concepts>
#include <limits>
#include <type_traits>
#include <utility>

namespace snicholls::math
{

/// \brief How a root search ended.
/// \ingroup core
enum class root_status
{
    converged,       ///< The bracket closed to the tolerance, or f was exactly zero (for `bracket`: a sign change was found).
    not_bracketed,   ///< f(lo) and f(hi) have the same sign, or one is NaN (for `bracket`: the step was zero or not finite).
    max_iterations   ///< The evaluation budget ran out first; the best point so far is returned.
};

/// \brief The result of a root search: the point, f there, the evaluations spent and how it ended.
/// \ingroup core
/// \tparam T The number type.
template <typename T>
struct root_result
{
    T x;                  ///< The root found, or the best point when not converged.
    T f;                  ///< f(x).
    int iterations;       ///< Evaluations of f after the two endpoints.
    root_status status;   ///< How the search ended.
};

/// \brief An interval over which f changes sign, from `bracket`.
/// \ingroup core
/// \tparam T The number type.
template <typename T>
struct bracket_result
{
    T lo;                 ///< The left end.
    T hi;                 ///< The right end.
    T f_lo;               ///< f(lo).
    T f_hi;               ///< f(hi).
    int iterations;       ///< Expansions made.
    root_status status;   ///< `converged` when f(lo) and f(hi) differ in sign (or one is zero).
};

/// \brief The evaluation budget the root finders default to: enough for bisection to reach adjacent numbers from any finite bracket.
/// \ingroup core
/// \tparam T The number type.
template <typename T>
inline constexpr int default_root_iterations =
    std::numeric_limits<T>::digits + std::numeric_limits<T>::max_exponent - std::numeric_limits<T>::min_exponent + 4;

namespace detail
{
template <typename T>
constexpr bool opposite_(T a, T b) { return (a < T(0) && b > T(0)) || (a > T(0) && b < T(0)); }

// Brent's convergence radius: half the caller's absolute tolerance plus two units of relative precision, never 0.
template <typename T>
constexpr T root_tol_(T x, T tolerance)
{
    return T(2) * std::numeric_limits<T>::epsilon() * abs_(x) + tolerance / T(2) + std::numeric_limits<T>::min();
}
} // namespace detail

/// \brief A root of f in [lo, hi] by bisection: one bit per evaluation, for any continuous f with a sign change.
/// \ingroup core
/// \tparam T The number type.
/// \tparam F A callable from T to T.
/// \param f The function.
/// \param lo One end of the bracket.
/// \param hi The other end (the ends may be given in either order).
/// \param tolerance The absolute width the bracket closes to; 0 for the type's full precision.
/// \param max_iterations The evaluation budget.
/// \return The root: the endpoint of the final bracket with the smaller |f|.
template <math_number T, typename F>
    requires std::invocable<F&, T>
constexpr root_result<T> bisect(F&& f, T lo, T hi, std::type_identity_t<T> tolerance = T(0),
                                int max_iterations = default_root_iterations<T>)
{
    if (hi < lo)
        std::swap(lo, hi);
    T flo = static_cast<T>(f(lo)), fhi = static_cast<T>(f(hi));
    if (flo == T(0)) return {lo, flo, 0, root_status::converged};
    if (fhi == T(0)) return {hi, fhi, 0, root_status::converged};
    if (!detail::opposite_(flo, fhi))
        return {lo, flo, 0, root_status::not_bracketed};
    const auto best = [&](int n, root_status s) {
        return detail::abs_(flo) <= detail::abs_(fhi) ? root_result<T>{lo, flo, n, s} : root_result<T>{hi, fhi, n, s};
    };
    for (int n = 0;; ++n) {
        const T mid = lo + (hi - lo) / T(2);
        if ((hi - lo) / T(2) <= detail::root_tol_(mid, tolerance) || mid == lo || mid == hi)
            return best(n, root_status::converged);
        if (n == max_iterations)
            return best(n, root_status::max_iterations);
        const T fm = static_cast<T>(f(mid));
        if (fm == T(0))
            return {mid, fm, n + 1, root_status::converged};
        if (detail::opposite_(flo, fm)) { hi = mid; fhi = fm; }
        else { lo = mid; flo = fm; }
    }
}

/// \brief A root of f in [lo, hi] by Brent's 1973 method: inverse quadratic interpolation and secant steps, safeguarded by bisection.
///
/// The algorithm of Brent, *Algorithms for Minimization without Derivatives* (1973), chapter 4: each step
/// interpolates through the last three points, and takes a bisection step instead whenever the
/// interpolated step falls outside the bracket or shrinks too slowly. Superlinear on a smooth f, never
/// much slower than `bisect`.
/// \ingroup core
/// \tparam T The number type.
/// \tparam F A callable from T to T.
/// \param f The function.
/// \param lo One end of the bracket.
/// \param hi The other end.
/// \param tolerance The absolute width the bracket closes to; 0 for the type's full precision.
/// \param max_iterations The evaluation budget.
/// \return The root, with f there.
template <math_number T, typename F>
    requires std::invocable<F&, T>
constexpr root_result<T> brent(F&& f, T lo, T hi, std::type_identity_t<T> tolerance = T(0),
                               int max_iterations = default_root_iterations<T>)
{
    using detail::abs_;
    T a = lo, b = hi;
    T fa = static_cast<T>(f(a)), fb = static_cast<T>(f(b));
    if (fa == T(0)) return {a, fa, 0, root_status::converged};
    if (fb == T(0)) return {b, fb, 0, root_status::converged};
    if (!detail::opposite_(fa, fb))
        return {a, fa, 0, root_status::not_bracketed};
    T c = a, fc = fa, d = b - a, e = d;
    for (int n = 0;; ++n) {
        if (detail::opposite_(fb, fc) == false) { c = a; fc = fa; d = e = b - a; }   // keep the root between b and c
        if (abs_(fc) < abs_(fb)) { a = b; b = c; c = a; fa = fb; fb = fc; fc = fa; }   // b is the best point
        const T tol1 = detail::root_tol_(b, tolerance);
        const T xm = (c - b) / T(2);
        if (abs_(xm) <= tol1 || fb == T(0))
            return {b, fb, n, root_status::converged};
        if (n == max_iterations)
            return {b, fb, n, root_status::max_iterations};
        if (abs_(e) >= tol1 && abs_(fa) > abs_(fb)) {
            T p, q;
            const T s = fb / fa;
            if (a == c) {   // the secant step
                p = T(2) * xm * s;
                q = T(1) - s;
            } else {        // inverse quadratic interpolation
                const T qa = fa / fc, r = fb / fc;
                p = s * (T(2) * xm * qa * (qa - r) - (b - a) * (r - T(1)));
                q = (qa - T(1)) * (r - T(1)) * (s - T(1));
            }
            if (p > T(0)) q = -q;
            else p = -p;
            const T bound1 = T(3) * xm * q - abs_(tol1 * q), bound2 = abs_(e * q);
            if (T(2) * p < (bound1 < bound2 ? bound1 : bound2)) { e = d; d = p / q; }   // accept the interpolation
            else { d = xm; e = d; }                                                    // bisect
        } else {
            d = xm;
            e = d;
        }
        a = b;
        fa = fb;
        b += abs_(d) > tol1 ? d : (xm > T(0) ? tol1 : -tol1);
        fb = static_cast<T>(f(b));
    }
}

/// \brief A root of f in [lo, hi] by Newton's method from x0, falling back to bisection when a step leaves the bracket or stalls.
///
/// Each evaluation narrows the bracket by the sign of f. The Newton step x - f/f' is taken when it lands
/// strictly inside the bracket and is at most half the step before it; otherwise the bracket is bisected.
/// So it converges quadratically near a simple root and never diverges.
/// \ingroup core
/// \tparam T The number type.
/// \tparam FDF A callable from T to a pair-like {f(x), f'(x)} (anything a structured binding of two unpacks).
/// \param f_and_df The function and its derivative.
/// \param lo One end of the bracket.
/// \param hi The other end.
/// \param x0 The first guess; the midpoint is used when it is not inside the bracket.
/// \param tolerance The absolute width the search closes to; 0 for the type's full precision.
/// \param max_iterations The evaluation budget.
/// \return The root, with f there.
template <math_number T, typename FDF>
    requires std::invocable<FDF&, T>
constexpr root_result<T> newton_safeguarded(FDF&& f_and_df, T lo, T hi, std::type_identity_t<T> x0,
                                            std::type_identity_t<T> tolerance = T(0),
                                            int max_iterations = default_root_iterations<T>)
{
    using detail::abs_;
    const auto [flo_, dlo_] = f_and_df(lo);
    const auto [fhi_, dhi_] = f_and_df(hi);
    (void)dlo_;
    (void)dhi_;
    const T flo = static_cast<T>(flo_), fhi = static_cast<T>(fhi_);
    if (flo == T(0)) return {lo, flo, 0, root_status::converged};
    if (fhi == T(0)) return {hi, fhi, 0, root_status::converged};
    if (!detail::opposite_(flo, fhi))
        return {lo, flo, 0, root_status::not_bracketed};
    T neg = flo < T(0) ? lo : hi, pos = flo < T(0) ? hi : lo;   // f(neg) < 0 < f(pos)
    const T low = lo < hi ? lo : hi, high = lo < hi ? hi : lo;
    T x = x0 > low && x0 < high ? x0 : low + (high - low) / T(2);
    T step = high - low, step_before = step;
    T best_x = abs_(flo) <= abs_(fhi) ? lo : hi, best_f = abs_(flo) <= abs_(fhi) ? flo : fhi;
    for (int n = 1;; ++n) {
        const auto [fx_, dx_] = f_and_df(x);
        const T fx = static_cast<T>(fx_), dfx = static_cast<T>(dx_);
        if (!(abs_(fx) > abs_(best_f))) { best_x = x; best_f = fx; }
        const T tol1 = detail::root_tol_(x, tolerance);
        if (fx == T(0) || abs_(step) <= tol1 || abs_(pos - neg) / T(2) <= tol1)
            return {x, fx, n, root_status::converged};
        if (n == max_iterations)
            return {best_x, best_f, n, root_status::max_iterations};
        if (fx < T(0)) neg = x;
        else pos = x;
        const T a = neg < pos ? neg : pos, b = neg < pos ? pos : neg;
        const T xn = x - fx / dfx;
        step_before = step;
        if (dfx != T(0) && xn > a && xn < b && abs_(T(2) * fx) <= abs_(step_before * dfx)) {
            step = x - xn;
            x = xn;
        } else {
            step = (b - a) / T(2);
            x = a + step;
        }
    }
}

/// \brief Widens an interval around x0 until f changes sign across it.
///
/// Starts from [x0 - step, x0 + step] and, while f has the same sign at both ends, moves the end where
/// |f| is smaller outward by `growth` times the current width, as Press et al.'s `zbrac` does.
/// \ingroup core
/// \tparam T The number type.
/// \tparam F A callable from T to T.
/// \param f The function.
/// \param x0 The centre of the first interval.
/// \param step The first interval's half-width; its sign is ignored.
/// \param growth The factor the width grows by at each expansion, greater than 0.
/// \param max_iterations The number of expansions allowed.
/// \return The interval and f at its ends; `converged` when f changes sign across it, `max_iterations` when none was found, `not_bracketed` when `step` is zero or not finite.
template <math_number T, typename F>
    requires std::invocable<F&, T>
constexpr bracket_result<T> bracket(F&& f, T x0, std::type_identity_t<T> step, std::type_identity_t<T> growth = T(1.6),
                                    int max_iterations = 60)
{
    step = detail::abs_(step);
    T a = x0 - step, b = x0 + step;
    if (!(step > T(0)) || detail::inf_(step))
        return {a, b, T(0), T(0), 0, root_status::not_bracketed};
    T fa = static_cast<T>(f(a)), fb = static_cast<T>(f(b));
    for (int n = 0;; ++n) {
        if (fa == T(0) || fb == T(0) || detail::opposite_(fa, fb))
            return {a, b, fa, fb, n, root_status::converged};
        if (n == max_iterations)
            return {a, b, fa, fb, n, root_status::max_iterations};
        if (detail::abs_(fa) < detail::abs_(fb)) { a -= growth * (b - a); fa = static_cast<T>(f(a)); }
        else { b += growth * (b - a); fb = static_cast<T>(f(b)); }
    }
}

} // namespace snicholls::math

#endif /* snicholls_core_math_roots_hpp */
