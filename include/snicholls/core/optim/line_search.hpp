//
//  line_search.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::optim::backtracking_armijo` and `strong_wolfe`: line searches along a descent direction, over callables and the caller's storage, all `constexpr`.
/// \ingroup core
///
/// **Overview.** A descent method picks a direction `d` at `x` and then a step `α` along it. With
/// `φ(α) = f(x + α d)` and `φ'(0) = ∇f(x)ᵀd < 0`:
///
/// - `backtracking_armijo(f, x, d, f(x), φ'(0), trial, options)` tries `α = initial_step, ρ α, ρ² α, …` until the
///   sufficient-decrease (Armijo) condition `φ(α) ≤ φ(0) + c₁ α φ'(0)` holds. It needs values only. Use it for
///   Newton-type directions whose unit step is usually right, and for projected and proximal methods.
/// - `strong_wolfe(fg, x, d, f(x), φ'(0), trial, trial_gradient, options)` finds `α` with sufficient decrease and the
///   strong curvature condition `|φ'(α)| ≤ c₂ |φ'(0)|`, by bracketing and zoom (Nocedal and Wright, Numerical
///   Optimization, 2nd ed., algorithms 3.5 and 3.6). The trial steps inside the bracket come from the cubic that
///   interpolates `φ` and `φ'` at both ends (their equation 3.59), else the quadratic through `φ(lo)`, `φ'(lo)` and
///   `φ(hi)`, else bisection, and are kept at least a tenth of the bracket from either end. Quasi-Newton methods need
///   it: the curvature condition makes `sᵀy > 0`, so the BFGS update stays positive definite (`optim/lbfgs.hpp`).
///
/// `f(x)` is a callable `T(std::span<const T>)`; `fg(x, g)` is `T(std::span<const T>, std::span<T>)`, returning
/// `f(x)` and writing `∇f(x)` into `g`. `trial` (and `trial_gradient`) are `n` values of the caller's: on return they
/// hold `x + α d` (and `∇f` there) for the returned `α`, so the caller moves to it without evaluating again.
///
/// \snippet optim/line_search_demo.cpp line_search
///
/// **Relationships.** `optim/lbfgs.hpp` (its line search), `linalg/types.hpp` (`scalar`), `math/constexpr_math.hpp`
/// (`sqrt`). Group: \ref core.
///
/// **Implementations.** Armijo: plain contraction by `shrink`. Strong Wolfe: the bracketing phase expands the step by
/// `expansion` up to `max_step`; the zoom keeps the invariants of algorithm 3.6 (`lo` satisfies sufficient decrease
/// and has the lowest `φ` seen, `φ'(lo) (hi - lo) < 0`). A value that is not a number fails the sufficient-decrease
/// test, so a step into a region where `f` is undefined is shortened rather than taken.
///
/// **Guarantees.** `constexpr`; `noexcept` when the callable is and `T`'s arithmetic is. No allocation, no recursion.
/// At most `max_evaluations` evaluations, plus one when a failed strong-Wolfe search re-evaluates its best point so
/// that `trial` holds it. On `line_search_status::satisfied` the returned step meets the documented conditions with
/// the documented constants.
///
/// **Errors.** Nothing throws unless the callable does. The outcome is a `line_search_status`:
/// `not_descent_direction` (`φ'(0) ≥ 0`, or not a number: nothing is evaluated, the step is 0 and `trial` is
/// untouched), `evaluation_limit`, `step_at_maximum` (strong Wolfe: `max_step` satisfies sufficient decrease and
/// `φ` is still falling: the function may be unbounded below), `interval_collapsed` (strong Wolfe: the bracket
/// shrank to rounding). After a failure the step is the best one found with sufficient decrease (0 if none).
///
/// **Performance.** Not benchmarked: the cost is the callable's evaluations, which the result counts.
///
/// **Tests and demos.** `tests/optim/line_search_test.cpp`, `demos/optim/line_search_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<cstdint>`, `<span>`, `<type_traits>`, `linalg/types.hpp`,
/// `math/constexpr_math.hpp`.
///
/// **Known issues.** `c₁ < c₂ < 1` is the caller's to ensure (the defaults are 1e-4 and 0.9, for quasi-Newton
/// methods; use `c₂ = 0.1` for nonlinear conjugate gradients). The searches do not check that `trial` is `n` long.
///
/// **Status.** Experimental, new in 0.21.0.

#ifndef snicholls_core_optim_line_search_hpp
#define snicholls_core_optim_line_search_hpp

#include <snicholls/core/linalg/types.hpp>
#include <snicholls/core/math/constexpr_math.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace snicholls::optim {

    /// \brief How a line search ended.
    /// \ingroup core
    enum class line_search_status : std::uint8_t {
        satisfied = 0,              ///< The step meets the conditions.
        not_descent_direction = 1,  ///< `φ'(0) ≥ 0` (or not a number): nothing was evaluated.
        evaluation_limit = 2,       ///< `max_evaluations` ran out.
        step_at_maximum = 3,        ///< Strong Wolfe: `max_step` decreases `f` enough and `φ` is still falling.
        interval_collapsed = 4,     ///< Strong Wolfe: the bracket shrank to rounding without meeting the conditions.
    };

    /// \brief The settings of `backtracking_armijo`.
    /// \ingroup core
    /// \tparam T The value type.
    template <linalg::scalar T>
    struct armijo_options {
        T initial_step = T(1);             ///< The first step tried.
        T c1 = T(1e-4);                    ///< The sufficient-decrease constant, in (0, 1).
        T shrink = T(0.5);                 ///< The contraction per rejected step, in (0, 1).
        std::size_t max_evaluations = 50;  ///< The most evaluations of `f`.
    };

    /// \brief The outcome of `backtracking_armijo`.
    /// \ingroup core
    /// \tparam T The value type.
    template <linalg::scalar T>
    struct armijo_result {
        line_search_status status = line_search_status::not_descent_direction;  ///< How it ended.
        T step = T(0);                ///< The step `α` (on failure: the last tried, or 0 if none).
        T value = T(0);               ///< `φ(step)`.
        std::size_t evaluations = 0;  ///< The evaluations of `f`.

        /// \brief Whether the step meets the condition. \return `status == line_search_status::satisfied`.
        [[nodiscard]] constexpr bool satisfied() const noexcept { return status == line_search_status::satisfied; }
    };

    /// \brief The settings of `strong_wolfe`.
    /// \ingroup core
    /// \tparam T The value type.
    template <linalg::scalar T>
    struct wolfe_options {
        T initial_step = T(1);             ///< The first step tried (clamped to `max_step`).
        T c1 = T(1e-4);                    ///< The sufficient-decrease constant, in (0, c₂).
        T c2 = T(0.9);                     ///< The curvature constant, in (c₁, 1).
        T expansion = T(2);                ///< The growth of the step while bracketing, above 1.
        T max_step = T(1e10);              ///< The largest step.
        std::size_t max_evaluations = 40;  ///< The most evaluations of `fg`.
    };

    /// \brief The outcome of `strong_wolfe`.
    /// \ingroup core
    /// \tparam T The value type.
    template <linalg::scalar T>
    struct wolfe_result {
        line_search_status status = line_search_status::not_descent_direction;  ///< How it ended.
        T step = T(0);                ///< The step `α`.
        T value = T(0);               ///< `φ(step)`.
        T derivative = T(0);          ///< `φ'(step) = ∇f(x + step d)ᵀ d`.
        std::size_t evaluations = 0;  ///< The evaluations of `fg`.

        /// \brief Whether the step meets the conditions. \return `status == line_search_status::satisfied`.
        [[nodiscard]] constexpr bool satisfied() const noexcept { return status == line_search_status::satisfied; }
    };

    namespace detail {
        template <typename T>
        constexpr T ls_abs(T v) noexcept
        {
            return v < T(0) ? -v : v;
        }
        // A point on the line: the step, φ and φ'
        template <typename T>
        struct ls_point {
            T a, f, d;
        };
        // The minimiser of the cubic interpolating φ, φ' at p and q (Nocedal and Wright 3.59), else that of the
        // quadratic through φ(p), φ'(p), φ(q), else the midpoint; kept a tenth of the interval from either end
        template <typename T>
        constexpr T ls_interpolate(const ls_point<T>& p, const ls_point<T>& q) noexcept
        {
            const T width = ls_abs(q.a - p.a);
            const T left = (p.a < q.a ? p.a : q.a) + width / T(10), right = (p.a < q.a ? q.a : p.a) - width / T(10);
            const auto inside = [&](T a) { return a >= left && a <= right; };
            const T d1 = p.d + q.d - T(3) * (p.f - q.f) / (p.a - q.a);
            const T disc = d1 * d1 - p.d * q.d;
            if (disc >= T(0)) {
                const T root = static_cast<T>(math::sqrt(disc));
                const T d2 = q.a > p.a ? root : -root;
                const T den = q.d - p.d + T(2) * d2;
                if (den != T(0)) {
                    const T a = q.a - (q.a - p.a) * (q.d + d2 - d1) / den;
                    if (inside(a))
                        return a;
                }
            }
            const T h = q.a - p.a;
            const T curv = T(2) * (q.f - p.f - p.d * h);
            if (curv > T(0)) {
                const T a = p.a - p.d * h * h / curv;
                if (inside(a))
                    return a;
            }
            return (p.a + q.a) / T(2);
        }
    } // namespace detail

    /// \brief Backtracking line search for the Armijo (sufficient-decrease) condition `φ(α) ≤ φ(0) + c₁ α φ'(0)`.
    /// \ingroup core
    /// \tparam F The objective, `T(std::span<const T>)`.
    /// \tparam T The value type.
    /// \param f The objective.
    /// \param x The current point, `n` values.
    /// \param direction The direction `d`, `n` values.
    /// \param value `f(x)`.
    /// \param gradient_dot_direction `φ'(0) = ∇f(x)ᵀ d`, negative.
    /// \param trial `n` values: on return, `x + step · d`.
    /// \param options The first step, `c₁`, the contraction and the evaluation limit.
    /// \return The status, the step, `φ(step)` and the evaluations.
    template <typename F, linalg::scalar T>
    [[nodiscard]] constexpr armijo_result<T> backtracking_armijo(F&& f, std::type_identity_t<std::span<const T>> x, std::type_identity_t<std::span<const T>> direction, T value,
                                                                 T gradient_dot_direction, std::type_identity_t<std::span<T>> trial,
                                                                 const std::type_identity_t<armijo_options<T>>& options = {}) noexcept(std::is_nothrow_invocable_v<F&, std::span<const T>> &&
                                                                                                                                    linalg::detail::nothrow_scalar<T>)
    {
        if (!(gradient_dot_direction < T(0)))
            return {line_search_status::not_descent_direction, T(0), value, 0};
        T alpha = options.initial_step, phi = value;
        for (std::size_t ev = 1; ev <= options.max_evaluations; ++ev) {
            for (std::size_t i = 0; i < x.size(); ++i)
                trial[i] = x[i] + alpha * direction[i];
            phi = f(std::span<const T>(trial));
            if (phi <= value + options.c1 * alpha * gradient_dot_direction)
                return {line_search_status::satisfied, alpha, phi, ev};
            if (ev < options.max_evaluations)
                alpha *= options.shrink;
        }
        return {line_search_status::evaluation_limit, options.max_evaluations > 0 ? alpha : T(0), phi, options.max_evaluations};
    }

    /// \brief Line search for the strong Wolfe conditions, `φ(α) ≤ φ(0) + c₁ α φ'(0)` and `|φ'(α)| ≤ c₂ |φ'(0)|`, by
    /// bracketing and zoom with safeguarded cubic interpolation (Nocedal and Wright, algorithms 3.5 and 3.6).
    /// \ingroup core
    /// \tparam F The objective with its gradient, `T(std::span<const T> x, std::span<T> g)`.
    /// \tparam T The value type.
    /// \param fg The objective: returns `f(x)`, writes `∇f(x)` into `g`.
    /// \param x The current point, `n` values.
    /// \param direction The direction `d`, `n` values.
    /// \param value `f(x)`.
    /// \param gradient_dot_direction `φ'(0) = ∇f(x)ᵀ d`, negative.
    /// \param trial `n` values: on return, `x + step · d`.
    /// \param trial_gradient `n` values: on return, `∇f(x + step · d)`.
    /// \param options The constants, the first and largest steps, the expansion and the evaluation limit.
    /// \return The status, the step, `φ(step)`, `φ'(step)` and the evaluations.
    template <typename F, linalg::scalar T>
    [[nodiscard]] constexpr wolfe_result<T> strong_wolfe(F&& fg, std::type_identity_t<std::span<const T>> x, std::type_identity_t<std::span<const T>> direction, T value,
                                                         T gradient_dot_direction, std::type_identity_t<std::span<T>> trial, std::type_identity_t<std::span<T>> trial_gradient,
                                                         const std::type_identity_t<wolfe_options<T>>& options = {}) noexcept(std::is_nothrow_invocable_v<F&, std::span<const T>, std::span<T>> &&
                                                                                                                             linalg::detail::nothrow_scalar<T>)
    {
        using point = detail::ls_point<T>;
        const T d0 = gradient_dot_direction;
        if (!(d0 < T(0)))
            return {line_search_status::not_descent_direction, T(0), value, d0, 0};
        std::size_t evaluations = 0;
        T last = T(-1);   // the step whose point trial holds
        const auto eval = [&](T a) -> point {
            for (std::size_t i = 0; i < x.size(); ++i)
                trial[i] = x[i] + a * direction[i];
            const T phi = fg(std::span<const T>(trial), trial_gradient);
            T dphi(0);
            for (std::size_t i = 0; i < x.size(); ++i)
                dphi += trial_gradient[i] * direction[i];
            ++evaluations;
            last = a;
            return {a, phi, dphi};
        };
        const auto decreases = [&](const point& p) { return p.f <= value + options.c1 * p.a * d0; };
        const auto curved = [&](const point& p) { return detail::ls_abs(p.d) <= -options.c2 * d0; };
        // A failure returns the best point with sufficient decrease, re-evaluated if trial holds another
        const auto fail = [&](line_search_status s, const point& lo) -> wolfe_result<T> {
            if (last != lo.a)
                eval(lo.a);
            return {s, lo.a, lo.f, lo.d, evaluations};
        };
        // Algorithm 3.6: lo has sufficient decrease and the lowest φ seen; φ'(lo) (hi - lo) < 0
        const auto zoom = [&](point lo, point hi) -> wolfe_result<T> {
            for (;;) {
                if (evaluations >= options.max_evaluations)
                    return fail(line_search_status::evaluation_limit, lo);
                const T aj = detail::ls_interpolate(lo, hi);
                if (!(aj != lo.a && aj != hi.a))
                    return fail(line_search_status::interval_collapsed, lo);
                const point pj = eval(aj);
                if (!decreases(pj) || pj.f >= lo.f)
                    hi = pj;
                else {
                    if (curved(pj))
                        return {line_search_status::satisfied, pj.a, pj.f, pj.d, evaluations};
                    if (pj.d * (hi.a - lo.a) >= T(0))
                        hi = lo;
                    lo = pj;
                }
            }
        };
        // Algorithm 3.5: expand until a bracket is found or the conditions hold
        point prev{T(0), value, d0};
        T a = options.initial_step < options.max_step ? options.initial_step : options.max_step;
        for (std::size_t i = 1;; ++i) {
            if (evaluations >= options.max_evaluations)
                return fail(line_search_status::evaluation_limit, prev);
            const point p = eval(a);
            if (!decreases(p) || (i > 1 && p.f >= prev.f))
                return zoom(prev, p);
            if (curved(p))
                return {line_search_status::satisfied, p.a, p.f, p.d, evaluations};
            if (p.d >= T(0))
                return zoom(p, prev);
            if (a >= options.max_step)
                return {line_search_status::step_at_maximum, p.a, p.f, p.d, evaluations};
            prev = p;
            a = a * options.expansion < options.max_step ? a * options.expansion : options.max_step;
        }
    }

} // namespace snicholls::optim

#endif /* snicholls_core_optim_line_search_hpp */
