//
//  lbfgs.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::optim::lbfgs_minimize`: limited-memory BFGS for smooth unconstrained minimisation, with its history of curvature pairs and the two-loop recursion, over the caller's storage, all `constexpr`.
/// \ingroup core
///
/// **Overview.** L-BFGS (Nocedal, 1980; Liu and Nocedal, 1989) minimises a smooth `f: ℝⁿ → ℝ` from values and
/// gradients alone. It keeps the last `M` pairs `sₖ = xₖ₊₁ - xₖ`, `yₖ = ∇fₖ₊₁ - ∇fₖ` and applies the BFGS inverse
/// Hessian they define to the gradient in O(Mn), without forming an `n x n` matrix. It is the workhorse for
/// calibration, maximum likelihood and smooth penalties when `n` is too large for Newton's method or the Hessian is
/// unavailable. It is not for constrained problems (bounds need L-BFGS-B; general constraints a QP or interior-point
/// solver), nor for non-smooth objectives.
///
/// - `lbfgs_history<T, M>`: a ring buffer of up to `M` pairs over storage of the caller's (`n · M` values for the
///   `s` vectors and `n · M` for the `y` vectors); `lbfgs_history<T, M, N>` keeps them inside, for a size fixed at
///   compile time. `push(s, y)` adds a pair (dropping the oldest when full) unless `sᵀy ≤ ε yᵀy`, which would spoil
///   positive definiteness.
/// - `two_loop(history, g, d)`: `d = -H g`, `H` the L-BFGS inverse Hessian with `H₀ = γ I`, `γ = sᵀy / yᵀy` of the
///   newest pair (1 with no pairs), by the two-loop recursion (Nocedal and Wright, algorithm 7.4).
/// - `lbfgs_minimize<M>(fg, x, options, work)`: from `x`, repeat `d = -H ∇f`, a strong-Wolfe line search along `d`
///   (`optim/line_search.hpp`), `push(s, y)`, until `‖∇f‖∞ ≤ gradient_tolerance`. `work` is `lbfgs_workspace(n, M)`
///   values. The first step is `min(1, 1 / ‖∇f‖₂)` along `-∇f`; later steps start at 1.
///
/// \snippet optim/lbfgs_demo.cpp lbfgs
///
/// **Relationships.** `optim/line_search.hpp` (`strong_wolfe`), `linalg/types.hpp` (`scalar`),
/// `linalg/krylov.hpp` (CG, which L-BFGS matches on a quadratic with exact line searches). Group: \ref core.
///
/// **Implementations.** The history holds `ρₖ = 1 / yₖᵀsₖ` with each pair. When `d` is not a descent direction
/// (`∇fᵀd ≥ 0`, possible only through rounding), the history is cleared and `d = -∇f`. When a line search fails
/// without progress and the history is not empty, the history is cleared and the iteration retried once along
/// `-∇f`; a failure with some decrease is accepted as a step.
///
/// **Guarantees.** `constexpr`; `noexcept` when `fg` is and `T`'s arithmetic is. No allocation, no recursion. On
/// return `x` is the best point reached (the last accepted iterate), and the result's value and gradient norm are
/// at `x`. Per iteration O(Mn) besides the evaluations. The history's storage must outlive it.
///
/// **Errors.** Nothing throws unless `fg` does. The outcome is an `lbfgs_status`: `gradient_converged`,
/// `value_converged` (`|fₖ - fₖ₊₁| ≤ value_tolerance · max(|fₖ|, |fₖ₊₁|, 1)`, when that tolerance is positive),
/// `iteration_limit`, `line_search_failed` (no decrease along `-∇f`: usually the gradient tolerance is below what
/// rounding allows, or the gradient is wrong), `invalid_input` (`work` too short; `x` untouched).
///
/// **Performance.** Not benchmarked: the cost is the objective's evaluations, which the result counts.
///
/// **Tests and demos.** `tests/optim/lbfgs_test.cpp`, `demos/optim/lbfgs_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<cstdint>`, `<limits>`, `<span>`, `<type_traits>`, `linalg/types.hpp`,
/// `math/constexpr_math.hpp`, `optim/line_search.hpp`.
///
/// **Known issues.** `M` between 3 and 20 is usual; beyond that the O(Mn) work grows without much gain. The gradient
/// tolerance is absolute: scale the objective, or the variables, so that 1e-6 means something.
///
/// **Status.** Experimental, new in 0.21.0.

#ifndef snicholls_core_optim_lbfgs_hpp
#define snicholls_core_optim_lbfgs_hpp

#include <snicholls/core/linalg/types.hpp>
#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/optim/line_search.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

namespace snicholls::optim {

    /// \brief The last `M` curvature pairs `(s, y)` of L-BFGS, in a ring buffer: over the caller's storage when `N` is
    /// `std::dynamic_extent`, inside when `N` is the dimension.
    /// \ingroup core
    /// \tparam T The value type.
    /// \tparam M The most pairs kept, at least 1.
    /// \tparam N The dimension if fixed at compile time, else `std::dynamic_extent`.
    template <linalg::scalar T, std::size_t M, std::size_t N = std::dynamic_extent>
    class lbfgs_history {
        static_assert(M > 0, "snicholls::optim::lbfgs_history: keep at least one pair");
        using storage = std::conditional_t<N == std::dynamic_extent, std::span<T>, std::array<T, (N == std::dynamic_extent ? 1 : N * M)>>;

    public:
        static constexpr std::size_t capacity = M;   ///< The most pairs kept.

        /// \brief An empty history of dimension `N`, its storage inside.
        constexpr lbfgs_history() noexcept
            requires(N != std::dynamic_extent)
        = default;

        /// \brief An empty history of dimension `n` over the caller's storage.
        /// \param s_storage At least `n · M` values for the `s` vectors.
        /// \param y_storage At least `n · M` values for the `y` vectors.
        /// \param n The dimension.
        /// \pre Both spans are at least `n · M` long and do not overlap.
        constexpr lbfgs_history(std::span<T> s_storage, std::span<T> y_storage, std::size_t n) noexcept
            requires(N == std::dynamic_extent)
            : s_(s_storage), y_(y_storage), n_(n)
        {
        }

        /// \brief The dimension. \return `n`.
        [[nodiscard]] constexpr std::size_t dimension() const noexcept { return n_; }
        /// \brief The pairs held. \return Between 0 and `M`.
        [[nodiscard]] constexpr std::size_t size() const noexcept { return count_; }
        /// \brief Whether no pair is held. \return `size() == 0`.
        [[nodiscard]] constexpr bool empty() const noexcept { return count_ == 0; }
        /// \brief Forgets every pair.
        constexpr void clear() noexcept
        {
            count_ = 0;
            head_ = 0;
            gamma_ = T(1);
        }
        /// \brief The initial inverse Hessian's scale `γ = sᵀy / yᵀy` of the newest pair. \return It, or 1 when empty.
        [[nodiscard]] constexpr T gamma() const noexcept { return gamma_; }

        /// \brief Pair `k`'s `s`, oldest first. \param k In `[0, size())`. \return Its `n` values.
        [[nodiscard]] constexpr std::span<const T> s(std::size_t k) const noexcept { return std::span<const T>(s_).subspan(slot(k) * n_, n_); }
        /// \brief Pair `k`'s `y`, oldest first. \param k In `[0, size())`. \return Its `n` values.
        [[nodiscard]] constexpr std::span<const T> y(std::size_t k) const noexcept { return std::span<const T>(y_).subspan(slot(k) * n_, n_); }
        /// \brief Pair `k`'s `ρ = 1 / yᵀs`, oldest first. \param k In `[0, size())`. \return It.
        [[nodiscard]] constexpr T rho(std::size_t k) const noexcept { return rho_[slot(k)]; }

        /// \brief Adds a pair, dropping the oldest when full, unless its curvature `sᵀy` is at most `ε yᵀy`.
        /// \param s `x_new - x_old`, `n` values.
        /// \param y `∇f_new - ∇f_old`, `n` values.
        /// \return True if the pair was added.
        constexpr bool push(std::span<const T> s, std::span<const T> y) noexcept(linalg::detail::nothrow_scalar<T>)
        {
            T sy(0), yy(0);
            for (std::size_t i = 0; i < n_; ++i) {
                sy += s[i] * y[i];
                yy += y[i] * y[i];
            }
            if (!(sy > std::numeric_limits<T>::epsilon() * yy) || !(yy > T(0)))
                return false;
            const std::size_t at = head_;
            for (std::size_t i = 0; i < n_; ++i) {
                s_[at * n_ + i] = s[i];
                y_[at * n_ + i] = y[i];
            }
            rho_[at] = T(1) / sy;
            gamma_ = sy / yy;
            head_ = (head_ + 1) % M;
            count_ = count_ < M ? count_ + 1 : M;
            return true;
        }

    private:
        // Pair k, oldest first, is in slot (head - count + k) mod M
        [[nodiscard]] constexpr std::size_t slot(std::size_t k) const noexcept { return (head_ + M - count_ + k) % M; }

        storage s_{};
        storage y_{};
        std::array<T, M> rho_{};
        std::size_t n_ = N == std::dynamic_extent ? 0 : N;
        std::size_t count_ = 0;
        std::size_t head_ = 0;
        T gamma_ = T(1);
    };

    /// \brief `d = -H g` by the two-loop recursion, `H` the L-BFGS inverse Hessian of the history with `H₀ = γ I`.
    /// \ingroup core
    /// \tparam T The value type.
    /// \tparam M The history's capacity.
    /// \tparam N The history's fixed dimension, or `std::dynamic_extent`.
    /// \param history The pairs.
    /// \param gradient `g`, `n` values.
    /// \param direction Receives `d`, `n` values (not aliasing `gradient`).
    template <linalg::scalar T, std::size_t M, std::size_t N>
    constexpr void two_loop(const lbfgs_history<T, M, N>& history, std::type_identity_t<std::span<const T>> gradient, std::type_identity_t<std::span<T>> direction) noexcept(
        linalg::detail::nothrow_scalar<T>)
    {
        const std::size_t n = history.dimension(), m = history.size();
        std::array<T, M> alpha{};
        for (std::size_t i = 0; i < n; ++i)
            direction[i] = gradient[i];
        for (std::size_t k = m; k-- > 0;) {
            const auto s = history.s(k), y = history.y(k);
            T a(0);
            for (std::size_t i = 0; i < n; ++i)
                a += s[i] * direction[i];
            a *= history.rho(k);
            alpha[k] = a;
            for (std::size_t i = 0; i < n; ++i)
                direction[i] -= a * y[i];
        }
        const T gamma = history.gamma();
        for (std::size_t i = 0; i < n; ++i)
            direction[i] *= gamma;
        for (std::size_t k = 0; k < m; ++k) {
            const auto s = history.s(k), y = history.y(k);
            T b(0);
            for (std::size_t i = 0; i < n; ++i)
                b += y[i] * direction[i];
            b *= history.rho(k);
            for (std::size_t i = 0; i < n; ++i)
                direction[i] += s[i] * (alpha[k] - b);
        }
        for (std::size_t i = 0; i < n; ++i)
            direction[i] = -direction[i];
    }

    /// \brief How `lbfgs_minimize` ended.
    /// \ingroup core
    enum class lbfgs_status : std::uint8_t {
        gradient_converged = 0,  ///< `‖∇f‖∞ ≤ gradient_tolerance`.
        value_converged = 1,     ///< The relative change in `f` fell to `value_tolerance`.
        iteration_limit = 2,     ///< `max_iterations` ran out.
        line_search_failed = 3,  ///< No decrease even along `-∇f`.
        invalid_input = 4,       ///< `work` too short; `x` untouched.
    };

    /// \brief The settings of `lbfgs_minimize`.
    /// \ingroup core
    /// \tparam T The value type.
    template <linalg::scalar T>
    struct lbfgs_options {
        T gradient_tolerance = T(1e-6);    ///< Stop when `‖∇f‖∞` is at most this.
        T value_tolerance = T(0);          ///< Stop when `|Δf| ≤ value_tolerance · max(|f|, |f_new|, 1)`; 0 for never.
        std::size_t max_iterations = 1000; ///< The most iterations (accepted steps).
        wolfe_options<T> line_search{};    ///< The line search's settings (its `initial_step` is set per iteration).
    };

    /// \brief The outcome of `lbfgs_minimize`.
    /// \ingroup core
    /// \tparam T The value type.
    template <linalg::scalar T>
    struct lbfgs_result {
        lbfgs_status status = lbfgs_status::invalid_input;  ///< How it ended.
        std::size_t iterations = 0;                         ///< The accepted steps.
        std::size_t evaluations = 0;                        ///< The evaluations of `fg`.
        T value = T(0);                                     ///< `f(x)` at the returned `x`.
        T gradient_norm = T(0);                             ///< `‖∇f(x)‖∞` at the returned `x`.

        /// \brief Whether a tolerance was met.
        /// \return `status` is `gradient_converged` or `value_converged`.
        [[nodiscard]] constexpr bool converged() const noexcept { return status == lbfgs_status::gradient_converged || status == lbfgs_status::value_converged; }
    };

    /// \brief The working storage `lbfgs_minimize<M>` needs for `n` variables: `(4 + 2M) n` values.
    /// \ingroup core
    /// \param n The variables.
    /// \param m The history's capacity `M`.
    /// \return `(4 + 2m) n`.
    [[nodiscard]] constexpr std::size_t lbfgs_workspace(std::size_t n, std::size_t m) noexcept { return (4 + 2 * m) * n; }

    /// \brief Minimises a smooth `f` by L-BFGS with a strong-Wolfe line search, from `x`.
    /// \ingroup core
    /// \tparam M The pairs kept.
    /// \tparam F The objective with its gradient, `T(std::span<const T> x, std::span<T> g)`.
    /// \tparam T The value type.
    /// \param fg The objective: returns `f(x)`, writes `∇f(x)` into `g`.
    /// \param x The starting point on entry, the best point reached on exit; `n` values.
    /// \param options The tolerances, the iteration limit and the line search's settings.
    /// \param work At least `lbfgs_workspace(n, M)` values, not aliasing `x`.
    /// \return The status, the iterations, the evaluations, and `f` and `‖∇f‖∞` at `x`.
    template <std::size_t M, typename F, linalg::scalar T>
    [[nodiscard]] constexpr lbfgs_result<T> lbfgs_minimize(F&& fg, std::span<T> x, const std::type_identity_t<lbfgs_options<T>>& options, std::type_identity_t<std::span<T>> work) noexcept(
        std::is_nothrow_invocable_v<F&, std::span<const T>, std::span<T>> && linalg::detail::nothrow_scalar<T>)
    {
        using cspan = std::span<const T>;
        const std::size_t n = x.size();
        if (work.size() < lbfgs_workspace(n, M))
            return {};
        const auto g = work.subspan(0, n), d = work.subspan(n, n), xt = work.subspan(2 * n, n), gt = work.subspan(3 * n, n);
        lbfgs_history<T, M> history(work.subspan(4 * n, n * M), work.subspan((4 + M) * n, n * M), n);
        const auto inf_norm = [&](std::span<const T> v) {
            T m(0);
            for (const T e : v)
                m = detail::ls_abs(e) > m ? detail::ls_abs(e) : m;
            return m;
        };
        T f = fg(cspan(x), g);
        std::size_t evaluations = 1, k = 0;
        T gnorm = inf_norm(cspan(g));
        for (;;) {
            if (gnorm <= options.gradient_tolerance)
                return {lbfgs_status::gradient_converged, k, evaluations, f, gnorm};
            if (k >= options.max_iterations)
                return {lbfgs_status::iteration_limit, k, evaluations, f, gnorm};
            wolfe_result<T> ls{};
            for (int attempt = 0; attempt < 2; ++attempt) {
                two_loop(history, cspan(g), d);
                T gd(0), gg(0);
                for (std::size_t i = 0; i < n; ++i) {
                    gd += g[i] * d[i];
                    gg += g[i] * g[i];
                }
                if (!(gd < T(0))) {
                    history.clear();
                    for (std::size_t i = 0; i < n; ++i)
                        d[i] = -g[i];
                    gd = -gg;
                }
                wolfe_options<T> lo = options.line_search;
                if (history.empty()) {
                    const T inv = T(1) / static_cast<T>(math::sqrt(gg));
                    lo.initial_step = inv < T(1) ? inv : T(1);
                } else
                    lo.initial_step = T(1);
                ls = strong_wolfe(fg, cspan(x), cspan(d), f, gd, xt, gt, lo);
                evaluations += ls.evaluations;
                if (ls.step > T(0) && ls.value < f)
                    break;
                if (history.empty())
                    return {lbfgs_status::line_search_failed, k, evaluations, f, gnorm};
                history.clear();
            }
            if (!(ls.step > T(0) && ls.value < f))
                return {lbfgs_status::line_search_failed, k, evaluations, f, gnorm};
            // s = x_new - x into d, y = g_new - g into g, then move
            for (std::size_t i = 0; i < n; ++i) {
                d[i] = xt[i] - x[i];
                g[i] = gt[i] - g[i];
            }
            history.push(cspan(d), cspan(g));
            for (std::size_t i = 0; i < n; ++i) {
                x[i] = xt[i];
                g[i] = gt[i];
            }
            const T f_old = f;
            f = ls.value;
            gnorm = inf_norm(cspan(g));
            ++k;
            if (options.value_tolerance > T(0)) {
                T scale = detail::ls_abs(f_old) > detail::ls_abs(f) ? detail::ls_abs(f_old) : detail::ls_abs(f);
                scale = scale > T(1) ? scale : T(1);
                if (detail::ls_abs(f_old - f) <= options.value_tolerance * scale && !(gnorm <= options.gradient_tolerance))
                    return {lbfgs_status::value_converged, k, evaluations, f, gnorm};
            }
        }
    }

} // namespace snicholls::optim

#endif /* snicholls_core_optim_lbfgs_hpp */
