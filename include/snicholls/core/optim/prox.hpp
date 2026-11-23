//
//  prox.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::optim`: proximal operators `prox_{λf}(v) = argmin_x λ f(x) + ½‖x − v‖²` in place, Moreau envelopes, and the prox of a conjugate by Moreau's identity.
/// \ingroup core
///
/// **Overview.** Proximal gradient (ISTA, FISTA), ADMM and the primal-dual methods take a step by the prox of each
/// nonsmooth term. These are the common ones, each in place on a `std::span` (the input `v` becomes `prox_{λf}(v)`),
/// with `λ > 0`:
///
/// | f | Function | `prox_{λf}(v)` |
/// |---|---|---|
/// | `‖x‖₁` | `prox_l1` | soft threshold `sign(v) max(|v| − λ, 0)` |
/// | `‖x‖₂` | `prox_l2` | block soft threshold `max(1 − λ/‖v‖, 0) v` |
/// | `Σ_g ‖x_g‖₂`, consecutive groups | `prox_group_l2` | `prox_l2` per group |
/// | `‖x‖∞` | `prox_linf` | `v − P_{‖·‖₁ ≤ λ}(v)` (Moreau), i.e. `clamp(v, −τ, τ)` |
/// | `Σ huber(xᵢ, M)` (CVXPY: `x²` for `|x| ≤ M`, `2M|x| − M²` beyond) | `prox_huber` | `v/(1 + 2λ)` or `v − 2λM sign(v)` |
/// | `‖x‖₂²` | `prox_squared_l2` | `v / (1 + 2λ)` |
/// | indicator of `[l, u]`, of `x ≥ 0` | `prox_box`, `prox_nonnegative` | the projection |
/// | `‖x‖₁ + (μ/2)‖x‖₂²` | `prox_elastic_net` | `soft(v, λ) / (1 + λμ)` |
/// | `Σ xᵢ log xᵢ` | `prox_negative_entropy` | `λ W₀(e^{v/λ − 1} / λ)` |
/// | `−Σ log xᵢ` | `prox_log_barrier` | `(v + √(v² + 4λ)) / 2` |
/// | `Σ ½dᵢxᵢ² + cᵢxᵢ`, `dᵢ ≥ 0` | `prox_quadratic` | `(v − λc) / (1 + λd)` |
///
/// `moreau_envelope_*` gives `M_{λf}(v) = min_x f(x) + ‖x − v‖²/(2λ)` where its closed form is cheap (ℓ1, ℓ2,
/// squared ℓ2, Huber, a box, the orthant). `prox_conjugate(prox_f, v, λ, workspace)` gives
/// `prox_{λf*}(v) = v − λ prox_{f/λ}(v/λ)` from any prox of `f`.
///
/// \snippet optim/prox_demo.cpp prox
///
/// **Relationships.** `optim/cones.hpp` (the projections, which are the prox of indicators; the ℓ1-ball projection
/// behind `prox_linf`). Group: \ref core.
///
/// **Implementations.** Closed forms. The ℓ∞ prox finds the ℓ1-ball threshold by Condat's algorithm with a
/// workspace, or Michelot's without. The negative-entropy prox computes `W₀(e^a)` directly as the root of
/// `w + log w = a` by Halley's iteration (from `a − log a` above 1, `e^a / (1 + e^a)` below), so no `e^a` is formed and
/// `v/λ` up to the type's range works; below `e^a < ε` it returns `e^a`. The log barrier's root avoids cancellation
/// for negative `v`.
///
/// **Guarantees.** Every function is `constexpr` and `noexcept` for the standard floating-point types; no allocation
/// (`prox_conjugate` and the Condat form of `prox_linf` take the caller's workspace); O(n), and the ℓ∞ prox expected
/// O(n). `prox_conjugate` is `noexcept` when the given prox is.
///
/// **Errors.** `optim::error::dimension_mismatch` when group sizes do not add up, or bounds or coefficients differ in
/// length from `v`; `workspace_too_small`. Nothing throws. `λ ≤ 0` is a precondition violation, not checked.
///
/// **Performance.** Not benchmarked.
///
/// **Tests and demos.** `tests/optim/prox_test.cpp`, `demos/optim/prox_demo.cpp` (a LASSO by ISTA, with its KKT
/// conditions checked).
///
/// **Dependencies.** `<cstddef>`, `<limits>`, `<span>`, `<type_traits>`, `<utility>`, `optim/cones.hpp`, `math/constexpr_math.hpp`, `result.hpp`.
///
/// **Known issues.** The negative-entropy prox evaluates `W₀(eᵃ)` in log form (Wright's ω), so it never overflows;
/// `math::lambert_w0` (`math/special.hpp`) is the general function. The negative-entropy prox underflows towards 0 (a subnormal, then 0) for `v/λ` below about `−700`
/// in double, where the exact answer is below the least normal number.
///
/// **Status.** Experimental, new in 0.21.0.

#ifndef snicholls_core_optim_prox_hpp
#define snicholls_core_optim_prox_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/optim/cones.hpp>
#include <snicholls/core/result.hpp>

#include <cstddef>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>

namespace snicholls::optim {

    namespace detail::prox {
        template <typename T>
        inline constexpr bool nt = linalg::detail::nothrow_scalar<T>;

        template <typename T>
        constexpr T soft(T v, T t) { return v > t ? v - t : v < -t ? v + t : T(0); }

        // W0(e^a): the w > 0 with w + log w = a, by Halley's iteration on f(w) = w + log w - a
        // (f' = 1 + 1/w, f'' = -1/w^2).
        template <typename T>
        constexpr T w0_exp(T a)
        {
            if (a != a)
                return a;
            if (a == std::numeric_limits<T>::infinity())
                return a;
            if (a < T(1)) {
                const T e = math::exp(a);
                if (e < std::numeric_limits<T>::epsilon())
                    return e * (T(1) - e);   // W(x) = x - x^2 + ...
            }
            T w = a >= T(1) ? a - math::log(a) : math::exp(a) / (T(1) + math::exp(a));
            for (int i = 0; i < 50; ++i) {
                const T f = w + math::log(w) - a, d1 = T(1) + T(1) / w, d2 = -T(1) / (w * w);
                const T step = T(2) * f * d1 / (T(2) * d1 * d1 - f * d2);
                T next = w - step;
                if (!(next > T(0)))
                    next = w / T(2);
                if (math::abs(next - w) <= T(4) * std::numeric_limits<T>::epsilon() * next)
                    return next;
                w = next;
            }
            return w;
        }

        template <typename T>
        constexpr T huber(T x, T m) { return math::abs(x) <= m ? x * x : T(2) * m * math::abs(x) - m * m; }

        template <typename T>
        constexpr T huber_prox(T v, T lambda, T m)
        {
            if (math::abs(v) <= m * (T(1) + T(2) * lambda))
                return v / (T(1) + T(2) * lambda);
            return v > T(0) ? v - T(2) * lambda * m : v + T(2) * lambda * m;
        }
    } // namespace detail::prox

    /// \brief The prox of `λ‖x‖₁`: the soft threshold `vᵢ ← sign(vᵢ) max(|vᵢ| − λ, 0)`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param v The point, overwritten.
    /// \param lambda λ, positive.
    template <linalg::scalar T>
    constexpr void prox_l1(std::span<T> v, std::type_identity_t<T> lambda) noexcept(detail::prox::nt<T>)
    {
        for (T& x : v)
            x = detail::prox::soft(x, lambda);
    }

    /// \brief The prox of `λ‖x‖₂` (the norm, not its square): the block soft threshold `v ← max(1 − λ/‖v‖, 0) v`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param v The point, overwritten.
    /// \param lambda λ, positive.
    template <linalg::scalar T>
    constexpr void prox_l2(std::span<T> v, std::type_identity_t<T> lambda) noexcept(detail::prox::nt<T>)
    {
        const T n = detail::cones::norm(std::span<const T>(v));
        const T k = n > lambda ? T(1) - lambda / n : T(0);
        for (T& x : v)
            x *= k;
    }

    /// \brief The prox of the group lasso `λ Σ_g ‖x_g‖₂` over consecutive groups: `prox_l2` on each.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param v The point, overwritten.
    /// \param sizes The group sizes, in order, summing to `v.size()`.
    /// \param lambda λ, positive.
    /// \return Nothing, or `error::dimension_mismatch` (and `v` unchanged).
    template <linalg::scalar T>
    constexpr result<void, error> prox_group_l2(std::span<T> v, std::span<const std::size_t> sizes, std::type_identity_t<T> lambda) noexcept(detail::prox::nt<T>)
    {
        std::size_t total = 0;
        for (const std::size_t s : sizes)
            total += s;
        if (total != v.size())
            return err(error::dimension_mismatch);
        std::size_t offset = 0;
        for (const std::size_t s : sizes) {
            prox_l2(v.subspan(offset, s), lambda);
            offset += s;
        }
        return ok();
    }

    /// \brief The prox of `λ‖x‖∞`, by Moreau's decomposition: `v − P_{‖·‖₁ ≤ λ}(v)`, which is `clamp(v, −τ, τ)` for the ℓ1 ball's threshold `τ` (Condat's algorithm).
    /// \ingroup core
    /// \tparam T The number type.
    /// \param v The point, overwritten (zero when `‖v‖₁ ≤ λ`).
    /// \param lambda λ, positive.
    /// \param workspace At least `v.size()` elements, overwritten.
    /// \return Nothing, or `error::workspace_too_small`.
    template <linalg::scalar T>
    constexpr result<void, error> prox_linf(std::span<T> v, std::type_identity_t<T> lambda, std::span<T> workspace) noexcept(detail::prox::nt<T>)
    {
        if (workspace.size() < v.size())
            return err(error::workspace_too_small);
        T s(0);
        for (const T& x : v)
            s += math::abs(x);
        if (s <= lambda) {
            project_zero(v);
            return ok();
        }
        const T tau = detail::cones::condat_tau(v.size(), [&](std::size_t i) { return math::abs(v[i]); }, lambda, workspace);
        project_box(v, -tau, tau);
        return ok();
    }

    /// \brief The prox of `λ‖x‖∞` without a workspace (Michelot's threshold).
    /// \ingroup core
    /// \tparam T The number type.
    /// \param v The point, overwritten.
    /// \param lambda λ, positive.
    template <linalg::scalar T>
    constexpr void prox_linf(std::span<T> v, std::type_identity_t<T> lambda) noexcept(detail::prox::nt<T>)
    {
        T s(0);
        for (const T& x : v)
            s += math::abs(x);
        if (s <= lambda) {
            project_zero(v);
            return;
        }
        const T tau = detail::cones::michelot_tau(v.size(), [&](std::size_t i) { return math::abs(v[i]); }, lambda);
        project_box(v, -tau, tau);
    }

    /// \brief The prox of `λ Σ huber(xᵢ, M)`, CVXPY's Huber (`x²` for `|x| ≤ M`, `2M|x| − M²` beyond).
    /// \ingroup core
    /// \tparam T The number type.
    /// \param v The point, overwritten by `v/(1 + 2λ)` where `|v| ≤ M(1 + 2λ)` and by `v − 2λM sign(v)` elsewhere.
    /// \param lambda λ, positive.
    /// \param m M, positive (CVXPY's default is 1).
    template <linalg::scalar T>
    constexpr void prox_huber(std::span<T> v, std::type_identity_t<T> lambda, std::type_identity_t<T> m = T(1)) noexcept(detail::prox::nt<T>)
    {
        for (T& x : v)
            x = detail::prox::huber_prox(x, lambda, m);
    }

    /// \brief The prox of `λ‖x‖₂²`: `v ← v / (1 + 2λ)`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param v The point, overwritten.
    /// \param lambda λ, positive.
    template <linalg::scalar T>
    constexpr void prox_squared_l2(std::span<T> v, std::type_identity_t<T> lambda) noexcept(detail::prox::nt<T>)
    {
        for (T& x : v)
            x /= T(1) + T(2) * lambda;
    }

    /// \brief The prox of the indicator of the box `[l, u]` (any λ): the projection, infinite bounds allowed.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param v The point, overwritten.
    /// \param l The lower bounds.
    /// \param u The upper bounds.
    /// \return Nothing, or `error::dimension_mismatch`.
    template <linalg::scalar T>
    constexpr result<void, error> prox_box(std::span<T> v, std::span<const std::type_identity_t<T>> l, std::span<const std::type_identity_t<T>> u) noexcept(detail::prox::nt<T>)
    {
        return project_box(v, l, u);
    }

    /// \brief The prox of the indicator of the non-negative orthant (any λ): `vᵢ ← max(vᵢ, 0)`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param v The point, overwritten.
    template <linalg::scalar T>
    constexpr void prox_nonnegative(std::span<T> v) noexcept(detail::prox::nt<T>)
    {
        project_nonnegative(v);
    }

    /// \brief The prox of the elastic net `λ(‖x‖₁ + (μ/2)‖x‖₂²)`: `soft(v, λ) / (1 + λμ)`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param v The point, overwritten.
    /// \param lambda λ, positive.
    /// \param mu μ, the ridge weight, at least 0.
    template <linalg::scalar T>
    constexpr void prox_elastic_net(std::span<T> v, std::type_identity_t<T> lambda, std::type_identity_t<T> mu) noexcept(detail::prox::nt<T>)
    {
        for (T& x : v)
            x = detail::prox::soft(x, lambda) / (T(1) + lambda * mu);
    }

    /// \brief The prox of the negative entropy `λ Σ xᵢ log xᵢ`: `xᵢ = λ W₀(e^{vᵢ/λ − 1} / λ)`, always positive.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param v The point, overwritten.
    /// \param lambda λ, positive.
    ///
    /// From the optimality condition `λ(log x + 1) + x − v = 0`. `W₀` of the exponential is computed without forming
    /// it, so a large `v/λ` does not overflow.
    template <linalg::scalar T>
    constexpr void prox_negative_entropy(std::span<T> v, std::type_identity_t<T> lambda) noexcept(detail::prox::nt<T>)
    {
        const T shift = T(1) + math::log(lambda);
        for (T& x : v)
            x = lambda * detail::prox::w0_exp(x / lambda - shift);
    }

    /// \brief The prox of the log barrier `−λ Σ log xᵢ`: the positive root `(v + √(v² + 4λ)) / 2`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param v The point, overwritten.
    /// \param lambda λ, positive.
    template <linalg::scalar T>
    constexpr void prox_log_barrier(std::span<T> v, std::type_identity_t<T> lambda) noexcept(detail::prox::nt<T>)
    {
        for (T& x : v) {
            const T root = math::sqrt(x * x + T(4) * lambda);
            x = x >= T(0) ? (x + root) / T(2) : T(2) * lambda / (root - x);
        }
    }

    /// \brief The prox of the separable quadratic `λ Σ (½dᵢxᵢ² + cᵢxᵢ)`: `(v − λc) / (1 + λd)`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param v The point, overwritten.
    /// \param d The curvatures, at least 0.
    /// \param c The linear coefficients.
    /// \param lambda λ, positive.
    /// \return Nothing, or `error::dimension_mismatch`.
    template <linalg::scalar T>
    constexpr result<void, error> prox_quadratic(std::span<T> v, std::span<const std::type_identity_t<T>> d, std::span<const std::type_identity_t<T>> c, std::type_identity_t<T> lambda) noexcept(detail::prox::nt<T>)
    {
        if (d.size() != v.size() || c.size() != v.size())
            return err(error::dimension_mismatch);
        for (std::size_t i = 0; i < v.size(); ++i)
            v[i] = (v[i] - lambda * c[i]) / (T(1) + lambda * d[i]);
        return ok();
    }

    // ---- Moreau envelopes ----

    /// \brief The Moreau envelope of `‖x‖₁`: `Σ` of the Huber function `v²/(2λ)` for `|v| ≤ λ`, `|v| − λ/2` beyond.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param v The point.
    /// \param lambda λ, positive.
    /// \return `M_{λ‖·‖₁}(v)`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr T moreau_envelope_l1(std::span<const std::type_identity_t<T>> v, T lambda) noexcept(detail::prox::nt<T>)
    {
        T s(0);
        for (const T& x : v)
            s += math::abs(x) <= lambda ? x * x / (T(2) * lambda) : math::abs(x) - lambda / T(2);
        return s;
    }

    /// \brief The Moreau envelope of `‖x‖₂`: `‖v‖²/(2λ)` for `‖v‖ ≤ λ`, `‖v‖ − λ/2` beyond.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param v The point.
    /// \param lambda λ, positive.
    /// \return `M_{λ‖·‖₂}(v)`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr T moreau_envelope_l2(std::span<const std::type_identity_t<T>> v, T lambda) noexcept(detail::prox::nt<T>)
    {
        const T n = detail::cones::norm(v);
        return n <= lambda ? n * n / (T(2) * lambda) : n - lambda / T(2);
    }

    /// \brief The Moreau envelope of `‖x‖₂²`: `‖v‖² / (1 + 2λ)`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param v The point.
    /// \param lambda λ, positive.
    /// \return `M_{λ‖·‖₂²}(v)`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr T moreau_envelope_squared_l2(std::span<const std::type_identity_t<T>> v, T lambda) noexcept(detail::prox::nt<T>)
    {
        T s(0);
        for (const T& x : v)
            s += x * x;
        return s / (T(1) + T(2) * lambda);
    }

    /// \brief The Moreau envelope of CVXPY's `Σ huber(xᵢ, M)`: `huber(p) + (p − v)²/(2λ)` summed, `p` its prox.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param v The point.
    /// \param lambda λ, positive.
    /// \param m M, positive.
    /// \return `M_{λ huber}(v)`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr T moreau_envelope_huber(std::span<const std::type_identity_t<T>> v, T lambda, T m = T(1)) noexcept(detail::prox::nt<T>)
    {
        T s(0);
        for (const T& x : v) {
            const T p = detail::prox::huber_prox(x, lambda, m);
            s += detail::prox::huber(p, m) + (p - x) * (p - x) / (T(2) * lambda);
        }
        return s;
    }

    /// \brief The Moreau envelope of the indicator of a box: the squared distance to it over `2λ`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param v The point.
    /// \param l The lower bounds (as long as `v`).
    /// \param u The upper bounds (as long as `v`).
    /// \param lambda λ, positive.
    /// \return `dist(v, [l, u])² / (2λ)`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr T moreau_envelope_box(std::span<const std::type_identity_t<T>> v, std::span<const std::type_identity_t<T>> l, std::span<const std::type_identity_t<T>> u, T lambda) noexcept(detail::prox::nt<T>)
    {
        T s(0);
        for (std::size_t i = 0; i < v.size(); ++i) {
            const T d = v[i] < l[i] ? l[i] - v[i] : u[i] < v[i] ? v[i] - u[i] : T(0);
            s += d * d;
        }
        return s / (T(2) * lambda);
    }

    /// \brief The Moreau envelope of the indicator of the non-negative orthant: `Σ min(vᵢ, 0)² / (2λ)`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param v The point.
    /// \param lambda λ, positive.
    /// \return `dist(v, R₊ⁿ)² / (2λ)`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr T moreau_envelope_nonnegative(std::span<const std::type_identity_t<T>> v, T lambda) noexcept(detail::prox::nt<T>)
    {
        T s(0);
        for (const T& x : v)
            s += x < T(0) ? x * x : T(0);
        return s / (T(2) * lambda);
    }

    // ---- Conjugates ----

    /// \brief The prox of the convex conjugate, by Moreau's identity: `prox_{λf*}(v) = v − λ prox_{f/λ}(v/λ)`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \tparam Prox A callable `prox(std::span<T> x, T t)` that overwrites `x` with `prox_{t f}(x)`.
    /// \param prox_f The prox of `f`.
    /// \param v The point, overwritten by `prox_{λf*}(v)`.
    /// \param lambda λ, positive.
    /// \param workspace At least `v.size()` elements, overwritten.
    /// \return Nothing, or `error::workspace_too_small`.
    template <linalg::scalar T, typename Prox>
    constexpr result<void, error> prox_conjugate(Prox&& prox_f, std::span<T> v, std::type_identity_t<T> lambda, std::span<T> workspace) noexcept(
        noexcept(prox_f(std::declval<std::span<T>>(), std::declval<T>())) && detail::prox::nt<T>)
    {
        if (workspace.size() < v.size())
            return err(error::workspace_too_small);
        const std::span<T> y = workspace.first(v.size());
        for (std::size_t i = 0; i < v.size(); ++i)
            y[i] = v[i] / lambda;
        prox_f(y, T(1) / lambda);
        for (std::size_t i = 0; i < v.size(); ++i)
            v[i] -= lambda * y[i];
        return ok();
    }

} // namespace snicholls::optim

#endif /* snicholls_core_optim_prox_hpp */
