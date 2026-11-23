//
//  cones.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::optim`: Euclidean projections onto the cones and sets of conic optimisation, their membership tests and dual cones, and Nesterov-Todd scaling.
/// \ingroup core
///
/// **Overview.** A first-order conic solver (OSQP's ADMM, SCS's ADMM on the homogeneous embedding, PDLP) spends
/// its iterations projecting onto a product of simple cones, and an interior-point solver (Clarabel's) scales by
/// the Nesterov-Todd point of each. These are those primitives, in place on a `std::span`:
///
/// | Set | Projection | Membership | Dual |
/// |---|---|---|---|
/// | zero `{0}` / free `Rⁿ` | `project_zero`, `project_free` | `in_zero_cone`, `in_free_cone` | each other |
/// | non-negative orthant | `project_nonnegative` | `in_nonnegative_cone` | self |
/// | box `[l, u]` (infinite bounds allowed) | `project_box` | `in_box` | not a cone |
/// | second-order cone `{(t, x) : ‖x‖ ≤ t}` | `project_second_order` | `in_second_order_cone` | self |
/// | rotated `{(u, v, w) : 2uv ≥ ‖w‖², u, v ≥ 0}` | `project_rotated_second_order` | `in_rotated_second_order_cone` | self |
/// | PSD, scaled svec | `project_psd`, `project_psd_fixed<N>` | `in_psd_cone`, `in_psd_cone_fixed<N>` | self |
/// | exponential `cl{(x, y, z) : y > 0, y e^{x/y} ≤ z}` | `project_exponential` | `in_exponential_cone` | `project_dual_exponential` |
/// | power `{(x, y, z) : x^α y^{1-α} ≥ |z|, x, y ≥ 0}` | `project_power` | `in_power_cone` | `project_dual_power` |
/// | simplex `{x ≥ 0, Σx = r}` | `project_simplex` | `in_simplex` | not a cone |
/// | ℓ1 ball, ℓ∞ ball, capped simplex | `project_l1_ball`, `project_linf_ball`, `project_capped_simplex` | `in_l1_ball`, ... | not cones |
///
/// A product of cones laid out one after another is described by an array of `cone_kind` (type, dimension,
/// parameter) and projected by `project(cones, x, workspace)`, onto its dual by `project_dual`, and tested by
/// `in_cone`. `nt_scaling_nonnegative` and `nt_scaling_second_order` compute the Nesterov-Todd scaling of a pair of
/// interior points, `nt_apply_*` apply W or W⁻¹ in place.
///
/// **Conventions.**
/// - PSD matrices are in *scaled svec* form, as SCS and Clarabel store them: the lower triangle column by column,
///   `(0,0), (1,0), ..., (n-1,0), (1,1), (2,1), ...`, each off-diagonal element times √2, so that
///   `svec(A) · svec(B) = tr(AB)` and the projection in svec space is the projection in the Frobenius norm. A matrix
///   (for `svec` / `smat`) is `n × n`, row-major. A `cone_kind` of PSD has `dimension = n(n+1)/2`.
/// - The second-order cone's first element is `t`; the rotated cone's first two are `u` and `v` (MOSEK's
///   convention, `2uv ≥ ‖w‖²`).
/// - Nesterov-Todd: for `s` and `z` in the interior of a self-dual cone, `W` is the symmetric positive definite
///   matrix with `W z = W⁻¹ s` (the scaled variable `λ`); equivalently `W² z = s`. For the orthant
///   `W = diag(w)`, `wᵢ = √(sᵢ / zᵢ)`, `λᵢ = √(sᵢ zᵢ)`. For the second-order cone, with `J = diag(1, -1, ..., -1)`,
///   `‖s‖_J = √(s₀² - ‖s₁‖²)`, `s̄ = s / ‖s‖_J`, `z̄ = z / ‖z‖_J`, `γ = √((1 + s̄ᵀz̄) / 2)`, the normalised scaling
///   point is `w̄ = (s̄ + J z̄) / (2γ)` (so `‖w̄‖_J = 1`), `η = √(‖s‖_J / ‖z‖_J)` and
///   `W = η [[w̄₀, w̄₁ᵀ], [w̄₁, I + w̄₁w̄₁ᵀ / (1 + w̄₀)]]`, `W⁻¹ = η⁻¹ [[w̄₀, -w̄₁ᵀ], [-w̄₁, I + w̄₁w̄₁ᵀ / (1 + w̄₀)]]`,
///   `W² = η² (2w̄w̄ᵀ - J)` (Vandenberghe's CVXOPT notes, ECOS and Clarabel use the same).
///
/// \snippet optim/cones_demo.cpp cones
///
/// **Relationships.** `optim/prox.hpp` (proximal operators; the ℓ∞ prox uses the ℓ1-ball projection here),
/// `linalg/fixed.hpp` (`symmetric_eigen` for `project_psd_fixed`), `linalg/reference.hpp` (the dense eigen
/// decomposition for run-time `n`), `math/constexpr_math.hpp`. Group: \ref core.
///
/// **Implementations.**
/// - Second-order cone: the closed form. Rotated: the orthogonal map `(u, v) ↦ ((u + v)/√2, (u − v)/√2)` to the
///   ordinary cone and back.
/// - PSD: eigen decomposition, then the sum over whichever of the positive or the negative eigenvalues is the shorter
///   (`X₊ = Σ_{λ>0} λ v vᵀ = X − Σ_{λ<0} λ v vᵀ`). A matrix already PSD is returned unchanged.
/// - Exponential cone: Friberg (2021), "Projection onto the exponential cone: a univariate root-finding problem".
///   After the analytic cases (in the cone, in the polar cone, `x ≤ 0` and `y ≤ 0`), the projection is
///   `s (ρ, 1, e^ρ)` where `ρ` is the root of `h(ρ) = ((ρ−1)x₀ + y₀) e^ρ − (x₀ − ρ y₀) e^{−ρ} − (ρ² − ρ + 1) z₀`
///   in the interval where both `(ρ−1)x₀ + y₀` and `x₀ − ρ y₀` are positive (both are linear in ρ, so the interval
///   is explicit; an unbounded end is bracketed by doubling). Any root there gives a Moreau decomposition, so it is
///   the projection. `h` is evaluated scaled by `e^{−|ρ|}`, so nothing overflows, and solved by Newton safeguarded by
///   bisection. The point is the projection onto the ray at the root, compared with Friberg's heuristic
///   `(min(x₀, 0), 0, max(z₀, 0))`, and the nearer kept. The input is scaled by its largest magnitude first (the
///   projection is positively homogeneous), so 1e±300 work as 1 does.
/// - Power cone: Hien (2015), "Differential properties of Euclidean projection onto power cone": the projection is
///   `(x(r), y(r), sign(z₀) r)` with `x(r) = (x₀ + √(x₀² + 4α r(|z₀| − r)))/2`, `y(r)` likewise with `1 − α`,
///   and `r ∈ (0, |z₀|)` the root of `x(r)^α y(r)^{1−α} = r`, by Newton safeguarded by bisection. `x` and `y` depend on
///   `r` through `r(|z₀| − r)`, so the root is sought as `r` or as `|z₀| − r`, whichever is below `|z₀|/2`: a projection
///   with `x` or `y` near 0 is then resolved to the type's precision, not to one ulp of `|z₀|`.
/// - Simplex and ℓ1 ball: Condat (2016), "Fast projection onto the simplex and the ℓ1 ball", expected O(n), with a
///   caller workspace of `n`; without a workspace, Michelot's (1986) method, O(n) a sweep and a few sweeps.
/// - Capped simplex: bisection on the threshold to adjacent floats, then the exact threshold of the final active set.
///
/// **Guarantees.** Every function is `constexpr` and `noexcept` for the standard floating-point types (exactly when
/// `T`'s arithmetic is). No allocation: the PSD projection takes a workspace of `psd_workspace_size(n)` elements, the
/// simplex projection one of `n`. Projections are idempotent up to rounding. A point already in a cone, the box or
/// the ℓ∞ ball is returned unchanged, bit for bit; one already in a simplex or a capped simplex to rounding. Complexity: O(n) for all but the PSD cone
/// (O(n³) per Jacobi sweep) and the capped simplex (O(n) per bisection step, at most 200).
///
/// **Errors.** `optim::error::dimension_mismatch` for spans of different lengths, an svec length that is not
/// triangular, an exponential or power cone of dimension other than 3, a rotated cone of dimension below 2;
/// `workspace_too_small`; `infeasible` for a negative radius or an empty capped simplex; `not_interior` from the
/// Nesterov-Todd scalings; `did_not_converge` from the eigen decomposition. Nothing throws.
///
/// **Performance.** Not benchmarked.
///
/// **Tests and demos.** `tests/optim/cones_test.cpp`, `demos/optim/cones_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<cstdint>`, `<limits>`, `<span>`, `<type_traits>`, `<utility>`,
/// `linalg/fixed.hpp`, `linalg/reference.hpp` (the whole reference backend, for the run-time PSD projection),
/// `math/constexpr_math.hpp`, `result.hpp`.
///
/// **Known issues.** Membership tests with a tolerance are by distance for the exponential and power cones (the
/// distance to the projection is at most `tol`) and by the defining inequality otherwise. The exponential and power
/// projections are accurate to a few units in the last place of the input's largest magnitude, not of each element:
/// an element 1e-20 times the largest is known only absolutely. The PSD projection's accuracy is the Jacobi
/// decomposition's, about `n ε ‖X‖`. The power cone needs `0 < α < 1`.
///
/// **Status.** Experimental.

#ifndef snicholls_core_optim_cones_hpp
#define snicholls_core_optim_cones_hpp

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/result.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>

namespace snicholls::optim {

    /// \brief Why a projection or a scaling could not be done. A reported value, never undefined behaviour.
    /// \ingroup core
    enum class error : std::uint8_t {
        dimension_mismatch = 1,   ///< Spans of different lengths, or a length the set does not have.
        workspace_too_small = 2,  ///< The caller's workspace is shorter than the operation needs.
        infeasible = 3,           ///< The set is empty (a negative radius, a capped simplex whose total cannot be met).
        not_interior = 4,         ///< A Nesterov-Todd scaling was asked of a point not strictly inside the cone.
        did_not_converge = 5,     ///< The eigen decomposition used its sweeps.
    };

    /// \brief The cones of `cone_kind`.
    /// \ingroup core
    enum class cone : std::uint8_t {
        zero = 0,                  ///< `{0}`.
        free = 1,                  ///< `Rⁿ`, the dual of the zero cone.
        nonnegative = 2,           ///< The non-negative orthant.
        second_order = 3,          ///< `{(t, x) : ‖x‖ ≤ t}`.
        rotated_second_order = 4,  ///< `{(u, v, w) : 2uv ≥ ‖w‖², u, v ≥ 0}`.
        psd = 5,                   ///< Positive semidefinite matrices, scaled svec; dimension `n(n+1)/2`.
        exponential = 6,           ///< `cl{(x, y, z) : y > 0, y e^{x/y} ≤ z}`; dimension 3.
        dual_exponential = 7,      ///< Its dual, `cl{(u, v, w) : u < 0, −u e^{v/u} ≤ e w}`; dimension 3.
        power = 8,                 ///< `{(x, y, z) : x^α y^{1−α} ≥ |z|, x, y ≥ 0}`, α the parameter; dimension 3.
        dual_power = 9,            ///< Its dual, `{(u, v, w) : (u/α)^α (v/(1−α))^{1−α} ≥ |w|, u, v ≥ 0}`; dimension 3.
    };

    /// \brief One cone of a product laid out consecutively: its type, its dimension, and α for a power cone.
    /// \ingroup core
    struct cone_kind {
        cone type = cone::zero;     ///< The cone.
        std::size_t dimension = 0;  ///< Its elements in the vector (for PSD the svec length).
        double parameter = 0;       ///< α in (0, 1) for `power` and `dual_power`; unused otherwise.

        /// \brief Member-wise equality. \return Whether type, dimension and parameter are equal.
        friend constexpr bool operator==(const cone_kind&, const cone_kind&) = default;
    };

    /// \brief The dual of a cone: zero and free swap, the exponential and power cones go to their duals, the rest are self-dual.
    /// \ingroup core
    /// \param c The cone.
    /// \return Its dual.
    [[nodiscard]] constexpr cone dual_of(cone c) noexcept
    {
        switch (c) {
        case cone::zero: return cone::free;
        case cone::free: return cone::zero;
        case cone::exponential: return cone::dual_exponential;
        case cone::dual_exponential: return cone::exponential;
        case cone::power: return cone::dual_power;
        case cone::dual_power: return cone::power;
        default: return c;
        }
    }

    namespace detail::cones {
        template <typename T>
        inline constexpr bool nt = linalg::detail::nothrow_scalar<T>;

        template <typename T>
        constexpr T min_(T a, T b) { return b < a ? b : a; }
        template <typename T>
        constexpr T max_(T a, T b) { return a < b ? b : a; }
        template <typename T>
        constexpr T inf() { return std::numeric_limits<T>::infinity(); }
        template <typename T>
        constexpr T sqrt2() { return T(1.4142135623730950488016887242096981L); }

        template <typename T>
        constexpr T norm(std::span<const T> x)
        {
            T s(0);
            for (const T& v : x)
                s += v * v;
            return math::sqrt(s);
        }

        template <typename T>
        constexpr T dist3(const std::array<T, 3>& a, T x, T y, T z) { return math::sqrt((a[0] - x) * (a[0] - x) + (a[1] - y) * (a[1] - y) + (a[2] - z) * (a[2] - z)); }

        template <typename T>
        constexpr bool finite3(const std::array<T, 3>& a) { return math::abs(a[0]) < inf<T>() && math::abs(a[1]) < inf<T>() && math::abs(a[2]) < inf<T>(); }

        // The order n of an svec length k = n(n+1)/2, or k + 1 when k is not triangular.
        constexpr std::size_t order_of(std::size_t k) noexcept
        {
            std::size_t n = 0;
            while (n * (n + 1) / 2 < k)
                ++n;
            return n * (n + 1) / 2 == k ? n : k + 1;
        }

        // A root of an increasing function on (a, b), f(x) returning {value, derivative}: Newton, kept inside the
        // bracket by bisection. f(a) <= 0 <= f(b) is assumed, never evaluated.
        template <typename T, typename F>
        constexpr T increasing_root(F f, T a, T b)
        {
            const T eps = std::numeric_limits<T>::epsilon();
            T x = a + (b - a) / T(2);
            for (int i = 0; i < 400; ++i) {
                const std::pair<T, T> g = f(x);
                if (g.first == T(0))
                    return x;
                if (g.first < T(0))
                    a = x;
                else
                    b = x;
                T next = x - g.first / g.second;
                if (!(next > a && next < b))
                    next = a + (b - a) / T(2);
                if (next == x || math::abs(next - x) <= T(2) * eps * math::abs(next) || !(b - a > T(2) * eps * max_(math::abs(a), math::abs(b))))
                    return next;
                x = next;
            }
            return x;
        }

        // ---- Exponential cone ----
        template <typename T>
        constexpr bool in_exp(T r, T s, T t)
        {
            if (s > T(0) && t > T(0))
                return r <= s * math::log(t / s);
            return s == T(0) && r <= T(0) && t >= T(0);
        }
        template <typename T>
        constexpr bool in_exp_dual(T u, T v, T w)
        {
            if (u < T(0) && w > T(0))
                return v >= u * (T(1) + math::log(w / -u));
            return u == T(0) && v >= T(0) && w >= T(0);
        }

        // The projection of (r0, s0, t0), its largest magnitude 1, onto the exponential cone (Friberg 2021).
        template <typename T>
        constexpr std::array<T, 3> exp_unit(T r0, T s0, T t0)
        {
            if (in_exp(r0, s0, t0))
                return {r0, s0, t0};
            if (in_exp_dual(-r0, -s0, -t0))
                return {T(0), T(0), T(0)};
            const std::array<T, 3> heuristic{min_(r0, T(0)), T(0), max_(t0, T(0))};
            if (r0 <= T(0) && s0 <= T(0))
                return heuristic;
            // The interval where (rho - 1) r0 + s0 > 0 and r0 - rho s0 > 0
            T lo = -inf<T>(), hi = inf<T>();
            if (r0 > T(0))
                lo = T(1) - s0 / r0;
            else if (r0 < T(0))
                hi = T(1) - s0 / r0;
            if (s0 > T(0))
                hi = min_(hi, r0 / s0);
            else if (s0 < T(0))
                lo = max_(lo, r0 / s0);
            // h(rho) e^{-|rho|} and its derivative
            auto g = [&](T p) {
                const T e = math::exp(-math::abs(p)), e2 = e * e;
                const T ep = p >= T(0) ? T(1) : e2, em = p >= T(0) ? e2 : T(1);
                const T a = (p - T(1)) * r0 + s0, b = r0 - p * s0, q = p * p - p + T(1);
                const T h = a * ep - b * em - q * t0 * e;
                const T dh = (p * r0 + s0) * ep + (r0 - (p - T(1)) * s0) * em - (T(2) * p - T(1)) * t0 * e;
                return std::pair<T, T>{h, p >= T(0) ? dh - h : dh + h};
            };
            // An unbounded end: step out, doubling, until h changes sign (it must: h e^{-|rho|} tends to a positive
            // value above and a negative one below). 60 doublings keep rho^2 finite in float.
            if (!(lo > -inf<T>())) {
                T step(1);
                lo = hi - step;
                for (int i = 0; i < 60 && !(g(lo).first < T(0)); ++i) {
                    step *= T(2);
                    lo = hi - step;
                }
            }
            if (!(hi < inf<T>())) {
                T step(1);
                hi = lo + step;
                for (int i = 0; i < 60 && !(g(hi).first > T(0)); ++i) {
                    step *= T(2);
                    hi = lo + step;
                }
            }
            const T rho = increasing_root(g, lo, hi);
            // The projection onto the ray (rho, 1, e^rho), normalised so that nothing overflows
            std::array<T, 3> ray{};
            if (rho > T(0)) {
                const T e = math::exp(-rho);
                ray = {rho * e, e, T(1)};
            } else {
                const T k = max_(-rho, T(1));
                ray = {rho / k, T(1) / k, math::exp(rho) / k};
            }
            const T scale = (r0 * ray[0] + s0 * ray[1] + t0 * ray[2]) / (ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]);
            const std::array<T, 3> candidate{scale * ray[0], scale * ray[1], scale * ray[2]};
            if (scale > T(0) && finite3(candidate) && dist3(candidate, r0, s0, t0) <= dist3(heuristic, r0, s0, t0))
                return candidate;
            return heuristic;
        }

        // ---- Power cone ----
        template <typename T>
        constexpr T pow_mean(T x, T y, T alpha) { return math::exp(alpha * math::log(x) + (T(1) - alpha) * math::log(y)); }

        template <typename T>
        constexpr bool in_pow(T x, T y, T z, T alpha)
        {
            if (x < T(0) || y < T(0))
                return false;
            if (x == T(0) || y == T(0))
                return z == T(0);
            return pow_mean(x, y, alpha) >= math::abs(z);
        }
        template <typename T>
        constexpr bool in_pow_dual(T u, T v, T w, T alpha)
        {
            if (u < T(0) || v < T(0))
                return false;
            if (u == T(0) || v == T(0))
                return w == T(0);
            return pow_mean(u / alpha, v / (T(1) - alpha), alpha) >= math::abs(w);
        }

        // (a + sqrt(a^2 + c)) / 2 for c >= 0, without cancellation when a < 0
        template <typename T>
        constexpr T half_root(T a, T c, T root)
        {
            return a >= T(0) ? (a + root) / T(2) : c / (T(2) * (root - a));
        }

        // The projection of (x0, y0, z0), its largest magnitude 1, onto the power cone (Hien 2015).
        template <typename T>
        constexpr std::array<T, 3> pow_unit(T x0, T y0, T z0, T alpha)
        {
            if (in_pow(x0, y0, z0, alpha))
                return {x0, y0, z0};
            if (in_pow_dual(-x0, -y0, -z0, alpha))
                return {T(0), T(0), T(0)};
            const std::array<T, 3> heuristic{max_(x0, T(0)), max_(y0, T(0)), T(0)};
            const T az = math::abs(z0);
            if (az == T(0))
                return heuristic;
            // x and y depend on r through r(|z0| - r), symmetric about |z0|/2. The root is found in t on (0, |z0|/2),
            // r = t below the middle and r = |z0| - t above it, so that a root next to either end (x or y near 0) is
            // resolved to the type's precision rather than to one ulp of |z0|.
            const T half = az / T(2);
            auto xy = [&](T t) {
                const T cx = T(4) * alpha * t * (az - t), cy = T(4) * (T(1) - alpha) * t * (az - t);
                const T rx = math::sqrt(x0 * x0 + cx), ry = math::sqrt(y0 * y0 + cy);
                return std::array<T, 4>{half_root(x0, cx, rx), half_root(y0, cy, ry), rx, ry};
            };
            const std::array<T, 4> mid = xy(half);
            const bool upper = !(pow_mean(mid[0], mid[1], alpha) < half);
            auto g = [&](T t) {
                const std::array<T, 4> p = xy(t);
                const T m = pow_mean(p[0], p[1], alpha);
                const T dx = alpha * (az - T(2) * t) / p[2], dy = (T(1) - alpha) * (az - T(2) * t) / p[3];
                const T dm = m * (alpha * dx / p[0] + (T(1) - alpha) * dy / p[1]);
                return upper ? std::pair<T, T>{m - (az - t), dm + T(1)} : std::pair<T, T>{t - m, T(1) - dm};
            };
            const T t = increasing_root(g, T(0), half);
            const T r = upper ? az - t : t;
            const std::array<T, 4> p = xy(t);
            const std::array<T, 3> candidate{p[0], p[1], z0 < T(0) ? -r : r};
            if (finite3(candidate) && dist3(candidate, x0, y0, z0) <= dist3(heuristic, x0, y0, z0))
                return candidate;
            return heuristic;
        }

        // Scale a 3-vector to largest magnitude 1, project with f, scale back
        template <typename T, typename F>
        constexpr void project3(std::span<T, 3> v, F f)
        {
            const T m = max_(max_(math::abs(v[0]), math::abs(v[1])), math::abs(v[2]));
            if (!(m > T(0)) || !(m < inf<T>()))
                return;
            const std::array<T, 3> p = f(v[0] / m, v[1] / m, v[2] / m);
            for (std::size_t i = 0; i < 3; ++i)
                v[i] = p[i] * m;
        }

        // ---- Simplex thresholds ----
        // Condat (2016), Algorithm 1: the threshold tau of the projection of y onto {x >= 0, sum x = r}, r > 0. The
        // workspace holds Condat's two lists side by side: the waiting list at [0, nt), the candidate list at
        // [nt, nt + nv), so that moving the candidates to the waiting list is nt += nv.
        template <typename T, typename Y>
        constexpr T condat_tau(std::size_t n, Y y, T r, std::span<T> ws)
        {
            std::size_t nt = 0, nv = 1;
            ws[0] = y(0);
            T rho = ws[0] - r;
            for (std::size_t i = 1; i < n; ++i) {
                const T yi = y(i);
                if (yi > rho) {
                    rho += (yi - rho) / static_cast<T>(nv + 1);
                    if (rho > yi - r) {
                        ws[nt + nv++] = yi;
                    } else {
                        nt += nv;
                        ws[nt] = yi;
                        nv = 1;
                        rho = yi - r;
                    }
                }
            }
            for (std::size_t k = nt; k-- > 0;) {
                const T yk = ws[k];
                if (yk > rho) {
                    ws[k] = ws[nt - 1];
                    ws[nt - 1] = yk;
                    --nt;
                    ++nv;
                    rho += (yk - rho) / static_cast<T>(nv);
                }
            }
            for (bool changed = true; changed;) {
                changed = false;
                for (std::size_t k = 0; k < nv && nv > 1;) {
                    const T yk = ws[nt + k];
                    if (yk <= rho) {
                        ws[nt + k] = ws[nt + nv - 1];
                        --nv;
                        rho += (rho - yk) / static_cast<T>(nv);
                        changed = true;
                    } else {
                        ++k;
                    }
                }
            }
            return rho;
        }

        // Michelot (1986): the same threshold without a workspace, a sweep per change of the active set.
        template <typename T, typename Y>
        constexpr T michelot_tau(std::size_t n, Y y, T r)
        {
            T sum(0);
            for (std::size_t i = 0; i < n; ++i)
                sum += y(i);
            T tau = (sum - r) / static_cast<T>(n);
            std::size_t count = n;
            for (std::size_t sweep = 0; sweep <= n; ++sweep) {
                T s(0);
                std::size_t c = 0;
                for (std::size_t i = 0; i < n; ++i)
                    if (y(i) > tau) {
                        s += y(i);
                        ++c;
                    }
                if (c == count || c == 0)
                    break;
                count = c;
                tau = (s - r) / static_cast<T>(c);
            }
            return tau;
        }
    } // namespace detail::cones

    // ---- Zero and free cones ----

    /// \brief Projects onto the zero cone `{0}`: every element becomes 0.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param x The point, overwritten.
    template <linalg::scalar T>
    constexpr void project_zero(std::span<T> x) noexcept(detail::cones::nt<T>)
    {
        for (T& v : x)
            v = T(0);
    }

    /// \brief Projects onto the free cone `Rⁿ`, the zero cone's dual: nothing changes.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param x The point, unchanged.
    template <linalg::scalar T>
    constexpr void project_free(std::span<T> x) noexcept
    {
        static_cast<void>(x);
    }

    /// \brief Whether every element is within `tol` of 0.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param x The point.
    /// \param tol The tolerance.
    /// \return Whether `max |xᵢ| ≤ tol`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr bool in_zero_cone(std::span<const std::type_identity_t<T>> x, T tol) noexcept(detail::cones::nt<T>)
    {
        for (const T& v : x)
            if (!(math::abs(v) <= tol))
                return false;
        return true;
    }

    /// \brief Whether `x` is in the free cone: always, unless an element is NaN.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param x The point.
    /// \param tol The tolerance (unused).
    /// \return Whether no element is NaN.
    template <linalg::scalar T>
    [[nodiscard]] constexpr bool in_free_cone(std::span<const std::type_identity_t<T>> x, T tol) noexcept(detail::cones::nt<T>)
    {
        static_cast<void>(tol);
        for (const T& v : x)
            if (v != v)
                return false;
        return true;
    }

    // ---- Non-negative orthant and box ----

    /// \brief Projects onto the non-negative orthant: `xᵢ ← max(xᵢ, 0)`. Self-dual.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param x The point, overwritten.
    template <linalg::scalar T>
    constexpr void project_nonnegative(std::span<T> x) noexcept(detail::cones::nt<T>)
    {
        for (T& v : x)
            if (v < T(0))
                v = T(0);
    }

    /// \brief Whether every element is at least `-tol`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param x The point.
    /// \param tol The tolerance.
    /// \return Whether `min xᵢ ≥ −tol`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr bool in_nonnegative_cone(std::span<const std::type_identity_t<T>> x, T tol) noexcept(detail::cones::nt<T>)
    {
        for (const T& v : x)
            if (!(v >= -tol))
                return false;
        return true;
    }

    /// \brief Projects onto the box `[l, u]`: `xᵢ ← clamp(xᵢ, lᵢ, uᵢ)`; a bound may be infinite, as OSQP's are.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param x The point, overwritten.
    /// \param l The lower bounds (`-∞` for none).
    /// \param u The upper bounds (`+∞` for none), `lᵢ ≤ uᵢ`.
    /// \return Nothing, or `error::dimension_mismatch` (and `x` unchanged).
    template <linalg::scalar T>
    constexpr result<void, error> project_box(std::span<T> x, std::span<const std::type_identity_t<T>> l, std::span<const std::type_identity_t<T>> u) noexcept(detail::cones::nt<T>)
    {
        if (l.size() != x.size() || u.size() != x.size())
            return err(error::dimension_mismatch);
        for (std::size_t i = 0; i < x.size(); ++i)
            x[i] = x[i] < l[i] ? l[i] : u[i] < x[i] ? u[i] : x[i];
        return ok();
    }

    /// \brief Projects onto the box `[l, u]ⁿ` with the same bounds for every element.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param x The point, overwritten.
    /// \param l The lower bound.
    /// \param u The upper bound, `l ≤ u`.
    template <linalg::scalar T>
    constexpr void project_box(std::span<T> x, std::type_identity_t<T> l, std::type_identity_t<T> u) noexcept(detail::cones::nt<T>)
    {
        for (T& v : x)
            v = v < l ? l : u < v ? u : v;
    }

    /// \brief Whether `x` is within `tol` of the box `[l, u]`, element by element.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param x The point.
    /// \param l The lower bounds.
    /// \param u The upper bounds.
    /// \param tol The tolerance.
    /// \return Whether `lᵢ − tol ≤ xᵢ ≤ uᵢ + tol` for every `i` (false for spans of different lengths).
    template <linalg::scalar T>
    [[nodiscard]] constexpr bool in_box(std::span<const std::type_identity_t<T>> x, std::span<const std::type_identity_t<T>> l, std::span<const std::type_identity_t<T>> u,
                                        T tol) noexcept(detail::cones::nt<T>)
    {
        if (l.size() != x.size() || u.size() != x.size())
            return false;
        for (std::size_t i = 0; i < x.size(); ++i)
            if (!(x[i] >= l[i] - tol && x[i] <= u[i] + tol))
                return false;
        return true;
    }

    // ---- Second-order cones ----

    /// \brief Projects onto the second-order cone `{(t, x) : ‖x‖ ≤ t}`, `t = x[0]`, in closed form. Self-dual.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param x The point `(t, x₁)`, overwritten; an empty span is left alone.
    ///
    /// Inside, unchanged; in the polar cone (`‖x₁‖ ≤ −t`), zero; otherwise `((t + ‖x₁‖)/2) (1, x₁/‖x₁‖)`.
    template <linalg::scalar T>
    constexpr void project_second_order(std::span<T> x) noexcept(detail::cones::nt<T>)
    {
        if (x.empty())
            return;
        const T t = x[0];
        const T n = detail::cones::norm(std::span<const T>(x.subspan(1)));
        if (n <= t)
            return;
        if (n <= -t) {
            project_zero(x);
            return;
        }
        const T a = (t + n) / T(2);
        x[0] = a;
        for (std::size_t i = 1; i < x.size(); ++i)
            x[i] *= a / n;
    }

    /// \brief Whether `(t, x₁)` is within `tol` of the second-order cone: `‖x₁‖ ≤ t + tol`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param x The point (empty is in the cone).
    /// \param tol The tolerance.
    /// \return Whether `‖x₁‖ ≤ t + tol`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr bool in_second_order_cone(std::span<const std::type_identity_t<T>> x, T tol) noexcept(detail::cones::nt<T>)
    {
        return x.empty() || detail::cones::norm(x.subspan(1)) <= x[0] + tol;
    }

    /// \brief Projects onto the rotated second-order cone `{(u, v, w) : 2uv ≥ ‖w‖², u, v ≥ 0}`. Self-dual.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param x The point `(u, v, w)`, at least 2 elements, overwritten.
    /// \return Nothing, or `error::dimension_mismatch` for fewer than 2 elements.
    ///
    /// The orthogonal map `(u, v) ↦ ((u + v)/√2, (u − v)/√2)` takes it to the ordinary cone, which is projected and
    /// mapped back (the map is its own inverse).
    template <linalg::scalar T>
    constexpr result<void, error> project_rotated_second_order(std::span<T> x) noexcept(detail::cones::nt<T>)
    {
        if (x.size() < 2)
            return err(error::dimension_mismatch);
        const T c = T(1) / detail::cones::sqrt2<T>();
        const T u = x[0], v = x[1];
        x[0] = (u + v) * c;
        x[1] = (u - v) * c;
        const T a = x[0], b = x[1];
        project_second_order(x);
        if (x[0] == a && x[1] == b) {   // it was inside: restore the original bits
            x[0] = u;
            x[1] = v;
            return ok();
        }
        const T p = x[0], q = x[1];
        x[0] = (p + q) * c;
        x[1] = (p - q) * c;
        return ok();
    }

    /// \brief Whether `(u, v, w)` is within `tol` of the rotated second-order cone, measured as the ordinary cone it maps to.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param x The point, at least 2 elements.
    /// \param tol The tolerance.
    /// \return Whether `‖((u − v)/√2, w)‖ ≤ (u + v)/√2 + tol` (false for fewer than 2 elements).
    template <linalg::scalar T>
    [[nodiscard]] constexpr bool in_rotated_second_order_cone(std::span<const std::type_identity_t<T>> x, T tol) noexcept(detail::cones::nt<T>)
    {
        if (x.size() < 2)
            return false;
        const T c = T(1) / detail::cones::sqrt2<T>();
        const T b = (x[0] - x[1]) * c;
        const T n = detail::cones::norm(x.subspan(2));
        return math::sqrt(b * b + n * n) <= (x[0] + x[1]) * c + tol;
    }

    // ---- PSD cone ----

    /// \brief The svec length of an `n × n` symmetric matrix, `n(n+1)/2`.
    /// \ingroup core
    /// \param n The order.
    /// \return `n(n+1)/2`.
    [[nodiscard]] constexpr std::size_t svec_size(std::size_t n) noexcept { return n * (n + 1) / 2; }

    /// \brief The workspace `project_psd` and `in_psd_cone` need for order `n`: `2n² + n` elements.
    /// \ingroup core
    /// \param n The order.
    /// \return `2n² + n`.
    [[nodiscard]] constexpr std::size_t psd_workspace_size(std::size_t n) noexcept { return 2 * n * n + n; }

    /// \brief The order `n` of a scaled svec of length `k = n(n+1)/2`.
    /// \ingroup core
    /// \param k The svec length.
    /// \return `n`, or `error::dimension_mismatch` when `k` is not triangular.
    [[nodiscard]] constexpr result<std::size_t, error> psd_order(std::size_t k) noexcept
    {
        const std::size_t n = detail::cones::order_of(k);
        if (n > k)
            return err(error::dimension_mismatch);
        return ok(n);
    }

    /// \brief The scaled svec of a symmetric matrix: its lower triangle column by column, off-diagonals times √2.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param m The `n × n` matrix, row-major (the lower triangle is read).
    /// \param v Receives the svec, `n(n+1)/2` elements.
    /// \return Nothing, or `error::dimension_mismatch` when `m.size() ≠ n²` for `v`'s `n`.
    template <linalg::scalar T>
    constexpr result<void, error> svec(std::span<const std::type_identity_t<T>> m, std::span<T> v) noexcept(detail::cones::nt<T>)
    {
        const std::size_t n = detail::cones::order_of(v.size());
        if (n > v.size() || m.size() != n * n)
            return err(error::dimension_mismatch);
        std::size_t k = 0;
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t i = j; i < n; ++i)
                v[k++] = i == j ? m[i * n + j] : m[i * n + j] * detail::cones::sqrt2<T>();
        return ok();
    }

    /// \brief The symmetric matrix of a scaled svec: the inverse of `svec`, both triangles written.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param v The svec, `n(n+1)/2` elements.
    /// \param m Receives the `n × n` matrix, row-major.
    /// \return Nothing, or `error::dimension_mismatch`.
    template <linalg::scalar T>
    constexpr result<void, error> smat(std::span<const std::type_identity_t<T>> v, std::span<T> m) noexcept(detail::cones::nt<T>)
    {
        const std::size_t n = detail::cones::order_of(v.size());
        if (n > v.size() || m.size() != n * n)
            return err(error::dimension_mismatch);
        std::size_t k = 0;
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t i = j; i < n; ++i) {
                const T e = i == j ? v[k] : v[k] / detail::cones::sqrt2<T>();
                ++k;
                m[i * n + j] = e;
                m[j * n + i] = e;
            }
        return ok();
    }

    namespace detail::cones {
        // X+ from an eigen decomposition (values w, vectors in the columns of V, row-major n x n), written as an
        // svec into x, from whichever of the positive or the negative eigenvalues are fewer (x holds svec(X)).
        template <typename T, typename V>
        constexpr void psd_from_eigen(std::span<T> x, std::size_t n, const T* w, V vec)
        {
            std::size_t negative = 0;
            for (std::size_t k = 0; k < n; ++k)
                negative += w[k] < T(0) ? 1 : 0;
            if (negative == 0)
                return;
            const bool from_negative = negative <= n - negative;
            std::size_t idx = 0;
            for (std::size_t j = 0; j < n; ++j)
                for (std::size_t i = j; i < n; ++i) {
                    T s(0);
                    for (std::size_t k = 0; k < n; ++k)
                        if (from_negative ? w[k] < T(0) : w[k] > T(0))
                            s += w[k] * vec(i, k) * vec(j, k);
                    const T e = i == j ? s : s * sqrt2<T>();
                    x[idx] = from_negative ? x[idx] - e : e;
                    ++idx;
                }
        }
    } // namespace detail::cones

    /// \brief Projects a scaled svec onto the PSD cone, for an order known at run time, by the dense Jacobi eigen decomposition. Self-dual.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param x The svec, `n(n+1)/2` elements, overwritten by that of `V max(Λ, 0) Vᵀ`.
    /// \param workspace At least `psd_workspace_size(n)` elements, overwritten.
    /// \return Nothing, `error::dimension_mismatch`, `error::workspace_too_small` or `error::did_not_converge` (and `x` unchanged).
    template <linalg::scalar T>
    constexpr result<void, error> project_psd(std::span<T> x, std::span<T> workspace) noexcept(detail::cones::nt<T>)
    {
        const std::size_t n = detail::cones::order_of(x.size());
        if (n > x.size())
            return err(error::dimension_mismatch);
        if (workspace.size() < psd_workspace_size(n))
            return err(error::workspace_too_small);
        if (n == 0)
            return ok();
        const std::span<T> a = workspace.first(n * n), v = workspace.subspan(n * n, n * n), w = workspace.subspan(2 * n * n, n);
        static_cast<void>(smat(std::span<const T>(x), a));
        const linalg::matrix_view<T, linalg::layout_right> av(a.data(), {n, n}), vv(v.data(), {n, n});
        if (!linalg::symmetric_eigen(linalg::reference, av, w, vv))
            return err(error::did_not_converge);
        detail::cones::psd_from_eigen(x, n, w.data(), [&](std::size_t i, std::size_t j) { return v[i * n + j]; });
        return ok();
    }

    /// \brief Projects a scaled svec of a fixed order `N` onto the PSD cone, by `linalg::symmetric_eigen`. Self-dual.
    /// \ingroup core
    /// \tparam N The order.
    /// \tparam T The number type.
    /// \param x The svec, `N(N+1)/2` elements, overwritten.
    /// \return Nothing, or `error::did_not_converge` (and `x` unchanged).
    template <std::size_t N, linalg::scalar T>
    constexpr result<void, error> project_psd_fixed(std::span<T, N*(N + 1) / 2> x) noexcept(detail::cones::nt<T>)
    {
        linalg::fixed_matrix<T, N, N> m{};
        static_cast<void>(smat(std::span<const T>(x), std::span<T>(m.elements)));
        const auto e = linalg::symmetric_eigen(m);
        if (!e)
            return err(error::did_not_converge);
        detail::cones::psd_from_eigen(std::span<T>(x), N, e->values.elements.data(), [&](std::size_t i, std::size_t j) { return e->vectors(i, j); });
        return ok();
    }

    /// \brief Whether a scaled svec is within `tol` of the PSD cone: its least eigenvalue is at least `−tol`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param x The svec.
    /// \param tol The tolerance.
    /// \param workspace At least `psd_workspace_size(n)` elements, overwritten.
    /// \return Whether `λ_min ≥ −tol` (false on any error).
    template <linalg::scalar T>
    [[nodiscard]] constexpr bool in_psd_cone(std::span<const std::type_identity_t<T>> x, T tol, std::span<T> workspace) noexcept(detail::cones::nt<T>)
    {
        const std::size_t n = detail::cones::order_of(x.size());
        if (n > x.size() || workspace.size() < psd_workspace_size(n))
            return false;
        if (n == 0)
            return true;
        const std::span<T> a = workspace.first(n * n), v = workspace.subspan(n * n, n * n), w = workspace.subspan(2 * n * n, n);
        static_cast<void>(smat(x, a));
        const linalg::matrix_view<T, linalg::layout_right> av(a.data(), {n, n}), vv(v.data(), {n, n});
        return linalg::symmetric_eigen(linalg::reference, av, w, vv) && w[0] >= -tol;
    }

    /// \brief Whether a scaled svec of fixed order `N` is within `tol` of the PSD cone.
    /// \ingroup core
    /// \tparam N The order.
    /// \tparam T The number type.
    /// \param x The svec, `N(N+1)/2` elements.
    /// \param tol The tolerance.
    /// \return Whether `λ_min ≥ −tol`.
    template <std::size_t N, linalg::scalar T>
    [[nodiscard]] constexpr bool in_psd_cone_fixed(std::span<const std::type_identity_t<T>, N*(N + 1) / 2> x, T tol) noexcept(detail::cones::nt<T>)
    {
        linalg::fixed_matrix<T, N, N> m{};
        static_cast<void>(smat(std::span<const T>(x), std::span<T>(m.elements)));
        const auto e = linalg::symmetric_eigen(m);
        return e && e->values[0] >= -tol;
    }

    // ---- Exponential cone ----

    /// \brief Projects onto the exponential cone `cl{(x, y, z) : y > 0, y e^{x/y} ≤ z}` by Friberg's (2021) univariate root finding.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param v The point `(x, y, z)`, overwritten.
    template <linalg::scalar T>
    constexpr void project_exponential(std::span<T, 3> v) noexcept(detail::cones::nt<T>)
    {
        if (!detail::cones::in_exp(v[0], v[1], v[2]))
            detail::cones::project3(v, [](T a, T b, T c) { return detail::cones::exp_unit(a, b, c); });
    }

    /// \brief Projects onto the dual exponential cone `cl{(u, v, w) : u < 0, −u e^{v/u} ≤ e w}`, by Moreau: `P_{K*}(v) = v + P_K(−v)`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param v The point, overwritten.
    template <linalg::scalar T>
    constexpr void project_dual_exponential(std::span<T, 3> v) noexcept(detail::cones::nt<T>)
    {
        std::array<T, 3> m{-v[0], -v[1], -v[2]};
        project_exponential(std::span<T, 3>(m));
        for (std::size_t i = 0; i < 3; ++i)
            v[i] += m[i];
    }

    /// \brief Whether `(x, y, z)` is within distance `tol` of the exponential cone.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param v The point.
    /// \param tol The tolerance (0 for exact membership by the defining inequality).
    /// \return Whether `‖v − P(v)‖ ≤ tol`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr bool in_exponential_cone(std::span<const std::type_identity_t<T>, 3> v, T tol) noexcept(detail::cones::nt<T>)
    {
        if (detail::cones::in_exp(v[0], v[1], v[2]))
            return true;
        std::array<T, 3> p{v[0], v[1], v[2]};
        project_exponential(std::span<T, 3>(p));
        return detail::cones::dist3(p, v[0], v[1], v[2]) <= tol;
    }

    /// \brief Whether `(u, v, w)` is within distance `tol` of the dual exponential cone.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param v The point.
    /// \param tol The tolerance.
    /// \return Whether `‖v − P_{K*}(v)‖ = ‖P_K(−v)‖ ≤ tol`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr bool in_dual_exponential_cone(std::span<const std::type_identity_t<T>, 3> v, T tol) noexcept(detail::cones::nt<T>)
    {
        if (detail::cones::in_exp_dual(v[0], v[1], v[2]))
            return true;
        std::array<T, 3> m{-v[0], -v[1], -v[2]};
        project_exponential(std::span<T, 3>(m));
        return detail::cones::dist3(m, T(0), T(0), T(0)) <= tol;
    }

    // ---- Power cone ----

    /// \brief Projects onto the 3-D power cone `{(x, y, z) : x^α y^{1−α} ≥ |z|, x, y ≥ 0}` by Hien's (2015) Newton method, safeguarded.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param v The point, overwritten.
    /// \param alpha α, in (0, 1).
    template <linalg::scalar T>
    constexpr void project_power(std::span<T, 3> v, std::type_identity_t<T> alpha) noexcept(detail::cones::nt<T>)
    {
        if (!detail::cones::in_pow(v[0], v[1], v[2], alpha))
            detail::cones::project3(v, [alpha](T a, T b, T c) { return detail::cones::pow_unit(a, b, c, alpha); });
    }

    /// \brief Projects onto the dual power cone `{(u, v, w) : (u/α)^α (v/(1−α))^{1−α} ≥ |w|, u, v ≥ 0}`, by Moreau.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param v The point, overwritten.
    /// \param alpha α, in (0, 1).
    template <linalg::scalar T>
    constexpr void project_dual_power(std::span<T, 3> v, std::type_identity_t<T> alpha) noexcept(detail::cones::nt<T>)
    {
        std::array<T, 3> m{-v[0], -v[1], -v[2]};
        project_power(std::span<T, 3>(m), alpha);
        for (std::size_t i = 0; i < 3; ++i)
            v[i] += m[i];
    }

    /// \brief Whether `(x, y, z)` is within distance `tol` of the power cone.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param v The point.
    /// \param alpha α, in (0, 1).
    /// \param tol The tolerance.
    /// \return Whether `‖v − P(v)‖ ≤ tol`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr bool in_power_cone(std::span<const std::type_identity_t<T>, 3> v, T alpha, T tol) noexcept(detail::cones::nt<T>)
    {
        if (detail::cones::in_pow(v[0], v[1], v[2], alpha))
            return true;
        std::array<T, 3> p{v[0], v[1], v[2]};
        project_power(std::span<T, 3>(p), alpha);
        return detail::cones::dist3(p, v[0], v[1], v[2]) <= tol;
    }

    /// \brief Whether `(u, v, w)` is within distance `tol` of the dual power cone.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param v The point.
    /// \param alpha α, in (0, 1).
    /// \param tol The tolerance.
    /// \return Whether `‖P_K(−v)‖ ≤ tol`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr bool in_dual_power_cone(std::span<const std::type_identity_t<T>, 3> v, T alpha, T tol) noexcept(detail::cones::nt<T>)
    {
        if (detail::cones::in_pow_dual(v[0], v[1], v[2], alpha))
            return true;
        std::array<T, 3> m{-v[0], -v[1], -v[2]};
        project_power(std::span<T, 3>(m), alpha);
        return detail::cones::dist3(m, T(0), T(0), T(0)) <= tol;
    }

    // ---- Simplex, l1 ball, l-infinity ball, capped simplex ----

    /// \brief Projects onto the simplex `{x ≥ 0, Σx = r}` by Condat's (2016) algorithm, expected O(n).
    /// \ingroup core
    /// \tparam T The number type.
    /// \param x The point, overwritten by `max(x − τ, 0)`.
    /// \param r The total, at least 0.
    /// \param workspace At least `x.size()` elements, overwritten.
    /// \return Nothing, `error::infeasible` for `r < 0` or an empty `x` with `r > 0`, or `error::workspace_too_small`.
    template <linalg::scalar T>
    constexpr result<void, error> project_simplex(std::span<T> x, std::type_identity_t<T> r, std::span<T> workspace) noexcept(detail::cones::nt<T>)
    {
        if (!(r >= T(0)) || (x.empty() && r > T(0)))
            return err(error::infeasible);
        if (workspace.size() < x.size())
            return err(error::workspace_too_small);
        if (x.empty())
            return ok();
        if (r == T(0)) {
            project_zero(x);
            return ok();
        }
        const T tau = detail::cones::condat_tau(x.size(), [&](std::size_t i) { return x[i]; }, r, workspace);
        for (T& v : x)
            v = v > tau ? v - tau : T(0);
        return ok();
    }

    /// \brief Projects onto the simplex `{x ≥ 0, Σx = r}` without a workspace, by Michelot's (1986) method: O(n) a sweep, a sweep per change of the active set.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param x The point, overwritten.
    /// \param r The total, at least 0.
    /// \return Nothing, or `error::infeasible` for `r < 0` or an empty `x` with `r > 0`.
    template <linalg::scalar T>
    constexpr result<void, error> project_simplex(std::span<T> x, std::type_identity_t<T> r) noexcept(detail::cones::nt<T>)
    {
        if (!(r >= T(0)) || (x.empty() && r > T(0)))
            return err(error::infeasible);
        if (x.empty())
            return ok();
        if (r == T(0)) {
            project_zero(x);
            return ok();
        }
        const T tau = detail::cones::michelot_tau(x.size(), [&](std::size_t i) { return x[i]; }, r);
        for (T& v : x)
            v = v > tau ? v - tau : T(0);
        return ok();
    }

    /// \brief Whether `x` is within `tol` of the simplex: every element at least `−tol` and the sum within `tol` of `r`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param x The point.
    /// \param r The total.
    /// \param tol The tolerance.
    /// \return Whether `min xᵢ ≥ −tol` and `|Σx − r| ≤ tol`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr bool in_simplex(std::span<const std::type_identity_t<T>> x, T r, T tol) noexcept(detail::cones::nt<T>)
    {
        T s(0);
        for (const T& v : x) {
            if (!(v >= -tol))
                return false;
            s += v;
        }
        return math::abs(s - r) <= tol;
    }

    /// \brief Projects onto the ℓ1 ball `{‖x‖₁ ≤ r}` by Condat's algorithm on `|x|`, signs restored.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param x The point, overwritten by `sign(x) max(|x| − τ, 0)` (unchanged if inside).
    /// \param r The radius, at least 0.
    /// \param workspace At least `x.size()` elements, overwritten.
    /// \return Nothing, `error::infeasible` for `r < 0`, or `error::workspace_too_small`.
    template <linalg::scalar T>
    constexpr result<void, error> project_l1_ball(std::span<T> x, std::type_identity_t<T> r, std::span<T> workspace) noexcept(detail::cones::nt<T>)
    {
        if (!(r >= T(0)))
            return err(error::infeasible);
        if (workspace.size() < x.size())
            return err(error::workspace_too_small);
        T s(0);
        for (const T& v : x)
            s += math::abs(v);
        if (s <= r)
            return ok();
        const T tau = r == T(0) ? s : detail::cones::condat_tau(x.size(), [&](std::size_t i) { return math::abs(x[i]); }, r, workspace);
        for (T& v : x)
            v = v > tau ? v - tau : v < -tau ? v + tau : T(0);
        return ok();
    }

    /// \brief Projects onto the ℓ1 ball `{‖x‖₁ ≤ r}` without a workspace (Michelot's method on `|x|`).
    /// \ingroup core
    /// \tparam T The number type.
    /// \param x The point, overwritten.
    /// \param r The radius, at least 0.
    /// \return Nothing, or `error::infeasible` for `r < 0`.
    template <linalg::scalar T>
    constexpr result<void, error> project_l1_ball(std::span<T> x, std::type_identity_t<T> r) noexcept(detail::cones::nt<T>)
    {
        if (!(r >= T(0)))
            return err(error::infeasible);
        T s(0);
        for (const T& v : x)
            s += math::abs(v);
        if (s <= r)
            return ok();
        const T tau = r == T(0) ? s : detail::cones::michelot_tau(x.size(), [&](std::size_t i) { return math::abs(x[i]); }, r);
        for (T& v : x)
            v = v > tau ? v - tau : v < -tau ? v + tau : T(0);
        return ok();
    }

    /// \brief Whether `‖x‖₁ ≤ r + tol`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param x The point.
    /// \param r The radius.
    /// \param tol The tolerance.
    /// \return Whether `‖x‖₁ ≤ r + tol`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr bool in_l1_ball(std::span<const std::type_identity_t<T>> x, T r, T tol) noexcept(detail::cones::nt<T>)
    {
        T s(0);
        for (const T& v : x)
            s += math::abs(v);
        return s <= r + tol;
    }

    /// \brief Projects onto the ℓ∞ ball `{‖x‖∞ ≤ r}`: `xᵢ ← clamp(xᵢ, −r, r)`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param x The point, overwritten.
    /// \param r The radius, at least 0.
    /// \return Nothing, or `error::infeasible` for `r < 0`.
    template <linalg::scalar T>
    constexpr result<void, error> project_linf_ball(std::span<T> x, std::type_identity_t<T> r) noexcept(detail::cones::nt<T>)
    {
        if (!(r >= T(0)))
            return err(error::infeasible);
        project_box(x, -r, r);
        return ok();
    }

    /// \brief Whether `‖x‖∞ ≤ r + tol`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param x The point.
    /// \param r The radius.
    /// \param tol The tolerance.
    /// \return Whether every `|xᵢ| ≤ r + tol`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr bool in_linf_ball(std::span<const std::type_identity_t<T>> x, T r, T tol) noexcept(detail::cones::nt<T>)
    {
        for (const T& v : x)
            if (!(math::abs(v) <= r + tol))
                return false;
        return true;
    }

    /// \brief Projects onto the capped simplex `{0 ≤ x ≤ u, Σx = r}`: `xᵢ ← clamp(xᵢ − τ, 0, u)`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param x The point, overwritten.
    /// \param r The total, in `[0, n u]`.
    /// \param u The cap, positive (1 for the classic capped simplex).
    /// \return Nothing, or `error::infeasible` when `r` is outside `[0, n u]` or `u ≤ 0` (and `x` unchanged).
    ///
    /// `Σ clamp(xᵢ − τ, 0, u)` is non-increasing and piecewise linear in `τ`: bisection to adjacent floats (at most
    /// 200 steps), then the exact `τ` of the final active set, kept if its sum is nearer `r`.
    template <linalg::scalar T>
    constexpr result<void, error> project_capped_simplex(std::span<T> x, std::type_identity_t<T> r, std::type_identity_t<T> u = T(1)) noexcept(detail::cones::nt<T>)
    {
        const std::size_t n = x.size();
        if (!(u > T(0)) || !(r >= T(0)) || !(r <= static_cast<T>(n) * u))
            return err(error::infeasible);
        if (n == 0)
            return ok();
        auto clamp = [u](T v) { return v < T(0) ? T(0) : u < v ? u : v; };
        auto sum = [&](T tau) {
            T s(0);
            for (const T& v : x)
                s += clamp(v - tau);
            return s;
        };
        T lo = x[0], hi = x[0];
        for (const T& v : x) {
            lo = detail::cones::min_(lo, v);
            hi = detail::cones::max_(hi, v);
        }
        lo -= u;   // sum(lo) = n u >= r, sum(hi) = 0 <= r
        for (int i = 0; i < 200; ++i) {
            const T mid = lo + (hi - lo) / T(2);
            if (!(mid > lo && mid < hi))
                break;
            if (sum(mid) > r)
                lo = mid;
            else
                hi = mid;
        }
        T tau = lo + (hi - lo) / T(2);
        // The exact threshold for the active set at tau
        T fixed(0);
        std::size_t free = 0;
        for (const T& v : x) {
            const T d = v - tau;
            if (d >= u)
                fixed += u;
            else if (d > T(0)) {
                fixed += v;
                ++free;
            }
        }
        if (free > 0) {
            const T exact = (fixed - r) / static_cast<T>(free);
            if (math::abs(sum(exact) - r) <= math::abs(sum(tau) - r))
                tau = exact;
        }
        for (T& v : x)
            v = clamp(v - tau);
        return ok();
    }

    /// \brief Whether `x` is within `tol` of the capped simplex `{0 ≤ x ≤ u, Σx = r}`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param x The point.
    /// \param r The total.
    /// \param u The cap.
    /// \param tol The tolerance.
    /// \return Whether `−tol ≤ xᵢ ≤ u + tol` and `|Σx − r| ≤ tol`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr bool in_capped_simplex(std::span<const std::type_identity_t<T>> x, T r, T u, T tol) noexcept(detail::cones::nt<T>)
    {
        T s(0);
        for (const T& v : x) {
            if (!(v >= -tol && v <= u + tol))
                return false;
            s += v;
        }
        return math::abs(s - r) <= tol;
    }

    // ---- Products of cones ----

    /// \brief The workspace `project`, `project_dual` and `in_cone` need: the largest PSD block's `psd_workspace_size`.
    /// \ingroup core
    /// \param cones The product.
    /// \return The elements, 0 when there is no PSD block.
    [[nodiscard]] constexpr std::size_t product_workspace_size(std::span<const cone_kind> cones) noexcept
    {
        std::size_t w = 0;
        for (const cone_kind& c : cones)
            if (c.type == cone::psd) {
                const std::size_t n = detail::cones::order_of(c.dimension);
                w = n > c.dimension ? w : (psd_workspace_size(n) > w ? psd_workspace_size(n) : w);
            }
        return w;
    }

    /// \brief The total dimension of a product of cones.
    /// \ingroup core
    /// \param cones The product.
    /// \return `Σ dimension`.
    [[nodiscard]] constexpr std::size_t product_dimension(std::span<const cone_kind> cones) noexcept
    {
        std::size_t d = 0;
        for (const cone_kind& c : cones)
            d += c.dimension;
        return d;
    }

    namespace detail::cones {
        template <typename T>
        constexpr result<void, error> project_one(const cone_kind& c, std::span<T> x, std::span<T> ws)
        {
            const T alpha = static_cast<T>(c.parameter);
            const bool three = x.size() == 3;
            const bool good_alpha = alpha > T(0) && alpha < T(1);
            switch (c.type) {
            case cone::zero: project_zero(x); return ok();
            case cone::free: return ok();
            case cone::nonnegative: project_nonnegative(x); return ok();
            case cone::second_order: project_second_order(x); return ok();
            case cone::rotated_second_order: return project_rotated_second_order(x);
            case cone::psd: return project_psd(x, ws);
            case cone::exponential:
                if (!three)
                    return err(error::dimension_mismatch);
                project_exponential(x.template first<3>());
                return ok();
            case cone::dual_exponential:
                if (!three)
                    return err(error::dimension_mismatch);
                project_dual_exponential(x.template first<3>());
                return ok();
            case cone::power:
                if (!three || !good_alpha)
                    return err(error::dimension_mismatch);
                project_power(x.template first<3>(), alpha);
                return ok();
            case cone::dual_power:
                if (!three || !good_alpha)
                    return err(error::dimension_mismatch);
                project_dual_power(x.template first<3>(), alpha);
                return ok();
            }
            return err(error::dimension_mismatch);
        }

        template <typename T>
        constexpr bool in_one(const cone_kind& c, std::span<const T> x, T tol, std::span<T> ws)
        {
            const T alpha = static_cast<T>(c.parameter);
            const bool three = x.size() == 3;
            switch (c.type) {
            case cone::zero: return in_zero_cone(x, tol);
            case cone::free: return in_free_cone(x, tol);
            case cone::nonnegative: return in_nonnegative_cone(x, tol);
            case cone::second_order: return in_second_order_cone(x, tol);
            case cone::rotated_second_order: return in_rotated_second_order_cone(x, tol);
            case cone::psd: return in_psd_cone(x, tol, ws);
            case cone::exponential: return three && in_exponential_cone(x.template first<3>(), tol);
            case cone::dual_exponential: return three && in_dual_exponential_cone(x.template first<3>(), tol);
            case cone::power: return three && in_power_cone(x.template first<3>(), alpha, tol);
            case cone::dual_power: return three && in_dual_power_cone(x.template first<3>(), alpha, tol);
            }
            return false;
        }

        template <typename T>
        constexpr result<void, error> project_product(std::span<const cone_kind> cones, std::span<T> x, std::span<T> ws, bool dual)
        {
            if (product_dimension(cones) != x.size())
                return err(error::dimension_mismatch);
            if (ws.size() < product_workspace_size(cones))
                return err(error::workspace_too_small);
            std::size_t offset = 0;
            for (const cone_kind& c : cones) {
                const cone_kind k{dual ? dual_of(c.type) : c.type, c.dimension, c.parameter};
                const auto r = project_one(k, x.subspan(offset, c.dimension), ws);
                if (!r)
                    return r;
                offset += c.dimension;
            }
            return ok();
        }
    } // namespace detail::cones

    /// \brief Projects onto a product of cones laid out consecutively, each block onto its own cone.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param cones The product, in order.
    /// \param x The point, `product_dimension(cones)` elements, overwritten.
    /// \param workspace At least `product_workspace_size(cones)` elements (empty when there is no PSD block).
    /// \return Nothing, or the first block's error (`dimension_mismatch` for a size that does not add up, an
    ///         exponential or power block not of 3, a power α outside (0, 1)); blocks before it are projected.
    template <linalg::scalar T>
    constexpr result<void, error> project(std::span<const cone_kind> cones, std::span<T> x, std::span<T> workspace = {}) noexcept(detail::cones::nt<T>)
    {
        return detail::cones::project_product(cones, x, workspace, false);
    }

    /// \brief Projects onto the dual of a product of cones (the product of the duals), as SCS's iteration needs.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param cones The product, in order.
    /// \param x The point, overwritten.
    /// \param workspace At least `product_workspace_size(cones)` elements.
    /// \return Nothing, or the first block's error.
    template <linalg::scalar T>
    constexpr result<void, error> project_dual(std::span<const cone_kind> cones, std::span<T> x, std::span<T> workspace = {}) noexcept(detail::cones::nt<T>)
    {
        return detail::cones::project_product(cones, x, workspace, true);
    }

    /// \brief Whether every block of `x` is within `tol` of its cone.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param cones The product.
    /// \param x The point.
    /// \param tol The tolerance, per block as each membership test measures it.
    /// \param workspace At least `product_workspace_size(cones)` elements.
    /// \return Whether every block is in its cone (false when the sizes do not add up).
    template <linalg::scalar T>
    [[nodiscard]] constexpr bool in_cone(std::span<const cone_kind> cones, std::span<const std::type_identity_t<T>> x, T tol, std::span<T> workspace = {}) noexcept(detail::cones::nt<T>)
    {
        if (product_dimension(cones) != x.size() || workspace.size() < product_workspace_size(cones))
            return false;
        std::size_t offset = 0;
        for (const cone_kind& c : cones) {
            if (!detail::cones::in_one(c, x.subspan(offset, c.dimension), tol, workspace))
                return false;
            offset += c.dimension;
        }
        return true;
    }

    // ---- Nesterov-Todd scaling ----

    /// \brief The Nesterov-Todd scaling of the non-negative orthant: `wᵢ = √(sᵢ / zᵢ)`, so `W = diag(w)` and `W z = W⁻¹ s = √(s z)`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param s The primal slack, strictly positive.
    /// \param z The dual variable, strictly positive.
    /// \param w Receives the scaling.
    /// \return Nothing, `error::dimension_mismatch`, or `error::not_interior` when an `sᵢ` or `zᵢ` is not positive.
    template <linalg::scalar T>
    constexpr result<void, error> nt_scaling_nonnegative(std::span<const std::type_identity_t<T>> s, std::span<const std::type_identity_t<T>> z, std::span<T> w) noexcept(detail::cones::nt<T>)
    {
        if (s.size() != w.size() || z.size() != w.size())
            return err(error::dimension_mismatch);
        for (std::size_t i = 0; i < w.size(); ++i)
            if (!(s[i] > T(0) && z[i] > T(0)))
                return err(error::not_interior);
        for (std::size_t i = 0; i < w.size(); ++i)
            w[i] = math::sqrt(s[i] / z[i]);
        return ok();
    }

    /// \brief Applies the orthant's `W = diag(w)` in place: `xᵢ ← wᵢ xᵢ`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param w The scaling from `nt_scaling_nonnegative`.
    /// \param x The vector, overwritten (as long as `w`).
    template <linalg::scalar T>
    constexpr void nt_apply_nonnegative(std::span<const std::type_identity_t<T>> w, std::span<T> x) noexcept(detail::cones::nt<T>)
    {
        for (std::size_t i = 0; i < x.size(); ++i)
            x[i] *= w[i];
    }

    /// \brief Applies the orthant's `W⁻¹` in place: `xᵢ ← xᵢ / wᵢ`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param w The scaling.
    /// \param x The vector, overwritten.
    template <linalg::scalar T>
    constexpr void nt_apply_inverse_nonnegative(std::span<const std::type_identity_t<T>> w, std::span<T> x) noexcept(detail::cones::nt<T>)
    {
        for (std::size_t i = 0; i < x.size(); ++i)
            x[i] /= w[i];
    }

    namespace detail::cones {
        // sqrt(x0^2 - |x1|^2) as sqrt((x0 - |x1|)(x0 + |x1|)), or -1 outside the interior
        template <typename T>
        constexpr T j_norm(std::span<const T> x)
        {
            const T n = norm(x.subspan(1));
            return x[0] - n > T(0) ? math::sqrt((x[0] - n) * (x[0] + n)) : T(-1);
        }

        template <typename T>
        constexpr void soc_apply(std::span<const T> w, T eta, std::span<T> x, T sign)
        {
            T d(0);
            for (std::size_t i = 1; i < x.size(); ++i)
                d += w[i] * x[i];
            const T x0 = x[0];
            const T c = sign * x0 + d / (T(1) + w[0]);
            x[0] = eta * (w[0] * x0 + sign * d);
            for (std::size_t i = 1; i < x.size(); ++i)
                x[i] = eta * (x[i] + c * w[i]);
        }
    } // namespace detail::cones

    /// \brief The Nesterov-Todd scaling of the second-order cone: the normalised scaling point `w̄` and the factor `η`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param s The primal slack, strictly inside the cone (`s₀ > ‖s₁‖`).
    /// \param z The dual variable, strictly inside the cone.
    /// \param w Receives `w̄ = (s̄ + J z̄) / (2γ)`, `‖w̄‖_J = 1`, `w̄₀ ≥ 1`.
    /// \return `η = √(‖s‖_J / ‖z‖_J)`, or `error::dimension_mismatch` or `error::not_interior`.
    ///
    /// With them `W = η [[w̄₀, w̄₁ᵀ], [w̄₁, I + w̄₁w̄₁ᵀ / (1 + w̄₀)]]` is symmetric positive definite, maps the cone onto
    /// itself, and satisfies `W z = W⁻¹ s` (`W² z = s`).
    template <linalg::scalar T>
    constexpr result<T, error> nt_scaling_second_order(std::span<const std::type_identity_t<T>> s, std::span<const std::type_identity_t<T>> z, std::span<T> w) noexcept(detail::cones::nt<T>)
    {
        if (w.empty() || s.size() != w.size() || z.size() != w.size())
            return err(error::dimension_mismatch);
        const T sn = detail::cones::j_norm(s), zn = detail::cones::j_norm(z);
        if (!(sn > T(0) && zn > T(0)))
            return err(error::not_interior);
        T sz(0);
        for (std::size_t i = 0; i < w.size(); ++i)
            sz += (s[i] / sn) * (z[i] / zn);
        const T gamma = math::sqrt((T(1) + sz) / T(2));
        w[0] = (s[0] / sn + z[0] / zn) / (T(2) * gamma);
        for (std::size_t i = 1; i < w.size(); ++i)
            w[i] = (s[i] / sn - z[i] / zn) / (T(2) * gamma);
        return ok(math::sqrt(sn / zn));
    }

    /// \brief Applies the second-order cone's `W` in place: `x ← W x`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param w The normalised scaling point `w̄` from `nt_scaling_second_order`.
    /// \param eta `η` from `nt_scaling_second_order`.
    /// \param x The vector, as long as `w`, overwritten.
    template <linalg::scalar T>
    constexpr void nt_apply_second_order(std::span<const std::type_identity_t<T>> w, std::type_identity_t<T> eta, std::span<T> x) noexcept(detail::cones::nt<T>)
    {
        if (!x.empty())
            detail::cones::soc_apply(w, eta, x, T(1));
    }

    /// \brief Applies the second-order cone's `W⁻¹` in place: `x ← W⁻¹ x`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param w The normalised scaling point.
    /// \param eta `η`.
    /// \param x The vector, overwritten.
    template <linalg::scalar T>
    constexpr void nt_apply_inverse_second_order(std::span<const std::type_identity_t<T>> w, std::type_identity_t<T> eta, std::span<T> x) noexcept(detail::cones::nt<T>)
    {
        if (!x.empty())
            detail::cones::soc_apply(w, T(1) / eta, x, T(-1));
    }

} // namespace snicholls::optim

#endif /* snicholls_core_optim_cones_hpp */
