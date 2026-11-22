//
//  transition.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::tracking::constant_velocity`, `singer`, `constant_turn`, `combined` and the other motion models: how a state moves over a time step, and how uncertain the move is.
/// \ingroup core
///
/// **Overview.** A transition model maps a state over `dt` seconds and says how much process noise the step adds.
/// Every model here satisfies `transition_model`:
///
/// - `static constexpr std::size_t dim`, and `value_type`;
/// - `apply(x, dt)`: the predicted state;
/// - `jacobian(x, dt)`: `∂apply/∂x` (the matrix `F(dt)` itself for a linear model);
/// - `covariance(dt)`: the process noise `Q(dt)`.
///
/// A `linear_transition` also has `matrix(dt)`. Models compose with `combined<Ms...>`, a block-diagonal type, so a
/// 2-D constant-velocity target is `combined<constant_velocity<double>, constant_velocity<double>>`, state
/// `(x, ẋ, y, ẏ)`, with no run-time cost for the composition.
///
/// | Model | State | Noise |
/// |---|---|---|
/// | `constant_nth_derivative<T, N>{q}` | a value and its first N derivatives | white N-th derivative, closed form |
/// | `random_walk`, `constant_velocity`, `constant_acceleration` | N = 0, 1, 2 | |
/// | `nth_derivative_decay<T, N>{q, k}` | the N-th derivative decays at rate `k` | integrated by Gauss-Legendre |
/// | `ornstein_uhlenbeck` (N = 1), `singer` (N = 2) | | |
/// | `singer_approximate<T>{q, k}` | Singer's matrix, constant acceleration's noise | closed form |
/// | `known_turn_rate<T>{q₁, q₂, ω}` | `(x, ẋ, y, ẏ)` turning at a known rate | per axis, as constant velocity |
/// | `constant_turn<T>{q_x, q_y, q_ω}` | `(x, ẋ, y, ẏ, ω)`, nonlinear | analytic Jacobian |
/// | `combined<Ms...>` | the models side by side | block diagonal |
///
/// Formulas, state order and noise are Stone Soup's (`models/transition/linear.py`, `nonlinear.py`), so the
/// matrices compare with it element for element. The process noise uses `|dt|`, so a negative step (retrodiction)
/// has the same noise as the positive one.
///
/// \snippet tracking/transition_demo.cpp transition
///
/// **Relationships.** `linalg/fixed.hpp`, `math/constexpr_math.hpp`; `tracking/kalman.hpp` and
/// `tracking/unscented.hpp` predict with them. Group: \ref core.
///
/// **Implementations.** `constant_nth_derivative`: `Fᵢⱼ = dt^(j-i) / (j-i)!`,
/// `Qᵢⱼ = q |dt|^(2N+1-i-j) / ((N-i)! (N-j)! (2N+1-i-j))`. `nth_derivative_decay`: Stone Soup's closed-form `F`, and
/// `Q = q ∫₀^|dt| F(τ) e_N e_Nᵀ F(τ)ᵀ dτ` by 16-point Gauss-Legendre on as many panels as `k |dt|` needs (at least
/// one, at most 64). The nodes are computed at compile time. `constant_turn` and `known_turn_rate` switch to series
/// when `|ω dt| < 0.5`, where `sin(ω dt) / ω` and the Jacobian's turn-rate column would cancel; twelve terms
/// are exact to rounding there.
///
/// **Guarantees.** `constexpr`, `noexcept` for the standard floating-point types, no allocation, no recursion. Every
/// model but `combined` is an aggregate of its parameters, trivially copyable; `combined` holds its models in a
/// `std::tuple`. `covariance(dt)` is symmetric positive semi-definite.
///
/// **Errors.** None: parameters are the caller's to choose (a decay rate `k > 0`).
///
/// **Performance.** Not benchmarked on their own. Form `matrix(dt)` and `covariance(dt)` once for many targets
/// stepped over the same `dt`, and use `predict(prior, F, Q)` (`tracking/kalman.hpp`, whose benchmark does so).
///
/// **Tests and demos.** `tests/tracking/transition_test.cpp`, `demos/tracking/transition_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<concepts>`, `<cstddef>`, `<tuple>`, `<type_traits>`, `<utility>`, `linalg/fixed.hpp`, `math/constexpr_math.hpp`.
///
/// **Known issues.** Stone Soup integrates `nth_derivative_decay`'s noise with QUADPACK; the two agree to about 1e-12
/// relative.
///
/// **Status.** Experimental, new in 0.19.0.

#ifndef snicholls_core_tracking_transition_hpp
#define snicholls_core_tracking_transition_hpp

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/math/constexpr_math.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace snicholls::tracking {

    /// \brief A motion model: `dim`, `apply(x, dt)`, `jacobian(x, dt)` and `covariance(dt)`.
    /// \ingroup core
    /// \tparam M The model.
    template <typename M>
    concept transition_model = requires(const M& m, const linalg::fixed_vector<typename M::value_type, M::dim>& x, typename M::value_type dt) {
        { m.apply(x, dt) } -> std::same_as<linalg::fixed_vector<typename M::value_type, M::dim>>;
        { m.jacobian(x, dt) } -> std::same_as<linalg::fixed_matrix<typename M::value_type, M::dim, M::dim>>;
        { m.covariance(dt) } -> std::same_as<linalg::fixed_matrix<typename M::value_type, M::dim, M::dim>>;
    };

    /// \brief A linear motion model: a `transition_model` with `matrix(dt)`, and `apply(x, dt) = matrix(dt) x`.
    /// \ingroup core
    /// \tparam M The model.
    template <typename M>
    concept linear_transition = transition_model<M> && requires(const M& m, typename M::value_type dt) {
        { m.matrix(dt) } -> std::same_as<linalg::fixed_matrix<typename M::value_type, M::dim, M::dim>>;
    };

    namespace detail::transition {
        constexpr double factorial(std::size_t n) noexcept
        {
            double f = 1;
            for (std::size_t i = 2; i <= n; ++i)
                f *= static_cast<double>(i);
            return f;
        }

        template <typename T>
        constexpr T power(T x, std::size_t n) noexcept(linalg::detail::nothrow_scalar<T>)
        {
            T r(1);
            for (std::size_t i = 0; i < n; ++i)
                r *= x;
            return r;
        }

        template <typename T>
        constexpr T abs(T x) noexcept(linalg::detail::nothrow_scalar<T>)
        {
            return x < T(0) ? -x : x;
        }

        // Gauss-Legendre nodes and weights on [-1, 1], by Newton's method on P_n, at compile time
        template <std::size_t N>
        struct gauss_legendre {
            std::array<double, N> x{}, w{};
        };
        template <std::size_t N>
        consteval gauss_legendre<N> make_gauss_legendre()
        {
            gauss_legendre<N> g{};
            for (std::size_t i = 0; i < N; ++i) {
                double z = math::cos(3.14159265358979323846 * (static_cast<double>(i) + 0.75) / (static_cast<double>(N) + 0.5));
                double dp = 0;
                for (int it = 0; it < 100; ++it) {
                    double p0 = 1, p1 = z;
                    for (std::size_t k = 2; k <= N; ++k) {
                        const double p2 = ((2.0 * static_cast<double>(k) - 1.0) * z * p1 - (static_cast<double>(k) - 1.0) * p0) / static_cast<double>(k);
                        p0 = p1;
                        p1 = p2;
                    }
                    dp = static_cast<double>(N) * (z * p1 - p0) / (z * z - 1.0);
                    const double dz = p1 / dp;
                    z -= dz;
                    if (dz < 1e-17 && dz > -1e-17)
                        break;
                }
                g.x[i] = z;
                g.w[i] = 2.0 / ((1.0 - z * z) * dp * dp);
            }
            return g;
        }
        inline constexpr gauss_legendre<16> gl16 = make_gauss_legendre<16>();

        // The series of sin a / a, (1 - cos a) / a and their derivatives, 12 terms: exact to rounding for |a| < 0.5,
        // where the closed forms lose digits to cancellation
        template <typename T>
        struct turn_series {
            T f, g, df, dg;
        };
        template <typename T>
        constexpr turn_series<T> turn_series_at(T a) noexcept(linalg::detail::nothrow_scalar<T>)
        {
            turn_series<T> r{T(0), T(0), T(0), T(0)};
            T p(1);       // a^(2k)
            T prev(0);    // a^(2k - 1), for the derivative of sin a / a (no division by a, so a = 0 is fine)
            T fact(1);    // (2k + 1)!
            for (int k = 0; k < 12; ++k) {
                const T sign = (k % 2 == 0) ? T(1) : T(-1);
                const T f2 = fact * T(2 * k + 2);  // (2k + 2)!
                r.f += sign * p / fact;
                r.g += sign * p * a / f2;
                r.dg += sign * T(2 * k + 1) * p / f2;
                r.df += sign * T(2 * k) * prev / fact;
                prev = p * a;
                p *= a * a;
                fact = f2 * T(2 * k + 3);
            }
            return r;
        }

        // sin(a)/w and (1 - cos(a))/w for a = w dt, by series when |a| is small
        template <typename T>
        struct turn_terms {
            T s, c, sw, cw;  // sin a, cos a, sin a / w, (1 - cos a) / w
        };
        template <typename T>
        constexpr turn_terms<T> turn(T w, T dt) noexcept(linalg::detail::nothrow_scalar<T>)
        {
            const T a = w * dt;
            const T s = static_cast<T>(math::sin(a)), c = static_cast<T>(math::cos(a));
            if (abs(a) < T(0.5)) {
                const auto r = turn_series_at(a);
                return {s, c, dt * r.f, dt * r.g};
            }
            return {s, c, s / w, (T(1) - c) / w};
        }
    } // namespace detail::transition

    /// \brief A value and its first `N` derivatives, the `N`-th constant but for white noise of diffusion `q`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \tparam N The derivative held constant: 0 random walk, 1 constant velocity, 2 constant acceleration.
    template <linalg::scalar T, std::size_t N>
    struct constant_nth_derivative {
        using value_type = T;                  ///< The number type.
        static constexpr std::size_t dim = N + 1;  ///< The state dimension.
        T q{};                                  ///< The noise diffusion coefficient (Stone Soup's `noise_diff_coeff`).

        /// \brief `F(dt)`. \param dt The step, seconds. \return `Fᵢⱼ = dt^(j-i) / (j-i)!` above the diagonal.
        [[nodiscard]] constexpr linalg::fixed_matrix<T, dim, dim> matrix(T dt) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            auto f = linalg::fixed_matrix<T, dim, dim>::identity();
            for (std::size_t i = 0; i < dim; ++i)
                for (std::size_t j = i + 1; j < dim; ++j)
                    f(i, j) = detail::transition::power(dt, j - i) / static_cast<T>(detail::transition::factorial(j - i));
            return f;
        }
        /// \brief `F(dt) x`. \param x The state. \param dt The step. \return The predicted state.
        [[nodiscard]] constexpr linalg::fixed_vector<T, dim> apply(const linalg::fixed_vector<T, dim>& x, T dt) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            return matrix(dt) * x;
        }
        /// \brief `F(dt)`. \param x The state (unused: the model is linear). \param dt The step. \return `F(dt)`.
        [[nodiscard]] constexpr linalg::fixed_matrix<T, dim, dim> jacobian(const linalg::fixed_vector<T, dim>& x, T dt) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            static_cast<void>(x);
            return matrix(dt);
        }
        /// \brief `Q(dt)`. \param dt The step, seconds.
        /// \return `Qᵢⱼ = q |dt|^(2N+1-i-j) / ((N-i)! (N-j)! (2N+1-i-j))`.
        [[nodiscard]] constexpr linalg::fixed_matrix<T, dim, dim> covariance(T dt) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            const T t = detail::transition::abs(dt);
            linalg::fixed_matrix<T, dim, dim> m{};
            for (std::size_t i = 0; i < dim; ++i)
                for (std::size_t j = 0; j < dim; ++j) {
                    const std::size_t e = 2 * N + 1 - i - j;
                    m(i, j) = q * detail::transition::power(t, e) /
                              static_cast<T>(detail::transition::factorial(N - i) * detail::transition::factorial(N - j) * static_cast<double>(e));
                }
            return m;
        }
    };

    /// \brief A random walk: the value itself diffuses. \tparam T The number type.
    /// \ingroup core
    template <linalg::scalar T>
    using random_walk = constant_nth_derivative<T, 0>;
    /// \brief Constant velocity (white-noise acceleration): `(x, ẋ)`. \tparam T The number type.
    /// \ingroup core
    template <linalg::scalar T>
    using constant_velocity = constant_nth_derivative<T, 1>;
    /// \brief Constant acceleration (white-noise jerk): `(x, ẋ, ẍ)`. \tparam T The number type.
    /// \ingroup core
    template <linalg::scalar T>
    using constant_acceleration = constant_nth_derivative<T, 2>;

    /// \brief A value and its first `N` derivatives, the `N`-th decaying at rate `k` and driven by white noise of
    /// diffusion `q` (Stone Soup's `NthDerivativeDecay`).
    /// \ingroup core
    /// \tparam T The number type.
    /// \tparam N The decaying derivative: 1 Ornstein-Uhlenbeck velocity, 2 Singer acceleration.
    template <linalg::scalar T, std::size_t N>
    struct nth_derivative_decay {
        static_assert(N >= 1, "snicholls::tracking::nth_derivative_decay: N is at least 1");
        using value_type = T;                      ///< The number type.
        static constexpr std::size_t dim = N + 1;  ///< The state dimension.
        T q{};                                      ///< The noise diffusion coefficient.
        T k{};                                      ///< The damping coefficient (Singer's 1 / decorrelation time), above zero.

        /// \brief `F(dt)`, Stone Soup's closed form. \param dt The step, seconds. \return The transition matrix.
        [[nodiscard]] constexpr linalg::fixed_matrix<T, dim, dim> matrix(T dt) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            auto f = linalg::fixed_matrix<T, dim, dim>::identity();
            const T e = static_cast<T>(math::exp(-k * dt));
            for (std::size_t i = 0; i < dim; ++i) {
                const std::size_t r = N - i;
                T v = e * ((r % 2 == 0) ? T(1) : T(-1)) / detail::transition::power(k, r);
                for (std::size_t n = 1; n <= r; ++n)
                    v -= ((n % 2 == 0) ? T(1) : T(-1)) * detail::transition::power(dt, r - n) / (static_cast<T>(detail::transition::factorial(r - n)) * detail::transition::power(k, n));
                f(i, N) = v;
                for (std::size_t j = i + 1; j < N; ++j)
                    f(i, j) = detail::transition::power(dt, j - i) / static_cast<T>(detail::transition::factorial(j - i));
            }
            return f;
        }
        /// \brief `F(dt) x`. \param x The state. \param dt The step. \return The predicted state.
        [[nodiscard]] constexpr linalg::fixed_vector<T, dim> apply(const linalg::fixed_vector<T, dim>& x, T dt) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            return matrix(dt) * x;
        }
        /// \brief `F(dt)`. \param x The state (unused). \param dt The step. \return `F(dt)`.
        [[nodiscard]] constexpr linalg::fixed_matrix<T, dim, dim> jacobian(const linalg::fixed_vector<T, dim>& x, T dt) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            static_cast<void>(x);
            return matrix(dt);
        }
        /// \brief `Q(dt) = q ∫₀^|dt| F(τ) e_N e_Nᵀ F(τ)ᵀ dτ`. \param dt The step, seconds. \return The process noise.
        [[nodiscard]] constexpr linalg::fixed_matrix<T, dim, dim> covariance(T dt) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            const T t = detail::transition::abs(dt);
            linalg::fixed_matrix<T, dim, dim> m{};
            if (t == T(0))
                return m;
            const T kt = k * t;
            std::size_t panels = 1;
            while (panels < 64 && static_cast<T>(panels) < kt)
                panels *= 2;
            const T h = t / static_cast<T>(panels);
            for (std::size_t p = 0; p < panels; ++p) {
                const T a = h * static_cast<T>(p);
                for (std::size_t g = 0; g < 16; ++g) {
                    const T tau = a + h * (T(1) + static_cast<T>(detail::transition::gl16.x[g])) / T(2);
                    const auto f = matrix(tau);
                    const T w = static_cast<T>(detail::transition::gl16.w[g]) * h / T(2);
                    for (std::size_t i = 0; i < dim; ++i)
                        for (std::size_t j = 0; j < dim; ++j)
                            m(i, j) += w * f(i, N) * f(j, N);
                }
            }
            return linalg::symmetrise(m * q);
        }
    };

    /// \brief Ornstein-Uhlenbeck: `(x, ẋ)`, the velocity decaying. \tparam T The number type.
    /// \ingroup core
    template <linalg::scalar T>
    using ornstein_uhlenbeck = nth_derivative_decay<T, 1>;
    /// \brief Singer: `(x, ẋ, ẍ)`, the acceleration decaying. \tparam T The number type.
    /// \ingroup core
    template <linalg::scalar T>
    using singer = nth_derivative_decay<T, 2>;

    /// \brief Singer's transition with constant acceleration's process noise (Stone Soup's `SingerApproximate`).
    /// \ingroup core
    /// \tparam T The number type.
    template <linalg::scalar T>
    struct singer_approximate {
        using value_type = T;                  ///< The number type.
        static constexpr std::size_t dim = 3;  ///< The state dimension.
        T q{};                                  ///< The noise diffusion coefficient.
        T k{};                                  ///< The damping coefficient.

        /// \brief `F(dt)`, Singer's. \param dt The step. \return The transition matrix.
        [[nodiscard]] constexpr linalg::fixed_matrix<T, 3, 3> matrix(T dt) const noexcept(linalg::detail::nothrow_scalar<T>) { return singer<T>{q, k}.matrix(dt); }
        /// \brief `F(dt) x`. \param x The state. \param dt The step. \return The predicted state.
        [[nodiscard]] constexpr linalg::fixed_vector<T, 3> apply(const linalg::fixed_vector<T, 3>& x, T dt) const noexcept(linalg::detail::nothrow_scalar<T>) { return matrix(dt) * x; }
        /// \brief `F(dt)`. \param x The state (unused). \param dt The step. \return `F(dt)`.
        [[nodiscard]] constexpr linalg::fixed_matrix<T, 3, 3> jacobian(const linalg::fixed_vector<T, 3>& x, T dt) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            static_cast<void>(x);
            return matrix(dt);
        }
        /// \brief Constant acceleration's `Q(dt)`. \param dt The step. \return The process noise.
        [[nodiscard]] constexpr linalg::fixed_matrix<T, 3, 3> covariance(T dt) const noexcept(linalg::detail::nothrow_scalar<T>) { return constant_acceleration<T>{q}.covariance(dt); }
    };

    /// \brief `(x, ẋ, y, ẏ)` turning at a known rate `ω` (Stone Soup's `KnownTurnRate`).
    /// \ingroup core
    /// \tparam T The number type.
    template <linalg::scalar T>
    struct known_turn_rate {
        using value_type = T;                  ///< The number type.
        static constexpr std::size_t dim = 4;  ///< The state dimension.
        T q1{};                                 ///< The noise diffusion coefficient along x.
        T q2{};                                 ///< The noise diffusion coefficient along y.
        T omega{};                              ///< The turn rate, radians per second.

        /// \brief `F(dt)`. \param dt The step, seconds. \return The transition matrix.
        [[nodiscard]] constexpr linalg::fixed_matrix<T, 4, 4> matrix(T dt) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            const auto t = detail::transition::turn(omega, dt);
            return {{T(1), t.sw, T(0), -t.cw, T(0), t.c, T(0), -t.s, T(0), t.cw, T(1), t.sw, T(0), t.s, T(0), t.c}};
        }
        /// \brief `F(dt) x`. \param x The state. \param dt The step. \return The predicted state.
        [[nodiscard]] constexpr linalg::fixed_vector<T, 4> apply(const linalg::fixed_vector<T, 4>& x, T dt) const noexcept(linalg::detail::nothrow_scalar<T>) { return matrix(dt) * x; }
        /// \brief `F(dt)`. \param x The state (unused). \param dt The step. \return `F(dt)`.
        [[nodiscard]] constexpr linalg::fixed_matrix<T, 4, 4> jacobian(const linalg::fixed_vector<T, 4>& x, T dt) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            static_cast<void>(x);
            return matrix(dt);
        }
        /// \brief `Q(dt)`: constant velocity's per axis. \param dt The step. \return The process noise.
        [[nodiscard]] constexpr linalg::fixed_matrix<T, 4, 4> covariance(T dt) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            return linalg::block_diagonal(constant_velocity<T>{q1}.covariance(dt), constant_velocity<T>{q2}.covariance(dt));
        }
    };

    /// \brief `(x, ẋ, y, ẏ, ω)`, turning at a rate that is part of the state (Stone Soup's `ConstantTurn`): nonlinear.
    /// \ingroup core
    /// \tparam T The number type.
    template <linalg::scalar T>
    struct constant_turn {
        using value_type = T;                  ///< The number type.
        static constexpr std::size_t dim = 5;  ///< The state dimension.
        T q_x{};                                ///< The acceleration noise diffusion coefficient along x.
        T q_y{};                                ///< The acceleration noise diffusion coefficient along y.
        T q_omega{};                            ///< The turn-rate noise coefficient.

        /// \brief The state after `dt`. \param x `(x, ẋ, y, ẏ, ω)`. \param dt The step, seconds. \return The predicted state.
        [[nodiscard]] constexpr linalg::fixed_vector<T, 5> apply(const linalg::fixed_vector<T, 5>& x, T dt) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            const auto t = detail::transition::turn(x[4], dt);
            return {{x[0] + t.sw * x[1] - t.cw * x[3], x[1] * t.c - x[3] * t.s, t.cw * x[1] + x[2] + t.sw * x[3], x[1] * t.s + x[3] * t.c, x[4]}};
        }
        /// \brief `∂apply/∂x`, analytic. \param x The state. \param dt The step. \return The Jacobian.
        [[nodiscard]] constexpr linalg::fixed_matrix<T, 5, 5> jacobian(const linalg::fixed_vector<T, 5>& x, T dt) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            const T w = x[4], vx = x[1], vy = x[3];
            const auto t = detail::transition::turn(w, dt);
            // d(sin a / w)/dw and d((1 - cos a)/w)/dw, by series when a = w dt is small
            const T a = w * dt;
            T dsw, dcw;
            if (detail::transition::abs(a) < T(0.5)) {
                // dt² times the derivatives of sin a / a and (1 - cos a) / a
                const auto r = detail::transition::turn_series_at(a);
                dsw = dt * dt * r.df;
                dcw = dt * dt * r.dg;
            } else {
                dsw = (dt * t.c * w - t.s) / (w * w);
                dcw = (dt * t.s * w - (T(1) - t.c)) / (w * w);
            }
            return {{T(1), t.sw, T(0), -t.cw, vx * dsw - vy * dcw,
                     T(0), t.c, T(0), -t.s, -dt * (t.s * vx + t.c * vy),
                     T(0), t.cw, T(1), t.sw, vx * dcw + vy * dsw,
                     T(0), t.s, T(0), t.c, dt * (t.c * vx - t.s * vy),
                     T(0), T(0), T(0), T(0), T(1)}};
        }
        /// \brief `Q(dt)`: constant velocity's per axis and `q_ω |dt|` for the rate. \param dt The step.
        /// \return The process noise.
        [[nodiscard]] constexpr linalg::fixed_matrix<T, 5, 5> covariance(T dt) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            auto m = linalg::block_diagonal(constant_velocity<T>{q_x}.covariance(dt), constant_velocity<T>{q_y}.covariance(dt));
            linalg::fixed_matrix<T, 5, 5> r{};
            linalg::set_block<0, 0>(r, m);
            r(4, 4) = q_omega * detail::transition::abs(dt);
            return r;
        }
    };

    namespace detail::transition {
        template <typename... Ms>
        constexpr std::array<std::size_t, sizeof...(Ms) + 1> offsets() noexcept
        {
            std::array<std::size_t, sizeof...(Ms) + 1> o{};
            const std::array<std::size_t, sizeof...(Ms)> d{Ms::dim...};
            for (std::size_t i = 0; i < d.size(); ++i)
                o[i + 1] = o[i] + d[i];
            return o;
        }
    } // namespace detail::transition

    /// \brief Several models side by side, each over its own block of the state: block-diagonal `F` and `Q`
    /// (Stone Soup's `CombinedLinearGaussianTransitionModel` and `CombinedGaussianTransitionModel`).
    /// \ingroup core
    /// \tparam M The first model.
    /// \tparam Ms The others, of the same `value_type`.
    template <transition_model M, transition_model... Ms>
    struct combined {
        using value_type = typename M::value_type;                     ///< The number type.
        static constexpr std::size_t dim = (M::dim + ... + Ms::dim);   ///< The state dimension.
        std::tuple<M, Ms...> models{};                                  ///< The models, in state order.

    private:
        using T = value_type;
        static constexpr auto offset = detail::transition::offsets<M, Ms...>();

        template <typename F>
        constexpr void each(F&& f) const
        {
            [&]<std::size_t... I>(std::index_sequence<I...>) { (f(std::integral_constant<std::size_t, I>{}, std::get<I>(models)), ...); }(std::index_sequence_for<M, Ms...>{});
        }

    public:
        /// \brief `F(dt)`, block diagonal; for linear models only. \param dt The step. \return The transition matrix.
        [[nodiscard]] constexpr linalg::fixed_matrix<T, dim, dim> matrix(T dt) const noexcept(linalg::detail::nothrow_scalar<T>)
            requires(linear_transition<M> && (linear_transition<Ms> && ...))
        {
            linalg::fixed_matrix<T, dim, dim> f{};
            each([&](auto i, const auto& m) { linalg::set_block<offset[i], offset[i]>(f, m.matrix(dt)); });
            return f;
        }
        /// \brief Each model applied to its block. \param x The state. \param dt The step. \return The predicted state.
        [[nodiscard]] constexpr linalg::fixed_vector<T, dim> apply(const linalg::fixed_vector<T, dim>& x, T dt) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            linalg::fixed_vector<T, dim> y{};
            each([&](auto i, const auto& m) {
                using Mi = std::remove_cvref_t<decltype(m)>;
                linalg::set_block<offset[i], 0>(y, m.apply(linalg::block<offset[i], 0, Mi::dim, 1>(x), dt));
            });
            return y;
        }
        /// \brief The block-diagonal Jacobian. \param x The state. \param dt The step. \return The Jacobian.
        [[nodiscard]] constexpr linalg::fixed_matrix<T, dim, dim> jacobian(const linalg::fixed_vector<T, dim>& x, T dt) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            linalg::fixed_matrix<T, dim, dim> j{};
            each([&](auto i, const auto& m) {
                using Mi = std::remove_cvref_t<decltype(m)>;
                linalg::set_block<offset[i], offset[i]>(j, m.jacobian(linalg::block<offset[i], 0, Mi::dim, 1>(x), dt));
            });
            return j;
        }
        /// \brief The block-diagonal `Q(dt)`. \param dt The step. \return The process noise.
        [[nodiscard]] constexpr linalg::fixed_matrix<T, dim, dim> covariance(T dt) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            linalg::fixed_matrix<T, dim, dim> q{};
            each([&](auto i, const auto& m) { linalg::set_block<offset[i], offset[i]>(q, m.covariance(dt)); });
            return q;
        }
    };

    /// \brief Builds a `combined` from model values. \tparam M The first model type.
    /// \ingroup core
    /// \tparam Ms The others. \param m The first model. \param ms The others. \return The combination.
    template <transition_model M, transition_model... Ms>
    [[nodiscard]] constexpr combined<M, Ms...> combine(const M& m, const Ms&... ms) noexcept(linalg::detail::nothrow_scalar<typename M::value_type>)
    {
        return combined<M, Ms...>{std::tuple<M, Ms...>{m, ms...}};
    }

} // namespace snicholls::tracking

#endif /* snicholls_core_tracking_transition_hpp */
