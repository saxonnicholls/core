//
//  angle.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::geometry::wrap_pi`, `wrap_elevation`, `angle_kind` and angle-aware residuals and means: arithmetic on vectors some of whose components are angles.
/// \ingroup core
///
/// **Overview.** A bearing of 179° and one of -179° are 2° apart, not 358°. Filters that subtract a predicted
/// measurement from a measured one, or average sigma points, must know which components are angles and how each
/// wraps. Here that is data known at compile time: an `angle_kinds<N>` array, one `angle_kind` per component
/// (`linear`, `bearing` wrapping to `[-π, π)`, `elevation` folding into `[-π/2, π/2]`), passed as a template argument.
/// A component marked `linear` costs nothing; an angle costs one wrap.
///
/// - `wrap_pi(x)` into `[-π, π)`, `wrap_two_pi(x)` into `[0, 2π)`, `wrap_elevation(x)` into `[-π/2, π/2]` (an
///   elevation past the pole comes back down the other side), `wrap(kind, x)`.
/// - `difference(kind, a, b)`: `a - b`, wrapped as `kind` says.
/// - `residual<Kinds>(a, b)` and `wrapped<Kinds>(x)`: the same per component of a `fixed_vector`.
/// - `circular_mean(angles, weights)`: `atan2(Σ wᵢ sin aᵢ, Σ wᵢ cos aᵢ)`; weights may be negative (the unscented
///   transform's centre weight can be).
/// - `weighted_mean<Kinds>(points, weights)`: the weighted mean of vectors, circular in the angle components.
///
/// The conventions are Stone Soup's (`mod_bearing`, `mod_elevation`, `Angle.average`), so results compare with it
/// directly.
///
/// \snippet geometry/angle_demo.cpp angle
///
/// **Relationships.** `linalg/fixed.hpp` (`fixed_vector`), `math/constexpr_math.hpp` (`floor`, `sin`, `cos`,
/// `atan2`). Used by `tracking/measurement.hpp` (each model's kinds), `tracking/kalman.hpp` and
/// `tracking/unscented.hpp` (residuals and means). Group: \ref core.
///
/// **Implementations.** One floor per wrap: `x - 2π ⌊(x + π) / 2π⌋`, with the rounding case that lands on `π` folded
/// back to `-π`.
///
/// **Guarantees.** `constexpr`, `noexcept` for `float`, `double` and `long double`, no allocation. `wrap_pi` returns a
/// value in `[-π, π)` for every finite input; NaN and infinity give NaN, in every wrap.
///
/// **Errors.** None.
///
/// **Performance.** Not benchmarked: a floor and a few multiplications.
///
/// **Tests and demos.** `tests/geometry/angle_test.cpp`, `demos/geometry/angle_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<cstdint>`, `<limits>`, `<numbers>`, `<span>`, `linalg/fixed.hpp`,
/// `math/constexpr_math.hpp`.
///
/// **Known issues.** For `|x|` beyond about 1e9 the reduction loses the angle's low bits, as any argument reduction
/// by a rounded `2π` does.
///
/// **Status.** Experimental, new in 0.19.0.

#ifndef snicholls_core_geometry_angle_hpp
#define snicholls_core_geometry_angle_hpp

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/math/constexpr_math.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <span>

namespace snicholls::geometry {

    /// \brief How one component of a vector differs and averages.
    /// \ingroup core
    enum class angle_kind : std::uint8_t {
        linear = 0,     ///< Not an angle: subtracted and averaged as it is.
        bearing = 1,    ///< An angle on the full circle, wrapped into `[-π, π)`.
        elevation = 2,  ///< An angle from the horizontal, folded into `[-π/2, π/2]`.
    };

    /// \brief The kind of each of `N` components, as a template argument of the functions that take it.
    /// \ingroup core
    /// \tparam N The components.
    template <std::size_t N>
    using angle_kinds = std::array<angle_kind, N>;

    /// \brief `N` linear components: no angles. \tparam N The components.
    /// \ingroup core
    template <std::size_t N>
    inline constexpr angle_kinds<N> all_linear{};

    /// \brief π in `T`. \tparam T The number type.
    /// \ingroup core
    template <typename T>
    inline constexpr T pi = static_cast<T>(std::numbers::pi_v<long double>);

    /// \brief Degrees in radians. \tparam T The number type. \param degrees The angle in degrees.
    /// \ingroup core
    /// \return The angle in radians.
    template <linalg::scalar T>
    [[nodiscard]] constexpr T radians(T degrees) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        return degrees * (pi<T> / T(180));
    }

    /// \brief Radians in degrees. \tparam T The number type. \param radians The angle in radians.
    /// \ingroup core
    /// \return The angle in degrees.
    template <linalg::scalar T>
    [[nodiscard]] constexpr T degrees(T radians) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        return radians * (T(180) / pi<T>);
    }

    /// \brief `x` wrapped into `[-π, π)`, Stone Soup's `mod_bearing`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param x An angle in radians.
    /// \return The same direction in `[-π, π)`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr T wrap_pi(T x) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const T two_pi = T(2) * pi<T>;
        T y = x + pi<T>;
        y -= two_pi * static_cast<T>(math::floor(y / two_pi));
        if (y >= two_pi)
            y -= two_pi;
        if (y < T(0))
            y = T(0);
        return y - pi<T>;
    }

    /// \brief `x` wrapped into `[0, 2π)`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param x An angle in radians.
    /// \return The same direction in `[0, 2π)`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr T wrap_two_pi(T x) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        return wrap_pi(x - pi<T>) + pi<T>;
    }

    /// \brief `x` folded into `[-π/2, π/2]`, Stone Soup's `mod_elevation`: past the zenith, the elevation comes down
    /// the other side.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param x An elevation in radians.
    /// \return The elevation in `[-π/2, π/2]`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr T wrap_elevation(T x) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        if (!(x == x) || x == std::numeric_limits<T>::infinity() || x == -std::numeric_limits<T>::infinity())
            return std::numeric_limits<T>::quiet_NaN();  // no direction; and the fold below would cast NaN to int
        const T two_pi = T(2) * pi<T>;
        T y = x - two_pi * static_cast<T>(math::floor(x / two_pi));
        if (y >= two_pi)
            y -= two_pi;
        const T quarter = pi<T> / T(2);
        const auto n = static_cast<int>(math::floor(y / quarter));
        if (n == 1 || n == 2)
            return pi<T> - y;
        if (n == 3)
            return y - two_pi;
        if (n >= 4)
            return T(0);
        return y;
    }

    /// \brief `x` wrapped as `kind` says. \tparam T The number type. \param kind The component's kind.
    /// \ingroup core
    /// \param x The value. \return `x`, `wrap_pi(x)` or `wrap_elevation(x)`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr T wrap(angle_kind kind, T x) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        return kind == angle_kind::bearing ? wrap_pi(x) : kind == angle_kind::elevation ? wrap_elevation(x) : x;
    }

    /// \brief `a - b` wrapped as `kind` says: the shorter way round for a bearing. \tparam T The number type.
    /// \ingroup core
    /// \param kind The component's kind. \param a The minuend. \param b The subtrahend. \return The difference.
    template <linalg::scalar T>
    [[nodiscard]] constexpr T difference(angle_kind kind, T a, T b) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        return wrap(kind, a - b);
    }

    /// \brief `a - b` component by component, each wrapped as `Kinds` says.
    /// \ingroup core
    /// \tparam Kinds An `angle_kinds<N>`.
    /// \tparam T The number type.
    /// \tparam N The components.
    /// \param a The minuend, such as a measurement.
    /// \param b The subtrahend, such as a predicted measurement.
    /// \return The residual.
    template <auto Kinds, linalg::scalar T, std::size_t N>
    [[nodiscard]] constexpr linalg::fixed_vector<T, N> residual(const linalg::fixed_vector<T, N>& a, const linalg::fixed_vector<T, N>& b) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        static_assert(Kinds.size() == N, "snicholls::geometry::residual: one angle_kind per component");
        linalg::fixed_vector<T, N> r{};
        for (std::size_t i = 0; i < N; ++i)
            r[i] = difference(Kinds[i], a[i], b[i]);
        return r;
    }

    /// \brief `x` with each component wrapped as `Kinds` says.
    /// \ingroup core
    /// \tparam Kinds An `angle_kinds<N>`.
    /// \tparam T The number type.
    /// \tparam N The components.
    /// \param x The vector.
    /// \return `x`, wrapped.
    template <auto Kinds, linalg::scalar T, std::size_t N>
    [[nodiscard]] constexpr linalg::fixed_vector<T, N> wrapped(const linalg::fixed_vector<T, N>& x) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        static_assert(Kinds.size() == N, "snicholls::geometry::wrapped: one angle_kind per component");
        linalg::fixed_vector<T, N> r{};
        for (std::size_t i = 0; i < N; ++i)
            r[i] = wrap(Kinds[i], x[i]);
        return r;
    }

    /// \brief The weighted circular mean, `atan2(Σ wᵢ sin aᵢ, Σ wᵢ cos aᵢ)`, Stone Soup's `Angle.average`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param angles The angles.
    /// \param weights One weight per angle (any sign); empty for equal weights.
    /// \return The mean direction in `[-π, π]`; 0 when there are no angles or the resultant is zero.
    template <linalg::scalar T>
    [[nodiscard]] constexpr T circular_mean(std::span<const T> angles, std::span<const T> weights = {}) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        T s(0), c(0);
        for (std::size_t i = 0; i < angles.size(); ++i) {
            const T w = weights.empty() ? T(1) : weights[i];
            s += w * static_cast<T>(math::sin(angles[i]));
            c += w * static_cast<T>(math::cos(angles[i]));
        }
        if (s == T(0) && c == T(0))
            return T(0);
        return static_cast<T>(math::atan2(s, c));
    }

    /// \brief The weighted mean of vectors: arithmetic in the linear components, circular in the angles, then wrapped.
    /// \ingroup core
    /// \tparam Kinds An `angle_kinds<N>`.
    /// \tparam T The number type.
    /// \tparam N The components.
    /// \tparam P The points.
    /// \param points The vectors.
    /// \param weights One weight per vector; for the linear components they should sum to 1.
    /// \return The mean.
    template <auto Kinds, linalg::scalar T, std::size_t N, std::size_t P>
    [[nodiscard]] constexpr linalg::fixed_vector<T, N> weighted_mean(const std::array<linalg::fixed_vector<T, N>, P>& points, const std::array<T, P>& weights) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        static_assert(Kinds.size() == N, "snicholls::geometry::weighted_mean: one angle_kind per component");
        linalg::fixed_vector<T, N> m{};
        for (std::size_t i = 0; i < N; ++i) {
            if (Kinds[i] == angle_kind::linear) {
                T s(0);
                for (std::size_t p = 0; p < P; ++p)
                    s += weights[p] * points[p][i];
                m[i] = s;
            } else {
                std::array<T, P> a{};
                for (std::size_t p = 0; p < P; ++p)
                    a[p] = points[p][i];
                m[i] = wrap(Kinds[i], circular_mean<T>(std::span<const T>(a), std::span<const T>(weights)));
            }
        }
        return m;
    }

} // namespace snicholls::geometry

#endif /* snicholls_core_geometry_angle_hpp */
