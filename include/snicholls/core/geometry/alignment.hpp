//
//  alignment.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::geometry::align_2d` and `align_3d`: the least-squares rigid or similarity transform between two sets of corresponding points (Umeyama in 2-D, Horn in 3-D).
/// \ingroup core
///
/// **Overview.** Two sensors that watch the same targets each see the targets' tracks in their own frame. The
/// transform that superimposes the two minimises `Σ wᵢ |dᵢ - (s R sᵢ + t)|²` over a rotation `R`, a translation
/// `t` and, optionally, a scale `s`. That transform registers one sensor to the other: two phones watching the
/// same swimmers, a camera against surveyed landmarks.
///
/// - `align_2d(source, target, weights, with_scale)`: SE(2), or Sim(2) with scale. Closed form: the angle is
///   `atan2(Σ w (s'ₓ d'_y - s'_y d'ₓ), Σ w s'·d')` over the centred points.
/// - `align_3d(source, target, weights, with_scale)`: SE(3) or Sim(3), by Horn's unit quaternion, the eigenvector of
///   the largest eigenvalue of a 4 x 4 symmetric matrix built from the cross-covariance.
/// - `transform_2d` and `transform_3d`: `apply(x)` = `s R x + t`, `inverse()`; `rms_error(x, source, target)` is the weighted RMS misfit.
///
/// Wrap them in `stats/ransac.hpp` when some correspondences are wrong.
///
/// \snippet geometry/alignment_demo.cpp alignment
///
/// **Relationships.** `linalg/fixed.hpp` (`symmetric_eigen`), `stats/ransac.hpp`. Group: \ref core.
///
/// **Guarantees.** `constexpr`, `noexcept` for the standard floating-point types, no allocation. `R` is a proper
/// rotation (determinant +1, no reflection). The result is exact for noise-free points. Complexity is O(n) in the
/// points, plus one fixed 4 x 4 eigen decomposition in 3-D.
///
/// **Errors.** `alignment_error::too_few_points`: fewer than two points in 2-D, or three in 3-D, or a weight list
/// of the wrong length. `alignment_error::degenerate`: the source points coincide (no spread), or in 3-D are
/// collinear, so the rotation is not determined.
///
/// **Performance.** Not benchmarked: O(n) arithmetic.
///
/// **Tests and demos.** `tests/geometry/alignment_test.cpp`, `demos/geometry/alignment_demo.cpp`.
///
/// **Dependencies.** `<cstdint>`, `<span>`, `linalg/fixed.hpp`, `math/constexpr_math.hpp`, `result.hpp`.
///
/// **Known issues.** None known.
///
/// **Status.** Experimental, new in 0.19.0.

#ifndef snicholls_core_geometry_alignment_hpp
#define snicholls_core_geometry_alignment_hpp

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/result.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace snicholls::geometry {

    /// \brief Why an alignment has no answer.
    /// \ingroup core
    enum class alignment_error : std::uint8_t {
        too_few_points = 1,  ///< Fewer points than the dimension needs, or source, target and weights differ in length.
        degenerate = 2,      ///< The source points do not determine a rotation (they coincide, or in 3-D are collinear).
    };

    /// \brief `x ↦ s R x + t` in the plane.
    /// \ingroup core
    /// \tparam T The number type.
    template <linalg::scalar T>
    struct transform_2d {
        linalg::fixed_matrix<T, 2, 2> rotation = linalg::fixed_matrix<T, 2, 2>::identity();  ///< `R`.
        linalg::fixed_vector<T, 2> translation{};                                           ///< `t`.
        T scale = T(1);                                                                     ///< `s`.

        /// \brief The rotation's angle. \return `atan2(R₁₀, R₀₀)`, radians.
        [[nodiscard]] constexpr T angle() const noexcept(linalg::detail::nothrow_scalar<T>) { return static_cast<T>(math::atan2(rotation(1, 0), rotation(0, 0))); }
        /// \brief The transformed point. \param x The point. \return `s R x + t`.
        [[nodiscard]] constexpr linalg::fixed_vector<T, 2> apply(const linalg::fixed_vector<T, 2>& x) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            return rotation * x * scale + translation;
        }
        /// \brief The inverse transform. \return `x ↦ s⁻¹ Rᵀ (x - t)`.
        [[nodiscard]] constexpr transform_2d inverse() const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            const auto rt = linalg::transpose(rotation);
            return {rt, -(rt * translation) / scale, T(1) / scale};
        }
    };

    /// \brief `x ↦ s R x + t` in space.
    /// \ingroup core
    /// \tparam T The number type.
    template <linalg::scalar T>
    struct transform_3d {
        linalg::fixed_matrix<T, 3, 3> rotation = linalg::fixed_matrix<T, 3, 3>::identity();  ///< `R`.
        linalg::fixed_vector<T, 3> translation{};                                           ///< `t`.
        T scale = T(1);                                                                     ///< `s`.

        /// \brief The transformed point. \param x The point. \return `s R x + t`.
        [[nodiscard]] constexpr linalg::fixed_vector<T, 3> apply(const linalg::fixed_vector<T, 3>& x) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            return rotation * x * scale + translation;
        }
        /// \brief The inverse transform. \return `x ↦ s⁻¹ Rᵀ (x - t)`.
        [[nodiscard]] constexpr transform_3d inverse() const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            const auto rt = linalg::transpose(rotation);
            return {rt, -(rt * translation) / scale, T(1) / scale};
        }
    };

    namespace detail::alignment {
        template <typename T, std::size_t N>
        struct centred {
            linalg::fixed_vector<T, N> source_mean{};
            linalg::fixed_vector<T, N> target_mean{};
            T weight{};
            T source_spread{};  // Σ w |s - s̄|²
        };

        template <typename T, std::size_t N>
        constexpr centred<T, N> centre(std::span<const linalg::fixed_vector<T, N>> s, std::span<const linalg::fixed_vector<T, N>> d, std::span<const T> w) noexcept(linalg::detail::nothrow_scalar<T>)
        {
            centred<T, N> c{};
            for (std::size_t i = 0; i < s.size(); ++i) {
                const T wi = w.empty() ? T(1) : w[i];
                c.weight += wi;
                c.source_mean += s[i] * wi;
                c.target_mean += d[i] * wi;
            }
            c.source_mean = c.source_mean / c.weight;
            c.target_mean = c.target_mean / c.weight;
            for (std::size_t i = 0; i < s.size(); ++i)
                c.source_spread += (w.empty() ? T(1) : w[i]) * linalg::squared_norm(s[i] - c.source_mean);
            return c;
        }

        constexpr bool lengths_fit(std::size_t need, std::size_t s, std::size_t d, std::size_t w) noexcept
        {
            return s >= need && s == d && (w == 0 || w == s);
        }
    } // namespace detail::alignment

    /// \brief The least-squares rigid (or similarity) transform taking `source` onto `target` in the plane.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param source The points in the frame to be moved.
    /// \param target The corresponding points in the reference frame.
    /// \param weights One non-negative weight per pair; empty for equal weights.
    /// \param with_scale Whether to fit a scale too (Sim(2)); otherwise the scale is 1 (SE(2)).
    /// \return The transform, or an `alignment_error`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr result<transform_2d<T>, alignment_error> align_2d(std::span<const linalg::fixed_vector<T, 2>> source, std::span<const linalg::fixed_vector<T, 2>> target,
                                                                            std::span<const T> weights = {}, bool with_scale = false) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        if (!detail::alignment::lengths_fit(2, source.size(), target.size(), weights.size()))
            return err(alignment_error::too_few_points);
        const auto c = detail::alignment::centre<T, 2>(source, target, weights);
        if (!(c.source_spread > T(0)) || !(c.weight > T(0)))
            return err(alignment_error::degenerate);
        T a(0), b(0);
        for (std::size_t i = 0; i < source.size(); ++i) {
            const T wi = weights.empty() ? T(1) : weights[i];
            const auto s = source[i] - c.source_mean;
            const auto d = target[i] - c.target_mean;
            a += wi * (s[0] * d[0] + s[1] * d[1]);
            b += wi * (s[0] * d[1] - s[1] * d[0]);
        }
        const T h = static_cast<T>(math::sqrt(a * a + b * b));
        if (!(h > T(0)))
            return err(alignment_error::degenerate);
        transform_2d<T> x{};
        x.rotation = {{a / h, -b / h, b / h, a / h}};
        x.scale = with_scale ? h / c.source_spread : T(1);
        x.translation = c.target_mean - x.rotation * c.source_mean * x.scale;
        return ok(x);
    }

    /// \brief The least-squares rigid (or similarity) transform taking `source` onto `target` in space, by Horn's quaternion.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param source The points in the frame to be moved.
    /// \param target The corresponding points in the reference frame.
    /// \param weights One non-negative weight per pair; empty for equal weights.
    /// \param with_scale Whether to fit a scale too (Sim(3)); otherwise the scale is 1 (SE(3)).
    /// \return The transform, or an `alignment_error`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr result<transform_3d<T>, alignment_error> align_3d(std::span<const linalg::fixed_vector<T, 3>> source, std::span<const linalg::fixed_vector<T, 3>> target,
                                                                            std::span<const T> weights = {}, bool with_scale = false) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        if (!detail::alignment::lengths_fit(3, source.size(), target.size(), weights.size()))
            return err(alignment_error::too_few_points);
        const auto c = detail::alignment::centre<T, 3>(source, target, weights);
        if (!(c.source_spread > T(0)) || !(c.weight > T(0)))
            return err(alignment_error::degenerate);
        linalg::fixed_matrix<T, 3, 3> m{};  // Σ w s' d'ᵀ
        for (std::size_t i = 0; i < source.size(); ++i) {
            const T wi = weights.empty() ? T(1) : weights[i];
            m += linalg::outer(source[i] - c.source_mean, target[i] - c.target_mean) * wi;
        }
        const T sxx = m(0, 0), sxy = m(0, 1), sxz = m(0, 2), syx = m(1, 0), syy = m(1, 1), syz = m(1, 2), szx = m(2, 0), szy = m(2, 1), szz = m(2, 2);
        const linalg::fixed_matrix<T, 4, 4> n{{sxx + syy + szz, syz - szy, szx - sxz, sxy - syx,
                                               syz - szy, sxx - syy - szz, sxy + syx, szx + sxz,
                                               szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy,
                                               sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz}};
        const auto e = linalg::symmetric_eigen(n);
        if (!e)
            return err(alignment_error::degenerate);
        // A unique rotation needs the largest eigenvalue to stand clear of the next (collinear points tie them)
        const T top = e->values[3], next = e->values[2];
        const T scale = linalg::norm(n);
        if (!(top - next > T(64) * std::numeric_limits<T>::epsilon() * scale))
            return err(alignment_error::degenerate);
        const T q0 = e->vectors(0, 3), qx = e->vectors(1, 3), qy = e->vectors(2, 3), qz = e->vectors(3, 3);
        transform_3d<T> x{};
        x.rotation = {{q0 * q0 + qx * qx - qy * qy - qz * qz, T(2) * (qx * qy - q0 * qz), T(2) * (qx * qz + q0 * qy),
                       T(2) * (qy * qx + q0 * qz), q0 * q0 - qx * qx + qy * qy - qz * qz, T(2) * (qy * qz - q0 * qx),
                       T(2) * (qz * qx - q0 * qy), T(2) * (qz * qy + q0 * qx), q0 * q0 - qx * qx - qy * qy + qz * qz}};
        if (with_scale) {
            T num(0);
            for (std::size_t i = 0; i < source.size(); ++i) {
                const T wi = weights.empty() ? T(1) : weights[i];
                num += wi * linalg::dot(target[i] - c.target_mean, x.rotation * (source[i] - c.source_mean));
            }
            x.scale = num / c.source_spread;
        }
        x.translation = c.target_mean - x.rotation * c.source_mean * x.scale;
        return ok(x);
    }

    /// \brief The weighted root-mean-square distance between `target` and the transformed `source`.
    /// \ingroup core
    /// \tparam X A `transform_2d` or `transform_3d`.
    /// \tparam T The number type.
    /// \tparam N The dimension.
    /// \param x The transform.
    /// \param source The points to transform.
    /// \param target The corresponding points.
    /// \param weights One weight per pair; empty for equal weights.
    /// \return `√(Σ wᵢ |dᵢ - x(sᵢ)|² / Σ wᵢ)`; 0 for no points.
    template <typename X, linalg::scalar T, std::size_t N>
    [[nodiscard]] constexpr T rms_error(const X& x, std::span<const linalg::fixed_vector<T, N>> source, std::span<const linalg::fixed_vector<T, N>> target,
                                       std::span<const T> weights = {}) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        T s(0), w(0);
        for (std::size_t i = 0; i < source.size() && i < target.size(); ++i) {
            const T wi = weights.empty() ? T(1) : weights[i];
            s += wi * linalg::squared_norm(target[i] - x.apply(source[i]));
            w += wi;
        }
        return w > T(0) ? static_cast<T>(math::sqrt(s / w)) : T(0);
    }

} // namespace snicholls::geometry

#endif /* snicholls_core_geometry_alignment_hpp */
