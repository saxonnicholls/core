//
//  camera.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::geometry::pinhole`, `project`, `pixel_ray`, `ray_plane` and `plane_fix`: a camera's pixel as a point on a plane, with its covariance.
/// \ingroup core
///
/// **Overview.** A camera alone measures a direction, not a range. When the target is known to lie on a plane (a
/// water surface, a road, a floor), the pixel's ray meets the plane in one point, so one camera gives a 3-D fix.
/// With camera centre `C`, orientation `R` (camera to world) and intrinsics `K`, a pixel `u` has the world ray
/// `d = R K⁻¹ ũ`. The plane `n·x = h` meets it at
///
///     x = C + t d,   t = (h - n·C) / (n·d).
///
/// The fix's uncertainty follows from the pixel's: `plane_fix` propagates a pixel covariance (and, optionally, a
/// variance of the plane's height, for a target that rides above it) through the exact Jacobian of `x(u, h)`. The
/// error is long along the ray and short across it, growing like `r σ / tan γ` at grazing angle `γ`.
///
/// - `pinhole<T>`: focal lengths and principal point, in pixels; `matrix()` is `K`.
/// - `project(K, R, C, x)`: the pixel of a world point, or `camera_error::behind_camera`.
/// - `pixel_ray(K, R, u)`: the world direction of a pixel (not normalised).
/// - `ray_plane(C, d, n, h)`: the intersection, or `camera_error::parallel_to_plane` / `behind_camera`.
/// - `plane_fix(K, R, C, u, Σ_u, n, h, σ_h²)`: the point, its 3 x 3 covariance, its range and grazing angle.
/// - `grazing_angle(d, n)`: the angle between a ray and the plane.
///
/// Camera axes are the computer-vision convention: x right, y down, z forward along the optical axis. A camera
/// orientation error `σ_θ` (radians) enters as pixel noise `f σ_θ`.
///
/// \snippet geometry/camera_demo.cpp camera
///
/// **Relationships.** `linalg/fixed.hpp`, `result.hpp`; `geometry/coordinates.hpp` for building `R`;
/// `geometry/geodetic.hpp` for the world frame; the fix's covariance is a measurement covariance for
/// `tracking/kalman.hpp`. Group: \ref core.
///
/// **Guarantees.** `constexpr`, `noexcept` for the standard floating-point types, no allocation. The covariance is
/// symmetric and positive semi-definite (rank 2, or 3 with a height variance).
///
/// **Errors.** `camera_error::behind_camera`: the point (or the intersection) is behind the camera.
/// `camera_error::parallel_to_plane`: the ray does not meet the plane (`|n·d|` at or below `1e-12 |n| |d|`).
///
/// **Performance.** Not benchmarked: a few 3 x 3 products.
///
/// **Tests and demos.** `tests/geometry/camera_test.cpp`, `demos/geometry/camera_demo.cpp`.
///
/// **Dependencies.** `<cstdint>`, `linalg/fixed.hpp`, `math/constexpr_math.hpp`, `result.hpp`.
///
/// **Known issues.** No lens distortion: undistort pixels first.
///
/// **Status.** Experimental, new in 0.19.0.

#ifndef snicholls_core_geometry_camera_hpp
#define snicholls_core_geometry_camera_hpp

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/result.hpp>

#include <cstdint>

namespace snicholls::geometry {

    /// \brief Why a camera computation has no answer.
    /// \ingroup core
    enum class camera_error : std::uint8_t {
        behind_camera = 1,      ///< The point, or the ray's meeting with the plane, is behind the camera.
        parallel_to_plane = 2,  ///< The ray is parallel to the plane.
    };

    /// \brief Pinhole intrinsics, in pixels, no skew.
    /// \ingroup core
    /// \tparam T The number type.
    template <linalg::scalar T>
    struct pinhole {
        T fx;  ///< The focal length along x, pixels.
        T fy;  ///< The focal length along y, pixels.
        T cx;  ///< The principal point's x, pixels.
        T cy;  ///< The principal point's y, pixels.

        /// \brief `K`. \return `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`.
        [[nodiscard]] constexpr linalg::fixed_matrix<T, 3, 3> matrix() const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            return {{fx, T(0), cx, T(0), fy, cy, T(0), T(0), T(1)}};
        }
    };

    /// \brief The pixel at which a world point appears.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param k The intrinsics.
    /// \param r The orientation, camera to world.
    /// \param c The camera centre, world.
    /// \param x The point, world.
    /// \return The pixel `(u, v)`, or `camera_error::behind_camera`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr result<linalg::fixed_vector<T, 2>, camera_error> project(const pinhole<T>& k, const linalg::fixed_matrix<T, 3, 3>& r, const linalg::fixed_vector<T, 3>& c,
                                                                                   const linalg::fixed_vector<T, 3>& x) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const auto p = linalg::transpose(r) * (x - c);
        if (!(p[2] > T(0)))
            return err(camera_error::behind_camera);
        return ok(linalg::fixed_vector<T, 2>{{k.fx * p[0] / p[2] + k.cx, k.fy * p[1] / p[2] + k.cy}});
    }

    /// \brief The world direction of a pixel's ray, `R K⁻¹ (u, v, 1)`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param k The intrinsics.
    /// \param r The orientation, camera to world.
    /// \param u The pixel.
    /// \return The direction, not normalised (its camera-frame z is 1).
    template <linalg::scalar T>
    [[nodiscard]] constexpr linalg::fixed_vector<T, 3> pixel_ray(const pinhole<T>& k, const linalg::fixed_matrix<T, 3, 3>& r, const linalg::fixed_vector<T, 2>& u) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        return r * linalg::fixed_vector<T, 3>{{(u[0] - k.cx) / k.fx, (u[1] - k.cy) / k.fy, T(1)}};
    }

    /// \brief Where a ray from `c` along `d` meets the plane `n·x = h`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param c The ray's origin.
    /// \param d The ray's direction.
    /// \param n The plane's normal.
    /// \param h The plane's offset.
    /// \return The point, or `camera_error::parallel_to_plane` or `camera_error::behind_camera`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr result<linalg::fixed_vector<T, 3>, camera_error> ray_plane(const linalg::fixed_vector<T, 3>& c, const linalg::fixed_vector<T, 3>& d,
                                                                                     const linalg::fixed_vector<T, 3>& n, T h) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const T nd = linalg::dot(n, d);
        const T lim = T(1e-12) * linalg::norm(n) * linalg::norm(d);
        if (!(nd > lim || nd < -lim))
            return err(camera_error::parallel_to_plane);
        const T t = (h - linalg::dot(n, c)) / nd;
        if (!(t > T(0)))
            return err(camera_error::behind_camera);
        return ok(c + d * t);
    }

    /// \brief The angle between a ray and a plane: `asin(|n·d| / (|n| |d|))`, radians; π/2 looking straight down.
    /// \tparam T The number type. \param d The ray's direction. \param n The plane's normal. \return The angle.
    /// \ingroup core
    template <linalg::scalar T>
    [[nodiscard]] constexpr T grazing_angle(const linalg::fixed_vector<T, 3>& d, const linalg::fixed_vector<T, 3>& n) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        T s = linalg::dot(n, d) / (linalg::norm(n) * linalg::norm(d));
        if (s < T(0))
            s = -s;
        return static_cast<T>(math::asin(s > T(1) ? T(1) : s));
    }

    /// \brief A fix on a plane from one pixel.
    /// \ingroup core
    /// \tparam T The number type.
    template <linalg::scalar T>
    struct plane_fix_result {
        linalg::fixed_vector<T, 3> point{};          ///< The point on the plane, world.
        linalg::fixed_matrix<T, 3, 3> covariance{};  ///< Its covariance, from the pixel's and the height's.
        T range{};                                   ///< The distance from the camera centre.
        T grazing{};                                 ///< The grazing angle, radians.
    };

    /// \brief The point where a pixel's ray meets the plane `n·x = h`, with its covariance by the exact Jacobian.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param k The intrinsics.
    /// \param r The orientation, camera to world.
    /// \param c The camera centre, world.
    /// \param u The pixel.
    /// \param pixel_covariance The pixel's 2 x 2 covariance, pixels² (add `(f σ_θ)²` for an orientation error σ_θ).
    /// \param n The plane's normal, unit length for `height_variance` to be in metres².
    /// \param h The plane's offset (the target's height on it).
    /// \param height_variance The variance of `h` (a target riding above the plane by an uncertain amount); 0 for none.
    /// \return The fix, or `camera_error::parallel_to_plane` or `camera_error::behind_camera`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr result<plane_fix_result<T>, camera_error> plane_fix(const pinhole<T>& k, const linalg::fixed_matrix<T, 3, 3>& r, const linalg::fixed_vector<T, 3>& c,
                                                                              const linalg::fixed_vector<T, 2>& u, const linalg::fixed_matrix<T, 2, 2>& pixel_covariance,
                                                                              const linalg::fixed_vector<T, 3>& n, T h, T height_variance = T(0)) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const auto d = pixel_ray(k, r, u);
        const auto x = ray_plane(c, d, n, h);
        if (!x)
            return err(x.unwrap_err_unchecked());
        const T nd = linalg::dot(n, d);
        const T t = (h - linalg::dot(n, c)) / nd;
        // ∂d/∂u = R e_x / fx, ∂d/∂v = R e_y / fy; ∂x/∂u_k = t (∂d_k - (n·∂d_k / n·d) d)
        linalg::fixed_matrix<T, 3, 2> j{};
        for (std::size_t q = 0; q < 2; ++q) {
            const T f = q == 0 ? k.fx : k.fy;
            linalg::fixed_vector<T, 3> dd{{r(0, q) / f, r(1, q) / f, r(2, q) / f}};
            const auto col = (dd - d * (linalg::dot(n, dd) / nd)) * t;
            for (std::size_t i = 0; i < 3; ++i)
                j(i, q) = col[i];
        }
        auto cov = linalg::sandwich(j, pixel_covariance);
        if (height_variance > T(0)) {
            const auto dh = d / nd;  // ∂x/∂h
            cov += linalg::outer(dh, dh) * height_variance;
        }
        return ok(plane_fix_result<T>{*x, cov, linalg::norm(*x - c), grazing_angle(d, n)});
    }

} // namespace snicholls::geometry

#endif /* snicholls_core_geometry_camera_hpp */
