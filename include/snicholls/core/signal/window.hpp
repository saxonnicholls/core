//
//  window.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::signal` window functions - rectangular, Hann, Hamming, Blackman, Blackman-Harris, Kaiser, Gaussian, Tukey - periodic or symmetric, as `constexpr` values, filled spans and compile-time tables; and the COLA and NOLA conditions of an overlap-add.
/// \ingroup core
///
/// **Overview.** Each window is a small type whose `at<T>(i, m)` is the value at sample `i` with denominator `m`:
/// the formulas of `scipy.signal.windows` (which PyTorch's `*_window` share), with `x = i / m`:
///
/// | Window | `w(i)` |
/// |---|---|
/// | `rectangular` | 1 |
/// | `hann` | `0.5 - 0.5 cos 2 pi x` |
/// | `hamming` | `0.54 - 0.46 cos 2 pi x` |
/// | `blackman` | `0.42 - 0.5 cos 2 pi x + 0.08 cos 4 pi x` |
/// | `blackman_harris` | `0.35875 - 0.48829 cos 2 pi x + 0.14128 cos 4 pi x - 0.01168 cos 6 pi x` (four terms, minimum sidelobe) |
/// | `kaiser{beta}` | `I0(beta sqrt(1 - (2x - 1)^2)) / I0(beta)` |
/// | `gaussian{sigma}` | `exp(-(i - m / 2)^2 / (2 sigma^2))`, `sigma` in samples |
/// | `tukey{alpha}` | a cosine taper over `alpha / 2` of each end, flat between (scipy's index arithmetic) |
///
/// A **symmetric** window of length `n` has `m = n - 1` and is the one FIR design uses; a **periodic** window has
/// `m = n` (a symmetric window of `n + 1` points without its last), the one spectral analysis uses, so that
/// overlap-adding it at a hop of `n / 2` (Hann) is constant. `window_symmetry::periodic` is the default, as in
/// PyTorch and `scipy.signal.get_window`. A window of length 1 is `{1}` either way.
///
/// `fill_window(out, w, symmetry)` fills a span; `make_window<T, N>(w, symmetry)` returns a `std::array`, so
/// `constexpr auto hann400 = make_window<float, 400>(hann{});` is computed by the compiler. `is_cola` and `is_nola`
/// check that a window and hop overlap-add to a constant (constant overlap-add, for reconstruction without dividing)
/// or to something non-zero everywhere (nonzero overlap-add, which `istft` needs), as `scipy.signal.check_COLA` and
/// `check_NOLA` do.
///
/// \snippet signal/window_demo.cpp window
///
/// **Relationships.** `signal/stft.hpp`, `signal/fir.hpp` and `signal/resample.hpp` take these windows;
/// `math::bessel_i0` is Kaiser's. Group: \ref core.
///
/// **Implementations.** One formula per window, in `compute_t<T>` (`double` for `float`) and rounded once.
///
/// **Guarantees.** Every function is `constexpr` and `noexcept`; the window types are trivially copyable and
/// structural. The cosine sums, Kaiser and Gaussian are symmetric to the bit in their symmetric form (`w(i)` is
/// computed from `min(i, m - i)`); Tukey follows scipy's index arithmetic. Values agree with scipy's to a few units in
/// the last place (scipy sums the cosines in another order). Complexity:
/// linear in the length; `is_cola` and `is_nola` are linear in the window.
///
/// **Errors.** None; a Kaiser `beta` beyond about 700 overflows `I0` to infinity and gives NaN.
///
/// **Performance.** Not benchmarked: a window is built once, usually at compile time.
///
/// **Tests and demos.** `tests/signal/window_test.cpp` (scipy's values, the symmetries, compile-time tables, COLA
/// and NOLA), `demos/signal/window_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<span>`, `snicholls/core/math/constexpr_math.hpp`,
/// `signal/complex.hpp` (for `compute_t`).
///
/// **Known issues.** scipy's Tukey window computes its taper's width with `floor`, so two lengths that differ by one
/// can have tapers of the same width; this is copied, not corrected.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_signal_window_hpp
#define snicholls_core_signal_window_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/signal/complex.hpp>

#include <array>
#include <cstddef>
#include <span>

namespace snicholls::signal
{

/// \brief Whether a window is symmetric (for filter design) or periodic (for spectral analysis).
/// \ingroup core
enum class window_symmetry
{
    periodic,    ///< `m = n`: one period of a window of `n + 1` points, without its last.
    symmetric,   ///< `m = n - 1`: `w(i) = w(n - 1 - i)`.
};

namespace detail::window
{
    template <typename C>
    constexpr C cos2pi(C x) noexcept
    {
        return math::cos(C(2) * math::math_limits<C>::pi * x);
    }

    // a0 - a1 cos 2 pi x + a2 cos 4 pi x - a3 cos 6 pi x, with i folded to min(i, m - i) so the symmetric form is exact.
    template <typename T>
    constexpr T cosine_sum(std::size_t i, std::size_t m, double a0, double a1, double a2, double a3) noexcept
    {
        using C = compute_t<T>;
        if (m == 0)
            return T(1);
        const std::size_t f = 2 * i > m ? m - i : i;
        const C x = C(static_cast<long long>(f)) / C(static_cast<long long>(m));
        return static_cast<T>(C(a0) - C(a1) * cos2pi(x) + C(a2) * cos2pi(C(2) * x) - C(a3) * cos2pi(C(3) * x));
    }
} // namespace detail::window

/// \brief The rectangular window: 1 everywhere.
/// \ingroup core
struct rectangular
{
    /// \brief The value at sample `i`.
    /// \tparam T A `math::math_number`.
    /// \return 1.
    template <math::math_number T>
    constexpr T at(std::size_t, std::size_t) const noexcept { return T(1); }
};

/// \brief The Hann window, `0.5 - 0.5 cos 2 pi i / m`.
/// \ingroup core
struct hann
{
    /// \brief The value at sample `i`.
    /// \tparam T A `math::math_number`.
    /// \param i The sample.
    /// \param m The denominator: `n` periodic, `n - 1` symmetric.
    /// \return The value.
    template <math::math_number T>
    constexpr T at(std::size_t i, std::size_t m) const noexcept { return detail::window::cosine_sum<T>(i, m, 0.5, 0.5, 0, 0); }
};

/// \brief The Hamming window, `0.54 - 0.46 cos 2 pi i / m`.
/// \ingroup core
struct hamming
{
    /// \brief The value at sample `i`.
    /// \tparam T A `math::math_number`.
    /// \param i The sample.
    /// \param m The denominator.
    /// \return The value.
    template <math::math_number T>
    constexpr T at(std::size_t i, std::size_t m) const noexcept { return detail::window::cosine_sum<T>(i, m, 0.54, 0.46, 0, 0); }
};

/// \brief The Blackman window, `0.42 - 0.5 cos 2 pi x + 0.08 cos 4 pi x`.
/// \ingroup core
struct blackman
{
    /// \brief The value at sample `i`.
    /// \tparam T A `math::math_number`.
    /// \param i The sample.
    /// \param m The denominator.
    /// \return The value.
    template <math::math_number T>
    constexpr T at(std::size_t i, std::size_t m) const noexcept { return detail::window::cosine_sum<T>(i, m, 0.42, 0.5, 0.08, 0); }
};

/// \brief The four-term Blackman-Harris window (minimum sidelobe, about -92 dB), scipy's coefficients.
/// \ingroup core
struct blackman_harris
{
    /// \brief The value at sample `i`.
    /// \tparam T A `math::math_number`.
    /// \param i The sample.
    /// \param m The denominator.
    /// \return The value.
    template <math::math_number T>
    constexpr T at(std::size_t i, std::size_t m) const noexcept
    {
        return detail::window::cosine_sum<T>(i, m, 0.35875, 0.48829, 0.14128, 0.01168);
    }
};

/// \brief The Kaiser window, `I0(beta sqrt(1 - (2 i / m - 1)^2)) / I0(beta)`.
/// \ingroup core
struct kaiser
{
    double beta = 12.0;   ///< The shape: 0 is rectangular, 5 is like Hamming, 8.6 like Blackman; larger is narrower.

    /// \brief The value at sample `i`.
    /// \tparam T A `math::math_number`.
    /// \param i The sample.
    /// \param m The denominator.
    /// \return The value.
    template <math::math_number T>
    constexpr T at(std::size_t i, std::size_t m) const noexcept
    {
        using C = compute_t<T>;
        if (m == 0)
            return T(1);
        const std::size_t f = 2 * i > m ? m - i : i;
        const C r = C(2) * C(static_cast<long long>(f)) / C(static_cast<long long>(m)) - C(1);
        const C b = C(beta);
        return static_cast<T>(math::bessel_i0(b * math::sqrt(C(1) - r * r)) / math::bessel_i0(b));
    }
};

/// \brief The Gaussian window, `exp(-(i - m / 2)^2 / (2 sigma^2))`.
/// \ingroup core
struct gaussian
{
    double sigma = 1.0;   ///< The standard deviation, in samples.

    /// \brief The value at sample `i`.
    /// \tparam T A `math::math_number`.
    /// \param i The sample.
    /// \param m The denominator.
    /// \return The value.
    template <math::math_number T>
    constexpr T at(std::size_t i, std::size_t m) const noexcept
    {
        using C = compute_t<T>;
        const std::size_t f = 2 * i > m ? m - i : i;
        const C d = C(static_cast<long long>(f)) - C(static_cast<long long>(m)) / C(2);
        const C s = C(sigma);
        return static_cast<T>(math::exp(-d * d / (C(2) * s * s)));
    }
};

/// \brief The Tukey (tapered cosine) window: a cosine taper over a fraction `alpha` of the window, flat between; `alpha` 0 is rectangular, 1 is Hann.
/// \ingroup core
struct tukey
{
    double alpha = 0.5;   ///< The fraction of the window inside the tapers.

    /// \brief The value at sample `i`, as `scipy.signal.windows.tukey` computes it.
    /// \tparam T A `math::math_number`.
    /// \param i The sample.
    /// \param m The denominator.
    /// \return The value.
    template <math::math_number T>
    constexpr T at(std::size_t i, std::size_t m) const noexcept
    {
        using C = compute_t<T>;
        if (alpha <= 0 || m == 0)
            return T(1);
        if (alpha >= 1)
            return hann{}.at<T>(i, m);
        const C a = C(alpha), mm = C(static_cast<long long>(m)), pi = math::math_limits<C>::pi;
        const auto width = static_cast<std::size_t>(math::floor(a * mm / C(2)));
        const C x = C(static_cast<long long>(i));
        if (i <= width)
            return static_cast<T>(C(0.5) * (C(1) + math::cos(pi * (C(-1) + C(2) * x / a / mm))));
        if (i < m - width)
            return T(1);
        return static_cast<T>(C(0.5) * (C(1) + math::cos(pi * (C(-2) / a + C(1) + C(2) * x / a / mm))));
    }
};

/// \brief The denominator of a window of `n` points: `n` periodic, `n - 1` symmetric.
/// \param n The length.
/// \param symmetry The form.
/// \return `m`.
constexpr std::size_t window_denominator(std::size_t n, window_symmetry symmetry) noexcept
{
    return symmetry == window_symmetry::periodic ? n : (n == 0 ? 0 : n - 1);
}

/// \brief One value of a window of `n` points.
/// \tparam T A `math::math_number`.
/// \tparam W A window type.
/// \param w The window.
/// \param i The sample, below `n`.
/// \param n The length.
/// \param symmetry The form.
/// \return The value; 1 for a window of one point.
template <math::math_number T, typename W>
constexpr T window_value(const W& w, std::size_t i, std::size_t n, window_symmetry symmetry = window_symmetry::periodic) noexcept
{
    return n == 1 ? T(1) : w.template at<T>(i, window_denominator(n, symmetry));
}

/// \brief Fills `out` with a window of `out.size()` points.
/// \tparam T A `math::math_number`.
/// \tparam W A window type.
/// \param out The window.
/// \param w The window type's value.
/// \param symmetry The form.
template <math::math_number T, typename W>
constexpr void fill_window(std::span<T> out, const W& w, window_symmetry symmetry = window_symmetry::periodic) noexcept
{
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = window_value<T>(w, i, out.size(), symmetry);
}

/// \brief A window of `N` points as a `std::array`: a compile-time table when the result is `constexpr`.
/// \tparam T A `math::math_number`.
/// \tparam N The length.
/// \tparam W A window type.
/// \param w The window.
/// \param symmetry The form.
/// \return The table.
template <math::math_number T, std::size_t N, typename W>
constexpr std::array<T, N> make_window(const W& w, window_symmetry symmetry = window_symmetry::periodic) noexcept
{
    std::array<T, N> out{};
    fill_window(std::span<T>(out), w, symmetry);
    return out;
}

namespace detail::window
{
    // The sums of w^Power over every position modulo the hop, as scipy's check_COLA / check_NOLA fold them.
    template <int Power, typename T, typename Check>
    constexpr bool fold(std::span<const T> w, std::size_t hop, Check check) noexcept
    {
        if (hop == 0 || hop > w.size())
            return false;
        for (std::size_t r = 0; r < hop; ++r) {
            T s = T(0);
            for (std::size_t i = r; i < w.size(); i += hop)
                s += Power == 1 ? w[i] : w[i] * w[i];
            if (!check(r, s))
                return false;
        }
        return true;
    }
} // namespace detail::window

/// \brief Whether a window overlap-added at `hop` sums to a constant (constant overlap-add): `scipy.signal.check_COLA`.
/// \tparam T A `math::math_number`.
/// \param w The window.
/// \param hop The hop, between 1 and the window's length.
/// \param tolerance The largest deviation of a position's sum from the first's.
/// \return The result; `false` for a hop of 0 or longer than the window.
template <math::math_number T>
constexpr bool is_cola(std::span<const T> w, std::size_t hop, T tolerance = T(1e-10)) noexcept
{
    T first = T(0);
    return detail::window::fold<1>(w, hop, [&](std::size_t r, T s) {
        if (r == 0)
            first = s;
        return math::abs(s - first) <= tolerance;
    });
}

/// \brief Whether a window's square overlap-added at `hop` is above `tolerance` everywhere (nonzero overlap-add), the condition `istft` needs: `scipy.signal.check_NOLA`.
/// \tparam T A `math::math_number`.
/// \param w The window.
/// \param hop The hop, between 1 and the window's length.
/// \param tolerance The least sum allowed.
/// \return The result; `false` for a hop of 0 or longer than the window.
template <math::math_number T>
constexpr bool is_nola(std::span<const T> w, std::size_t hop, T tolerance = T(1e-10)) noexcept
{
    return detail::window::fold<2>(w, hop, [tolerance](std::size_t, T s) { return s > tolerance; });
}

} // namespace snicholls::signal

#endif // snicholls_core_signal_window_hpp
