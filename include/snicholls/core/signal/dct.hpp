//
//  dct.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::signal` discrete cosine and sine transforms of types II, III and IV through the FFT, with scipy's definitions and its orthonormal option, and the backward pass of the DCT-II.
/// \ingroup core
///
/// **Overview.** With `scipy.fft`'s definitions (`norm=None`), for `n` points:
///
/// | Transform | `y_k` |
/// |---|---|
/// | `dct2` | `2 sum_j x_j cos(pi k (2j + 1) / 2n)` |
/// | `dct3` | `x_0 + 2 sum_{j>=1} x_j cos(pi j (2k + 1) / 2n)` |
/// | `dct4` | `2 sum_j x_j cos(pi (2j + 1)(2k + 1) / 4n)` |
/// | `dst2` | `2 sum_j x_j sin(pi (k + 1)(2j + 1) / 2n)` |
/// | `dst3` | `(-1)^k x_{n-1} + 2 sum_{j<n-1} x_j sin(pi (j + 1)(2k + 1) / 2n)` |
/// | `dst4` | `2 sum_j x_j sin(pi (2j + 1)(2k + 1) / 4n)` |
///
/// `dct_norm::ortho` scales each to an orthonormal matrix, as scipy's `norm="ortho"`: the DCT-II's `y_0` by
/// `sqrt(1 / 4n)` and the rest by `sqrt(1 / 2n)`, the DCT-III by the inverse of that, the type IV by `sqrt(1 / 2n)`;
/// then the DCT-III inverts the DCT-II exactly and the type IV is its own inverse. Unnormalised, `dct3(dct2(x)) = 2n x`.
/// The DCT-II with `ortho` is the MFCC's transform (`signal/mel.hpp`), and it may return only its first coefficients.
///
/// Types II and III run Makhoul's permutation through one complex transform of length `n` (an `fft_plan` of `n`);
/// type IV a pre-twiddled complex transform of length `2n` (an `fft_plan` of `2n`). The DSTs are DCTs of the reversed
/// or alternated sequence, so every transform is `O(n log n)` for every `n`.
///
/// \snippet signal/dct_demo.cpp dct
///
/// **Relationships.** `signal/fft.hpp` (the plans); `signal/mel.hpp` (`mfcc`). Group: \ref core.
///
/// **Implementations.** One per type, with `scipy.fft`'s conventions; types I are not provided.
///
/// **Guarantees.** `constexpr`; no allocation (work: `dct_work_size(plan)` complex elements). Complexity: one
/// transform of the plan's length and `O(n)` twiddles computed as they are used (a sine and a cosine each).
///
/// **Errors.** Preconditions, asserted: the plan's length is `n` (`2n` for type IV); the output has `n` elements
/// (`dct2` may have fewer); the work buffer is large enough.
///
/// **Performance.** Not benchmarked separately: one complex transform (`signal/fft.hpp`'s benchmark) and `O(n)`
/// trigonometry.
///
/// **Tests and demos.** `tests/signal/dct_test.cpp` (against the definitions in `long double` and scipy's values,
/// inverses, orthonormality, the backward pass by central differences), `demos/signal/dct_demo.cpp`.
///
/// **Dependencies.** `<cassert>`, `<cstddef>`, `<span>`, `<type_traits>`, `signal/fft.hpp`.
///
/// **Known issues.** A real-input transform of half the length would halve the cost of types II and III; the
/// complex transform keeps the code short.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_signal_dct_hpp
#define snicholls_core_signal_dct_hpp

#include <snicholls/core/signal/fft.hpp>

#include <cassert>
#include <cstddef>
#include <span>
#include <type_traits>

namespace snicholls::signal
{

/// \brief How a cosine or sine transform is scaled, as scipy's `norm`.
/// \ingroup core
enum class dct_norm
{
    none,    ///< scipy's `norm=None` (`"backward"`).
    ortho,   ///< An orthonormal matrix.
};

/// \brief The work buffer a cosine or sine transform needs with this plan: its length and its work.
/// \tparam T A `math::math_number`.
/// \param plan The plan (of `n`, or `2n` for type IV).
/// \return The size, in complex elements.
template <math::math_number T>
constexpr std::size_t dct_work_size(const fft_plan<T>& plan) noexcept
{
    return plan.size() + plan.work_size();
}

namespace detail::dct
{
    template <typename T>
    constexpr T inv_sqrt(std::size_t m) noexcept
    {
        return T(1) / math::sqrt(T(static_cast<long long>(m)));
    }

    // DCT-II of in(j), j < n, handing y_k for k < count to out(k, y_k).
    template <typename T, typename In, typename Out>
    constexpr void dct2(const fft_plan<T>& plan, In&& in, std::size_t count, Out&& out, std::span<complex<T>> work, dct_norm norm) noexcept
    {
        const std::size_t n = plan.size();
        assert(work.size() >= dct_work_size(plan) && count <= n);
        const auto v = work.first(n);
        for (std::size_t j = 0; 2 * j < n; ++j)
            v[j] = {in(2 * j), T(0)};
        for (std::size_t j = 0; 2 * j + 1 < n; ++j)
            v[n - 1 - j] = {in(2 * j + 1), T(0)};
        plan.template execute<false>(v, work.subspan(n));
        const T s0 = norm == dct_norm::ortho ? inv_sqrt<T>(4 * n) : T(1), s = norm == dct_norm::ortho ? inv_sqrt<T>(2 * n) : T(1);
        for (std::size_t k = 0; k < count; ++k)
            out(k, T(2) * (twiddle<T>(k, 4 * n) * v[k]).re * (k == 0 ? s0 : s));
    }

    // DCT-III of in(j), j < n, handing y_k for every k to out(k, y_k).
    template <typename T, typename In, typename Out>
    constexpr void dct3(const fft_plan<T>& plan, In&& in, Out&& out, std::span<complex<T>> work, dct_norm norm) noexcept
    {
        const std::size_t n = plan.size();
        assert(work.size() >= dct_work_size(plan));
        const auto v = work.first(n);
        const T s0 = norm == dct_norm::ortho ? inv_sqrt<T>(n) : T(1), s = norm == dct_norm::ortho ? inv_sqrt<T>(2 * n) : T(1);
        for (std::size_t k = 0; k < n; ++k) {
            const T a = in(k) * (k == 0 ? s0 : s);
            const T b = k == 0 ? T(0) : in(n - k) * s;
            v[k] = conj(twiddle<T>(k, 4 * n)) * complex<T>{a, -b};
        }
        plan.template execute<true>(v, work.subspan(n));
        for (std::size_t j = 0; 2 * j < n; ++j)
            out(2 * j, v[j].re);
        for (std::size_t j = 0; 2 * j + 1 < n; ++j)
            out(2 * j + 1, v[n - 1 - j].re);
    }

    // DCT-IV of in(j), j < n, through a transform of length 2n.
    template <typename T, typename In, typename Out>
    constexpr void dct4(const fft_plan<T>& plan, In&& in, Out&& out, std::span<complex<T>> work, dct_norm norm) noexcept
    {
        const std::size_t m = plan.size(), n = m / 2;
        assert(m % 2 == 0 && work.size() >= dct_work_size(plan));
        const auto v = work.first(m);
        for (std::size_t j = 0; j < n; ++j)
            v[j] = twiddle<T>(j, 4 * n) * in(j);
        for (std::size_t j = n; j < m; ++j)
            v[j] = {};
        plan.template execute<false>(v, work.subspan(m));
        const T s = norm == dct_norm::ortho ? inv_sqrt<T>(2 * n) : T(1);
        for (std::size_t k = 0; k < n; ++k)
            out(k, T(2) * (twiddle<T>(2 * k + 1, 8 * n) * v[k]).re * s);
    }

    constexpr bool odd(std::size_t k) noexcept { return k % 2 == 1; }
} // namespace detail::dct

/// \brief The DCT-II: `y_k = 2 sum_j x_j cos(pi k (2j + 1) / 2n)`, or its orthonormal form; the first `out.size()` coefficients.
/// \tparam T A `math::math_number`.
/// \param plan The complex plan of length `n`.
/// \param x `n` values.
/// \param out At most `n` coefficients: the first ones.
/// \param work At least `dct_work_size(plan)` elements.
/// \param norm The scaling.
template <math::math_number T>
constexpr void dct2(const fft_plan<T>& plan, std::span<const std::type_identity_t<T>> x, std::span<std::type_identity_t<T>> out,
                    std::span<complex<std::type_identity_t<T>>> work, dct_norm norm = dct_norm::none) noexcept
{
    assert(x.size() == plan.size());
    detail::dct::dct2(plan, [x](std::size_t j) { return x[j]; }, out.size(), [out](std::size_t k, T v) { out[k] = v; }, work, norm);
}

/// \brief The DCT-III: `y_k = x_0 + 2 sum_{j>=1} x_j cos(pi j (2k + 1) / 2n)`, or its orthonormal form (the inverse of `dct2`'s).
/// \tparam T A `math::math_number`.
/// \param plan The complex plan of length `n`.
/// \param x `n` values.
/// \param out `n` values.
/// \param work At least `dct_work_size(plan)` elements.
/// \param norm The scaling.
template <math::math_number T>
constexpr void dct3(const fft_plan<T>& plan, std::span<const std::type_identity_t<T>> x, std::span<std::type_identity_t<T>> out,
                    std::span<complex<std::type_identity_t<T>>> work, dct_norm norm = dct_norm::none) noexcept
{
    assert(x.size() == plan.size() && out.size() == plan.size());
    detail::dct::dct3(plan, [x](std::size_t j) { return x[j]; }, [out](std::size_t k, T v) { out[k] = v; }, work, norm);
}

/// \brief The DCT-IV: `y_k = 2 sum_j x_j cos(pi (2j + 1)(2k + 1) / 4n)`, or its orthonormal form (its own inverse).
/// \tparam T A `math::math_number`.
/// \param plan The complex plan of length `2n`.
/// \param x `n` values.
/// \param out `n` values.
/// \param work At least `dct_work_size(plan)` elements.
/// \param norm The scaling.
template <math::math_number T>
constexpr void dct4(const fft_plan<T>& plan, std::span<const std::type_identity_t<T>> x, std::span<std::type_identity_t<T>> out,
                    std::span<complex<std::type_identity_t<T>>> work, dct_norm norm = dct_norm::none) noexcept
{
    assert(2 * x.size() == plan.size() && out.size() == x.size());
    detail::dct::dct4(plan, [x](std::size_t j) { return x[j]; }, [out](std::size_t k, T v) { out[k] = v; }, work, norm);
}

/// \brief The DST-II: `y_k = 2 sum_j x_j sin(pi (k + 1)(2j + 1) / 2n)`, or its orthonormal form.
/// \tparam T A `math::math_number`.
/// \param plan The complex plan of length `n`.
/// \param x `n` values.
/// \param out `n` values.
/// \param work At least `dct_work_size(plan)` elements.
/// \param norm The scaling.
template <math::math_number T>
constexpr void dst2(const fft_plan<T>& plan, std::span<const std::type_identity_t<T>> x, std::span<std::type_identity_t<T>> out,
                    std::span<complex<std::type_identity_t<T>>> work, dct_norm norm = dct_norm::none) noexcept
{
    const std::size_t n = plan.size();
    assert(x.size() == n && out.size() == n);
    detail::dct::dct2(plan, [x](std::size_t j) { return detail::dct::odd(j) ? -x[j] : x[j]; }, n,
                      [out, n](std::size_t k, T v) { out[n - 1 - k] = v; }, work, norm);
}

/// \brief The DST-III: `y_k = (-1)^k x_{n-1} + 2 sum_{j<n-1} x_j sin(pi (j + 1)(2k + 1) / 2n)`, or its orthonormal form (the inverse of `dst2`'s).
/// \tparam T A `math::math_number`.
/// \param plan The complex plan of length `n`.
/// \param x `n` values.
/// \param out `n` values.
/// \param work At least `dct_work_size(plan)` elements.
/// \param norm The scaling.
template <math::math_number T>
constexpr void dst3(const fft_plan<T>& plan, std::span<const std::type_identity_t<T>> x, std::span<std::type_identity_t<T>> out,
                    std::span<complex<std::type_identity_t<T>>> work, dct_norm norm = dct_norm::none) noexcept
{
    const std::size_t n = plan.size();
    assert(x.size() == n && out.size() == n);
    detail::dct::dct3(plan, [x, n](std::size_t j) { return x[n - 1 - j]; },
                      [out](std::size_t k, T v) { out[k] = detail::dct::odd(k) ? -v : v; }, work, norm);
}

/// \brief The DST-IV: `y_k = 2 sum_j x_j sin(pi (2j + 1)(2k + 1) / 4n)`, or its orthonormal form (its own inverse).
/// \tparam T A `math::math_number`.
/// \param plan The complex plan of length `2n`.
/// \param x `n` values.
/// \param out `n` values.
/// \param work At least `dct_work_size(plan)` elements.
/// \param norm The scaling.
template <math::math_number T>
constexpr void dst4(const fft_plan<T>& plan, std::span<const std::type_identity_t<T>> x, std::span<std::type_identity_t<T>> out,
                    std::span<complex<std::type_identity_t<T>>> work, dct_norm norm = dct_norm::none) noexcept
{
    const std::size_t n = x.size();
    assert(2 * n == plan.size() && out.size() == n);
    detail::dct::dct4(plan, [x, n](std::size_t j) { return x[n - 1 - j]; },
                      [out](std::size_t k, T v) { out[k] = detail::dct::odd(k) ? -v : v; }, work, norm);
}

/// \brief The backward pass of `dct2`: given `dL/dy` for its first `grad_y.size()` coefficients, `dL/dx` (the transpose, a DCT-III).
/// \tparam T A `math::math_number`.
/// \param plan The complex plan of length `n`.
/// \param grad_y `dL/dy`, at most `n` values (the rest are taken as 0).
/// \param grad_x `n` values of `dL/dx`, written.
/// \param work At least `dct_work_size(plan)` elements.
/// \param norm The forward transform's scaling.
template <math::math_number T>
constexpr void dct2_backward(const fft_plan<T>& plan, std::span<const std::type_identity_t<T>> grad_y, std::span<std::type_identity_t<T>> grad_x,
                             std::span<complex<std::type_identity_t<T>>> work, dct_norm norm = dct_norm::none) noexcept
{
    assert(grad_y.size() <= plan.size() && grad_x.size() == plan.size());
    // Unnormalised, C2^T = C3 diag(2, 1, ..., 1); orthonormal, the transpose is the inverse, the orthonormal DCT-III.
    const T first = norm == dct_norm::ortho ? T(1) : T(2);
    detail::dct::dct3(plan, [grad_y, first](std::size_t k) { return k < grad_y.size() ? grad_y[k] * (k == 0 ? first : T(1)) : T(0); },
                      [grad_x](std::size_t j, T v) { grad_x[j] = v; }, work, norm);
}

} // namespace snicholls::signal

#endif // snicholls_core_signal_dct_hpp
