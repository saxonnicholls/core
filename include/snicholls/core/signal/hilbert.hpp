//
//  hilbert.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::signal` analytic signal and Hilbert transform through the FFT, as `scipy.signal.hilbert`, with the envelope and the backward pass.
/// \ingroup core
///
/// **Overview.** `analytic_signal(plan, x, out, work)` is `scipy.signal.hilbert(x)`: the transform of `x`, the
/// negative frequencies set to zero and the positive ones doubled (bin 0 and, for even `n`, bin `n / 2` kept), and the
/// inverse. Its real part is `x`; its imaginary part is the Hilbert transform of `x` (a quarter-period phase shift of
/// every component), which `hilbert_transform` returns alone; its magnitude is the envelope (`envelope`), the
/// amplitude of a modulated carrier. `analytic_signal_backward` is the adjoint of the linear map from `x` to the
/// analytic signal, for training through it.
///
/// \snippet signal/hilbert_demo.cpp hilbert
///
/// **Relationships.** `signal/fft.hpp` (the plan). Group: \ref core.
///
/// **Implementations.** One: the FFT of the whole signal (scipy's), so the signal is treated as periodic.
///
/// **Guarantees.** `constexpr`, no allocation (work: `plan.work_size()`, plus `n` for `hilbert_transform` and
/// `envelope`). Complexity: two transforms of length `n`.
///
/// **Errors.** Preconditions, asserted: the plan's length is the signal's; the outputs' sizes are `n`.
///
/// **Performance.** Not benchmarked separately: two transforms (`signal/fft.hpp`'s benchmark).
///
/// **Tests and demos.** `tests/signal/hilbert_test.cpp` (a cosine's transform is the sine, scipy's values, the
/// envelope of a modulated tone, the backward pass by central differences), `demos/signal/hilbert_demo.cpp`.
///
/// **Dependencies.** `<cassert>`, `<cstddef>`, `<span>`, `<type_traits>`, `signal/fft.hpp`.
///
/// **Known issues.** The ends wrap around: a signal that does not fit a whole number of periods has ringing near
/// both ends, as with scipy. Window or pad it first.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_signal_hilbert_hpp
#define snicholls_core_signal_hilbert_hpp

#include <snicholls/core/signal/fft.hpp>

#include <cassert>
#include <cstddef>
#include <span>
#include <type_traits>

namespace snicholls::signal
{

namespace detail::hilbert
{
    // scipy's h: 1 at bin 0 and (even n) n / 2, 2 at the positive frequencies, 0 at the negative.
    template <typename T>
    constexpr T weight(std::size_t k, std::size_t n) noexcept
    {
        if (k == 0 || 2 * k == n)
            return T(1);
        return 2 * k < n ? T(2) : T(0);
    }

    template <typename T>
    constexpr void apply(const fft_plan<T>& plan, std::span<complex<T>> z, std::span<complex<T>> work) noexcept
    {
        const std::size_t n = plan.size();
        plan.template execute<false>(z, work);
        const T inv = T(1) / T(static_cast<long long>(n));
        for (std::size_t k = 0; k < n; ++k)
            z[k] *= weight<T>(k, n) * inv;
        plan.template execute<true>(z, work);
    }
} // namespace detail::hilbert

/// \brief The analytic signal `x + i H(x)`: `scipy.signal.hilbert(x)`.
/// \tparam T A `math::math_number`.
/// \param plan The plan of the signal's length.
/// \param x The real signal.
/// \param out `x.size()` complex values.
/// \param work At least `plan.work_size()` elements.
template <math::math_number T>
constexpr void analytic_signal(const fft_plan<T>& plan, std::span<const std::type_identity_t<T>> x, std::span<complex<std::type_identity_t<T>>> out,
                               std::span<complex<std::type_identity_t<T>>> work) noexcept
{
    const std::size_t n = plan.size();
    assert(x.size() == n && out.size() == n);
    if (n == 0)
        return;
    for (std::size_t i = 0; i < n; ++i)
        out[i] = {x[i], T(0)};
    detail::hilbert::apply(plan, out, work);
}

/// \brief The Hilbert transform `H(x)` alone: the imaginary part of the analytic signal.
/// \tparam T A `math::math_number`.
/// \param plan The plan of the signal's length.
/// \param x The real signal.
/// \param out `x.size()` values.
/// \param work At least `plan.size() + plan.work_size()` elements.
template <math::math_number T>
constexpr void hilbert_transform(const fft_plan<T>& plan, std::span<const std::type_identity_t<T>> x, std::span<std::type_identity_t<T>> out,
                                 std::span<complex<std::type_identity_t<T>>> work) noexcept
{
    const std::size_t n = plan.size();
    assert(out.size() == n && work.size() >= n + plan.work_size());
    const auto z = work.first(n);
    analytic_signal(plan, x, z, work.subspan(n));
    for (std::size_t i = 0; i < n; ++i)
        out[i] = z[i].im;
}

/// \brief The envelope `|x + i H(x)|`: the instantaneous amplitude.
/// \tparam T A `math::math_number`.
/// \param plan The plan of the signal's length.
/// \param x The real signal.
/// \param out `x.size()` values.
/// \param work At least `plan.size() + plan.work_size()` elements.
template <math::math_number T>
constexpr void envelope(const fft_plan<T>& plan, std::span<const std::type_identity_t<T>> x, std::span<std::type_identity_t<T>> out,
                        std::span<complex<std::type_identity_t<T>>> work) noexcept
{
    const std::size_t n = plan.size();
    assert(out.size() == n && work.size() >= n + plan.work_size());
    const auto z = work.first(n);
    analytic_signal(plan, x, z, work.subspan(n));
    for (std::size_t i = 0; i < n; ++i)
        out[i] = abs(z[i]);
}

/// \brief The backward pass of `analytic_signal`: given `dL/dz` (PyTorch's convention), `dL/dx`.
/// \tparam T A `math::math_number`.
/// \param plan The plan of the signal's length.
/// \param grad_z `dL/dz`, `n` complex values; overwritten (used as the buffer).
/// \param grad_x `dL/dx`, `n` values, written.
/// \param work At least `plan.work_size()` elements.
template <math::math_number T>
constexpr void analytic_signal_backward(const fft_plan<T>& plan, std::span<complex<std::type_identity_t<T>>> grad_z,
                                        std::span<std::type_identity_t<T>> grad_x, std::span<complex<std::type_identity_t<T>>> work) noexcept
{
    // z = A x with A = F^-1 D F (D real, diagonal); for real x, dL/dx = Re(A^H g) = Re(F^H D F^-H g) = Re(F^-1 D F g),
    // since F^H = n F^-1 and F^-H = F / n. So it is the same map applied to g, real part taken.
    const std::size_t n = plan.size();
    assert(grad_z.size() == n && grad_x.size() == n);
    if (n == 0)
        return;
    detail::hilbert::apply(plan, grad_z, work);
    for (std::size_t i = 0; i < n; ++i)
        grad_x[i] = grad_z[i].re;
}

} // namespace snicholls::signal

#endif // snicholls_core_signal_hilbert_hpp
