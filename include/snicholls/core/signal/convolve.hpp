//
//  convolve.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::signal` convolution and correlation of real sequences, direct and through the FFT, in scipy's `full`, `same` and `valid` modes, and block filtering by overlap-add and overlap-save.
/// \ingroup core
///
/// **Overview.** `convolve(a, b, out, mode)` is `scipy.signal.convolve`: `full` gives `na + nb - 1` values
/// `c_k = sum_j a_j b_(k-j)`; `same` the `na` values centred on it (starting at `(nb - 1) / 2`); `valid` the
/// `max - min + 1` values where one sequence covers the other. `correlate` is `scipy.signal.correlate`,
/// `c_k = sum_j a_(j+k) b_j`, the convolution with `b` reversed (lags from `-(nb - 1)` in `full` mode).
///
/// - `convolve` / `correlate`: direct, `O(na nb)`, the reference and the fast path for short kernels; `constexpr`.
/// - `fft_convolve` / `fft_correlate`: one complex transform of both real inputs packed as `a + i b`, the product of
///   the two spectra separated from it, and one inverse; the plan's length must be at least `na + nb - 1`.
/// - `overlap_add(plan, h, x, out)`: the full convolution of a long signal with a filter of `L` taps, in blocks of
///   `plan.size() - L + 1` samples, each block's tail added to the next.
/// - `overlap_save(plan, h, x, out)`: the causal filter output `y_n = sum_k h_k x_(n-k)`, `x.size()` samples, each
///   block transformed with the `L - 1` samples before it and the wrapped part discarded.
///
/// \snippet signal/convolve_demo.cpp convolve
///
/// **Relationships.** `signal/fft.hpp` (the plans); `signal/fir.hpp` (filters to apply); `signal/resample.hpp`.
/// Group: \ref core.
///
/// **Implementations.** Direct; FFT of the full length; overlap-add and overlap-save over blocks. For a kernel of
/// fewer than about 32 taps the direct form is faster; above, overlap-save wins, and the full-length transform for
/// two long sequences.
///
/// **Guarantees.** `constexpr`; no allocation (work buffers from the caller: `fft_convolve_work_size(plan)`,
/// `block_filter_work_size(plan)`). The FFT forms agree with the direct one within
/// `4 eps ceil(log2 n) ||a||_2 ||b||_2`. Complexity: `O(n log n)` for `fft_convolve`, `O(N log n)` for a signal of `N`
/// samples through the block forms.
///
/// **Errors.** Preconditions, asserted: the output's size is the mode's; the plan is long enough
/// (`na + nb - 1`, or more than the filter for the block forms); `correlate` in `valid` mode needs `na >= nb`.
/// Empty inputs give empty outputs.
///
/// **Performance.** `benchmarks/signal/convolve_benchmark.cpp`: `convolve` against the same loop by hand (bit-identical),
/// and `fft_convolve` and `overlap_save` against it, on 16,384 samples. Intel Xeon W-3245, AppleClang 17, Release,
/// 2026-11-20, a shared machine under load, so the pairs vary by 30% from run to run (`BENCHMARKS.md`): 16 taps,
/// direct 0.19 ms, overlap-save (blocks of 64) 0.42 ms; 64 taps, direct 0.81 ms, overlap-save 0.36 ms; 512 taps,
/// direct 8.5 ms, overlap-save 0.43 ms, `fft_convolve` 0.9 ms at every length.
///
/// **Tests and demos.** `tests/signal/convolve_test.cpp` (scipy's values for every mode, the FFT and block forms
/// against the direct one, compile-time use), `demos/signal/convolve_demo.cpp`.
///
/// **Dependencies.** `<cassert>`, `<cstddef>`, `<span>`, `<type_traits>`, `signal/fft.hpp`.
///
/// **Known issues.** Real sequences only. The block forms transform one block per pass; packing two real blocks
/// into one complex transform would halve their cost.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_signal_convolve_hpp
#define snicholls_core_signal_convolve_hpp

#include <snicholls/core/signal/fft.hpp>

#include <cassert>
#include <cstddef>
#include <span>
#include <type_traits>

namespace snicholls::signal
{

/// \brief Which part of a convolution to return, as scipy's `mode`.
/// \ingroup core
enum class convolve_mode
{
    full,    ///< All `na + nb - 1` values.
    same,    ///< `na` values, centred on the full output.
    valid,   ///< The `max(na, nb) - min(na, nb) + 1` values where one sequence covers the other.
};

/// \brief The output's length for a mode.
/// \param na The first sequence's length.
/// \param nb The second's.
/// \param mode The mode.
/// \return The length; 0 if either is empty.
constexpr std::size_t convolve_size(std::size_t na, std::size_t nb, convolve_mode mode) noexcept
{
    if (na == 0 || nb == 0)
        return 0;
    switch (mode) {
    case convolve_mode::full: return na + nb - 1;
    case convolve_mode::same: return na;
    case convolve_mode::valid: break;
    }
    return (na > nb ? na - nb : nb - na) + 1;
}

namespace detail::convolve
{
    // The index of the full output at which the mode's output starts.
    constexpr std::size_t start(std::size_t na, std::size_t nb, convolve_mode mode) noexcept
    {
        switch (mode) {
        case convolve_mode::full: return 0;
        case convolve_mode::same: return (nb - 1) / 2;
        case convolve_mode::valid: break;
        }
        return (na < nb ? na : nb) - 1;
    }

    // Direct: full-output index k = start + i of sum_j a_j b_(k-j), b read through get_b.
    template <typename T, typename B>
    constexpr void direct(std::span<const T> a, std::size_t nb, B get_b, std::span<T> out, convolve_mode mode) noexcept
    {
        const std::size_t na = a.size();
        assert(out.size() == convolve_size(na, nb, mode));
        if (out.empty())
            return;
        const std::size_t s = start(na, nb, mode);
        for (std::size_t i = 0; i < out.size(); ++i) {
            const std::size_t k = s + i;
            const std::size_t lo = k + 1 > nb ? k + 1 - nb : 0, hi = k < na - 1 ? k : na - 1;
            T acc = T(0);
            for (std::size_t j = lo; j <= hi; ++j)
                acc += a[j] * get_b(k - j);
            out[i] = acc;
        }
    }

    // Through one transform of a + i b and one inverse.
    template <typename T, typename B>
    constexpr void by_fft(const fft_plan<T>& plan, std::span<const T> a, std::size_t nb, B get_b, std::span<T> out,
                          std::span<complex<T>> work, convolve_mode mode) noexcept
    {
        const std::size_t na = a.size(), n = plan.size();
        assert(out.size() == convolve_size(na, nb, mode));
        if (out.empty())
            return;
        assert(n >= na + nb - 1 && work.size() >= n + plan.work_size());
        const auto z = work.first(n);
        for (std::size_t i = 0; i < n; ++i)
            z[i] = {i < na ? a[i] : T(0), i < nb ? get_b(i) : T(0)};
        plan.template execute<false>(z, work.subspan(n));
        // A_k = (Z_k + conj Z_(n-k)) / 2, B_k = (Z_k - conj Z_(n-k)) / 2i, so A_k B_k = (Z_k^2 - conj(Z_(n-k))^2) / 4i.
        for (std::size_t k = 0; 2 * k <= n; ++k) {
            const std::size_t r = (n - k) % n;
            const complex<T> zk = z[k], zr = z[r];
            const complex<T> pk = zk * zk - conj(zr) * conj(zr), pr = zr * zr - conj(zk) * conj(zk);
            z[k] = complex<T>{pk.im, -pk.re} * T(0.25);   // divided by 4i
            z[r] = complex<T>{pr.im, -pr.re} * T(0.25);
        }
        plan.template execute<true>(z, work.subspan(n));
        const T inv = T(1) / T(static_cast<long long>(n));
        const std::size_t s = start(na, nb, mode);
        for (std::size_t i = 0; i < out.size(); ++i)
            out[i] = z[s + i].re * inv;
    }
} // namespace detail::convolve

/// \brief The direct convolution, scipy's `convolve(a, b, mode)`.
/// \tparam T A `math::math_number`.
/// \param a The first sequence.
/// \param b The second.
/// \param out `convolve_size(a.size(), b.size(), mode)` values.
/// \param mode The mode.
template <math::math_number T>
constexpr void convolve(std::span<const T> a, std::span<const std::type_identity_t<T>> b, std::span<std::type_identity_t<T>> out,
                        convolve_mode mode = convolve_mode::full) noexcept
{
    detail::convolve::direct(a, b.size(), [b](std::size_t i) { return b[i]; }, out, mode);
}

/// \brief The direct correlation, scipy's `correlate(a, b, mode)`: `c_k = sum_j a_(j+k) b_j`.
/// \tparam T A `math::math_number`.
/// \param a The first sequence.
/// \param b The second; `valid` mode needs `a.size() >= b.size()`.
/// \param out `convolve_size(a.size(), b.size(), mode)` values.
/// \param mode The mode.
template <math::math_number T>
constexpr void correlate(std::span<const T> a, std::span<const std::type_identity_t<T>> b, std::span<std::type_identity_t<T>> out,
                         convolve_mode mode = convolve_mode::full) noexcept
{
    assert(mode != convolve_mode::valid || a.size() >= b.size());
    const std::size_t nb = b.size();
    detail::convolve::direct(a, nb, [b, nb](std::size_t i) { return b[nb - 1 - i]; }, out, mode);
}

/// \brief The work buffer `fft_convolve` and `fft_correlate` need with this plan.
/// \tparam T A `math::math_number`.
/// \param plan The plan.
/// \return The size, in complex elements.
template <math::math_number T>
constexpr std::size_t fft_convolve_work_size(const fft_plan<T>& plan) noexcept
{
    return plan.size() + plan.work_size();
}

/// \brief The convolution through the FFT: the same values as `convolve` to rounding.
/// \tparam T A `math::math_number`.
/// \param plan A plan of length at least `a.size() + b.size() - 1`.
/// \param a The first sequence.
/// \param b The second.
/// \param out `convolve_size(a.size(), b.size(), mode)` values.
/// \param work At least `fft_convolve_work_size(plan)` elements.
/// \param mode The mode.
template <math::math_number T>
constexpr void fft_convolve(const fft_plan<T>& plan, std::span<const std::type_identity_t<T>> a, std::span<const std::type_identity_t<T>> b,
                            std::span<std::type_identity_t<T>> out, std::span<complex<std::type_identity_t<T>>> work,
                            convolve_mode mode = convolve_mode::full) noexcept
{
    detail::convolve::by_fft(plan, a, b.size(), [b](std::size_t i) { return b[i]; }, out, work, mode);
}

/// \brief The correlation through the FFT: the same values as `correlate` to rounding.
/// \tparam T A `math::math_number`.
/// \param plan A plan of length at least `a.size() + b.size() - 1`.
/// \param a The first sequence.
/// \param b The second; `valid` mode needs `a.size() >= b.size()`.
/// \param out `convolve_size(a.size(), b.size(), mode)` values.
/// \param work At least `fft_convolve_work_size(plan)` elements.
/// \param mode The mode.
template <math::math_number T>
constexpr void fft_correlate(const fft_plan<T>& plan, std::span<const std::type_identity_t<T>> a, std::span<const std::type_identity_t<T>> b,
                             std::span<std::type_identity_t<T>> out, std::span<complex<std::type_identity_t<T>>> work,
                             convolve_mode mode = convolve_mode::full) noexcept
{
    assert(mode != convolve_mode::valid || a.size() >= b.size());
    const std::size_t nb = b.size();
    detail::convolve::by_fft(plan, a, nb, [b, nb](std::size_t i) { return b[nb - 1 - i]; }, out, work, mode);
}

/// \brief The work buffer `overlap_add` and `overlap_save` need with this plan: the filter's spectrum, a block, the plan's work.
/// \tparam T A `math::math_number`.
/// \param plan The plan.
/// \return The size, in complex elements.
template <math::math_number T>
constexpr std::size_t block_filter_work_size(const fft_plan<T>& plan) noexcept
{
    return 2 * plan.size() + plan.work_size();
}

namespace detail::convolve
{
    template <typename T>
    constexpr void spectrum_of(const fft_plan<T>& plan, std::span<const T> h, std::span<complex<T>> out, std::span<complex<T>> work) noexcept
    {
        for (std::size_t i = 0; i < out.size(); ++i)
            out[i] = {i < h.size() ? h[i] : T(0), T(0)};
        plan.template execute<false>(out, work);
    }
} // namespace detail::convolve

/// \brief The full convolution of a long signal with a filter, block by block: overlap-add.
/// \tparam T A `math::math_number`.
/// \param plan A plan longer than the filter; each block carries `plan.size() - h.size() + 1` new samples.
/// \param h The filter's taps.
/// \param x The signal.
/// \param out `x.size() + h.size() - 1` values.
/// \param work At least `block_filter_work_size(plan)` elements.
template <math::math_number T>
constexpr void overlap_add(const fft_plan<T>& plan, std::span<const std::type_identity_t<T>> h, std::span<const std::type_identity_t<T>> x,
                           std::span<std::type_identity_t<T>> out, std::span<complex<std::type_identity_t<T>>> work) noexcept
{
    const std::size_t n = plan.size(), taps = h.size();
    assert(out.size() == convolve_size(x.size(), taps, convolve_mode::full));
    if (out.empty())
        return;
    assert(n >= taps && work.size() >= block_filter_work_size(plan));
    const std::size_t step = n - taps + 1;
    const auto hs = work.first(n), block = work.subspan(n, n), rest = work.subspan(2 * n);
    detail::convolve::spectrum_of(plan, std::span<const T>(h), hs, rest);
    const T inv = T(1) / T(static_cast<long long>(n));
    for (auto& v : out)
        v = T(0);
    for (std::size_t s = 0; s < x.size(); s += step) {
        for (std::size_t i = 0; i < n; ++i)
            block[i] = {i < step && s + i < x.size() ? x[s + i] : T(0), T(0)};
        plan.template execute<false>(block, rest);
        for (std::size_t i = 0; i < n; ++i)
            block[i] *= hs[i];
        plan.template execute<true>(block, rest);
        for (std::size_t i = 0; i < n && s + i < out.size(); ++i)
            out[s + i] += block[i].re * inv;
    }
}

/// \brief The causal filter output `y_n = sum_k h_k x_(n-k)`, `x.size()` samples, block by block: overlap-save.
/// \tparam T A `math::math_number`.
/// \param plan A plan longer than the filter; each block yields `plan.size() - h.size() + 1` samples.
/// \param h The filter's taps.
/// \param x The signal (taken as zero before its start).
/// \param out `x.size()` values.
/// \param work At least `block_filter_work_size(plan)` elements.
template <math::math_number T>
constexpr void overlap_save(const fft_plan<T>& plan, std::span<const std::type_identity_t<T>> h, std::span<const std::type_identity_t<T>> x,
                            std::span<std::type_identity_t<T>> out, std::span<complex<std::type_identity_t<T>>> work) noexcept
{
    const std::size_t n = plan.size(), taps = h.size();
    assert(out.size() == x.size());
    if (out.empty() || taps == 0) {
        for (auto& v : out)
            v = T(0);
        return;
    }
    assert(n >= taps && work.size() >= block_filter_work_size(plan));
    const std::size_t step = n - taps + 1, history = taps - 1;
    const auto hs = work.first(n), block = work.subspan(n, n), rest = work.subspan(2 * n);
    detail::convolve::spectrum_of(plan, std::span<const T>(h), hs, rest);
    const T inv = T(1) / T(static_cast<long long>(n));
    for (std::size_t s = 0; s < x.size(); s += step) {
        // the block is x[s - history, s + step), zero outside the signal
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t p = s + i;
            block[i] = {p >= history && p - history < x.size() ? x[p - history] : T(0), T(0)};
        }
        plan.template execute<false>(block, rest);
        for (std::size_t i = 0; i < n; ++i)
            block[i] *= hs[i];
        plan.template execute<true>(block, rest);
        for (std::size_t i = 0; i < step && s + i < out.size(); ++i)
            out[s + i] = block[history + i].re * inv;
    }
}

} // namespace snicholls::signal

#endif // snicholls_core_signal_convolve_hpp
