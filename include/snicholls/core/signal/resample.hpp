//
//  resample.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::signal` polyphase resampling by a rational factor `up / down`, as `scipy.signal.resample_poly`, with the windowed-sinc kernel built at compile time when the factor is known.
/// \ingroup core
///
/// **Overview.** `resample_poly(x, up, down, h, y)` gives `ceil(n up / down)` samples at `up / down` times the rate:
/// conceptually `x` with `up - 1` zeros after each sample, low-pass filtered by `h`, every `down`-th sample kept; in
/// fact only the taps that meet a non-zero input are summed, `(taps / up)` multiply-adds per output (the polyphase
/// form). The factor is reduced by its greatest common divisor first.
///
/// The kernel is scipy's: `firwin(20 max(up, down) + 1, 1 / max(up, down), window=('kaiser', 5.0))` times `up`, the
/// centre tap aligned with the first output, so the outputs equal scipy's (`padtype='constant'`). `resample_kernel`
/// fills it in a span; `polyphase_resampler<T, Up, Down>` builds it in a `consteval` constructor, so the
/// 44.1 kHz to 48 kHz kernel (`Up = 160`, `Down = 147`, 3,201 taps) is a table the compiler writes.
///
/// \snippet signal/resample_demo.cpp resample
///
/// **Relationships.** `signal/fir.hpp` (`fir_design`), `signal/window.hpp` (`kaiser`). Group: \ref core.
///
/// **Implementations.** The direct polyphase sum; one call over a whole signal (no streaming state).
///
/// **Guarantees.** `constexpr`, no allocation. Complexity: `O(n up / down x taps / up)`. A sinusoid well inside both
/// bands is reproduced at the new rate within the kernel's ripple (about `1e-3` with Kaiser 5; tested).
///
/// **Errors.** Preconditions, asserted: `up, down >= 1`; `h` has `resample_kernel_size(up, down)` taps; `y` has
/// `resample_size(x.size(), up, down)` samples.
///
/// **Performance.** Not benchmarked; a direct sum of `taps / up` terms per output.
///
/// **Tests and demos.** `tests/signal/resample_test.cpp` (scipy 1.13.1's outputs, sinusoids, the identity, compile-time
/// kernels), `demos/signal/resample_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cassert>`, `<cstddef>`, `<numeric>`, `<span>`, `<type_traits>`, `signal/fir.hpp`.
///
/// **Known issues.** scipy's other padding types (`mean`, `line`, ...) are not provided: the signal is zero outside.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_signal_resample_hpp
#define snicholls_core_signal_resample_hpp

#include <snicholls/core/signal/fir.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <numeric>
#include <span>
#include <type_traits>

namespace snicholls::signal
{

/// \brief A resampling factor `up / down` in lowest terms.
/// \ingroup core
struct resample_ratio
{
    std::size_t up = 1;     ///< The upsampling factor.
    std::size_t down = 1;   ///< The downsampling factor.

    /// \brief The ratio reduced by the greatest common divisor.
    /// \param up The upsampling factor, at least 1.
    /// \param down The downsampling factor, at least 1.
    /// \return The ratio.
    static constexpr resample_ratio of(std::size_t up, std::size_t down) noexcept
    {
        assert(up >= 1 && down >= 1);
        const std::size_t g = std::gcd(up, down);
        return {up / g, down / g};
    }
    /// \brief Half the kernel's length: `10 max(up, down)`, or 0 for the identity.
    /// \return The half length.
    [[nodiscard]] constexpr std::size_t half_length() const noexcept { return up == 1 && down == 1 ? 0 : 10 * (up > down ? up : down); }
    /// \brief The kernel's length, `2 half_length() + 1`.
    /// \return The number of taps.
    [[nodiscard]] constexpr std::size_t taps() const noexcept { return 2 * half_length() + 1; }
};

/// \brief The output's length: `ceil(n up / down)`.
/// \param n The input's length.
/// \param up The upsampling factor.
/// \param down The downsampling factor.
/// \return The length.
constexpr std::size_t resample_size(std::size_t n, std::size_t up, std::size_t down) noexcept
{
    const resample_ratio r = resample_ratio::of(up, down);
    return (n * r.up + r.down - 1) / r.down;
}

/// \brief The kernel's length for a factor.
/// \param up The upsampling factor.
/// \param down The downsampling factor.
/// \return `20 max(up, down) + 1` in lowest terms, 1 for the identity.
constexpr std::size_t resample_kernel_size(std::size_t up, std::size_t down) noexcept { return resample_ratio::of(up, down).taps(); }

/// \brief Fills scipy's resampling kernel: a windowed-sinc low-pass at `1 / max(up, down)` of Nyquist, times `up`.
/// \tparam T A `math::math_number`.
/// \tparam W A window type.
/// \param up The upsampling factor.
/// \param down The downsampling factor.
/// \param h `resample_kernel_size(up, down)` taps.
/// \param window The window; scipy's default is `kaiser{5.0}`.
template <math::math_number T, typename W = kaiser>
constexpr void resample_kernel(std::size_t up, std::size_t down, std::span<T> h, const W& window = W{5.0}) noexcept
{
    const resample_ratio r = resample_ratio::of(up, down);
    assert(h.size() == r.taps());
    if (r.half_length() == 0) {
        h[0] = T(1);
        return;
    }
    const std::size_t m = r.up > r.down ? r.up : r.down;
    fir_design(h, fir_band::lowpass, T(1) / T(static_cast<long long>(m)), T(0), window);
    for (auto& v : h)
        v *= T(static_cast<long long>(r.up));
}

/// \brief Resamples `x` by `up / down` through the kernel `h`: `scipy.signal.resample_poly(x, up, down)`.
/// \tparam T A `math::math_number`.
/// \param x The input.
/// \param up The upsampling factor.
/// \param down The downsampling factor.
/// \param h The kernel, `resample_kernel` of the same factor.
/// \param y `resample_size(x.size(), up, down)` outputs.
template <math::math_number T>
constexpr void resample_poly(std::span<const T> x, std::size_t up, std::size_t down, std::span<const std::type_identity_t<T>> h,
                             std::span<std::type_identity_t<T>> y) noexcept
{
    const resample_ratio r = resample_ratio::of(up, down);
    assert(h.size() == r.taps() && y.size() == resample_size(x.size(), up, down));
    const std::size_t taps = h.size(), half = r.half_length(), n = x.size();
    for (std::size_t o = 0; o < y.size(); ++o) {
        // y[o] = sum_i x[i] h[t - i up], t = o down + half: the centre tap on output o's position.
        const std::size_t t = o * r.down + half;
        const std::size_t lo = t + 1 > taps ? (t + 1 - taps + r.up - 1) / r.up : 0;
        const std::size_t hi = t / r.up < n ? t / r.up : n - 1;
        T acc = T(0);
        for (std::size_t i = lo; i <= hi && n > 0; ++i)
            acc += x[i] * h[t - i * r.up];
        y[o] = acc;
    }
}

/// \brief A resampler for a factor known at compile time, its kernel built by the compiler.
/// \ingroup core
/// \tparam T A `math::math_number`.
/// \tparam Up The upsampling factor.
/// \tparam Down The downsampling factor.
template <math::math_number T, std::size_t Up, std::size_t Down>
    requires(Up >= 1 && Down >= 1)
class polyphase_resampler
{
public:
    /// \brief The factor in lowest terms.
    static constexpr resample_ratio ratio = resample_ratio::of(Up, Down);
    /// \brief The kernel's length.
    static constexpr std::size_t taps = ratio.taps();

    /// \brief Builds the kernel, at compile time.
    consteval polyphase_resampler() { resample_kernel(Up, Down, std::span<T>(kernel_)); }

    /// \brief The kernel.
    /// \return The taps.
    [[nodiscard]] constexpr const std::array<T, taps>& kernel() const noexcept { return kernel_; }

    /// \brief The output's length for an input of `n` samples.
    /// \param n The input's length.
    /// \return `ceil(n Up / Down)`.
    [[nodiscard]] static constexpr std::size_t output_size(std::size_t n) noexcept { return resample_size(n, Up, Down); }

    /// \brief Resamples `x` into `y`.
    /// \param x The input.
    /// \param y `output_size(x.size())` outputs.
    constexpr void process(std::span<const T> x, std::span<T> y) const noexcept
    {
        resample_poly(x, Up, Down, std::span<const T>(kernel_), y);
    }

private:
    std::array<T, taps> kernel_{};
};

} // namespace snicholls::signal

#endif // snicholls_core_signal_resample_hpp
