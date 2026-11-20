//
//  stft.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::signal` short-time Fourier transform and its inverse, as `torch.stft` and `torch.istft` compute them, spectrograms, and their backward passes.
/// \ingroup core
///
/// **Overview.** `stft` cuts a real signal into frames of `n_fft` samples every `hop`, multiplies each by a window and
/// transforms it: with an `rfft_plan` the one-sided spectrum (`n_fft / 2 + 1` bins), with an `fft_plan` the two-sided
/// one (`n_fft` bins). As in `torch.stft`, `center` pads `n_fft / 2` samples on each side first, by reflection
/// (`pad_mode::reflect`, the default: `x[-k] = x[k]`, the edge not repeated) or with zeros (`pad_mode::constant`);
/// the number of frames is `1 + (padded length - n_fft) / hop`; `normalized` multiplies each spectrum by
/// `n_fft^(-1/2)`. The output is **frame-major**, `frames x bins` (row `t` is frame `t`), the transpose of
/// PyTorch's `(bins, frames)`: each frame's spectrum is contiguous.
///
/// `istft` is `torch.istft`: each frame inverse-transformed, multiplied by the window, overlap-added, and divided by
/// the overlap-added square of the window, then the `centre` padding removed. For a window and hop that satisfy
/// NOLA (`is_nola`), `istft(stft(x)) == x` to rounding. The divisor is computed per sample from the window, so no
/// envelope buffer is needed.
///
/// `spectrogram` gives `|X|` or `|X|^2` frame by frame without storing the complex spectrum. Each function reads the
/// padded signal through an index map, so no padded copy is made.
///
/// **Gradients.** `stft_backward` (the adjoint: each frame's `rfft_backward`, windowed and overlap-added back
/// through the padding, reflected samples folded onto their sources), `istft_backward`, and `power_backward` /
/// `magnitude_backward` for the spectrogram, with PyTorch's convention for complex gradients
/// (`g = dL/d re + i dL/d im`).
///
/// \snippet signal/stft_demo.cpp stft
///
/// **Relationships.** `signal/fft.hpp` (the plans), `signal/window.hpp` (the windows, `is_nola`), `signal/mel.hpp`
/// (mel spectrograms and Whisper's front end on top of these). Group: \ref core.
///
/// **Implementations.** One-sided (`rfft_plan`) and two-sided (`fft_plan`) overloads of `stft` and `istft`; the
/// backward passes are one-sided.
///
/// **Guarantees.** `constexpr`; no allocation (the caller gives the work buffer, `plan.work_size()` elements, or
/// `spectrogram_work_size` for `spectrogram`). Complexity: `frames x n_fft log n_fft`; `istft`'s divisor costs
/// `n_fft / hop` multiply-adds per sample.
///
/// **Errors.** Preconditions, asserted: the plan's and the window's lengths are `n_fft`; `hop >= 1`; the output's
/// size is `stft_frames(length) x bins`; reflection needs a signal longer than `n_fft / 2`, as in PyTorch; `istft`
/// needs the window's overlap-added square above `1e-11` at every output sample (PyTorch raises there).
///
/// **Performance.** Not benchmarked separately: one real transform per frame (`signal/fft.hpp`'s benchmark).
///
/// **Tests and demos.** `tests/signal/stft_test.cpp` (against a direct transform of each padded frame, round trips,
/// `torch.stft`'s frame counts, the backward passes against central differences), `demos/signal/stft_demo.cpp`.
///
/// **Dependencies.** `<cassert>`, `<cstddef>`, `<span>`, `<type_traits>`, `signal/fft.hpp`.
///
/// **Known issues.** A window shorter than `n_fft` (PyTorch's `win_length`) is not padded here: centre it in `n_fft`
/// zeros first. Complex input and the two-sided backward passes are not provided.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_signal_stft_hpp
#define snicholls_core_signal_stft_hpp

#include <snicholls/core/signal/fft.hpp>

#include <cassert>
#include <cstddef>
#include <span>
#include <type_traits>

namespace snicholls::signal
{

/// \brief How `stft` pads the signal when it centres the frames.
/// \ingroup core
enum class pad_mode
{
    reflect,    ///< `x[-k] = x[k]`, `x[n - 1 + k] = x[n - 1 - k]`: PyTorch's default.
    constant,   ///< Zeros.
};

/// \brief The parameters of a short-time Fourier transform, with `torch.stft`'s meanings.
/// \ingroup core
struct stft_options
{
    std::size_t n_fft = 400;            ///< The frame's length and the transform's.
    std::size_t hop = 160;              ///< The distance between frames.
    bool center = true;                 ///< Whether to pad `n_fft / 2` samples on each side first.
    pad_mode pad = pad_mode::reflect;   ///< How to pad.
    bool normalized = false;            ///< Whether to multiply each spectrum by `n_fft^(-1/2)`.

    /// \brief The padding on each side: `n_fft / 2` when centred, else 0.
    /// \return The padding.
    [[nodiscard]] constexpr std::size_t padding() const noexcept { return center ? n_fft / 2 : 0; }
};

/// \brief The number of frames `stft` gives for a signal of `length` samples: `1 + (padded - n_fft) / hop`, or 0.
/// \param length The signal's length.
/// \param o The options.
/// \return The number of frames.
constexpr std::size_t stft_frames(std::size_t length, const stft_options& o) noexcept
{
    const std::size_t padded = length + 2 * o.padding();
    return padded < o.n_fft || o.hop == 0 ? 0 : 1 + (padded - o.n_fft) / o.hop;
}

/// \brief The signal length `istft` recovers from `frames` frames by default: `n_fft + hop (frames - 1)` less the padding.
/// \param frames The number of frames.
/// \param o The options.
/// \return The length.
constexpr std::size_t istft_length(std::size_t frames, const stft_options& o) noexcept
{
    if (frames == 0)
        return 0;
    const std::size_t full = o.n_fft + o.hop * (frames - 1), pad = 2 * o.padding();
    return full > pad ? full - pad : 0;
}

namespace detail::stft
{
    // The index in x of sample p of the padded signal, or -1 for a zero of constant padding.
    constexpr long long source_index(std::size_t p, std::size_t pad, std::size_t n, pad_mode mode) noexcept
    {
        const long long q = static_cast<long long>(p) - static_cast<long long>(pad), nn = static_cast<long long>(n);
        if (q >= 0 && q < nn)
            return q;
        if (mode == pad_mode::constant)
            return -1;
        const long long r = q < 0 ? -q : 2 * (nn - 1) - q;
        assert(r >= 0 && r < nn);   // reflection needs a signal longer than the padding
        return r;
    }

    template <typename T>
    constexpr T sample(std::span<const T> x, std::size_t p, std::size_t pad, pad_mode mode) noexcept
    {
        const long long i = source_index(p, pad, x.size(), mode);
        return i < 0 ? T(0) : x[static_cast<std::size_t>(i)];
    }

    template <typename T>
    constexpr T spectrum_scale(const stft_options& o) noexcept
    {
        return o.normalized ? T(1) / math::sqrt(T(static_cast<long long>(o.n_fft))) : T(1);
    }

    // sum_t w[p - t hop]^2 over the frames covering padded sample p: istft's divisor.
    template <typename T>
    constexpr T envelope(std::span<const T> w, std::size_t p, std::size_t hop, std::size_t frames) noexcept
    {
        const std::size_t n = w.size();
        const std::size_t first = p + 1 > n ? (p + 1 - n + hop - 1) / hop : 0;
        const std::size_t last = p / hop < frames ? p / hop : frames - 1;
        T e = T(0);
        for (std::size_t t = first; t <= last && t < frames; ++t)
            e += w[p - t * hop] * w[p - t * hop];
        return e;
    }

    template <typename T>
    constexpr void check(std::size_t plan_n, std::span<const T> window, const stft_options& o) noexcept
    {
        assert(plan_n == o.n_fft && window.size() == o.n_fft && o.hop >= 1);
        (void)plan_n;
        (void)window;
        (void)o;
    }
} // namespace detail::stft

/// \brief The one-sided short-time Fourier transform: `frames x (n_fft / 2 + 1)` bins, frame-major.
/// \tparam T A `math::math_number`.
/// \param plan The real plan of length `n_fft`.
/// \param x The signal.
/// \param window `n_fft` values.
/// \param out `stft_frames(x.size(), o) * plan.bins()` elements.
/// \param work At least `plan.work_size()` elements.
/// \param o The options.
template <math::math_number T>
constexpr void stft(const rfft_plan<T>& plan, std::span<const std::type_identity_t<T>> x, std::span<const std::type_identity_t<T>> window,
                    std::span<complex<std::type_identity_t<T>>> out, std::span<complex<std::type_identity_t<T>>> work,
                    const stft_options& o) noexcept
{
    detail::stft::check(plan.size(), window, o);
    const std::size_t frames = stft_frames(x.size(), o), bins = plan.bins(), pad = o.padding();
    assert(out.size() == frames * bins);
    const T s = detail::stft::spectrum_scale<T>(o);
    for (std::size_t t = 0; t < frames; ++t) {
        const auto row = out.subspan(t * bins, bins);
        plan.forward_from([&](std::size_t j) { return window[j] * detail::stft::sample(x, t * o.hop + j, pad, o.pad); }, row, work);
        detail::fft::scale_all(row.data(), bins, s);
    }
}

/// \brief The two-sided short-time Fourier transform: `frames x n_fft` bins, frame-major.
/// \tparam T A `math::math_number`.
/// \param plan The complex plan of length `n_fft`.
/// \param x The signal.
/// \param window `n_fft` values.
/// \param out `stft_frames(x.size(), o) * n_fft` elements.
/// \param work At least `plan.work_size()` elements.
/// \param o The options.
template <math::math_number T>
constexpr void stft(const fft_plan<T>& plan, std::span<const std::type_identity_t<T>> x, std::span<const std::type_identity_t<T>> window,
                    std::span<complex<std::type_identity_t<T>>> out, std::span<complex<std::type_identity_t<T>>> work,
                    const stft_options& o) noexcept
{
    detail::stft::check(plan.size(), window, o);
    const std::size_t frames = stft_frames(x.size(), o), n = o.n_fft, pad = o.padding();
    assert(out.size() == frames * n);
    const T s = detail::stft::spectrum_scale<T>(o);
    for (std::size_t t = 0; t < frames; ++t) {
        const auto row = out.subspan(t * n, n);
        for (std::size_t j = 0; j < n; ++j)
            row[j] = {window[j] * detail::stft::sample(x, t * o.hop + j, pad, o.pad), T(0)};
        plan.template execute<false>(row, work);
        detail::fft::scale_all(row.data(), n, s);
    }
}

namespace detail::stft
{
    // Overlap-adds frame t's samples (already inverse-transformed and scaled) into out, then (once all are in) divides.
    template <typename T>
    constexpr void normalise(std::span<const T> window, std::span<T> out, std::size_t frames, const stft_options& o) noexcept
    {
        const std::size_t pad = o.padding(), covered = o.n_fft + o.hop * (frames == 0 ? 0 : frames - 1);
        for (std::size_t i = 0; i < out.size(); ++i) {
            const std::size_t p = i + pad;
            if (frames == 0 || p >= covered) {
                out[i] = T(0);
                continue;
            }
            const T e = envelope(window, p, o.hop, frames);
            assert(e > T(1e-11));   // NOLA: torch.istft raises here
            out[i] /= e;
        }
    }
} // namespace detail::stft

/// \brief The inverse of the one-sided `stft`: overlap-add of the windowed inverse transforms, divided by the window's overlap-added square.
/// \tparam T A `math::math_number`.
/// \param plan The real plan of length `n_fft`.
/// \param spectrum `frames * plan.bins()` elements, frame-major.
/// \param window The analysis window, `n_fft` values.
/// \param out The signal: `istft_length(frames, o)` samples recover what `stft` was given; samples no frame covers are 0.
/// \param work At least `plan.work_size()` elements.
/// \param o The options `stft` was given.
template <math::math_number T>
constexpr void istft(const rfft_plan<T>& plan, std::span<const complex<std::type_identity_t<T>>> spectrum,
                     std::span<const std::type_identity_t<T>> window, std::span<std::type_identity_t<T>> out,
                     std::span<complex<std::type_identity_t<T>>> work, const stft_options& o) noexcept
{
    detail::stft::check(plan.size(), window, o);
    const std::size_t bins = plan.bins(), frames = spectrum.size() / bins, pad = o.padding();
    assert(spectrum.size() == frames * bins);
    const T s = T(1) / T(static_cast<long long>(o.n_fft)) / detail::stft::spectrum_scale<T>(o);
    for (auto& v : out)
        v = T(0);
    for (std::size_t t = 0; t < frames; ++t)
        plan.inverse_to([&](std::size_t k) { return spectrum[t * bins + k]; },
                        [&](std::size_t j, T v) {
                            const std::size_t p = t * o.hop + j;
                            if (p >= pad && p - pad < out.size())
                                out[p - pad] += window[j] * v * s;
                        },
                        work);
    detail::stft::normalise(window, out, frames, o);
}

/// \brief The inverse of the two-sided `stft`: the real part of each frame's inverse transform, overlap-added and normalised.
/// \tparam T A `math::math_number`.
/// \param plan The complex plan of length `n_fft`.
/// \param spectrum `frames * n_fft` elements, frame-major.
/// \param window The analysis window, `n_fft` values.
/// \param out The signal.
/// \param work At least `n_fft + plan.work_size()` elements.
/// \param o The options `stft` was given.
template <math::math_number T>
constexpr void istft(const fft_plan<T>& plan, std::span<const complex<std::type_identity_t<T>>> spectrum,
                     std::span<const std::type_identity_t<T>> window, std::span<std::type_identity_t<T>> out,
                     std::span<complex<std::type_identity_t<T>>> work, const stft_options& o) noexcept
{
    detail::stft::check(plan.size(), window, o);
    const std::size_t n = o.n_fft, frames = n == 0 ? 0 : spectrum.size() / n, pad = o.padding();
    assert(spectrum.size() == frames * n && work.size() >= n + plan.work_size());
    const T s = T(1) / T(static_cast<long long>(n)) / detail::stft::spectrum_scale<T>(o);
    for (auto& v : out)
        v = T(0);
    const auto buf = work.first(n);
    for (std::size_t t = 0; t < frames; ++t) {
        for (std::size_t k = 0; k < n; ++k)
            buf[k] = spectrum[t * n + k];
        plan.template execute<true>(buf, work.subspan(n));
        for (std::size_t j = 0; j < n; ++j) {
            const std::size_t p = t * o.hop + j;
            if (p >= pad && p - pad < out.size())
                out[p - pad] += window[j] * buf[j].re * s;
        }
    }
    detail::stft::normalise(window, out, frames, o);
}

/// \brief The backward pass of the one-sided `stft`: given `dL/dX` (frame-major), `dL/dx`.
/// \tparam T A `math::math_number`.
/// \param plan The real plan of length `n_fft`.
/// \param grad_spectrum `frames * plan.bins()` elements of `dL/dX`.
/// \param window The window.
/// \param grad_x `dL/dx`, the signal's length; overwritten.
/// \param work At least `plan.work_size()` elements.
/// \param o The options.
template <math::math_number T>
constexpr void stft_backward(const rfft_plan<T>& plan, std::span<const complex<std::type_identity_t<T>>> grad_spectrum,
                             std::span<const std::type_identity_t<T>> window, std::span<std::type_identity_t<T>> grad_x,
                             std::span<complex<std::type_identity_t<T>>> work, const stft_options& o) noexcept
{
    detail::stft::check(plan.size(), window, o);
    const std::size_t bins = plan.bins(), frames = stft_frames(grad_x.size(), o), pad = o.padding(), n = o.n_fft;
    assert(grad_spectrum.size() == frames * bins);
    const T s = detail::stft::spectrum_scale<T>(o);
    for (auto& v : grad_x)
        v = T(0);
    for (std::size_t t = 0; t < frames; ++t)
        plan.inverse_to(
            [&](std::size_t k) {
                const complex<T> g = grad_spectrum[t * bins + k];
                return k == 0 || 2 * k == n ? g : g * T(0.5);
            },
            [&](std::size_t j, T v) {
                const long long i = detail::stft::source_index(t * o.hop + j, pad, grad_x.size(), o.pad);
                if (i >= 0)
                    grad_x[static_cast<std::size_t>(i)] += window[j] * v * s;
            },
            work);
}

/// \brief The backward pass of the one-sided `istft`: given `dL/dy` over its output, `dL/dX` (frame-major).
/// \tparam T A `math::math_number`.
/// \param plan The real plan of length `n_fft`.
/// \param grad_y `dL/dy`, the output's length.
/// \param window The window.
/// \param grad_spectrum `frames * plan.bins()` elements of `dL/dX`, written.
/// \param work At least `plan.work_size()` elements.
/// \param o The options.
template <math::math_number T>
constexpr void istft_backward(const rfft_plan<T>& plan, std::span<const std::type_identity_t<T>> grad_y,
                              std::span<const std::type_identity_t<T>> window, std::span<complex<std::type_identity_t<T>>> grad_spectrum,
                              std::span<complex<std::type_identity_t<T>>> work, const stft_options& o) noexcept
{
    detail::stft::check(plan.size(), window, o);
    const std::size_t bins = plan.bins(), frames = grad_spectrum.size() / bins, pad = o.padding(), n = o.n_fft;
    const std::size_t covered = n + o.hop * (frames == 0 ? 0 : frames - 1);
    const T s = T(1) / T(static_cast<long long>(n)) / detail::stft::spectrum_scale<T>(o);
    for (std::size_t t = 0; t < frames; ++t) {
        const auto row = grad_spectrum.subspan(t * bins, bins);
        plan.forward_from(
            [&](std::size_t j) {
                const std::size_t p = t * o.hop + j;
                if (p < pad || p - pad >= grad_y.size() || p >= covered)
                    return T(0);
                return window[j] * grad_y[p - pad] / detail::stft::envelope(window, p, o.hop, frames);
            },
            row, work);
        for (std::size_t k = 0; k < bins; ++k)
            row[k] *= k == 0 || 2 * k == n ? s : T(2) * s;
    }
}

/// \brief What a spectrogram holds per bin.
/// \ingroup core
enum class spectrum_kind
{
    magnitude,   ///< `|X|`.
    power,       ///< `|X|^2`.
};

/// \brief The work buffer `spectrogram` needs: one frame's spectrum and the plan's work.
/// \tparam T A `math::math_number`.
/// \param plan The real plan.
/// \return The size, in complex elements.
template <math::math_number T>
constexpr std::size_t spectrogram_work_size(const rfft_plan<T>& plan) noexcept
{
    return plan.bins() + plan.work_size();
}

/// \brief The magnitude or power spectrogram, `frames x bins`, frame-major, without storing the complex spectrum.
/// \tparam T A `math::math_number`.
/// \param plan The real plan of length `n_fft`.
/// \param x The signal.
/// \param window `n_fft` values.
/// \param out `stft_frames(x.size(), o) * plan.bins()` values.
/// \param work At least `spectrogram_work_size(plan)` elements.
/// \param o The options.
/// \param kind Magnitude or power.
template <math::math_number T>
constexpr void spectrogram(const rfft_plan<T>& plan, std::span<const std::type_identity_t<T>> x, std::span<const std::type_identity_t<T>> window,
                           std::span<std::type_identity_t<T>> out, std::span<complex<std::type_identity_t<T>>> work, const stft_options& o,
                           spectrum_kind kind = spectrum_kind::power) noexcept
{
    detail::stft::check(plan.size(), window, o);
    const std::size_t frames = stft_frames(x.size(), o), bins = plan.bins(), pad = o.padding();
    assert(out.size() == frames * bins && work.size() >= spectrogram_work_size(plan));
    const T s = detail::stft::spectrum_scale<T>(o);
    const auto row = work.first(bins);
    for (std::size_t t = 0; t < frames; ++t) {
        plan.forward_from([&](std::size_t j) { return window[j] * detail::stft::sample(x, t * o.hop + j, pad, o.pad); }, row,
                          work.subspan(bins));
        for (std::size_t k = 0; k < bins; ++k) {
            const T p = norm(row[k]) * (s * s);
            out[t * bins + k] = kind == spectrum_kind::power ? p : math::sqrt(p);
        }
    }
}

/// \brief `|X|^2` of each element.
/// \tparam T A `math::math_number`.
/// \param spectrum The spectrum.
/// \param out The powers, the same size.
template <math::math_number T>
constexpr void power(std::span<const complex<T>> spectrum, std::span<std::type_identity_t<T>> out) noexcept
{
    assert(out.size() == spectrum.size());
    for (std::size_t i = 0; i < spectrum.size(); ++i)
        out[i] = norm(spectrum[i]);
}

/// \brief `|X|` of each element.
/// \tparam T A `math::math_number`.
/// \param spectrum The spectrum.
/// \param out The magnitudes, the same size.
template <math::math_number T>
constexpr void magnitude(std::span<const complex<T>> spectrum, std::span<std::type_identity_t<T>> out) noexcept
{
    assert(out.size() == spectrum.size());
    for (std::size_t i = 0; i < spectrum.size(); ++i)
        out[i] = abs(spectrum[i]);
}

/// \brief The backward pass of `power`: `dL/dX = 2 X dL/dP`.
/// \tparam T A `math::math_number`.
/// \param spectrum The spectrum `power` was given.
/// \param grad_power `dL/dP`.
/// \param grad_spectrum `dL/dX`, written.
template <math::math_number T>
constexpr void power_backward(std::span<const complex<T>> spectrum, std::span<const std::type_identity_t<T>> grad_power,
                              std::span<complex<std::type_identity_t<T>>> grad_spectrum) noexcept
{
    assert(grad_power.size() == spectrum.size() && grad_spectrum.size() == spectrum.size());
    for (std::size_t i = 0; i < spectrum.size(); ++i)
        grad_spectrum[i] = spectrum[i] * (T(2) * grad_power[i]);
}

/// \brief The backward pass of `magnitude`: `dL/dX = X / |X| dL/dM`, 0 where `X` is 0 (PyTorch's choice).
/// \tparam T A `math::math_number`.
/// \param spectrum The spectrum `magnitude` was given.
/// \param grad_magnitude `dL/dM`.
/// \param grad_spectrum `dL/dX`, written.
template <math::math_number T>
constexpr void magnitude_backward(std::span<const complex<T>> spectrum, std::span<const std::type_identity_t<T>> grad_magnitude,
                                  std::span<complex<std::type_identity_t<T>>> grad_spectrum) noexcept
{
    assert(grad_magnitude.size() == spectrum.size() && grad_spectrum.size() == spectrum.size());
    for (std::size_t i = 0; i < spectrum.size(); ++i) {
        const T m = abs(spectrum[i]);
        grad_spectrum[i] = m == T(0) ? complex<T>{} : spectrum[i] * (grad_magnitude[i] / m);
    }
}

} // namespace snicholls::signal

#endif // snicholls_core_signal_stft_hpp
