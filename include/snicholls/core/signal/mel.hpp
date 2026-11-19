//
//  mel.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::signal` audio front ends: the mel scales (HTK and Slaney), mel filterbanks as librosa and torchaudio build them, mel spectrograms, log compression (decibels, Whisper's), OpenAI Whisper's log-mel spectrogram, MFCCs, and the backward passes.
/// \ingroup core
///
/// **Overview.**
///
/// - `hz_to_mel` / `mel_to_hz` with `mel_scale::htk` (`2595 log10(1 + f / 700)`) or `mel_scale::slaney` (linear
///   below 1 kHz at 3 mels per 200 Hz, logarithmic above at 27 mels per factor of 6.4: librosa's default and the
///   Auditory Toolbox's).
/// - `mel_filterbank(options, out)`: the `n_mels x (n_fft / 2 + 1)` triangular filters of `librosa.filters.mel`,
///   with the same arithmetic (`linspace` in mels, `rfftfreq`, the ramps, and Slaney's area normalisation `2 / (f_{m+2}
///   - f_m)`), so `float` filters match librosa's `float32` ones within one unit in the last place. torchaudio's
///   `melscale_fbanks` is the transpose of the same matrix for an even sample rate (with `mel_scale::htk` and
///   `mel_norm::none`, its defaults). `make_mel_filterbank<T, Mels, NFft>(...)` returns a `std::array`, a
///   compile-time table when it is `constexpr`.
/// - `mel_spectrogram(filters, power, out)` applies the filters to each frame of a power spectrogram (frame-major);
///   `mel_backward` is its transpose.
/// - `log_compress(in, out, c)`: `y = scale (multiplier log10(max(x, floor)) floored at (max - range) + offset)`, the
///   maximum taken over the whole array. `decibels(top_db)` is librosa's `power_to_db` (`10 log10`, `amin` 1e-10,
///   `top_db` 80); `whisper_compression` is Whisper's (`log10`, clamp 1e-10, floor at `max - 8`, `(x + 4) / 4`).
///   `log_compress_backward` is its gradient.
/// - `whisper_log_mel`: OpenAI Whisper's `log_mel_spectrogram` (whisper/audio.py): `padding` zeros appended, a
///   centred, reflect-padded STFT with a periodic Hann window of 400 points and a hop of 160, the last frame dropped,
///   `|X|^2`, the mel filters of `librosa.filters.mel(sr=16000, n_fft=400, n_mels=80 or 128)` (Slaney scale and
///   normalisation), and `whisper_compression`. The output is mel-major, `n_mels x frames`, as Whisper's tensor; a
///   30-second chunk (480,000 samples) gives 3,000 frames. `whisper_log_mel_backward` is its gradient with respect to
///   the audio, each frame's spectrum recomputed rather than stored.
/// - `mfcc`: librosa's MFCCs, the orthonormal DCT-II of each frame of a log-mel spectrogram (`decibels`), its first
///   `n_mfcc` coefficients.
///
/// \snippet signal/mel_demo.cpp mel
/// \snippet signal/mel_demo.cpp whisper
///
/// **Relationships.** `signal/stft.hpp` (spectrograms), `signal/window.hpp` (the Hann window), `signal/dct.hpp`
/// (MFCCs). Group: \ref core.
///
/// **Implementations.** One each; Whisper's front end is fixed by Whisper's constants.
///
/// **Guarantees.** `constexpr`; no allocation (work buffers from the caller: `whisper_work_size(plan)` complex
/// elements). Filterbanks are computed in `compute_t<T>` (`double` for `float`, as librosa computes in float64) and
/// rounded as librosa rounds. Complexity: a filterbank is `n_mels x bins`; a mel spectrogram `frames x n_mels x bins`.
///
/// **Errors.** Preconditions, asserted: sizes agree; `f_min < f_max <= sample_rate / 2` is not checked (librosa warns
/// of empty filters; so do the values). The audio given to `whisper_log_mel` must be longer than 200 samples after
/// padding (reflection, as PyTorch requires).
///
/// **Performance.** Not benchmarked separately; the STFT dominates.
///
/// **Tests and demos.** `tests/signal/mel_test.cpp`: the scales, the filterbanks of Whisper's 80 and 128 mels and an
/// HTK bank against librosa 0.11.0's (fixtures in `tests/fixtures/signal/`), Whisper's log-mel against an independent
/// float64 implementation of whisper/audio.py in numpy 2.0.2, MFCCs against librosa's, a filterbank and a log-mel at
/// compile time, every backward pass by central differences. `demos/signal/mel_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cassert>`, `<cstddef>`, `<span>`, `<type_traits>`, `signal/dct.hpp`,
/// `signal/stft.hpp`, `signal/window.hpp`.
///
/// **Known issues.** PyTorch splits the gradient of `max(y, max(y) - range)` evenly at a tie and so does
/// `log_compress_backward` for the global maximum; at an exact tie between a value and the floor the value takes it
/// all. Whisper's own STFT runs in `float32` in PyTorch's FFT; `whisper_log_mel<float>` agrees with it to rounding,
/// not to the bit.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_signal_mel_hpp
#define snicholls_core_signal_mel_hpp

#include <snicholls/core/signal/dct.hpp>
#include <snicholls/core/signal/stft.hpp>
#include <snicholls/core/signal/window.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <span>
#include <type_traits>

namespace snicholls::signal
{

/// \brief Which mel scale.
/// \ingroup core
enum class mel_scale
{
    htk,      ///< `2595 log10(1 + f / 700)`: HTK's, torchaudio's default.
    slaney,   ///< Linear below 1 kHz, logarithmic above: the Auditory Toolbox's, librosa's default, Whisper's.
};

/// \brief How each mel filter is normalised.
/// \ingroup core
enum class mel_norm
{
    none,     ///< Peak 1.
    slaney,   ///< Area normalised: each triangle multiplied by `2 / (f_{m+2} - f_m)`.
};

namespace detail::mel
{
    template <typename C>
    struct slaney_constants
    {
        static constexpr C f_sp = C(200) / C(3);
        static constexpr C min_log_hz = C(1000);
        static constexpr C min_log_mel = min_log_hz / f_sp;   // 15
    };

    template <typename C>
    constexpr C logstep() noexcept
    {
        return math::log(C(6.4)) / C(27);
    }
} // namespace detail::mel

/// \brief A frequency in mels.
/// \tparam T A `math::math_number`.
/// \param hz The frequency in hertz.
/// \param scale The scale.
/// \return The mels.
template <math::math_number T>
constexpr T hz_to_mel(T hz, mel_scale scale = mel_scale::slaney) noexcept
{
    using C = compute_t<T>;
    using K = detail::mel::slaney_constants<C>;
    const C f = C(hz);
    if (scale == mel_scale::htk)
        return static_cast<T>(C(2595) * math::log10(C(1) + f / C(700)));
    if (f >= K::min_log_hz)
        return static_cast<T>(K::min_log_mel + math::log(f / K::min_log_hz) / detail::mel::logstep<C>());
    return static_cast<T>(f / K::f_sp);
}

/// \brief A frequency in hertz.
/// \tparam T A `math::math_number`.
/// \param mel The frequency in mels.
/// \param scale The scale.
/// \return The hertz.
template <math::math_number T>
constexpr T mel_to_hz(T mel, mel_scale scale = mel_scale::slaney) noexcept
{
    using C = compute_t<T>;
    using K = detail::mel::slaney_constants<C>;
    const C m = C(mel);
    if (scale == mel_scale::htk)
        return static_cast<T>(C(700) * (math::pow(C(10), m / C(2595)) - C(1)));
    if (m >= K::min_log_mel)
        return static_cast<T>(K::min_log_hz * math::exp(detail::mel::logstep<C>() * (m - K::min_log_mel)));
    return static_cast<T>(K::f_sp * m);
}

/// \brief The parameters of a mel filterbank, with `librosa.filters.mel`'s meanings and defaults.
/// \ingroup core
/// \tparam T A `math::math_number`.
template <math::math_number T>
struct mel_filterbank_options
{
    T sample_rate = T(16000);               ///< The sample rate in hertz.
    std::size_t n_fft = 400;                ///< The transform's length: the filters have `n_fft / 2 + 1` bins.
    std::size_t n_mels = 80;                ///< The number of filters.
    T f_min = T(0);                         ///< The lowest frequency.
    T f_max = T(0);                         ///< The highest frequency; 0 means `sample_rate / 2`.
    mel_scale scale = mel_scale::slaney;    ///< The scale.
    mel_norm norm = mel_norm::slaney;       ///< The normalisation.

    /// \brief The number of bins of each filter.
    /// \return `n_fft / 2 + 1`.
    [[nodiscard]] constexpr std::size_t bins() const noexcept { return n_fft / 2 + 1; }
};

/// \brief The mel filterbank, `n_mels x (n_fft / 2 + 1)`, row-major: `librosa.filters.mel`.
/// \tparam T A `math::math_number`.
/// \param o The options.
/// \param out `n_mels * bins` values.
template <math::math_number T>
constexpr void mel_filterbank(const mel_filterbank_options<T>& o, std::span<std::type_identity_t<T>> out) noexcept
{
    using C = compute_t<T>;
    const std::size_t bins = o.bins(), mels = o.n_mels;
    assert(out.size() == mels * bins);
    const C f_max = o.f_max > T(0) ? C(o.f_max) : C(o.sample_rate) / C(2);
    const C lo = C(hz_to_mel(C(o.f_min), o.scale)), hi = C(hz_to_mel(f_max, o.scale));
    // np.linspace(lo, hi, mels + 2): i * step + lo, the last exactly hi; then back to hertz.
    const auto mel_f = [&](std::size_t i) {
        const C step = (hi - lo) / C(static_cast<long long>(mels + 1));
        const C m = i == mels + 1 ? hi : C(static_cast<long long>(i)) * step + lo;
        return mel_to_hz(m, o.scale);
    };
    // np.fft.rfftfreq(n_fft, 1 / sr): k * (1 / (n_fft * (1 / sr))).
    const C val = C(1) / (C(static_cast<long long>(o.n_fft)) * (C(1) / C(o.sample_rate)));
    for (std::size_t m = 0; m < mels; ++m) {
        const C f0 = mel_f(m), f1 = mel_f(m + 1), f2 = mel_f(m + 2);
        const C enorm = C(2) / (f2 - f0);
        for (std::size_t k = 0; k < bins; ++k) {
            const C fk = C(static_cast<long long>(k)) * val;
            const C lower = -(f0 - fk) / (f1 - f0), upper = (f2 - fk) / (f2 - f1);
            const C w = lower < upper ? lower : upper;
            const T wt = static_cast<T>(w > C(0) ? w : C(0));   // librosa stores the triangle, then scales it
            out[m * bins + k] = o.norm == mel_norm::slaney ? static_cast<T>(C(wt) * enorm) : wt;
        }
    }
}

/// \brief A mel filterbank as a `std::array`, `Mels x (NFft / 2 + 1)`: a compile-time table when the result is `constexpr`.
/// \tparam T A `math::math_number`.
/// \tparam Mels The number of filters.
/// \tparam NFft The transform's length.
/// \param sample_rate The sample rate.
/// \param f_min The lowest frequency.
/// \param f_max The highest; 0 for `sample_rate / 2`.
/// \param scale The scale.
/// \param norm The normalisation.
/// \return The filters.
template <math::math_number T, std::size_t Mels, std::size_t NFft>
constexpr std::array<T, Mels*(NFft / 2 + 1)> make_mel_filterbank(T sample_rate, T f_min = T(0), T f_max = T(0),
                                                                  mel_scale scale = mel_scale::slaney, mel_norm norm = mel_norm::slaney) noexcept
{
    std::array<T, Mels*(NFft / 2 + 1)> out{};
    mel_filterbank(mel_filterbank_options<T>{sample_rate, NFft, Mels, f_min, f_max, scale, norm}, std::span<T>(out));
    return out;
}

/// \brief The mel spectrogram of a power (or magnitude) spectrogram, frame by frame: `out[t][m] = sum_k F[m][k] P[t][k]`.
/// \tparam T A `math::math_number`.
/// \param filters `n_mels x bins`, row-major.
/// \param n_mels The number of filters.
/// \param spectrogram `frames x bins`, frame-major.
/// \param out `frames x n_mels`, frame-major.
template <math::math_number T>
constexpr void mel_spectrogram(std::span<const T> filters, std::size_t n_mels, std::span<const std::type_identity_t<T>> spectrogram,
                               std::span<std::type_identity_t<T>> out) noexcept
{
    const std::size_t bins = n_mels == 0 ? 0 : filters.size() / n_mels, frames = bins == 0 ? 0 : spectrogram.size() / bins;
    assert(filters.size() == n_mels * bins && spectrogram.size() == frames * bins && out.size() == frames * n_mels);
    for (std::size_t t = 0; t < frames; ++t)
        for (std::size_t m = 0; m < n_mels; ++m) {
            T s = T(0);
            for (std::size_t k = 0; k < bins; ++k)
                s += filters[m * bins + k] * spectrogram[t * bins + k];
            out[t * n_mels + m] = s;
        }
}

/// \brief The backward pass of `mel_spectrogram`: `dP[t][k] = sum_m F[m][k] dM[t][m]`.
/// \tparam T A `math::math_number`.
/// \param filters `n_mels x bins`.
/// \param n_mels The number of filters.
/// \param grad_mel `frames x n_mels`.
/// \param grad_spectrogram `frames x bins`, written.
template <math::math_number T>
constexpr void mel_backward(std::span<const T> filters, std::size_t n_mels, std::span<const std::type_identity_t<T>> grad_mel,
                            std::span<std::type_identity_t<T>> grad_spectrogram) noexcept
{
    const std::size_t bins = n_mels == 0 ? 0 : filters.size() / n_mels, frames = n_mels == 0 ? 0 : grad_mel.size() / n_mels;
    assert(filters.size() == n_mels * bins && grad_mel.size() == frames * n_mels && grad_spectrogram.size() == frames * bins);
    for (std::size_t t = 0; t < frames; ++t)
        for (std::size_t k = 0; k < bins; ++k) {
            T s = T(0);
            for (std::size_t m = 0; m < n_mels; ++m)
                s += filters[m * bins + k] * grad_mel[t * n_mels + m];
            grad_spectrogram[t * bins + k] = s;
        }
}

/// \brief A log compression: `y = scale (max(multiplier log10(max(x, floor)), max_y - range) + offset)`.
/// \ingroup core
/// \tparam T A `math::math_number`.
template <math::math_number T>
struct log_compression
{
    T floor = T(1e-10);                 ///< The least value taken before the logarithm (librosa's `amin`).
    T multiplier = T(10);               ///< 10 for decibels of power, 1 for Whisper.
    T range = T(80);                    ///< The dynamic range kept below the maximum (librosa's `top_db`); infinity for none.
    T offset = T(0);                    ///< Added after the floor.
    T scale = T(1);                     ///< The final factor.
};

/// \brief librosa's `power_to_db(S, ref=1.0, amin=1e-10, top_db)`.
/// \tparam T A `math::math_number`.
/// \param top_db The dynamic range kept.
/// \return The compression.
template <math::math_number T>
constexpr log_compression<T> decibels(T top_db = T(80)) noexcept
{
    return {T(1e-10), T(10), top_db, T(0), T(1)};
}

/// \brief Whisper's: `log10(clamp(x, 1e-10))`, floored at the maximum less 8, then `(x + 4) / 4`.
/// \tparam T A `math::math_number`.
/// \return The compression.
template <math::math_number T>
constexpr log_compression<T> whisper_compression() noexcept
{
    return {T(1e-10), T(1), T(8), T(4), T(0.25)};
}

/// \brief Applies a log compression to a whole array (its maximum taken over all of it).
/// \tparam T A `math::math_number`.
/// \param in The values (powers).
/// \param out The compressed values, the same size; may be `in`.
/// \param c The compression.
template <math::math_number T>
constexpr void log_compress(std::span<const T> in, std::span<std::type_identity_t<T>> out, const log_compression<std::type_identity_t<T>>& c) noexcept
{
    assert(out.size() == in.size());
    if (in.empty())
        return;
    T top = T(0);
    for (std::size_t i = 0; i < in.size(); ++i) {
        const T y = c.multiplier * math::log10(in[i] > c.floor ? in[i] : c.floor);
        out[i] = y;
        top = i == 0 || y > top ? y : top;
    }
    const T lowest = top - c.range;
    for (auto& y : out)
        y = ((y > lowest ? y : lowest) + c.offset) * c.scale;
}

namespace detail::mel
{
    // What log_compress's backward pass needs beyond each element: the maximum, the floor below it, and each tied
    // maximum's share of the gradients of the values floored to it.
    template <typename T>
    struct compression_backward
    {
        log_compression<T> c;
        T top = T(0), lowest = T(0), share = T(0);

        constexpr T log_of(T x) const noexcept { return c.multiplier * math::log10(x > c.floor ? x : c.floor); }

        template <typename In, typename Grad>
        constexpr compression_backward(const log_compression<T>& comp, std::size_t n, In in, Grad grad) noexcept : c(comp)
        {
            top = n == 0 ? T(0) : log_of(in(0));
            for (std::size_t i = 1; i < n; ++i)
                top = log_of(in(i)) > top ? log_of(in(i)) : top;
            lowest = top - c.range;
            T to_top = T(0);
            std::size_t ties = 0;
            for (std::size_t i = 0; i < n; ++i) {
                const T y = log_of(in(i));
                ties += y == top ? 1 : 0;
                to_top += y < lowest ? grad(i) * c.scale : T(0);
            }
            share = ties == 0 ? T(0) : to_top / T(static_cast<long long>(ties));
        }

        // dL/dx for one element.
        constexpr T operator()(T x, T g) const noexcept
        {
            const T y = log_of(x);
            const T dy = (y < lowest ? T(0) : g * c.scale) + (y == top ? share : T(0));
            return x >= c.floor ? dy * c.multiplier / (x * math::log(T(10))) : T(0);
        }
    };
} // namespace detail::mel

/// \brief The backward pass of `log_compress`: given the input and `dL/dy`, `dL/dx`.
///
/// A value above the floor passes `scale multiplier / (x ln 10)`; a value at or below `floor` passes nothing; a value
/// below `max - range` passes its gradient to the maximum, split evenly among ties, as PyTorch's.
/// \tparam T A `math::math_number`.
/// \param in The values `log_compress` was given.
/// \param grad_out `dL/dy`.
/// \param grad_in `dL/dx`, written; may be `grad_out`, not `in`.
/// \param c The compression.
template <math::math_number T>
constexpr void log_compress_backward(std::span<const T> in, std::span<const std::type_identity_t<T>> grad_out,
                                     std::span<std::type_identity_t<T>> grad_in, const log_compression<std::type_identity_t<T>>& c) noexcept
{
    assert(grad_out.size() == in.size() && grad_in.size() == in.size());
    const detail::mel::compression_backward<T> b(c, in.size(), [in](std::size_t i) { return in[i]; },
                                                 [grad_out](std::size_t i) { return grad_out[i]; });
    for (std::size_t i = 0; i < in.size(); ++i)
        grad_in[i] = b(in[i], grad_out[i]);
}

/// \brief Whisper's sample rate, 16 kHz.
inline constexpr std::size_t whisper_sample_rate = 16000;
/// \brief Whisper's transform length, 400 (25 ms).
inline constexpr std::size_t whisper_n_fft = 400;
/// \brief Whisper's hop, 160 (10 ms).
inline constexpr std::size_t whisper_hop_length = 160;
/// \brief The samples of one 30-second chunk, 480,000.
inline constexpr std::size_t whisper_n_samples = 30 * whisper_sample_rate;
/// \brief The frames of one chunk, 3,000.
inline constexpr std::size_t whisper_n_frames = whisper_n_samples / whisper_hop_length;

/// \brief Whisper's STFT: 400 points, hop 160, centred, reflect padding, not normalised.
/// \return The options.
constexpr stft_options whisper_stft_options() noexcept { return {whisper_n_fft, whisper_hop_length, true, pad_mode::reflect, false}; }

/// \brief The frames `whisper_log_mel` gives: `(samples + padding) / 160` (the STFT's frames less the last).
/// \param samples The audio's length.
/// \param padding The zeros appended.
/// \return The number of frames.
constexpr std::size_t whisper_frames(std::size_t samples, std::size_t padding = 0) noexcept
{
    return (samples + padding) / whisper_hop_length;
}

/// \brief Whisper's window: `torch.hann_window(400)`, periodic.
/// \tparam T A `math::math_number`.
/// \return The window.
template <math::math_number T>
constexpr std::array<T, whisper_n_fft> whisper_window() noexcept
{
    return make_window<T, whisper_n_fft>(hann{}, window_symmetry::periodic);
}

/// \brief Whisper's mel filters: `librosa.filters.mel(sr=16000, n_fft=400, n_mels=Mels)`, Mels 80 or 128.
/// \tparam T A `math::math_number`.
/// \tparam Mels 80 (Whisper's models to large-v2) or 128 (large-v3).
/// \return The filters, `Mels x 201`.
template <math::math_number T, std::size_t Mels>
    requires(Mels == 80 || Mels == 128)
constexpr std::array<T, Mels*(whisper_n_fft / 2 + 1)> whisper_mel_filters() noexcept
{
    return make_mel_filterbank<T, Mels, whisper_n_fft>(T(whisper_sample_rate));
}

/// \brief The work buffer `whisper_log_mel` needs: one frame's spectrum and the plan's work.
/// \tparam T A `math::math_number`.
/// \param plan The real plan of length 400.
/// \return The size, in complex elements.
template <math::math_number T>
constexpr std::size_t whisper_work_size(const rfft_plan<T>& plan) noexcept
{
    return plan.bins() + plan.work_size();
}

/// \brief OpenAI Whisper's `log_mel_spectrogram(audio, n_mels, padding)`: mel-major, `n_mels x whisper_frames(audio.size(), padding)`.
/// \tparam T A `math::math_number`.
/// \param plan The real plan of length 400.
/// \param window `whisper_window<T>()`.
/// \param filters `whisper_mel_filters<T, n_mels>()`, `n_mels x 201`.
/// \param n_mels 80 or 128.
/// \param audio The samples at 16 kHz.
/// \param padding Zeros appended first (Whisper's `padding`; `whisper_n_samples` to pad a chunk).
/// \param out `n_mels * whisper_frames(audio.size(), padding)` values, mel-major.
/// \param work At least `whisper_work_size(plan)` elements.
template <math::math_number T>
constexpr void whisper_log_mel(const rfft_plan<T>& plan, std::span<const std::type_identity_t<T>> window, std::span<const std::type_identity_t<T>> filters,
                               std::size_t n_mels, std::span<const std::type_identity_t<T>> audio, std::size_t padding,
                               std::span<std::type_identity_t<T>> out, std::span<complex<std::type_identity_t<T>>> work) noexcept
{
    const stft_options o = whisper_stft_options();
    const std::size_t bins = plan.bins(), length = audio.size() + padding, frames = whisper_frames(audio.size(), padding);
    assert(plan.size() == whisper_n_fft && window.size() == whisper_n_fft && filters.size() == n_mels * bins);
    assert(out.size() == n_mels * frames && work.size() >= whisper_work_size(plan));
    const auto row = work.first(bins);
    const auto sample = [&](std::size_t p) {
        const long long i = detail::stft::source_index(p, o.padding(), length, o.pad);
        return static_cast<std::size_t>(i) < audio.size() ? audio[static_cast<std::size_t>(i)] : T(0);
    };
    for (std::size_t t = 0; t < frames; ++t) {
        plan.forward_from([&](std::size_t j) { return window[j] * sample(t * o.hop + j); }, row, work.subspan(bins));
        for (std::size_t k = 0; k < bins; ++k)
            row[k].re = norm(row[k]);
        for (std::size_t m = 0; m < n_mels; ++m) {
            T s = T(0);
            for (std::size_t k = 0; k < bins; ++k)
                s += filters[m * bins + k] * row[k].re;
            out[m * frames + t] = s;
        }
    }
    log_compress(std::span<const T>(out), out, whisper_compression<T>());
}

/// \brief The backward pass of `whisper_log_mel`: given `dL/dy` (mel-major), `dL/d audio`, each frame's spectrum recomputed.
/// \tparam T A `math::math_number`.
/// \param plan The real plan of length 400.
/// \param window `whisper_window<T>()`.
/// \param filters The mel filters, `n_mels x 201`.
/// \param n_mels 80 or 128.
/// \param audio The samples `whisper_log_mel` was given.
/// \param padding The zeros it appended.
/// \param grad_out `dL/dy`, `n_mels x frames`, mel-major.
/// \param grad_audio `dL/d audio`, `audio.size()` values, written.
/// \param scratch `n_mels x frames` values: the mel spectrogram before compression, recomputed.
/// \param work At least `whisper_work_size(plan)` elements.
template <math::math_number T>
constexpr void whisper_log_mel_backward(const rfft_plan<T>& plan, std::span<const std::type_identity_t<T>> window,
                                        std::span<const std::type_identity_t<T>> filters, std::size_t n_mels,
                                        std::span<const std::type_identity_t<T>> audio, std::size_t padding,
                                        std::span<const std::type_identity_t<T>> grad_out, std::span<std::type_identity_t<T>> grad_audio,
                                        std::span<std::type_identity_t<T>> scratch, std::span<complex<std::type_identity_t<T>>> work) noexcept
{
    const stft_options o = whisper_stft_options();
    const std::size_t bins = plan.bins(), length = audio.size() + padding, frames = whisper_frames(audio.size(), padding), n = o.n_fft;
    assert(grad_out.size() == n_mels * frames && grad_audio.size() == audio.size() && scratch.size() == n_mels * frames);
    assert(work.size() >= whisper_work_size(plan));
    const auto row = work.first(bins);
    const auto sample = [&](std::size_t p) {
        const long long i = detail::stft::source_index(p, o.padding(), length, o.pad);
        return static_cast<std::size_t>(i) < audio.size() ? audio[static_cast<std::size_t>(i)] : T(0);
    };
    const auto spectrum_of = [&](std::size_t t) {
        plan.forward_from([&](std::size_t j) { return window[j] * sample(t * o.hop + j); }, row, work.subspan(bins));
    };
    for (std::size_t t = 0; t < frames; ++t) {   // the mel spectrogram, before compression
        spectrum_of(t);
        for (std::size_t m = 0; m < n_mels; ++m) {
            T s = T(0);
            for (std::size_t k = 0; k < bins; ++k)
                s += filters[m * bins + k] * norm(row[k]);
            scratch[m * frames + t] = s;
        }
    }
    const detail::mel::compression_backward<T> back(whisper_compression<T>(), scratch.size(), [scratch](std::size_t i) { return scratch[i]; },
                                                    [grad_out](std::size_t i) { return grad_out[i]; });
    for (std::size_t i = 0; i < scratch.size(); ++i)
        scratch[i] = back(scratch[i], grad_out[i]);   // now dL/d mel
    for (auto& g : grad_audio)
        g = T(0);
    for (std::size_t t = 0; t < frames; ++t) {
        spectrum_of(t);
        for (std::size_t k = 0; k < bins; ++k) {   // dL/dX_k = 2 X_k sum_m F[m][k] dL/dmel[m][t]
            T dp = T(0);
            for (std::size_t m = 0; m < n_mels; ++m)
                dp += filters[m * bins + k] * scratch[m * frames + t];
            row[k] *= T(2) * dp;
        }
        plan.inverse_to([&](std::size_t k) { return k == 0 || 2 * k == n ? row[k] : row[k] * T(0.5); },
                        [&](std::size_t j, T v) {
                            const long long i = detail::stft::source_index(t * o.hop + j, o.padding(), length, o.pad);
                            if (static_cast<std::size_t>(i) < audio.size())
                                grad_audio[static_cast<std::size_t>(i)] += window[j] * v;
                        },
                        work.subspan(bins));
    }
}

/// \brief MFCCs: the DCT-II of each frame of a log-mel spectrogram (frame-major), its first `n_mfcc` coefficients, as `librosa.feature.mfcc`.
/// \tparam T A `math::math_number`.
/// \param plan The complex plan of length `n_mels`.
/// \param log_mel `frames x n_mels`, frame-major (librosa's input is `log_compress` with `decibels()`).
/// \param out `frames x n_mfcc`, frame-major.
/// \param n_mfcc The coefficients kept, at most `n_mels`.
/// \param work At least `dct_work_size(plan)` elements.
/// \param norm The DCT's scaling; librosa's default is `ortho`.
template <math::math_number T>
constexpr void mfcc(const fft_plan<T>& plan, std::span<const std::type_identity_t<T>> log_mel, std::span<std::type_identity_t<T>> out,
                    std::size_t n_mfcc, std::span<complex<std::type_identity_t<T>>> work, dct_norm norm = dct_norm::ortho) noexcept
{
    const std::size_t n_mels = plan.size(), frames = n_mels == 0 ? 0 : log_mel.size() / n_mels;
    assert(log_mel.size() == frames * n_mels && out.size() == frames * n_mfcc && n_mfcc <= n_mels);
    for (std::size_t t = 0; t < frames; ++t)
        dct2(plan, log_mel.subspan(t * n_mels, n_mels), out.subspan(t * n_mfcc, n_mfcc), work, norm);
}

} // namespace snicholls::signal

#endif // snicholls_core_signal_mel_hpp
