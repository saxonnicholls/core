//
//  fir.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::signal` FIR filter design by the windowed sinc (`scipy.signal.firwin`): low-pass, high-pass, band-pass and band-stop, `constexpr` so a filter can be a compile-time table, with direct filtering and the frequency response.
/// \ingroup core
///
/// **Overview.** `fir_design(h, band, f1, f2, window)` is `scipy.signal.firwin(numtaps, cutoff, window,
/// pass_zero, scale=True)` with frequencies as fractions of the Nyquist frequency (scipy's default `fs = 2`): the
/// ideal response's impulse response, `sum over pass bands (right sinc(right m) - left sinc(left m))` with
/// `m = i - (taps - 1) / 2`, multiplied by a symmetric window (Hamming by default, as scipy's), then scaled to unit
/// gain at the centre of the first pass band (0 for a low-pass, Nyquist for a high-pass). A high-pass or band-stop
/// filter passes Nyquist, so it needs an odd number of taps.
///
/// `make_fir<T, N>(band, f1, f2, window)` returns the taps as a `std::array`: `constexpr auto h =
/// make_fir<float, 63>(fir_band::lowpass, 0.25);` is computed by the compiler. `fir_filter` applies taps directly
/// (`scipy.signal.lfilter(h, 1, x)` from rest); `fir_response` evaluates `H(e^(i omega))`.
///
/// \snippet signal/fir_demo.cpp fir
///
/// **Relationships.** `signal/window.hpp` (the windows), `signal/convolve.hpp` (FFT filtering for long filters),
/// `signal/resample.hpp` (its anti-aliasing filter is a `fir_design` low-pass). Group: \ref core.
///
/// **Implementations.** The windowed sinc only; least-squares and Parks-McClellan designs are not provided.
///
/// **Guarantees.** `constexpr`, no allocation. Taps are symmetric (linear phase). Computed in `compute_t<T>`.
/// Complexity: `O(taps)` to design, `O(taps)` per output sample to filter.
///
/// **Errors.** Preconditions, asserted: `0 < f1 < 1` (and `f1 < f2 < 1` for the band forms); an odd number of taps for
/// `highpass` and `bandstop`, as scipy requires.
///
/// **Performance.** Not benchmarked: design runs once; `fir_filter` is the direct convolution
/// (`signal/convolve.hpp`'s benchmark).
///
/// **Tests and demos.** `tests/signal/fir_test.cpp` (scipy 1.13.1's taps, the responses at sample frequencies,
/// compile-time design), `demos/signal/fir_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cassert>`, `<cstddef>`, `<span>`, `<type_traits>`, `signal/complex.hpp`,
/// `signal/window.hpp`.
///
/// **Known issues.** scipy also accepts several bands and a sample rate; here there is at most one band and the
/// frequencies are normalised.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_signal_fir_hpp
#define snicholls_core_signal_fir_hpp

#include <snicholls/core/signal/complex.hpp>
#include <snicholls/core/signal/window.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <span>
#include <type_traits>

namespace snicholls::signal
{

/// \brief Which band a filter passes.
/// \ingroup core
enum class fir_band
{
    lowpass,    ///< Below `f1`.
    highpass,   ///< Above `f1`.
    bandpass,   ///< Between `f1` and `f2`.
    bandstop,   ///< Outside `f1` to `f2`.
};

/// \brief The normalised sinc, `sin(pi x) / (pi x)`, 1 at 0 (numpy's `sinc`).
/// \tparam T A `math::math_number`.
/// \param x The argument.
/// \return The value.
template <math::math_number T>
constexpr T sinc(T x) noexcept
{
    if (x == T(0))
        return T(1);
    using C = compute_t<T>;
    const C y = math::math_limits<C>::pi * C(x);
    return static_cast<T>(math::sin(y) / y);
}

/// \brief Designs a filter by the windowed sinc into `h`: `scipy.signal.firwin(h.size(), cutoff, window, pass_zero, scale=True)`.
/// \tparam T A `math::math_number`.
/// \tparam W A window type.
/// \param h The taps.
/// \param band The band passed.
/// \param f1 The (lower) edge, a fraction of Nyquist.
/// \param f2 The upper edge for the band forms.
/// \param window The window, applied in its symmetric form.
template <math::math_number T, typename W = hamming>
constexpr void fir_design(std::span<T> h, fir_band band, std::type_identity_t<T> f1, std::type_identity_t<T> f2 = T(0), const W& window = W{}) noexcept
{
    using C = compute_t<T>;
    const std::size_t taps = h.size();
    [[maybe_unused]] const bool two_edges = band == fir_band::bandpass || band == fir_band::bandstop;
    assert(f1 > T(0) && f1 < T(1) && (!two_edges || (f2 > f1 && f2 < T(1))));
    assert((band != fir_band::highpass && band != fir_band::bandstop) || taps % 2 == 1);
    // The pass bands, as scipy's hstack of [0] * pass_zero, cutoffs, [1] * pass_nyquist.
    std::array<C, 4> edges{};
    std::size_t count = 0;
    const auto push = [&](C v) { edges[count++] = v; };
    switch (band) {
    case fir_band::lowpass: push(C(0)); push(C(f1)); break;
    case fir_band::highpass: push(C(f1)); push(C(1)); break;
    case fir_band::bandpass: push(C(f1)); push(C(f2)); break;
    case fir_band::bandstop: push(C(0)); push(C(f1)); push(C(f2)); push(C(1)); break;
    }
    const C alpha = C(0.5) * C(static_cast<long long>(taps) - 1);
    const C left0 = edges[0], right0 = edges[1];
    const C scale_frequency = left0 == C(0) ? C(0) : (right0 == C(1) ? C(1) : C(0.5) * (left0 + right0));
    // The taps are symmetric: compute the first half and mirror it (half the work, which matters in constant evaluation).
    C s = C(0);
    for (std::size_t i = 0; 2 * i < taps; ++i) {
        const C m = C(static_cast<long long>(i)) - alpha;
        C v = C(0);
        for (std::size_t b = 0; b < count; b += 2)
            v += edges[b + 1] * sinc(edges[b + 1] * m) - (edges[b] == C(0) ? C(0) : edges[b] * sinc(edges[b] * m));
        v *= C(window_value<T>(window, i, taps, window_symmetry::symmetric));
        h[i] = h[taps - 1 - i] = static_cast<T>(v);
        const C c = scale_frequency == C(0) ? C(1) : math::cos(math::math_limits<C>::pi * m * scale_frequency);
        s += (2 * i + 1 == taps ? C(1) : C(2)) * v * c;
    }
    for (auto& v : h)
        v = static_cast<T>(C(v) / s);
}

/// \brief A windowed-sinc filter of `N` taps as a `std::array`: a compile-time table when the result is `constexpr`.
/// \tparam T A `math::math_number`.
/// \tparam N The number of taps.
/// \tparam W A window type.
/// \param band The band passed.
/// \param f1 The (lower) edge, a fraction of Nyquist.
/// \param f2 The upper edge for the band forms.
/// \param window The window.
/// \return The taps.
template <math::math_number T, std::size_t N, typename W = hamming>
constexpr std::array<T, N> make_fir(fir_band band, T f1, T f2 = T(0), const W& window = W{}) noexcept
{
    std::array<T, N> h{};
    fir_design(std::span<T>(h), band, f1, f2, window);
    return h;
}

/// \brief The frequency response of taps at `omega` radians per sample: `H = sum_k h_k e^(-i omega k)`.
/// \tparam T A `math::math_number`.
/// \param h The taps.
/// \param omega The frequency, `pi` at Nyquist.
/// \return The response.
template <math::math_number T>
constexpr complex<T> fir_response(std::span<const T> h, std::type_identity_t<T> omega) noexcept
{
    complex<T> acc{};
    for (std::size_t k = 0; k < h.size(); ++k) {
        const T a = -omega * T(static_cast<long long>(k));
        acc += complex<T>{math::cos(a), math::sin(a)} * h[k];
    }
    return acc;
}

/// \brief Filters `x` by taps from rest: `y_n = sum_k h_k x_(n-k)`, `scipy.signal.lfilter(h, 1, x)`.
/// \tparam T A `math::math_number`.
/// \param h The taps.
/// \param x The signal.
/// \param y `x.size()` outputs; must not overlap `x`.
template <math::math_number T>
constexpr void fir_filter(std::span<const T> h, std::span<const std::type_identity_t<T>> x, std::span<std::type_identity_t<T>> y) noexcept
{
    assert(y.size() == x.size());
    for (std::size_t n = 0; n < x.size(); ++n) {
        T acc = T(0);
        const std::size_t kmax = n + 1 < h.size() ? n + 1 : h.size();
        for (std::size_t k = 0; k < kmax; ++k)
            acc += h[k] * x[n - k];
        y[n] = acc;
    }
}

} // namespace snicholls::signal

#endif // snicholls_core_signal_fir_hpp
