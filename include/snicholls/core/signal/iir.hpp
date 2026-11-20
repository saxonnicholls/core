//
//  iir.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::signal` IIR filters: biquads with the Audio EQ Cookbook's coefficients, Butterworth design as second-order sections (low-, high-, band-pass, band-stop), `sosfilt`, and frequency responses, all `constexpr`.
/// \ingroup core
///
/// **Overview.** A `biquad<T>` is one second-order section `H(z) = (b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2)`
/// (normalised so `a0 = 1`; a first-order section has `b2 = a2 = 0`). A filter is a cascade of sections, applied by
/// `sosfilt` (`scipy.signal.sosfilt`: each section in transposed direct form II, state from the caller, zero for a
/// filter at rest) and evaluated by `sos_response` at any frequency.
///
/// - `rbj_biquad(kind, f0, fs, q, gain_db)`: Robert Bristow-Johnson's Audio EQ Cookbook (low-pass, high-pass,
///   band-pass with 0 dB peak, notch, all-pass, peaking, low shelf, high shelf; `q` as the cookbook's Q, the
///   shelves' slope given through Q as well).
/// - `butterworth(order, band, w1, w2, sections)`: the digital Butterworth filter of `scipy.signal.butter(order, Wn,
///   btype, output='sos')`, edges as fractions of Nyquist, prewarped and mapped by the bilinear transform. The analog
///   prototype's poles `e^(i pi (2k + order + 1) / 2 order)` are placed in closed form, paired with their conjugates,
///   and each pair becomes one section (band forms: each prototype pole gives two). The cascade's response is
///   scipy's to rounding; the split of the gain among the sections differs (here each low- and high-pass section
///   has unit gain at DC or Nyquist, each band-stop section unit gain at DC).
/// - `make_butterworth<T, Order>(band, w1, w2)`: the sections as a `std::array`, a compile-time table when
///   `constexpr`.
///
/// \snippet signal/iir_demo.cpp iir
///
/// **Relationships.** `signal/complex.hpp`; `signal/fir.hpp` for linear-phase filters. Group: \ref core.
///
/// **Implementations.** The cookbook's eight biquads; Butterworth only among the classical designs (Chebyshev and
/// elliptic are not provided).
///
/// **Guarantees.** `constexpr`, no allocation; `biquad` and `biquad_state` are trivially copyable. `sosfilt` costs
/// five multiplications and four additions per section per sample and may filter in place. Sections:
/// `(order + 1) / 2` for low- and high-pass, `order` for the band forms.
///
/// **Errors.** Preconditions, asserted: `0 < f0 < fs / 2`; `0 < w1 (< w2) < 1`; `order >= 1`; the sections' span has
/// `butterworth_sections(order, band)` elements and the state one per section.
///
/// **Performance.** Not benchmarked: a section is the arithmetic written out.
///
/// **Tests and demos.** `tests/signal/iir_test.cpp` (the cookbook's responses at their defining frequencies,
/// Butterworth magnitudes against the closed form `1 / (1 + (Omega / Omega_c)^(2 order))` at sample frequencies and
/// scipy 1.13.1's `sosfreqz` and `sosfilt`, compile-time design), `demos/signal/iir_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cassert>`, `<cstddef>`, `<span>`, `<type_traits>`, `signal/complex.hpp`.
///
/// **Known issues.** High orders with narrow bands lose precision in `float` (poles near the unit circle); design in
/// `double` and round the coefficients if `float` is needed.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_signal_iir_hpp
#define snicholls_core_signal_iir_hpp

#include <snicholls/core/signal/complex.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <span>
#include <type_traits>

namespace snicholls::signal
{

/// \brief One second-order section, normalised so `a0 = 1`.
/// \ingroup core
/// \tparam T A `math::math_number`.
template <math::math_number T>
struct biquad
{
    T b0 = T(1);   ///< The numerator's constant term.
    T b1 = T(0);   ///< The numerator's `z^-1` term.
    T b2 = T(0);   ///< The numerator's `z^-2` term.
    T a1 = T(0);   ///< The denominator's `z^-1` term.
    T a2 = T(0);   ///< The denominator's `z^-2` term.

    /// \brief The response at `omega` radians per sample: `H(e^(i omega))`.
    /// \param omega The frequency, `pi` at Nyquist.
    /// \return The response.
    [[nodiscard]] constexpr complex<T> response(T omega) const noexcept
    {
        const complex<T> z1{math::cos(omega), -math::sin(omega)};   // e^(-i omega)
        const complex<T> z2 = z1 * z1;
        const complex<T> num = complex<T>{b0, T(0)} + z1 * b1 + z2 * b2;
        const complex<T> den = complex<T>{T(1), T(0)} + z1 * a1 + z2 * a2;
        return num / den;
    }

    /// \brief Equality of every coefficient.
    /// \param a The first section.
    /// \param b The second.
    /// \return Whether they are equal.
    friend constexpr bool operator==(const biquad& a, const biquad& b) noexcept = default;
};

/// \brief A section's state in transposed direct form II: zero for a filter at rest.
/// \ingroup core
/// \tparam T A `math::math_number`.
template <math::math_number T>
struct biquad_state
{
    T z1 = T(0);   ///< The first delay.
    T z2 = T(0);   ///< The second delay.

    /// \brief Filters one sample through a section, updating the state.
    /// \param s The section.
    /// \param x The input sample.
    /// \return The output sample.
    constexpr T process(const biquad<T>& s, T x) noexcept
    {
        const T y = s.b0 * x + z1;
        z1 = s.b1 * x - s.a1 * y + z2;
        z2 = s.b2 * x - s.a2 * y;
        return y;
    }
};

/// \brief The Audio EQ Cookbook's filters.
/// \ingroup core
enum class biquad_kind
{
    lowpass,     ///< `(1 - cos w0) / 2 (1 + 2 z^-1 + z^-2)`, `-3 dB` at `f0` for Q = 1/sqrt(2).
    highpass,    ///< The high-pass mirror.
    bandpass,    ///< Constant 0 dB peak gain at `f0`.
    notch,       ///< Zero gain at `f0`.
    allpass,     ///< Unit gain, phase `-pi` at `f0`.
    peaking,     ///< `gain_db` at `f0`.
    lowshelf,    ///< `gain_db` below `f0`.
    highshelf,   ///< `gain_db` above `f0`.
};

/// \brief A cookbook biquad.
/// \tparam T A `math::math_number`.
/// \param kind The filter.
/// \param f0 The centre or corner frequency, in hertz.
/// \param fs The sample rate.
/// \param q The quality factor (`1 / sqrt(2)` for a Butterworth-like low- or high-pass).
/// \param gain_db The gain of the peaking and shelving filters, in decibels.
/// \return The section, normalised by `a0`.
template <math::math_number T>
constexpr biquad<T> rbj_biquad(biquad_kind kind, T f0, std::type_identity_t<T> fs, std::type_identity_t<T> q,
                               std::type_identity_t<T> gain_db = T(0)) noexcept
{
    using C = compute_t<T>;
    assert(f0 > T(0) && f0 < fs / T(2) && q > T(0));
    const C w0 = C(2) * math::math_limits<C>::pi * C(f0) / C(fs), cw = math::cos(w0), sw = math::sin(w0);
    const C alpha = sw / (C(2) * C(q)), a = math::pow(C(10), C(gain_db) / C(40)), ra = C(2) * math::sqrt(a) * alpha;
    std::array<C, 6> c{};   // b0 b1 b2 a0 a1 a2
    switch (kind) {
    case biquad_kind::lowpass: c = {(1 - cw) / 2, 1 - cw, (1 - cw) / 2, 1 + alpha, -2 * cw, 1 - alpha}; break;
    case biquad_kind::highpass: c = {(1 + cw) / 2, -(1 + cw), (1 + cw) / 2, 1 + alpha, -2 * cw, 1 - alpha}; break;
    case biquad_kind::bandpass: c = {alpha, 0, -alpha, 1 + alpha, -2 * cw, 1 - alpha}; break;
    case biquad_kind::notch: c = {1, -2 * cw, 1, 1 + alpha, -2 * cw, 1 - alpha}; break;
    case biquad_kind::allpass: c = {1 - alpha, -2 * cw, 1 + alpha, 1 + alpha, -2 * cw, 1 - alpha}; break;
    case biquad_kind::peaking: c = {1 + alpha * a, -2 * cw, 1 - alpha * a, 1 + alpha / a, -2 * cw, 1 - alpha / a}; break;
    case biquad_kind::lowshelf:
        c = {a * ((a + 1) - (a - 1) * cw + ra), 2 * a * ((a - 1) - (a + 1) * cw), a * ((a + 1) - (a - 1) * cw - ra),
             (a + 1) + (a - 1) * cw + ra, -2 * ((a - 1) + (a + 1) * cw), (a + 1) + (a - 1) * cw - ra};
        break;
    case biquad_kind::highshelf:
        c = {a * ((a + 1) + (a - 1) * cw + ra), -2 * a * ((a - 1) + (a + 1) * cw), a * ((a + 1) + (a - 1) * cw - ra),
             (a + 1) - (a - 1) * cw + ra, 2 * ((a - 1) - (a + 1) * cw), (a + 1) - (a - 1) * cw - ra};
        break;
    }
    return {static_cast<T>(c[0] / c[3]), static_cast<T>(c[1] / c[3]), static_cast<T>(c[2] / c[3]), static_cast<T>(c[4] / c[3]),
            static_cast<T>(c[5] / c[3])};
}

/// \brief Which band a Butterworth filter passes.
/// \ingroup core
enum class iir_band
{
    lowpass,    ///< Below `w1`.
    highpass,   ///< Above `w1`.
    bandpass,   ///< Between `w1` and `w2`.
    bandstop,   ///< Outside `w1` to `w2`.
};

/// \brief The number of sections of a Butterworth filter: `(order + 1) / 2`, or `order` for the band forms.
/// \param order The prototype's order.
/// \param band The band.
/// \return The number of sections.
constexpr std::size_t butterworth_sections(std::size_t order, iir_band band) noexcept
{
    return band == iir_band::bandpass || band == iir_band::bandstop ? order : (order + 1) / 2;
}

namespace detail::iir
{
    // Bilinear transform s = (1 - z^-1) / (1 + z^-1) of (n2 s^2 + n1 s + n0) / (s^2 + d1 s + d0), normalised by a0.
    template <typename C, typename T>
    constexpr biquad<T> bilinear(C n2, C n1, C n0, C d1, C d0) noexcept
    {
        const C a0 = C(1) + d1 + d0, a1 = C(2) * d0 - C(2), a2 = C(1) - d1 + d0;
        const C b0 = n2 + n1 + n0, b1 = C(2) * n0 - C(2) * n2, b2 = n2 - n1 + n0;
        return {static_cast<T>(b0 / a0), static_cast<T>(b1 / a0), static_cast<T>(b2 / a0), static_cast<T>(a1 / a0), static_cast<T>(a2 / a0)};
    }

    // Bilinear transform of the first-order (n1 s + n0) / (s + d0).
    template <typename C, typename T>
    constexpr biquad<T> bilinear1(C n1, C n0, C d0) noexcept
    {
        const C a0 = C(1) + d0, a1 = d0 - C(1);
        return {static_cast<T>((n1 + n0) / a0), static_cast<T>((n0 - n1) / a0), T(0), static_cast<T>(a1 / a0), T(0)};
    }
} // namespace detail::iir

/// \brief Designs a digital Butterworth filter as second-order sections: `scipy.signal.butter(order, [w1, w2], band, output='sos')`.
/// \tparam T A `math::math_number`.
/// \param order The prototype's order, at least 1.
/// \param band The band.
/// \param w1 The (lower) edge, a fraction of Nyquist, where the gain is `1 / sqrt(2)`.
/// \param w2 The upper edge for the band forms.
/// \param sections `butterworth_sections(order, band)` sections, written in order.
template <math::math_number T>
constexpr void butterworth(std::size_t order, iir_band band, T w1, std::type_identity_t<T> w2, std::span<biquad<std::type_identity_t<T>>> sections) noexcept
{
    using C = compute_t<T>;
    const bool two = band == iir_band::bandpass || band == iir_band::bandstop;
    assert(order >= 1 && w1 > T(0) && w1 < T(1) && (!two || (w2 > w1 && w2 < T(1))));
    assert(sections.size() == butterworth_sections(order, band));
    const C pi = math::math_limits<C>::pi;
    const C o1 = math::tan(pi * C(w1) / C(2));   // prewarped edges for s = (1 - z^-1) / (1 + z^-1)
    const C o2 = two ? math::tan(pi * C(w2) / C(2)) : C(0);
    const C bw = o2 - o1, o0sq = o1 * o2;
    std::size_t out = 0;
    // The prototype's poles in the upper half-plane, p_k = e^(i theta), theta = pi (2k + order + 1) / (2 order), k < order / 2;
    // and for odd order the real pole -1.
    for (std::size_t k = 0; k < order / 2; ++k) {
        const C theta = pi * C(static_cast<long long>(2 * k + order + 1)) / C(static_cast<long long>(2 * order));
        const complex<C> p{math::cos(theta), math::sin(theta)};
        switch (band) {
        case iir_band::lowpass:   // Omega_c^2 / (s^2 - 2 Re(p) Omega_c s + Omega_c^2)
            sections[out++] = detail::iir::bilinear<C, T>(C(0), C(0), o1 * o1, -C(2) * p.re * o1, o1 * o1);
            break;
        case iir_band::highpass:   // s^2 / (s^2 - 2 Re(p) Omega_c s + Omega_c^2)
            sections[out++] = detail::iir::bilinear<C, T>(C(1), C(0), C(0), -C(2) * p.re * o1, o1 * o1);
            break;
        case iir_band::bandpass:
        case iir_band::bandstop: {
            // Band-pass: s^2 - p BW s + Omega_0^2 = 0; band-stop: s^2 - (BW / p) s + Omega_0^2 = 0. Two roots, each with its conjugate.
            const complex<C> c = band == iir_band::bandpass ? p * bw : conj(p) * bw;   // 1 / p = conj p on the unit circle
            const complex<C> disc = sqrt(c * c - complex<C>{C(4) * o0sq, C(0)});
            for (const complex<C> r : std::array<complex<C>, 2>{(c + disc) * C(0.5), (c - disc) * C(0.5)}) {
                const C d1 = -C(2) * r.re, d0 = norm(r);
                sections[out++] = band == iir_band::bandpass ? detail::iir::bilinear<C, T>(C(0), bw, C(0), d1, d0)
                                                             : detail::iir::bilinear<C, T>(d0 / o0sq, C(0), d0, d1, d0);
            }
            break;
        }
        }
    }
    if (order % 2 == 1) {   // the real pole p = -1
        switch (band) {
        case iir_band::lowpass: sections[out++] = detail::iir::bilinear1<C, T>(C(0), o1, o1); break;
        case iir_band::highpass: sections[out++] = detail::iir::bilinear1<C, T>(C(1), C(0), o1); break;
        case iir_band::bandpass: sections[out++] = detail::iir::bilinear<C, T>(C(0), bw, C(0), bw, o0sq); break;
        case iir_band::bandstop: sections[out++] = detail::iir::bilinear<C, T>(C(1), C(0), o0sq, bw, o0sq); break;
        }
    }
}

/// \brief A Butterworth filter's sections as a `std::array`: a compile-time table when the result is `constexpr`.
/// \tparam T A `math::math_number`.
/// \tparam Order The prototype's order.
/// \tparam Band The band.
/// \param w1 The (lower) edge, a fraction of Nyquist.
/// \param w2 The upper edge for the band forms.
/// \return The sections.
template <math::math_number T, std::size_t Order, iir_band Band>
constexpr std::array<biquad<T>, butterworth_sections(Order, Band)> make_butterworth(T w1, T w2 = T(0)) noexcept
{
    std::array<biquad<T>, butterworth_sections(Order, Band)> s{};
    butterworth(Order, Band, w1, w2, std::span<biquad<T>>(s));
    return s;
}

/// \brief The cascade's response at `omega` radians per sample: the product of the sections'.
/// \tparam T A `math::math_number`.
/// \param sections The sections.
/// \param omega The frequency, `pi` at Nyquist.
/// \return The response.
template <math::math_number T>
constexpr complex<T> sos_response(std::span<const biquad<T>> sections, std::type_identity_t<T> omega) noexcept
{
    complex<T> h{T(1), T(0)};
    for (const auto& s : sections)
        h *= s.response(omega);
    return h;
}

/// \brief Filters `x` through a cascade of sections, `scipy.signal.sosfilt`, carrying the state.
/// \tparam T A `math::math_number`.
/// \param sections The sections.
/// \param state One per section; zero for a filter at rest, and left as the filter ends so a stream can continue.
/// \param x The input.
/// \param y `x.size()` outputs; may be `x` itself (pass the same buffer as both).
template <math::math_number T>
constexpr void sosfilt(std::span<const biquad<T>> sections, std::span<biquad_state<std::type_identity_t<T>>> state,
                       std::span<const std::type_identity_t<T>> x, std::span<std::type_identity_t<T>> y) noexcept
{
    assert(state.size() == sections.size() && y.size() == x.size());
    for (std::size_t n = 0; n < x.size(); ++n) {
        T v = x[n];
        for (std::size_t s = 0; s < sections.size(); ++s)
            v = state[s].process(sections[s], v);
        y[n] = v;
    }
}

} // namespace snicholls::signal

#endif // snicholls_core_signal_iir_hpp
