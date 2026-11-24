//
//  goertzel.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::signal` Goertzel detectors: one frequency of a block without a full FFT, its coefficient computed by the compiler when the frequency is known when the program is written; a streaming state of two values, the generalised (non-integer bin) form, windows, and a bank of K frequencies in one pass.
/// \ingroup core
///
/// **Overview.** The Goertzel algorithm computes one term of the discrete-time Fourier transform of a block of `n`
/// samples, `X(w) = sum_j x_j e^(-i w j)`, with the second-order recurrence
///
///     s_j = x_j + 2 cos(w) s_(j-1) - s_(j-2)
///
/// one real multiplication and two additions per sample, and a final step `X(w) = e^(-i w n) (e^(i w) s_(n-1) -
/// s_(n-2))` once per block. With `w = 2 pi k / n` for an integer `k` it is bin `k` of the DFT, and the phase factor
/// `e^(-i w n)` is 1; with any other `w` (a frequency between bins: the *generalised* Goertzel algorithm) it is the
/// transform at exactly that frequency, aligned to the block's first sample. The power `|X|^2` is
/// `(cos w s_(n-1) - s_(n-2))^2 + (sin w s_(n-1))^2`, no complex arithmetic: the same four multiplications as the
/// textbook `s_(n-1)^2 + s_(n-2)^2 - 2 cos w s_(n-1) s_(n-2)`, without that form's cancellation at DC and Nyquist.
///
/// Use it to test for a handful of frequencies: DTMF's eight tones, a pilot tone, a mains-hum check, a carrier.
/// Prefer `signal/fft.hpp` when most of the spectrum is wanted: `K` detectors cost `K n` real multiply-adds, a real
/// FFT about `(n / 2) log2(n / 2)` complex ones, so the detectors win while `K` is below about `log2 n`. That is the
/// count of operations; measured (Performance, below) one detector is bound by the latency of its recurrence, so a
/// bank of 8 costs only about 1.3 times one detector, and at 1,024 points it still takes 0.6 of the real FFT's time.
/// The detectors also stream: they need no buffer of the block, and the answer is ready the sample the block ends.
///
/// Three forms, one state:
///
/// - `static_goertzel<T, Frequency, SampleRate, N, Window>`: the frequency, sample rate and block length are template
///   parameters, so the bin, `cos w`, `sin w` and the coefficient `2 cos w` are `static constexpr` members computed by
///   the compiler with `snicholls::math` (no trigonometry at run time: a GPU or FPGA kernel reads a constant), and so
///   is the window's table when there is one. The object is the state alone.
/// - `goertzel<T>`: the same detector for a frequency chosen at run time (its constructor is `constexpr`, so a
///   `constexpr goertzel<float>` also has its coefficient computed by the compiler).
/// - `goertzel_bank<T, K>`: `K` frequencies over the same block, every state updated by each sample in one pass.
///
/// The constants are `goertzel_coefficients<T>` (`at_frequency`, `at_bin`, `at_fractional_bin`), and the recurrence
/// is `goertzel_state<T>`, for a caller who keeps its own layout (one state per channel, a structure of arrays on a
/// device).
///
/// \snippet signal/goertzel_demo.cpp compile_time
/// \snippet signal/goertzel_demo.cpp run_time
/// \snippet signal/goertzel_demo.cpp dtmf
///
/// **Template parameters.** `Frequency` and `SampleRate` are `double` non-type template parameters, which C++20
/// admits and GCC 13, Clang 18 and Apple clang 17 compile (Clang from 18; checked for this header), so
/// `static_goertzel<float, 1209.0, 8000.0, 205>` reads as it is written and a frequency that is not a whole number of
/// hertz (59.94, 1633.5) needs no scaled integer. A rational frequency in integers would have compiled on older
/// compilers too, but at the cost of a numerator and denominator per frequency, and the ratio is reduced in the
/// compute type anyway. `Window` is a window *value* of `signal/window.hpp` (`signal::hann{}`, `signal::kaiser{8.6}`),
/// a structural class type, so its parameters are part of the type as well.
///
/// **Scaling: the same as `fft.hpp`'s bin.** No normalisation is applied, so `bin()` is `fft(x)[k]` with
/// `fft_norm::backward` (the default) for an integer `k`, and `power()` is its squared magnitude: the two are compared
/// directly (the tests do, for every `k`). A tone `A cos(2 pi k j / n + phi)` on an integer bin gives `|X_k| = A n / 2`
/// and phase `phi`; with a window `w`, `A sum(w) / 2`. Divide by `n` (or `sum(w)`) for `fft_norm::forward`'s scale.
///
/// **Relationships.** `signal/fft.hpp` (the full transform it is checked against), `signal/window.hpp` (the windows),
/// `signal/complex.hpp` (the bin's type and the exact integer-bin twiddle), `math/constexpr_math.hpp` (the compile-time
/// cosine and sine). Group: \ref core.
///
/// **Implementations.** One recurrence. The coefficients of an integer bin come from the octant-reduced root of
/// unity that `fft.hpp`'s tables use, so `cos w` and `sin w` are the same constants the FFT multiplies by; a frequency
/// whose bin `k = f n / fs` is within `4 eps n` of an integer is snapped to that bin. A non-integer bin, reduced into
/// `[0, n)`, computes `cos` and `sin` of `2 pi k / n` and the block phase from `frac(k)`. Constants are computed in
/// `compute_t<T>` (`double` for `float`) and rounded once; the recurrence runs in `T` (a `float` detector is all
/// `float`, as a device kernel is), written `s_j = (x_j - s_(j-2)) + coeff s_(j-1)` so that the subtraction is off the
/// recurrence's critical path (one multiplication and one addition of latency per sample: 1.6 times the textbook
/// order's speed, measured). Constants built at compile time use `math`'s own sine and cosine, those built at run time
/// `<cmath>`'s; the two may differ in the last place, so a detector built at run time and one built by the compiler
/// for the same frequency agree within an ulp of their coefficients, not always to the bit.
///
/// **Guarantees.** Every function is `constexpr` and `noexcept`; nothing allocates and nothing recurses.
/// `goertzel_state`, `goertzel_coefficients`, `goertzel`, `static_goertzel` and `goertzel_bank` are trivially
/// copyable and standard layout; `static_goertzel` is the state alone (two `T`s and a count). A detector is not
/// synchronised: one per thread, or one per channel. `push` is O(1), `update` O(samples), a bank's O(samples K); the
/// final step is O(1) per frequency.
///
/// Numerical behaviour. The recurrence is a resonator with both poles on the unit circle, so rounding errors are not
/// damped: in the tests' worst case over random blocks, `|X - X*| <= eps n ||x||_2 / max(|sin w|, 1 / n)`, a bound
/// that grows with `n` where the FFT's grows with `log2 n`, and fastest near `w = 0` and `w = pi`, where `sin w` is
/// small (at DC the states grow as `n^2` and the final difference cancels); measured, the worst case is 0.8 of the
/// bound. In `float` over 4,096 samples a mid-band bin is good to about 1e-4 relative to `||x||`; a long block in single
/// precision near DC or Nyquist is where to use `double` (or shorter blocks, which detection usually wants anyway).
/// `power()` is within `2 d (|X| + d)` of `|X|^2`, `d` that bound, and is `norm(bin())` to the bit for an integer bin.
///
/// **Errors.** None thrown. Preconditions, asserted: a block length of at least 1 and a positive sample rate; a window
/// table at least as long as the samples pushed through it. A frequency outside `[0, fs)` aliases into it (a
/// negative one is `fs` minus it), as sampling does. Samples pushed past the block length extend the sum: `bin()` is
/// then the transform of everything pushed since `reset()` (its phase factor computed at run time).
///
/// **Performance.** `benchmarks/signal/goertzel_benchmark.cpp`: one detector and a bank of eight against the same
/// recurrence written out by hand (bit-identical, a fairness check), and against `fft.hpp`'s full real transform of
/// the block reading the same bins (agreeing within the bound above). Intel Xeon W-3245, AppleClang 17, Release,
/// 2026-11-24, on a shared machine under heavy load (load average 24 to 50 on 32 hardware threads; `BENCHMARKS.md`), so the
/// numbers are indicative: `double`, a block of 1,024: one bin 2.0 us (by hand 2.0 us), a bank of 8 bins 2.6 us (by
/// hand 2.7 us), the real FFT of the block 4.2 us; 4,096: 8.2 us, 10.6 us, the FFT 18.3 us; a block of 205 (DTMF at
/// 8 kHz): one bin 0.39 us, 8 bins 0.52 us, the FFT 9.9 us (205 is 5 x 41, a Bluestein length); `float`, 1,024: one
/// bin 2.0 us, 8 bins 2.3 us, the FFT 3.3 us. One detector runs at about 500 million samples a second, the latency of
/// a multiplication and an addition per sample; the bank's eight independent recurrences overlap.
///
/// **Tests and demos.** `tests/signal/goertzel_test.cpp` (every integer bin against `fft.hpp` in three types,
/// numpy's DTFT at non-integer bins and with windows from `tests/fixtures/signal/goertzel.txt`, tones, noise, DC,
/// Nyquist, the 16 DTMF digits, compile-time constants), `demos/signal/goertzel_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cassert>`, `<cstddef>`, `<span>`, `<type_traits>`,
/// `snicholls/core/math/constexpr_math.hpp`, `signal/complex.hpp`, `signal/window.hpp`.
///
/// **Known issues.** A `static_goertzel` with a window builds its table of `N` values at compile time, within the
/// compiler's constant-evaluation limit (Clang's default admits a few thousand points of a cosine window; a Kaiser
/// window costs more per point). MSVC accepts `double` template parameters from Visual Studio 2019 16.8.
///
/// **Status.** Experimental, new in 0.24.0.

#ifndef snicholls_core_signal_goertzel_hpp
#define snicholls_core_signal_goertzel_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/signal/complex.hpp>
#include <snicholls/core/signal/window.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <span>
#include <type_traits>

namespace snicholls::signal
{

/// \brief What a Goertzel detector knows before it sees a sample: the coefficient `2 cos w`, `cos w`, `sin w`, the block's phase factor and the bin.
/// \ingroup core
/// \tparam T A `math::math_number`.
template <math::math_number T>
struct goertzel_coefficients
{
    T coeff = T(2);                               ///< `2 cos w`, the recurrence's only multiplier.
    T cos_w = T(1);                               ///< `cos w`.
    T sin_w = T(0);                               ///< `sin w`.
    complex<T> block_phase{T(1), T(0)};           ///< `e^(-i w n)`: 1 for an integer bin.
    T bin = T(0);                                 ///< `k = w n / (2 pi)`, in `[0, n)`; not a whole number for the generalised form.
    std::size_t n = 1;                            ///< The block length.

    /// \brief The coefficients of the integer bin `k` of a block of `n`: `fft.hpp`'s twiddle constants, exact at the quarter turns.
    /// \param k The bin; reduced modulo `n`.
    /// \param n The block length, at least 1.
    /// \return The coefficients.
    static constexpr goertzel_coefficients at_bin(std::size_t k, std::size_t n) noexcept
    {
        assert(n >= 1);
        using C = compute_t<T>;
        k %= n;
        const complex<C> u = detail::complex_::unit<C>(k, n);   // (cos, sin) of 2 pi k / n
        return {static_cast<T>(C(2) * u.re), static_cast<T>(u.re), static_cast<T>(u.im), {T(1), T(0)},
                static_cast<T>(static_cast<long long>(k)), n};
    }

    /// \brief The coefficients of a fractional bin `k` (the DFT of `n` evaluated between its bins): the generalised form, snapped to the integer bin within a few ulps of one.
    /// \param k The bin; any real, reduced modulo `n` into `[0, n)`.
    /// \param n The block length, at least 1.
    /// \return The coefficients.
    static constexpr goertzel_coefficients at_fractional_bin(compute_t<T> k, std::size_t n) noexcept
    {
        assert(n >= 1);
        using C = compute_t<T>;
        const C nn = C(static_cast<long long>(n));
        k -= nn * math::floor(k / nn);   // in [0, n]
        const C whole = math::round(k);
        if (math::abs(k - whole) <= C(8) * nn / C(math::math_limits<C>::whole_from))   // 4 eps n: on a bin
            return at_bin(static_cast<std::size_t>(static_cast<long long>(whole)), n);
        const C two_pi = C(2) * math::math_limits<C>::pi;
        const C c = math::cos(two_pi * (k / nn)), s = math::sin(two_pi * (k / nn));
        const C frac = two_pi * (k - math::floor(k));   // e^(-i w n) = e^(-2 pi i frac(k))
        return {static_cast<T>(C(2) * c), static_cast<T>(c), static_cast<T>(s),
                {static_cast<T>(math::cos(frac)), static_cast<T>(-math::sin(frac))}, static_cast<T>(k), n};
    }

    /// \brief The coefficients of `cycles` periods per sample (`f / fs`) over a block of `n`: `at_fractional_bin(cycles n, n)`.
    /// \param cycles The frequency over the sample rate; any real, aliased into `[0, 1)`.
    /// \param n The block length, at least 1.
    /// \return The coefficients.
    static constexpr goertzel_coefficients at_cycles(compute_t<T> cycles, std::size_t n) noexcept
    {
        return at_fractional_bin(cycles * compute_t<T>(static_cast<long long>(n)), n);
    }

    /// \brief The coefficients of a frequency in hertz: `at_fractional_bin(frequency n / sample_rate, n)`.
    /// \param frequency The frequency; any real, aliased into `[0, sample_rate)`.
    /// \param sample_rate The sample rate, positive.
    /// \param n The block length, at least 1.
    /// \return The coefficients.
    static constexpr goertzel_coefficients at_frequency(double frequency, double sample_rate, std::size_t n) noexcept
    {
        assert(sample_rate > 0);
        using C = compute_t<T>;
        return at_fractional_bin(C(frequency) * C(static_cast<long long>(n)) / C(sample_rate), n);
    }

    /// \brief The phase factor `e^(-i w c)` that aligns a sum of `c` samples to its first sample: `block_phase` when `c` is the block length, otherwise computed (one sine and cosine).
    /// \param c The number of samples.
    /// \return The factor.
    [[nodiscard]] constexpr complex<T> phase_after(std::size_t c) const noexcept
    {
        if (c == n)
            return block_phase;
        using C = compute_t<T>;
        const C t = C(bin) * C(static_cast<long long>(c)) / C(static_cast<long long>(n));   // w c / (2 pi)
        const C a = C(2) * math::math_limits<C>::pi * (t - math::floor(t));
        return {static_cast<T>(math::cos(a)), static_cast<T>(-math::sin(a))};
    }
};

/// \brief The Goertzel recurrence's state: two values and a count, trivially copyable.
/// \ingroup core
/// \tparam T A `math::math_number`.
template <math::math_number T>
struct goertzel_state
{
    T s1 = T(0);              ///< `s_(c-1)`, the latest state.
    T s2 = T(0);              ///< `s_(c-2)`.
    std::size_t count = 0;    ///< The samples pushed since the last reset, `c`.

    /// \brief One step: `s_c = (x - s_(c-2)) + coeff s_(c-1)`.
    /// \param coeff `2 cos w`.
    /// \param x The sample.
    constexpr void push(T coeff, T x) noexcept
    {
        const T s0 = (x - s2) + coeff * s1;   // x - s2 is off the critical path: one multiply and one add per sample
        s2 = s1;
        s1 = s0;
        ++count;
    }

    /// \brief Clears the state for a new block.
    constexpr void reset() noexcept { *this = goertzel_state{}; }

    /// \brief `|X|^2`, as `(cos w s1 - s2)^2 + (sin w s1)^2`: the cost of the textbook `s1^2 + s2^2 - coeff s1 s2` (four multiplications) without its cancellation near DC and Nyquist.
    /// \param c The coefficients the state was pushed with.
    /// \return The power; `norm(bin(c))` to the bit for an integer bin over a full block.
    [[nodiscard]] constexpr T power(const goertzel_coefficients<T>& c) const noexcept
    {
        const T re = c.cos_w * s1 - s2, im = c.sin_w * s1;
        return re * re + im * im;
    }

    /// \brief The transform `X(w)` of the samples pushed, aligned to the first: `e^(-i w c) (e^(i w) s1 - s2)`, the generalised Goertzel algorithm's final step.
    /// \param c The coefficients the state was pushed with.
    /// \return `X(w)`; `fft(x)[k]` for an integer bin over a full block.
    [[nodiscard]] constexpr complex<T> bin(const goertzel_coefficients<T>& c) const noexcept
    {
        const complex<T> y{c.cos_w * s1 - s2, c.sin_w * s1};
        return c.phase_after(count) * y;   // exactly y for an integer bin over a full block
    }
};

namespace detail::goertzel
{
    // The windowed sample: x times the window's value at the state's position.
    template <typename T>
    constexpr T weighted(T x, std::span<const T> window, std::size_t at) noexcept
    {
        assert(at < window.size());
        return x * window[at];
    }
} // namespace detail::goertzel

/// \brief A Goertzel detector for a frequency chosen at run time: the coefficients and the state.
///
/// The constructor is `constexpr`, so `constexpr goertzel<float> d(697.0, 8000.0, 205);` has its coefficient computed
/// by the compiler; copy it to push samples.
/// \ingroup core
/// \tparam T A `math::math_number`.
template <math::math_number T>
class goertzel
{
public:
    /// \brief The element type.
    using value_type = T;

    /// \brief A detector of DC over a block of 1.
    constexpr goertzel() noexcept = default;
    /// \brief A detector over precomputed coefficients.
    /// \param c The coefficients.
    constexpr explicit goertzel(const goertzel_coefficients<T>& c) noexcept : c_(c) {}
    /// \brief A detector of `frequency` at `sample_rate` over blocks of `n`.
    /// \param frequency The frequency, in the units of `sample_rate`.
    /// \param sample_rate The sample rate, positive.
    /// \param n The block length, at least 1.
    constexpr goertzel(double frequency, double sample_rate, std::size_t n) noexcept
        : c_(goertzel_coefficients<T>::at_frequency(frequency, sample_rate, n)) {}

    /// \brief A detector of the integer bin `k` of a block of `n`.
    /// \param k The bin.
    /// \param n The block length.
    /// \return The detector.
    static constexpr goertzel at_bin(std::size_t k, std::size_t n) noexcept { return goertzel(goertzel_coefficients<T>::at_bin(k, n)); }

    /// \brief Pushes one sample.
    /// \param x The sample.
    constexpr void push(T x) noexcept { s_.push(c_.coeff, x); }
    /// \brief Pushes a run of samples.
    /// \param xs The samples.
    constexpr void update(std::span<const std::type_identity_t<T>> xs) noexcept
    {
        for (const T x : xs)
            s_.push(c_.coeff, x);
    }
    /// \brief Pushes a run of samples, each multiplied by the window's value at its position in the block.
    /// \param xs The samples.
    /// \param window The window's table (`make_window` or `fill_window`), at least `count() + xs.size()` long.
    constexpr void update(std::span<const std::type_identity_t<T>> xs, std::span<const std::type_identity_t<T>> window) noexcept
    {
        for (const T x : xs)
            s_.push(c_.coeff, detail::goertzel::weighted(x, window, s_.count));
    }
    /// \brief Clears the state for the next block; the coefficients stay.
    constexpr void reset() noexcept { s_.reset(); }

    /// \brief `|X|^2`, `fft.hpp`'s bin power for an integer bin.
    /// \return The power.
    [[nodiscard]] constexpr T power() const noexcept { return s_.power(c_); }
    /// \brief `|X|`.
    /// \return The magnitude.
    [[nodiscard]] constexpr T magnitude() const noexcept { return math::sqrt(power()); }
    /// \brief `X(w)`, aligned to the block's first sample.
    /// \return The bin.
    [[nodiscard]] constexpr complex<T> bin() const noexcept { return s_.bin(c_); }
    /// \brief `arg X(w)`, in (-pi, pi].
    /// \return The phase.
    [[nodiscard]] constexpr T phase() const noexcept { return arg(bin()); }
    /// \brief The samples pushed since the last reset.
    /// \return The count.
    [[nodiscard]] constexpr std::size_t count() const noexcept { return s_.count; }
    /// \brief Whether a full block has been pushed.
    /// \return `count() >= n`.
    [[nodiscard]] constexpr bool complete() const noexcept { return s_.count >= c_.n; }
    /// \brief The coefficients.
    /// \return They.
    [[nodiscard]] constexpr const goertzel_coefficients<T>& coefficients() const noexcept { return c_; }
    /// \brief The state.
    /// \return It.
    [[nodiscard]] constexpr const goertzel_state<T>& state() const noexcept { return s_; }

private:
    goertzel_coefficients<T> c_;
    goertzel_state<T> s_;
};

/// \brief A Goertzel detector whose frequency, sample rate, block length and window are template parameters: every constant computed by the compiler, the object the state alone.
/// \ingroup core
/// \tparam T A `math::math_number`.
/// \tparam Frequency The frequency, in the units of `SampleRate`.
/// \tparam SampleRate The sample rate, positive.
/// \tparam N The block length, at least 1.
/// \tparam Window A window value of `signal/window.hpp` (`rectangular{}`, the default, applies none).
template <math::math_number T, double Frequency, double SampleRate, std::size_t N, auto Window = rectangular{}>
class static_goertzel
{
    static_assert(N >= 1, "snicholls::signal::static_goertzel: the block length must be at least 1");
    static_assert(SampleRate > 0, "snicholls::signal::static_goertzel: the sample rate must be positive");
    static constexpr bool windowed = !std::is_same_v<std::remove_cv_t<decltype(Window)>, rectangular>;

public:
    /// \brief The element type.
    using value_type = T;
    /// \brief The frequency.
    static constexpr double frequency = Frequency;
    /// \brief The sample rate.
    static constexpr double sample_rate = SampleRate;
    /// \brief The block length.
    static constexpr std::size_t size = N;
    /// \brief Every constant, computed by the compiler.
    static constexpr goertzel_coefficients<T> coefficients = goertzel_coefficients<T>::at_frequency(Frequency, SampleRate, N);
    /// \brief `2 cos w`, a compile-time constant.
    static constexpr T coeff = coefficients.coeff;
    /// \brief The bin, `f N / fs`.
    static constexpr T bin_index = coefficients.bin;
    /// \brief The window's periodic table, computed by the compiler (empty for the rectangular window).
    static constexpr std::array<T, windowed ? N : 0> window = [] {
        if constexpr (windowed)
            return make_window<T, N>(Window);
        else
            return std::array<T, 0>{};
    }();

    /// \brief Pushes one sample (times the window's value at its position, when there is a window).
    /// \param x The sample; with a window, at most `N` per block.
    constexpr void push(T x) noexcept
    {
        if constexpr (windowed)
            s_.push(coeff, detail::goertzel::weighted(x, std::span<const T>(window), s_.count));
        else
            s_.push(coeff, x);
    }
    /// \brief Pushes a run of samples.
    /// \param xs The samples.
    constexpr void update(std::span<const std::type_identity_t<T>> xs) noexcept
    {
        for (const T x : xs)
            push(x);
    }
    /// \brief Clears the state for the next block.
    constexpr void reset() noexcept { s_.reset(); }

    /// \brief `|X|^2`.
    /// \return The power.
    [[nodiscard]] constexpr T power() const noexcept { return s_.power(coefficients); }
    /// \brief `|X|`.
    /// \return The magnitude.
    [[nodiscard]] constexpr T magnitude() const noexcept { return math::sqrt(power()); }
    /// \brief `X(w)`, aligned to the block's first sample.
    /// \return The bin.
    [[nodiscard]] constexpr complex<T> bin() const noexcept { return s_.bin(coefficients); }
    /// \brief `arg X(w)`, in (-pi, pi].
    /// \return The phase.
    [[nodiscard]] constexpr T phase() const noexcept { return arg(bin()); }
    /// \brief The samples pushed since the last reset.
    /// \return The count.
    [[nodiscard]] constexpr std::size_t count() const noexcept { return s_.count; }
    /// \brief Whether a full block has been pushed.
    /// \return `count() >= N`.
    [[nodiscard]] constexpr bool complete() const noexcept { return s_.count >= N; }
    /// \brief The state.
    /// \return It.
    [[nodiscard]] constexpr const goertzel_state<T>& state() const noexcept { return s_; }

private:
    goertzel_state<T> s_;
};

/// \brief `K` Goertzel detectors over the same block, every state updated by each sample in one pass.
///
/// The constructor is `constexpr`: `constexpr goertzel_bank<float, 8> dtmf(freqs, 8000.0, 205);` is a table the
/// compiler fills. The states are a structure of arrays, so the inner loop over the `K` frequencies vectorises.
/// \ingroup core
/// \tparam T A `math::math_number`.
/// \tparam K The number of frequencies.
template <math::math_number T, std::size_t K>
class goertzel_bank
{
    static_assert(K >= 1, "snicholls::signal::goertzel_bank: a bank needs at least one frequency");

public:
    /// \brief The element type.
    using value_type = T;

    /// \brief A bank of DC detectors over a block of 1.
    constexpr goertzel_bank() noexcept { set(std::array<goertzel_coefficients<T>, K>{}); }
    /// \brief A bank over precomputed coefficients, all of the same block length.
    /// \param c The coefficients.
    constexpr explicit goertzel_bank(const std::array<goertzel_coefficients<T>, K>& c) noexcept { set(c); }
    /// \brief A bank of `frequencies` at `sample_rate` over blocks of `n`.
    /// \param frequencies The frequencies.
    /// \param sample_rate The sample rate, positive.
    /// \param n The block length, at least 1.
    constexpr goertzel_bank(const std::array<double, K>& frequencies, double sample_rate, std::size_t n) noexcept
    {
        std::array<goertzel_coefficients<T>, K> c{};
        for (std::size_t i = 0; i < K; ++i)
            c[i] = goertzel_coefficients<T>::at_frequency(frequencies[i], sample_rate, n);
        set(c);
    }

    /// \brief The number of frequencies.
    /// \return `K`.
    [[nodiscard]] static constexpr std::size_t size() noexcept { return K; }

    /// \brief Pushes one sample into every detector.
    /// \param x The sample.
    constexpr void push(T x) noexcept
    {
        for (std::size_t i = 0; i < K; ++i) {
            const T s0 = (x - s2_[i]) + coeff_[i] * s1_[i];
            s2_[i] = s1_[i];
            s1_[i] = s0;
        }
        ++count_;
    }
    /// \brief Pushes a run of samples.
    /// \param xs The samples.
    constexpr void update(std::span<const std::type_identity_t<T>> xs) noexcept
    {
        for (const T x : xs)
            push(x);
    }
    /// \brief Pushes a run of samples, each multiplied by the window's value at its position in the block.
    /// \param xs The samples.
    /// \param window The window's table, at least `count() + xs.size()` long.
    constexpr void update(std::span<const std::type_identity_t<T>> xs, std::span<const std::type_identity_t<T>> window) noexcept
    {
        for (const T x : xs)
            push(detail::goertzel::weighted(x, window, count_));
    }
    /// \brief Clears every state for the next block.
    constexpr void reset() noexcept
    {
        s1_ = {};
        s2_ = {};
        count_ = 0;
    }

    /// \brief Detector `i`'s state.
    /// \param i The detector, below `K`.
    /// \return Its state.
    [[nodiscard]] constexpr goertzel_state<T> state(std::size_t i) const noexcept { return {s1_[i], s2_[i], count_}; }
    /// \brief `|X|^2` of detector `i`.
    /// \param i The detector.
    /// \return The power.
    [[nodiscard]] constexpr T power(std::size_t i) const noexcept { return state(i).power(c_[i]); }
    /// \brief `|X|` of detector `i`.
    /// \param i The detector.
    /// \return The magnitude.
    [[nodiscard]] constexpr T magnitude(std::size_t i) const noexcept { return math::sqrt(power(i)); }
    /// \brief `X(w)` of detector `i`.
    /// \param i The detector.
    /// \return The bin.
    [[nodiscard]] constexpr complex<T> bin(std::size_t i) const noexcept { return state(i).bin(c_[i]); }
    /// \brief `arg X(w)` of detector `i`.
    /// \param i The detector.
    /// \return The phase.
    [[nodiscard]] constexpr T phase(std::size_t i) const noexcept { return arg(bin(i)); }
    /// \brief Every detector's power.
    /// \return The powers, in the order of the frequencies.
    [[nodiscard]] constexpr std::array<T, K> powers() const noexcept
    {
        std::array<T, K> p{};
        for (std::size_t i = 0; i < K; ++i)
            p[i] = power(i);
        return p;
    }
    /// \brief The detector with the largest power among `[first, last)`: a DTMF row or column.
    /// \param first The first detector.
    /// \param last One past the last, at most `K`, above `first`.
    /// \return Its index; the first of equal powers.
    [[nodiscard]] constexpr std::size_t strongest(std::size_t first = 0, std::size_t last = K) const noexcept
    {
        assert(first < last && last <= K);
        std::size_t best = first;
        for (std::size_t i = first + 1; i < last; ++i)
            if (power(i) > power(best))
                best = i;
        return best;
    }
    /// \brief The samples pushed since the last reset.
    /// \return The count.
    [[nodiscard]] constexpr std::size_t count() const noexcept { return count_; }
    /// \brief Whether a full block has been pushed.
    /// \return `count() >= n`.
    [[nodiscard]] constexpr bool complete() const noexcept { return count_ >= c_[0].n; }
    /// \brief The coefficients of detector `i`.
    /// \param i The detector.
    /// \return They.
    [[nodiscard]] constexpr const goertzel_coefficients<T>& coefficients(std::size_t i) const noexcept { return c_[i]; }

private:
    constexpr void set(const std::array<goertzel_coefficients<T>, K>& c) noexcept
    {
        c_ = c;
        for (std::size_t i = 0; i < K; ++i) {
            assert(c[i].n == c[0].n);
            coeff_[i] = c[i].coeff;
        }
    }

    std::array<goertzel_coefficients<T>, K> c_{};
    std::array<T, K> coeff_{};
    std::array<T, K> s1_{};
    std::array<T, K> s2_{};
    std::size_t count_ = 0;
};

} // namespace snicholls::signal

#endif // snicholls_core_signal_goertzel_hpp
