//
//  fft.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::signal` fast Fourier transforms of any length, at compile time and at run time: complex and real, forward and inverse, in one dimension, over rows and over N axes, with their backward passes.
/// \ingroup core
///
/// **Overview.** The discrete Fourier transform `X_k = sum_j x_j e^(-2 pi i jk / n)` in `O(n log n)` for every `n`:
///
/// - **Mixed radix** when every prime factor of `n` is at most `fft_max_radix` (31): a Stockham autosort transform,
///   one pass per factor, radix 4 and 2 by hand for the powers of two and a generic butterfly for 3, 5, 7, ..., 31.
///   Stockham needs no bit reversal: each pass reads one buffer and writes the other in natural order.
/// - **Bluestein's chirp z-transform** for any other `n` (a large prime factor): the transform as a convolution of
///   length `m`, the power of two at least `2n - 1`, done with three transforms of length `m`.
/// - **In place, radix 2, with a bit-reversal table**, for a power of two known at compile time
///   (`static_fft_plan<T, N>::forward_in_place`): no second buffer, the form an FPGA pipeline takes.
///
/// A **plan** holds what does not depend on the data: the factorisation and the twiddle factors (and, for
/// Bluestein, the chirp and the spectrum of its kernel). `fft_plan<T>` is built at run time over memory the caller
/// gives it (`fft_plan<T>::storage_size(n)` elements) and never allocates; `static_fft_plan<T, N>` is built by a
/// `consteval` constructor into `std::array`s, so for a size known when the program is written every table is computed by
/// the compiler. Both are usable in constant evaluation, so a whole transform can be a `static_assert`. Executing a
/// plan takes a work buffer of `work_size()` elements from the caller; nothing allocates.
///
/// Real input: `rfft` gives the `n / 2 + 1` bins of the half spectrum (the rest is their conjugate), through one complex
/// transform of length `n / 2` when `n` is even; `irfft` is its inverse, and uses only the real parts of bin 0 and,
/// for even `n`, bin `n / 2`, as numpy does. `fft_rows` transforms a batch of contiguous rows; `fftn` any number of
/// axes of a row-major array (`fft2` for two).
///
/// **Normalisation**, as numpy's and PyTorch's `norm`: `fft_norm::backward` (the default: the forward transform is
/// unscaled, the inverse divides by `n`), `fft_norm::forward` (the forward divides by `n`), `fft_norm::ortho` (both
/// divide by `sqrt(n)`, so the transform is unitary), and `fft_norm::none` (neither is scaled). The inverse is
/// `x_j = s sum_k X_k e^(+2 pi i jk / n)` with that scale `s`.
///
/// **Gradients.** For a real loss of a complex output, with the gradient written `g = dL/d re + i dL/d im` (PyTorch's
/// convention), the backward pass of each transform is its adjoint: `fft_backward` is the inverse transform with the
/// forward's scale, `ifft_backward` the forward with the inverse's, `rfft_backward` and `irfft_backward` the
/// corresponding real transforms with the bins between 0 and `n / 2` halved or doubled for the half spectrum.
///
/// \snippet signal/fft_demo.cpp compile_time
/// \snippet signal/fft_demo.cpp run_time
///
/// **Relationships.** `signal/complex.hpp` (the element type and `twiddle`); `signal/stft.hpp`, `signal/dct.hpp`,
/// `signal/convolve.hpp` and `signal/hilbert.hpp` run on these plans. Group: \ref core.
///
/// **Implementations.** Mixed-radix Stockham (radix 4, 2 and a generic odd prime up to 31), Bluestein for the rest,
/// and in-place radix-2 decimation in time with a bit-reversal table for `static_fft_plan` of a power of two. The
/// generic butterfly costs `p` complex multiplications per output, so a factor of 31 is about eight times a radix-4
/// pass; Bluestein costs about six transforms of length `m <= 4n`, so a prime `n` runs roughly ten times slower than a
/// power of two near it. Pad or choose a smooth length where the length is yours to choose.
///
/// **Guarantees.** `constexpr` throughout; no allocation, no recursion, every loop bounded by the length. Plans are
/// immutable after construction and may be shared by any number of threads; a transform touches only its data and
/// its work buffer. `fft_plan` is trivially copyable (a layout and a span). Tables built at compile time and at run
/// time agree to a few units in the last place (`math`'s compile-time sine and cosine against `<cmath>`'s), and each
/// is deterministic: the same plan gives the same result to the bit on every call.
///
/// Accuracy, against the direct transform in `long double` (`tests/signal/fft_test.cpp`: every length from 1 to 300,
/// the powers of two to 2^16 and primes to 10007 bin by bin, `float` and `double`):
/// `max_k |X_k - X*_k| <= 2 eps ceil(log2 L) ||x||_2`, `eps` the element type's epsilon and `L` the length the passes
/// run at (`n`, or Bluestein's `m`); measured, the worst case is a third of that. Round trips (`ifft(fft(x))`) return
/// `x` within the same bound, and Parseval holds to it.
///
/// **Errors.** Preconditions, asserted: the data's length is the plan's; the work buffer is at least `work_size()`;
/// the storage given to a plan is at least `storage_size(n)`. A plan of length 0 transforms nothing.
///
/// **Performance.** `benchmarks/signal/fft_benchmark.cpp`: the transform against the same Stockham passes written out
/// by hand (bit-identical, a fairness check) and against the direct `O(n^2)` DFT. Intel Xeon W-3245, AppleClang 17,
/// Release, 2026-11-20, on a shared machine under load (`BENCHMARKS.md`): `double`, 1,024 points 8.0 us (by hand 8.2
/// us; the direct DFT 1.3 ms), 4,096 points 34 us, 65,536 points 0.94 ms; 1,000 points (radices 4 2 5 5 5) 38 us;
/// 1,009 (Bluestein) 44 us; `rfft` of 1,024 4.2 us; the in-place radix-2 pass of `static_fft_plan<double, 1024>` 8.1
/// us.
///
/// **Tests and demos.** `tests/signal/fft_test.cpp`, `demos/signal/fft_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cassert>`, `<cstddef>`, `<cstdint>`, `<span>`, `<type_traits>`,
/// `snicholls/core/math/constexpr_math.hpp`, `signal/complex.hpp`.
///
/// **Known issues.** The generic butterfly (radix 3, 5, 7, ...) is not written out, so a length of 1,000 costs about
/// four times a power of two per point. Constant evaluation has a step limit: Clang's default (`-fconstexpr-steps`, about a million)
/// allows a `static_fft_plan` of a few thousand points and a transform of about a thousand; GCC's is larger. The
/// generic butterfly keeps its `p` inputs on the stack (at most 31 complex numbers).
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_signal_fft_hpp
#define snicholls_core_signal_fft_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/signal/complex.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace snicholls::signal
{

/// \brief How a transform pair is scaled, as numpy's and PyTorch's `norm`.
/// \ingroup core
enum class fft_norm
{
    backward,   ///< The forward transform is unscaled; the inverse divides by `n` (the default everywhere).
    forward,    ///< The forward transform divides by `n`; the inverse is unscaled.
    ortho,      ///< Both divide by `sqrt(n)`: the transform is unitary.
    none,       ///< Neither is scaled.
};

/// \brief The largest prime factor the mixed-radix transform handles itself; a length with a larger one uses Bluestein's algorithm.
/// \ingroup core
inline constexpr std::size_t fft_max_radix = 31;

/// \brief The scale a transform of length `n` applies for a normalisation.
/// \tparam T A `math::math_number`.
/// \param norm The normalisation.
/// \param n The length.
/// \param inverse Whether the transform is the inverse.
/// \return 1, `1 / n` or `1 / sqrt(n)`.
template <math::math_number T>
constexpr T fft_scale(fft_norm norm, std::size_t n, bool inverse) noexcept
{
    if (n == 0)
        return T(1);
    const T nn = T(static_cast<long long>(n));
    switch (norm) {
    case fft_norm::backward: return inverse ? T(1) / nn : T(1);
    case fft_norm::forward: return inverse ? T(1) : T(1) / nn;
    case fft_norm::ortho: return T(1) / math::sqrt(nn);
    case fft_norm::none: break;
    }
    return T(1);
}

/// \brief Whether `n` is a power of two (1 is; 0 is not).
/// \param n The number.
/// \return The result.
constexpr bool is_power_of_two(std::size_t n) noexcept { return n != 0 && (n & (n - 1)) == 0; }

/// \brief The least power of two at least `n` (1 for 0).
/// \param n The number.
/// \return The power of two.
constexpr std::size_t next_power_of_two(std::size_t n) noexcept
{
    std::size_t p = 1;
    while (p < n)
        p *= 2;
    return p;
}

/// \brief What a transform's plan knows before it sees data: its length, its factors, and whether it uses Bluestein's algorithm.
/// \ingroup core
struct fft_layout
{
    std::size_t n = 0;                          ///< The transform's length.
    std::size_t m = 0;                          ///< Bluestein's convolution length, a power of two at least `2n - 1`; 0 for a mixed-radix length.
    std::array<std::uint8_t, 64> factors{};     ///< The passes' radices, in order: of `n`, or of `m` for Bluestein.
    std::size_t factor_count = 0;               ///< How many of `factors` are used.

    /// \brief The layout of a transform of length `n`: 4s first, then a 2, then the odd primes in increasing order.
    /// \param n The length.
    /// \return The layout.
    static constexpr fft_layout of(std::size_t n) noexcept
    {
        fft_layout l;
        l.n = n;
        std::size_t r = n;
        const auto add = [&l](std::size_t p) { l.factors[l.factor_count++] = static_cast<std::uint8_t>(p); };
        while (r % 4 == 0 && r > 1) { add(4); r /= 4; }
        if (r % 2 == 0 && r > 1) { add(2); r /= 2; }
        for (std::size_t p = 3; p <= fft_max_radix && r > 1; p += 2)
            while (r % p == 0) { add(p); r /= p; }
        if (r > 1) {
            l = fft_layout{};
            l.n = n;
            l.m = next_power_of_two(2 * n - 1);
            for (std::size_t q = l.m; q > 1;) {
                const std::size_t p = q % 4 == 0 ? 4 : 2;
                add(p);
                q /= p;
            }
        }
        return l;
    }

    /// \brief Whether the transform uses Bluestein's algorithm.
    /// \return `m != 0`.
    [[nodiscard]] constexpr bool bluestein() const noexcept { return m != 0; }
    /// \brief The plan's table size, in complex elements: `n`, or `n + 2m` for Bluestein.
    /// \return The size.
    [[nodiscard]] constexpr std::size_t storage_size() const noexcept { return bluestein() ? n + 2 * m : n; }
    /// \brief The work buffer a transform needs, in complex elements: `n`, or `2m` for Bluestein.
    /// \return The size.
    [[nodiscard]] constexpr std::size_t work_size() const noexcept { return bluestein() ? 2 * m : n; }
    /// \brief The passes' radices.
    /// \return A span over `factors`.
    [[nodiscard]] constexpr std::span<const std::uint8_t> passes() const noexcept { return {factors.data(), factor_count}; }
};

namespace detail::fft
{
    template <typename T>
    using cx = complex<T>;

    // W_n^k for k < n, from the first octant: bit-identical to twiddle<T>(k, n), with an eighth of the trigonometry.
    template <typename T>
    constexpr void fill_twiddles(cx<T>* tw, std::size_t n) noexcept
    {
        if (n % 4 != 0) {
            for (std::size_t k = 0; k < n; ++k)
                tw[k] = twiddle<T>(k, n);
            return;
        }
        const std::size_t q = n / 4;
        for (std::size_t k = 0; k < q; ++k)
            tw[k] = 8 * k <= n ? twiddle<T>(k, n) : cx<T>{-tw[q - k].im, -tw[q - k].re};
        for (std::size_t k = q; k < n; ++k)
            tw[k] = cx<T>{tw[k - q].im, -tw[k - q].re};   // times -i
    }

    template <bool Inverse, typename T>
    constexpr cx<T> tw_at(const cx<T>* tw, std::size_t k) noexcept
    {
        return Inverse ? cx<T>{tw[k].re, -tw[k].im} : tw[k];
    }

    template <bool Inverse, typename T>
    constexpr void radix2(const cx<T>* src, cx<T>* dst, const cx<T>* tw, std::size_t m, std::size_t s) noexcept
    {
        for (std::size_t j = 0; j < m; ++j) {
            const cx<T> w = tw_at<Inverse>(tw, s * j);
            const cx<T>* a = src + s * j;
            const cx<T>* b = src + s * (j + m);
            cx<T>* y0 = dst + s * (2 * j);
            cx<T>* y1 = dst + s * (2 * j + 1);
            for (std::size_t q = 0; q < s; ++q) {
                const cx<T> u = a[q], v = b[q];
                y0[q] = u + v;
                y1[q] = (u - v) * w;
            }
        }
    }

    template <bool Inverse, typename T>
    constexpr void radix4(const cx<T>* src, cx<T>* dst, const cx<T>* tw, std::size_t m, std::size_t s) noexcept
    {
        for (std::size_t j = 0; j < m; ++j) {
            const cx<T> w1 = tw_at<Inverse>(tw, s * j), w2 = tw_at<Inverse>(tw, 2 * s * j), w3 = tw_at<Inverse>(tw, 3 * s * j);
            const cx<T>* a0 = src + s * j;
            const cx<T>* a1 = src + s * (j + m);
            const cx<T>* a2 = src + s * (j + 2 * m);
            const cx<T>* a3 = src + s * (j + 3 * m);
            cx<T>* y = dst + s * (4 * j);
            for (std::size_t q = 0; q < s; ++q) {
                const cx<T> t0 = a0[q] + a2[q], t1 = a0[q] - a2[q], t2 = a1[q] + a3[q], d = a1[q] - a3[q];
                const cx<T> t3 = Inverse ? cx<T>{-d.im, d.re} : cx<T>{d.im, -d.re};   // times +i or -i
                y[q] = t0 + t2;
                y[q + s] = (t1 + t3) * w1;
                y[q + 2 * s] = (t0 - t2) * w2;
                y[q + 3 * s] = (t1 - t3) * w3;
            }
        }
    }

    template <bool Inverse, typename T>
    constexpr void radix_p(const cx<T>* src, cx<T>* dst, const cx<T>* tw, std::size_t n, std::size_t p, std::size_t m,
                           std::size_t s) noexcept
    {
        const std::size_t root = n / p;   // omega_p^e = W_n^(e n / p)
        for (std::size_t j = 0; j < m; ++j)
            for (std::size_t q = 0; q < s; ++q) {
                std::array<cx<T>, fft_max_radix> a{};
                for (std::size_t r = 0; r < p; ++r)
                    a[r] = src[q + s * (j + r * m)];
                for (std::size_t t = 0; t < p; ++t) {
                    cx<T> acc = a[0];
                    std::size_t e = 0;
                    for (std::size_t r = 1; r < p; ++r) {
                        e += t;
                        if (e >= p)
                            e -= p;
                        acc += a[r] * tw_at<Inverse>(tw, e * root);
                    }
                    dst[q + s * (p * j + t)] = acc * tw_at<Inverse>(tw, s * j * t);
                }
            }
    }

    // The Stockham autosort transform: one pass per factor, from x to y and back; the result ends in x.
    template <bool Inverse, typename T>
    constexpr void stockham(const cx<T>* tw, std::size_t n, std::span<const std::uint8_t> factors, cx<T>* x, cx<T>* y) noexcept
    {
        std::size_t len = n, s = 1;
        cx<T>* src = x;
        cx<T>* dst = y;
        for (const std::uint8_t f : factors) {
            const std::size_t p = f, m = len / p;
            switch (p) {
            case 2: radix2<Inverse>(src, dst, tw, m, s); break;
            case 4: radix4<Inverse>(src, dst, tw, m, s); break;
            default: radix_p<Inverse>(src, dst, tw, n, p, m, s); break;
            }
            cx<T>* t = src;
            src = dst;
            dst = t;
            len = m;
            s *= p;
        }
        if (src != x)
            for (std::size_t i = 0; i < n; ++i)
                x[i] = src[i];
    }

    constexpr std::size_t reverse_bits(std::size_t i, std::size_t bits) noexcept
    {
        std::size_t r = 0;
        for (std::size_t b = 0; b < bits; ++b) {
            r = (r << 1) | (i & 1);
            i >>= 1;
        }
        return r;
    }

    constexpr std::size_t log2_of(std::size_t n) noexcept
    {
        std::size_t b = 0;
        while ((std::size_t{1} << b) < n)
            ++b;
        return b;
    }

    // In-place radix-2 decimation in time over a power of two, with a bit-reversal permutation given or computed.
    template <bool Inverse, typename T, typename Reverse>
    constexpr void in_place_pow2(cx<T>* x, const cx<T>* tw, std::size_t n, Reverse reverse) noexcept
    {
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t r = reverse(i);
            if (i < r) {
                const cx<T> t = x[i];
                x[i] = x[r];
                x[r] = t;
            }
        }
        for (std::size_t len = 2; len <= n; len *= 2) {
            const std::size_t half = len / 2, step = n / len;
            for (std::size_t start = 0; start < n; start += len)
                for (std::size_t k = 0; k < half; ++k) {
                    const cx<T> u = x[start + k], v = x[start + k + half] * tw_at<Inverse>(tw, k * step);
                    x[start + k] = u + v;
                    x[start + k + half] = u - v;
                }
        }
    }

    template <typename T>
    constexpr void conj_all(cx<T>* x, std::size_t n) noexcept
    {
        for (std::size_t i = 0; i < n; ++i)
            x[i].im = -x[i].im;
    }

    template <typename T>
    constexpr void scale_all(cx<T>* x, std::size_t n, T s) noexcept
    {
        if (s != T(1))
            for (std::size_t i = 0; i < n; ++i)
                x[i] *= s;
    }

    // Fills a plan's tables: W_n^k, or for Bluestein the chirp, W_m^k and the kernel's spectrum.
    template <typename T>
    constexpr void build(const fft_layout& l, cx<T>* st) noexcept
    {
        if (!l.bluestein()) {
            fill_twiddles(st, l.n);
            return;
        }
        const std::size_t n = l.n, m = l.m, two_n = 2 * n;
        cx<T>* chirp = st;
        cx<T>* tw = st + n;
        cx<T>* kernel = st + n + m;
        std::size_t sq = 0;   // i^2 mod 2n, so e^(-i pi i^2 / n) is a twiddle of order 2n
        for (std::size_t i = 0; i < n; ++i) {
            chirp[i] = twiddle<T>(sq, two_n);
            sq = (sq + 2 * i + 1) % two_n;
        }
        fill_twiddles(tw, m);
        for (std::size_t i = 0; i < m; ++i)
            kernel[i] = {};
        kernel[0] = conj(chirp[0]);
        for (std::size_t i = 1; i < n; ++i)
            kernel[i] = kernel[m - i] = conj(chirp[i]);
        const std::size_t bits = log2_of(m);
        in_place_pow2<false>(kernel, tw, m, [bits](std::size_t i) { return reverse_bits(i, bits); });
    }

    // An unnormalised transform of x (length l.n) in place.
    template <bool Inverse, typename T>
    constexpr void execute(const fft_layout& l, const cx<T>* st, cx<T>* x, cx<T>* work) noexcept
    {
        if (!l.bluestein()) {
            stockham<Inverse>(st, l.n, l.passes(), x, work);
            return;
        }
        const std::size_t n = l.n, m = l.m;
        const cx<T>* chirp = st;
        const cx<T>* tw = st + n;
        const cx<T>* kernel = st + n + m;
        cx<T>* a = work;
        cx<T>* b = work + m;
        if (Inverse)
            conj_all(x, n);
        for (std::size_t i = 0; i < n; ++i)
            a[i] = x[i] * chirp[i];
        for (std::size_t i = n; i < m; ++i)
            a[i] = {};
        stockham<false>(tw, m, l.passes(), a, b);
        for (std::size_t i = 0; i < m; ++i)
            a[i] *= kernel[i];
        stockham<true>(tw, m, l.passes(), a, b);
        const T inv_m = T(1) / T(static_cast<long long>(m));
        for (std::size_t k = 0; k < n; ++k)
            x[k] = chirp[k] * a[k] * inv_m;
        if (Inverse)
            conj_all(x, n);
    }
} // namespace detail::fft

/// \brief A plan for complex transforms of one length, over tables in memory the caller owns.
///
/// Construct it over `storage_size(n)` elements (a `std::array` in constant evaluation, any span at run time); the
/// constructor fills them. The plan refers to the storage, which must outlive it.
/// \ingroup core
/// \tparam T A `math::math_number`.
template <math::math_number T>
class fft_plan
{
public:
    /// \brief The element type.
    using value_type = T;

    /// \brief The table size a plan of length `n` needs, in complex elements.
    /// \param n The length.
    /// \return `n`, or `n + 2m` when Bluestein's algorithm is used.
    static constexpr std::size_t storage_size(std::size_t n) noexcept { return fft_layout::of(n).storage_size(); }
    /// \brief The work buffer a transform of length `n` needs, in complex elements.
    /// \param n The length.
    /// \return `n`, or `2m` when Bluestein's algorithm is used.
    static constexpr std::size_t work_size(std::size_t n) noexcept { return fft_layout::of(n).work_size(); }

    /// \brief Builds the plan's tables in `storage`.
    /// \param n The transform's length.
    /// \param storage At least `storage_size(n)` elements; overwritten, and referred to by the plan.
    constexpr fft_plan(std::size_t n, std::span<complex<T>> storage) noexcept
        : layout_(fft_layout::of(n)), tables_(storage.data(), layout_.storage_size())
    {
        assert(storage.size() >= layout_.storage_size());
        detail::fft::build(layout_, storage.data());
    }

    /// \brief A plan over tables already built for `layout` (a `static_fft_plan`'s, or a copy of another plan's).
    /// \param layout The layout.
    /// \param tables `layout.storage_size()` elements as `fft_plan`'s constructor fills them.
    constexpr fft_plan(const fft_layout& layout, std::span<const complex<T>> tables) noexcept : layout_(layout), tables_(tables)
    {
        assert(tables.size() >= layout.storage_size());
    }

    /// \brief The transform's length.
    /// \return `n`.
    [[nodiscard]] constexpr std::size_t size() const noexcept { return layout_.n; }
    /// \brief The work buffer a transform needs, in complex elements.
    /// \return The size.
    [[nodiscard]] constexpr std::size_t work_size() const noexcept { return layout_.work_size(); }
    /// \brief Whether the plan uses Bluestein's algorithm.
    /// \return The result.
    [[nodiscard]] constexpr bool uses_bluestein() const noexcept { return layout_.bluestein(); }
    /// \brief The layout.
    /// \return The layout.
    [[nodiscard]] constexpr const fft_layout& layout() const noexcept { return layout_; }
    /// \brief The tables.
    /// \return The span the plan reads.
    [[nodiscard]] constexpr std::span<const complex<T>> tables() const noexcept { return tables_; }

    /// \brief An unnormalised transform in place: the building block of every public transform.
    /// \tparam Inverse Whether the exponent's sign is positive.
    /// \param data `size()` elements.
    /// \param work At least `work_size()` elements.
    template <bool Inverse>
    constexpr void execute(std::span<complex<T>> data, std::span<complex<T>> work) const noexcept
    {
        assert(data.size() == layout_.n && work.size() >= layout_.work_size());
        if (layout_.n > 1)
            detail::fft::execute<Inverse>(layout_, tables_.data(), data.data(), work.data());
    }

private:
    fft_layout layout_;
    std::span<const complex<T>> tables_;
};

/// \brief The forward transform in place: `X_k = s sum_j x_j e^(-2 pi i jk / n)`.
/// \tparam T A `math::math_number`.
/// \param plan The plan of the data's length.
/// \param data The sequence, replaced by its transform.
/// \param work At least `plan.work_size()` elements.
/// \param norm The normalisation.
template <math::math_number T>
constexpr void fft(const fft_plan<T>& plan, std::span<complex<std::type_identity_t<T>>> data,
                   std::span<complex<std::type_identity_t<T>>> work, fft_norm norm = fft_norm::backward) noexcept
{
    plan.template execute<false>(data, work);
    detail::fft::scale_all(data.data(), data.size(), fft_scale<T>(norm, plan.size(), false));
}

/// \brief The inverse transform in place: `x_j = s sum_k X_k e^(+2 pi i jk / n)`.
/// \tparam T A `math::math_number`.
/// \param plan The plan of the data's length.
/// \param data The spectrum, replaced by the sequence.
/// \param work At least `plan.work_size()` elements.
/// \param norm The normalisation.
template <math::math_number T>
constexpr void ifft(const fft_plan<T>& plan, std::span<complex<std::type_identity_t<T>>> data,
                    std::span<complex<std::type_identity_t<T>>> work, fft_norm norm = fft_norm::backward) noexcept
{
    plan.template execute<true>(data, work);
    detail::fft::scale_all(data.data(), data.size(), fft_scale<T>(norm, plan.size(), true));
}

/// \brief The backward pass of `fft`: given `dL/dX`, `dL/dx` (the adjoint, the inverse transform with the forward's scale), in place.
/// \tparam T A `math::math_number`.
/// \param plan The forward transform's plan.
/// \param grad `dL/dX`, replaced by `dL/dx`.
/// \param work At least `plan.work_size()` elements.
/// \param norm The forward transform's normalisation.
template <math::math_number T>
constexpr void fft_backward(const fft_plan<T>& plan, std::span<complex<std::type_identity_t<T>>> grad,
                            std::span<complex<std::type_identity_t<T>>> work, fft_norm norm = fft_norm::backward) noexcept
{
    plan.template execute<true>(grad, work);
    detail::fft::scale_all(grad.data(), grad.size(), fft_scale<T>(norm, plan.size(), false));
}

/// \brief The backward pass of `ifft`: given `dL/dx`, `dL/dX` (the forward transform with the inverse's scale), in place.
/// \tparam T A `math::math_number`.
/// \param plan The inverse transform's plan.
/// \param grad `dL/dx`, replaced by `dL/dX`.
/// \param work At least `plan.work_size()` elements.
/// \param norm The inverse transform's normalisation.
template <math::math_number T>
constexpr void ifft_backward(const fft_plan<T>& plan, std::span<complex<std::type_identity_t<T>>> grad,
                             std::span<complex<std::type_identity_t<T>>> work, fft_norm norm = fft_norm::backward) noexcept
{
    plan.template execute<false>(grad, work);
    detail::fft::scale_all(grad.data(), grad.size(), fft_scale<T>(norm, plan.size(), true));
}

/// \brief Fills `out` with the twiddle factors `W_n^k = e^(-2 pi i k / n)`, `n = out.size()`: the table `dft` reads.
/// \tparam T A `math::math_number`.
/// \param out The table.
template <math::math_number T>
constexpr void twiddles(std::span<complex<T>> out) noexcept
{
    detail::fft::fill_twiddles(out.data(), out.size());
}

/// \brief The direct `O(n^2)` transform, the reference the fast ones are tested against, with a twiddle table.
/// \tparam T A `math::math_number`.
/// \param in The input, `n` elements.
/// \param out The output, `n` elements; must not overlap `in`.
/// \param table `twiddles` of length `n`.
/// \param inverse Whether the exponent's sign is positive.
/// \param norm The normalisation.
template <math::math_number T>
constexpr void dft(std::span<const complex<std::type_identity_t<T>>> in, std::span<complex<T>> out,
                   std::span<const complex<std::type_identity_t<T>>> table, bool inverse = false,
                   fft_norm norm = fft_norm::backward) noexcept
{
    const std::size_t n = in.size();
    assert(out.size() == n && table.size() == n);
    const T s = fft_scale<T>(norm, n, inverse);
    for (std::size_t k = 0; k < n; ++k) {
        complex<T> acc{};
        std::size_t e = 0;   // j k mod n
        for (std::size_t j = 0; j < n; ++j) {
            const complex<T> w = inverse ? conj(table[e]) : table[e];
            acc += in[j] * w;
            e += k;
            if (e >= n)
                e -= n;
        }
        out[k] = acc * s;
    }
}

/// \brief The direct `O(n^2)` transform, each twiddle computed as it is used (no table).
/// \tparam T A `math::math_number`.
/// \param in The input, `n` elements.
/// \param out The output, `n` elements; must not overlap `in`.
/// \param inverse Whether the exponent's sign is positive.
/// \param norm The normalisation.
template <math::math_number T>
constexpr void dft(std::span<const complex<std::type_identity_t<T>>> in, std::span<complex<T>> out, bool inverse = false,
                   fft_norm norm = fft_norm::backward) noexcept
{
    const std::size_t n = in.size();
    assert(out.size() == n);
    const T s = fft_scale<T>(norm, n, inverse);
    for (std::size_t k = 0; k < n; ++k) {
        complex<T> acc{};
        std::size_t e = 0;
        for (std::size_t j = 0; j < n; ++j) {
            const complex<T> w = twiddle<T>(e, n);
            acc += in[j] * (inverse ? conj(w) : w);
            e += k;
            if (e >= n)
                e -= n;
        }
        out[k] = acc * s;
    }
}

/// \brief The bit-reversal permutation of `N` (a power of two): entry `i` is `i` with its `log2 N` bits reversed.
/// \tparam N A power of two.
/// \return The table.
template <std::size_t N>
    requires(is_power_of_two(N))
consteval std::array<std::uint32_t, N> bit_reversal_permutation()
{
    std::array<std::uint32_t, N> r{};
    const std::size_t bits = detail::fft::log2_of(N);
    for (std::size_t i = 0; i < N; ++i)
        r[i] = static_cast<std::uint32_t>(detail::fft::reverse_bits(i, bits));
    return r;
}

/// \brief A plan for a length known at compile time, every table computed by the compiler.
///
/// The tables are `std::array`s filled by a `consteval` constructor, so `inline constexpr static_fft_plan<float, 512>
/// plan{};` costs nothing at run time. `plan()` is an `fft_plan` over them, for the free functions. For a power of two
/// the plan also holds the bit-reversal permutation and transforms in place with no work buffer.
/// \ingroup core
/// \tparam T A `math::math_number`.
/// \tparam N The length.
template <math::math_number T, std::size_t N>
class static_fft_plan
{
public:
    /// \brief The layout.
    static constexpr fft_layout layout = fft_layout::of(N);
    /// \brief The work buffer a transform needs, in complex elements.
    static constexpr std::size_t work_size = layout.work_size();

    /// \brief Builds every table, at compile time.
    consteval static_fft_plan()
    {
        detail::fft::build(layout, tables_.data());
        for (std::size_t i = 0; i < bit_reversal_.size(); ++i)   // empty unless N is a power of two
            bit_reversal_[i] = static_cast<std::uint32_t>(detail::fft::reverse_bits(i, detail::fft::log2_of(N)));
    }

    /// \brief An `fft_plan` over this plan's tables, for the free functions.
    /// \return The plan; it refers to this object.
    [[nodiscard]] constexpr fft_plan<T> plan() const noexcept { return fft_plan<T>(layout, tables_); }

    /// \brief The tables.
    /// \return The array.
    [[nodiscard]] constexpr const std::array<complex<T>, layout.storage_size()>& tables() const noexcept { return tables_; }

    /// \brief The forward transform in place, with a work buffer of `work_size` elements.
    /// \param data The sequence, replaced by its transform.
    /// \param work The work buffer.
    /// \param norm The normalisation.
    constexpr void forward(std::span<complex<T>, N> data, std::span<complex<T>, work_size> work,
                           fft_norm norm = fft_norm::backward) const noexcept
    {
        fft(plan(), std::span<complex<T>>(data), std::span<complex<T>>(work), norm);
    }

    /// \brief The inverse transform in place, with a work buffer of `work_size` elements.
    /// \param data The spectrum, replaced by the sequence.
    /// \param work The work buffer.
    /// \param norm The normalisation.
    constexpr void inverse(std::span<complex<T>, N> data, std::span<complex<T>, work_size> work,
                           fft_norm norm = fft_norm::backward) const noexcept
    {
        ifft(plan(), std::span<complex<T>>(data), std::span<complex<T>>(work), norm);
    }

    /// \brief The forward transform of an array, its work buffer on the stack: for small `N` and for constant evaluation.
    /// \param x The sequence.
    /// \param norm The normalisation.
    /// \return Its transform.
    [[nodiscard]] constexpr std::array<complex<T>, N> forward(std::array<complex<T>, N> x, fft_norm norm = fft_norm::backward) const noexcept
    {
        std::array<complex<T>, work_size> work{};
        forward(x, work, norm);
        return x;
    }

    /// \brief The inverse transform of an array, its work buffer on the stack.
    /// \param x The spectrum.
    /// \param norm The normalisation.
    /// \return The sequence.
    [[nodiscard]] constexpr std::array<complex<T>, N> inverse(std::array<complex<T>, N> x, fft_norm norm = fft_norm::backward) const noexcept
    {
        std::array<complex<T>, work_size> work{};
        inverse(x, work, norm);
        return x;
    }

    /// \brief The bit-reversal permutation (a power of two only).
    /// \return The table.
    [[nodiscard]] constexpr const std::array<std::uint32_t, N>& bit_reversal() const noexcept
        requires(is_power_of_two(N))
    {
        return bit_reversal_;
    }

    /// \brief The forward transform in place with no work buffer: radix-2 decimation in time over the bit-reversal table (a power of two only).
    /// \param data The sequence, replaced by its transform.
    /// \param norm The normalisation.
    constexpr void forward_in_place(std::span<complex<T>, N> data, fft_norm norm = fft_norm::backward) const noexcept
        requires(is_power_of_two(N))
    {
        detail::fft::in_place_pow2<false>(data.data(), tables_.data(), N, [this](std::size_t i) { return std::size_t{bit_reversal_[i]}; });
        detail::fft::scale_all(data.data(), N, fft_scale<T>(norm, N, false));
    }

    /// \brief The inverse transform in place with no work buffer (a power of two only).
    /// \param data The spectrum, replaced by the sequence.
    /// \param norm The normalisation.
    constexpr void inverse_in_place(std::span<complex<T>, N> data, fft_norm norm = fft_norm::backward) const noexcept
        requires(is_power_of_two(N))
    {
        detail::fft::in_place_pow2<true>(data.data(), tables_.data(), N, [this](std::size_t i) { return std::size_t{bit_reversal_[i]}; });
        detail::fft::scale_all(data.data(), N, fft_scale<T>(norm, N, true));
    }

private:
    std::array<complex<T>, layout.storage_size()> tables_{};
    std::array<std::uint32_t, is_power_of_two(N) ? N : 0> bit_reversal_{};
};

/// \brief The plan of length `N`, built once by the compiler: what `fft(std::array)` and `ifft(std::array)` use.
/// \ingroup core
/// \tparam T A `math::math_number`.
/// \tparam N The length.
template <math::math_number T, std::size_t N>
inline constexpr static_fft_plan<T, N> static_fft_plan_v{};

/// \brief The forward transform of an array whose length is known at compile time, its plan built by the compiler.
/// \tparam T A `math::math_number`.
/// \tparam N The length.
/// \param x The sequence.
/// \param norm The normalisation.
/// \return Its transform.
template <math::math_number T, std::size_t N>
constexpr std::array<complex<T>, N> fft(const std::array<complex<T>, N>& x, fft_norm norm = fft_norm::backward) noexcept
{
    return static_fft_plan_v<T, N>.forward(x, norm);
}

/// \brief The inverse transform of an array whose length is known at compile time.
/// \tparam T A `math::math_number`.
/// \tparam N The length.
/// \param x The spectrum.
/// \param norm The normalisation.
/// \return The sequence.
template <math::math_number T, std::size_t N>
constexpr std::array<complex<T>, N> ifft(const std::array<complex<T>, N>& x, fft_norm norm = fft_norm::backward) noexcept
{
    return static_fft_plan_v<T, N>.inverse(x, norm);
}

// ---------------------------------------------------------------------------------------------------------------
// Real transforms

/// \brief What a real transform's plan knows before it sees data.
/// \ingroup core
struct rfft_layout
{
    std::size_t n = 0;      ///< The real length.
    fft_layout inner;       ///< The complex transform's: of length `n / 2` for even `n`, `n` for odd.

    /// \brief The layout of a real transform of length `n`.
    /// \param n The length, at least 1.
    /// \return The layout.
    static constexpr rfft_layout of(std::size_t n) noexcept { return {n, fft_layout::of(n % 2 == 0 ? n / 2 : n)}; }
    /// \brief Whether `n` is even, so the transform runs at half length.
    /// \return The result.
    [[nodiscard]] constexpr bool even() const noexcept { return n % 2 == 0; }
    /// \brief The half spectrum's length, `n / 2 + 1`.
    /// \return The number of bins.
    [[nodiscard]] constexpr std::size_t bins() const noexcept { return n / 2 + 1; }
    /// \brief The plan's table size, in complex elements: the inner plan's, and `n / 2` twiddles for even `n`.
    /// \return The size.
    [[nodiscard]] constexpr std::size_t storage_size() const noexcept { return inner.storage_size() + (even() ? n / 2 : 0); }
    /// \brief The work buffer `rfft`, `irfft` and their backward passes need, in complex elements.
    /// \return The size.
    [[nodiscard]] constexpr std::size_t work_size() const noexcept { return inner.n + inner.work_size(); }
};

/// \brief A plan for real transforms of one length, over tables in memory the caller owns.
/// \ingroup core
/// \tparam T A `math::math_number`.
template <math::math_number T>
class rfft_plan
{
public:
    /// \brief The element type.
    using value_type = T;

    /// \brief The table size a plan of length `n` needs, in complex elements.
    /// \param n The length.
    /// \return The size.
    static constexpr std::size_t storage_size(std::size_t n) noexcept { return rfft_layout::of(n).storage_size(); }
    /// \brief The work buffer a real transform of length `n` needs, in complex elements.
    /// \param n The length.
    /// \return The size.
    static constexpr std::size_t work_size(std::size_t n) noexcept { return rfft_layout::of(n).work_size(); }

    /// \brief Builds the plan's tables in `storage`.
    /// \param n The real length, at least 1.
    /// \param storage At least `storage_size(n)` elements; overwritten, and referred to by the plan.
    constexpr rfft_plan(std::size_t n, std::span<complex<T>> storage) noexcept
        : layout_(rfft_layout::of(n)), tables_(storage.data(), layout_.storage_size())
    {
        assert(n >= 1 && storage.size() >= layout_.storage_size());
        detail::fft::build(layout_.inner, storage.data());
        if (layout_.even())
            for (std::size_t k = 0; k < n / 2; ++k)
                storage[layout_.inner.storage_size() + k] = twiddle<T>(k, n);
    }

    /// \brief A plan over tables already built for `layout`.
    /// \param layout The layout.
    /// \param tables `layout.storage_size()` elements as `rfft_plan`'s constructor fills them.
    constexpr rfft_plan(const rfft_layout& layout, std::span<const complex<T>> tables) noexcept : layout_(layout), tables_(tables)
    {
        assert(tables.size() >= layout.storage_size());
    }

    /// \brief The real length.
    /// \return `n`.
    [[nodiscard]] constexpr std::size_t size() const noexcept { return layout_.n; }
    /// \brief The half spectrum's length, `n / 2 + 1`.
    /// \return The number of bins.
    [[nodiscard]] constexpr std::size_t bins() const noexcept { return layout_.bins(); }
    /// \brief The work buffer a transform needs, in complex elements.
    /// \return The size.
    [[nodiscard]] constexpr std::size_t work_size() const noexcept { return layout_.work_size(); }
    /// \brief The layout.
    /// \return The layout.
    [[nodiscard]] constexpr const rfft_layout& layout() const noexcept { return layout_; }
    /// \brief The complex plan inside: of length `n / 2` for even `n`, `n` for odd.
    /// \return The plan; it refers to the same tables.
    [[nodiscard]] constexpr fft_plan<T> complex_plan() const noexcept { return fft_plan<T>(layout_.inner, tables_.first(layout_.inner.storage_size())); }

    /// \brief The unnormalised real transform of `in(j)`, `j < n`, into `out` (`bins()` elements): the building block of `rfft`.
    /// \tparam In A callable `T(std::size_t)`: sample `j` (a window applied, a frame of a longer signal, ...).
    /// \param in The samples.
    /// \param out The half spectrum.
    /// \param work At least `work_size()` elements.
    template <typename In>
    constexpr void forward_from(In&& in, std::span<complex<T>> out, std::span<complex<T>> work) const noexcept
    {
        assert(out.size() == bins() && work.size() >= work_size());
        const std::size_t n = layout_.n;
        const fft_plan<T> inner = complex_plan();
        if (!layout_.even()) {
            const auto buf = work.first(n);
            for (std::size_t j = 0; j < n; ++j)
                buf[j] = {in(j), T(0)};
            inner.template execute<false>(buf, work.subspan(n));
            for (std::size_t k = 0; k < bins(); ++k)
                out[k] = buf[k];
            out[0].im = T(0);   // exactly real, as the even path's
            return;
        }
        const std::size_t h = n / 2;
        for (std::size_t j = 0; j < h; ++j)
            out[j] = {in(2 * j), in(2 * j + 1)};
        inner.template execute<false>(out.first(h), work);
        const complex<T>* w = tables_.data() + layout_.inner.storage_size();
        const complex<T> z0 = out[0];
        out[0] = {z0.re + z0.im, T(0)};
        out[h] = {z0.re - z0.im, T(0)};
        for (std::size_t k = 1; 2 * k <= h; ++k) {
            const complex<T> zk = out[k], zh = out[h - k];
            const complex<T> e = (zk + conj(zh)) * T(0.5);
            const complex<T> d = zk - conj(zh);
            const complex<T> o = complex<T>{d.im, -d.re} * T(0.5);   // (zk - conj zh) / 2i
            const complex<T> wo = w[k] * o;
            out[k] = e + wo;
            out[h - k] = conj(e - wo);
        }
    }

    /// \brief The unnormalised inverse real transform of the half spectrum `in(k)`, `k <= n / 2`, each sample handed to `out(j, x_j)`: the building block of `irfft`.
    ///
    /// Only the real parts of bin 0 and (for even `n`) bin `n / 2` are read. The result is `n` times numpy's `irfft`.
    /// \tparam In A callable `complex<T>(std::size_t)`.
    /// \tparam Out A callable `void(std::size_t, T)`.
    /// \param in The half spectrum.
    /// \param out The sink of the samples, called once for each `j < n`.
    /// \param work At least `work_size()` elements.
    template <typename In, typename Out>
    constexpr void inverse_to(In&& in, Out&& out, std::span<complex<T>> work) const noexcept
    {
        assert(work.size() >= work_size());
        const std::size_t n = layout_.n;
        const fft_plan<T> inner = complex_plan();
        if (!layout_.even()) {
            const auto buf = work.first(n);
            buf[0] = {in(0).re, T(0)};
            for (std::size_t k = 1; k <= n / 2; ++k) {
                buf[k] = in(k);
                buf[n - k] = conj(buf[k]);
            }
            inner.template execute<true>(buf, work.subspan(n));
            for (std::size_t j = 0; j < n; ++j)
                out(j, buf[j].re);
            return;
        }
        const std::size_t h = n / 2;
        const auto buf = work.first(h);
        const complex<T>* w = tables_.data() + layout_.inner.storage_size();
        for (std::size_t k = 0; k < h; ++k) {
            const complex<T> xk = k == 0 ? complex<T>{in(0).re, T(0)} : in(k);
            const complex<T> xh = k == 0 ? complex<T>{in(h).re, T(0)} : in(h - k);
            const complex<T> e = xk + conj(xh);
            const complex<T> o = (xk - conj(xh)) * conj(w[k]);
            buf[k] = e + times_i(o);
        }
        inner.template execute<true>(buf, work.subspan(h));
        for (std::size_t j = 0; j < h; ++j) {
            out(2 * j, buf[j].re);
            out(2 * j + 1, buf[j].im);
        }
    }

private:
    rfft_layout layout_;
    std::span<const complex<T>> tables_;
};

/// \brief The real forward transform: the half spectrum `X_k`, `k <= n / 2`, of `n` real samples.
/// \tparam T A `math::math_number`.
/// \param plan The plan of the input's length.
/// \param in `n` samples.
/// \param out `n / 2 + 1` bins.
/// \param work At least `plan.work_size()` elements.
/// \param norm The normalisation.
template <math::math_number T>
constexpr void rfft(const rfft_plan<T>& plan, std::span<const std::type_identity_t<T>> in, std::span<complex<std::type_identity_t<T>>> out,
                    std::span<complex<std::type_identity_t<T>>> work, fft_norm norm = fft_norm::backward) noexcept
{
    assert(in.size() == plan.size());
    plan.forward_from([in](std::size_t j) { return in[j]; }, out, work);
    detail::fft::scale_all(out.data(), out.size(), fft_scale<T>(norm, plan.size(), false));
}

/// \brief The real inverse transform: `n` samples from the half spectrum (bin 0's and bin `n / 2`'s imaginary parts ignored).
/// \tparam T A `math::math_number`.
/// \param plan The plan of the output's length.
/// \param in `n / 2 + 1` bins.
/// \param out `n` samples.
/// \param work At least `plan.work_size()` elements.
/// \param norm The normalisation.
template <math::math_number T>
constexpr void irfft(const rfft_plan<T>& plan, std::span<const complex<std::type_identity_t<T>>> in, std::span<std::type_identity_t<T>> out,
                     std::span<complex<std::type_identity_t<T>>> work, fft_norm norm = fft_norm::backward) noexcept
{
    assert(in.size() == plan.bins() && out.size() == plan.size());
    const T s = fft_scale<T>(norm, plan.size(), true);
    plan.inverse_to([in](std::size_t k) { return in[k]; }, [out, s](std::size_t j, T v) { out[j] = v * s; }, work);
}

/// \brief The backward pass of `rfft`: given `dL/dX` over the half spectrum, `dL/dx` over the real input.
/// \tparam T A `math::math_number`.
/// \param plan The forward transform's plan.
/// \param grad_spectrum `n / 2 + 1` bins of `dL/dX`.
/// \param grad_x `n` samples of `dL/dx`, written.
/// \param work At least `plan.work_size()` elements.
/// \param norm The forward transform's normalisation.
template <math::math_number T>
constexpr void rfft_backward(const rfft_plan<T>& plan, std::span<const complex<std::type_identity_t<T>>> grad_spectrum,
                             std::span<std::type_identity_t<T>> grad_x, std::span<complex<std::type_identity_t<T>>> work,
                             fft_norm norm = fft_norm::backward) noexcept
{
    assert(grad_spectrum.size() == plan.bins() && grad_x.size() == plan.size());
    const std::size_t n = plan.size();
    const T s = fft_scale<T>(norm, n, false);
    plan.inverse_to([grad_spectrum, n](std::size_t k) { return k == 0 || 2 * k == n ? grad_spectrum[k] : grad_spectrum[k] * T(0.5); },
                    [grad_x, s](std::size_t j, T v) { grad_x[j] = v * s; }, work);
}

/// \brief The backward pass of `irfft`: given `dL/dx` over the real output, `dL/dX` over the half spectrum.
/// \tparam T A `math::math_number`.
/// \param plan The inverse transform's plan.
/// \param grad_x `n` samples of `dL/dx`.
/// \param grad_spectrum `n / 2 + 1` bins of `dL/dX`, written.
/// \param work At least `plan.work_size()` elements.
/// \param norm The inverse transform's normalisation.
template <math::math_number T>
constexpr void irfft_backward(const rfft_plan<T>& plan, std::span<const std::type_identity_t<T>> grad_x,
                              std::span<complex<std::type_identity_t<T>>> grad_spectrum, std::span<complex<std::type_identity_t<T>>> work,
                              fft_norm norm = fft_norm::backward) noexcept
{
    assert(grad_spectrum.size() == plan.bins() && grad_x.size() == plan.size());
    const std::size_t n = plan.size();
    plan.forward_from([grad_x](std::size_t j) { return grad_x[j]; }, grad_spectrum, work);
    const T s = fft_scale<T>(norm, n, true);
    for (std::size_t k = 0; k < grad_spectrum.size(); ++k)
        grad_spectrum[k] *= k == 0 || 2 * k == n ? s : T(2) * s;
}

/// \brief A real-transform plan for a length known at compile time, every table computed by the compiler.
/// \ingroup core
/// \tparam T A `math::math_number`.
/// \tparam N The real length, at least 1.
template <math::math_number T, std::size_t N>
    requires(N >= 1)
class static_rfft_plan
{
public:
    /// \brief The layout.
    static constexpr rfft_layout layout = rfft_layout::of(N);
    /// \brief The work buffer a transform needs, in complex elements.
    static constexpr std::size_t work_size = layout.work_size();
    /// \brief The half spectrum's length.
    static constexpr std::size_t bins = N / 2 + 1;

    /// \brief Builds every table, at compile time.
    consteval static_rfft_plan() { rfft_plan<T>(N, std::span<complex<T>>(tables_)); }

    /// \brief An `rfft_plan` over this plan's tables, for the free functions.
    /// \return The plan; it refers to this object.
    [[nodiscard]] constexpr rfft_plan<T> plan() const noexcept { return rfft_plan<T>(layout, tables_); }

    /// \brief The half spectrum of an array, its work buffer on the stack.
    /// \param x The samples.
    /// \param norm The normalisation.
    /// \return The `N / 2 + 1` bins.
    [[nodiscard]] constexpr std::array<complex<T>, bins> forward(const std::array<T, N>& x, fft_norm norm = fft_norm::backward) const noexcept
    {
        std::array<complex<T>, bins> out{};
        std::array<complex<T>, work_size> work{};
        rfft(plan(), std::span<const T>(x), std::span<complex<T>>(out), std::span<complex<T>>(work), norm);
        return out;
    }

    /// \brief The samples of a half spectrum, its work buffer on the stack.
    /// \param x The `N / 2 + 1` bins.
    /// \param norm The normalisation.
    /// \return The `N` samples.
    [[nodiscard]] constexpr std::array<T, N> inverse(const std::array<complex<T>, bins>& x, fft_norm norm = fft_norm::backward) const noexcept
    {
        std::array<T, N> out{};
        std::array<complex<T>, work_size> work{};
        irfft(plan(), std::span<const complex<T>>(x), std::span<T>(out), std::span<complex<T>>(work), norm);
        return out;
    }

private:
    std::array<complex<T>, layout.storage_size()> tables_{};
};

/// \brief The real plan of length `N`, built once by the compiler: what `rfft(std::array)` and `irfft<N>(std::array)` use.
/// \ingroup core
/// \tparam T A `math::math_number`.
/// \tparam N The real length.
template <math::math_number T, std::size_t N>
inline constexpr static_rfft_plan<T, N> static_rfft_plan_v{};

/// \brief The half spectrum of an array whose length is known at compile time.
/// \tparam T A `math::math_number`.
/// \tparam N The length.
/// \param x The samples.
/// \param norm The normalisation.
/// \return The `N / 2 + 1` bins.
template <math::math_number T, std::size_t N>
constexpr std::array<complex<T>, N / 2 + 1> rfft(const std::array<T, N>& x, fft_norm norm = fft_norm::backward) noexcept
{
    return static_rfft_plan_v<T, N>.forward(x, norm);
}

/// \brief The `N` samples of a half spectrum, `N` given (it is `2 (bins - 1)` or one more).
/// \tparam N The real length.
/// \tparam T A `math::math_number`.
/// \param x The `N / 2 + 1` bins.
/// \param norm The normalisation.
/// \return The samples.
template <std::size_t N, math::math_number T>
constexpr std::array<T, N> irfft(const std::array<complex<T>, N / 2 + 1>& x, fft_norm norm = fft_norm::backward) noexcept
{
    return static_rfft_plan_v<T, N>.inverse(x, norm);
}

// ---------------------------------------------------------------------------------------------------------------
// Several dimensions

/// \brief The forward transform of each of `rows` contiguous rows of `plan.size()` elements, in place.
/// \tparam T A `math::math_number`.
/// \param plan The rows' plan.
/// \param data `rows * plan.size()` elements, row-major.
/// \param work At least `plan.work_size()` elements.
/// \param norm The normalisation.
/// \param inverse Whether to apply the inverse instead.
template <math::math_number T>
constexpr void fft_rows(const fft_plan<T>& plan, std::span<complex<std::type_identity_t<T>>> data, std::span<complex<std::type_identity_t<T>>> work,
                        fft_norm norm = fft_norm::backward, bool inverse = false) noexcept
{
    const std::size_t n = plan.size();
    if (n == 0)
        return;
    assert(data.size() % n == 0);
    const T s = fft_scale<T>(norm, n, inverse);
    for (std::size_t r = 0; r < data.size() / n; ++r) {
        const auto row = data.subspan(r * n, n);
        if (inverse)
            plan.template execute<true>(row, work);
        else
            plan.template execute<false>(row, work);
        detail::fft::scale_all(row.data(), n, s);
    }
}

/// \brief The work buffer `fftn` needs for these plans, in complex elements: the longest axis and its plan's work.
/// \tparam T A `math::math_number`.
/// \param plans One plan per axis.
/// \return The size.
template <math::math_number T>
constexpr std::size_t fftn_work_size(std::span<const fft_plan<T>> plans) noexcept
{
    std::size_t w = 0;
    for (const auto& p : plans)
        w = p.size() + p.work_size() > w ? p.size() + p.work_size() : w;
    return w;
}

/// \brief The transform along every axis of a row-major array, in place: numpy's `fftn` over all axes.
/// \tparam T A `math::math_number`.
/// \param plans One plan per axis, of that axis's length.
/// \param data The array, `product of the lengths` elements, the last axis contiguous.
/// \param work At least `fftn_work_size(plans)` elements.
/// \param norm The normalisation, applied per axis (so the total scale is that of the whole size).
/// \param inverse Whether to apply the inverse instead.
template <math::math_number T>
constexpr void fftn(std::span<const fft_plan<T>> plans, std::span<complex<std::type_identity_t<T>>> data,
                    std::span<complex<std::type_identity_t<T>>> work, fft_norm norm = fft_norm::backward, bool inverse = false) noexcept
{
    std::size_t total = 1;
    for (const auto& p : plans)
        total *= p.size();
    assert(data.size() == total && work.size() >= fftn_work_size(plans));
    if (total == 0)
        return;
    std::size_t stride = total;   // the product of the lengths of this axis and those after it, divided as we go
    for (const auto& p : plans) {
        const std::size_t n = p.size();
        stride /= n;
        const T s = fft_scale<T>(norm, n, inverse);
        const auto line = work.first(n);
        const auto rest = work.subspan(n);
        for (std::size_t outer = 0; outer < total; outer += n * stride)
            for (std::size_t inner = 0; inner < stride; ++inner) {
                const std::size_t base = outer + inner;
                for (std::size_t i = 0; i < n; ++i)
                    line[i] = data[base + i * stride];
                if (inverse)
                    p.template execute<true>(line, rest);
                else
                    p.template execute<false>(line, rest);
                for (std::size_t i = 0; i < n; ++i)
                    data[base + i * stride] = line[i] * s;
            }
    }
}

/// \brief The two-dimensional transform of a row-major `rows x cols` array, in place.
/// \tparam T A `math::math_number`.
/// \param plan0 The plan of length `rows` (the first axis).
/// \param plan1 The plan of length `cols` (the second, contiguous, axis).
/// \param data `rows * cols` elements.
/// \param work At least `fftn_work_size` of the two plans.
/// \param norm The normalisation.
/// \param inverse Whether to apply the inverse instead.
template <math::math_number T>
constexpr void fft2(const fft_plan<T>& plan0, const fft_plan<std::type_identity_t<T>>& plan1, std::span<complex<std::type_identity_t<T>>> data,
                    std::span<complex<std::type_identity_t<T>>> work, fft_norm norm = fft_norm::backward, bool inverse = false) noexcept
{
    const std::array<fft_plan<T>, 2> plans{plan0, plan1};
    fftn<T>(std::span<const fft_plan<T>>(plans), data, work, norm, inverse);
}

/// \brief The backward pass of `fftn`: the adjoint, the opposite direction with the same per-axis scale.
/// \tparam T A `math::math_number`.
/// \param plans The forward transform's plans.
/// \param grad `dL/dX`, replaced by `dL/dx`.
/// \param work At least `fftn_work_size(plans)` elements.
/// \param norm The forward transform's normalisation.
/// \param inverse Whether the forward transform was the inverse.
template <math::math_number T>
constexpr void fftn_backward(std::span<const fft_plan<T>> plans, std::span<complex<std::type_identity_t<T>>> grad,
                             std::span<complex<std::type_identity_t<T>>> work, fft_norm norm = fft_norm::backward, bool inverse = false) noexcept
{
    // The adjoint of s F is s F^H: run the other direction unnormalised, then apply the forward pass's scale.
    fftn<T>(plans, grad, work, fft_norm::none, !inverse);
    std::size_t total = 1;
    for (const auto& p : plans)
        total *= p.size();
    T s = T(1);
    for (const auto& p : plans)
        s *= fft_scale<T>(norm, p.size(), inverse);
    detail::fft::scale_all(grad.data(), total, s);
}

} // namespace snicholls::signal

#endif // snicholls_core_signal_fft_hpp
