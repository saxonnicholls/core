//
//  minifloat.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml::minifloat<E, M, Kind>`: the low-precision floating-point formats of machine learning (bfloat16, float16, FP8 E4M3 and E5M2, the MX FP6 and FP4) as one template, with exact decoding, round-to-nearest-even and stochastic rounding, all `constexpr`.
/// \ingroup core
///
/// **Overview.** A `minifloat` is the bits of a narrow float in a trivially copyable 8- or 16-bit word. It
/// converts to `float` and `double` exactly, and from them with round-to-nearest-even or with stochastic
/// rounding from random bits the caller supplies (so a GPU kernel draws them by thread index). A format is three
/// numbers: exponent bits, mantissa bits, and what the all-ones exponent means.
///
/// | Alias | E | M | Kind | Largest finite | Use |
/// |---|---|---|---|---|---|
/// | `bfloat16` | 8 | 7 | `ieee` | 3.39e38 | training: float's range, 3 digits |
/// | `float16` | 5 | 10 | `ieee` | 65504 | inference, gradients with loss scaling |
/// | `float8_e5m2` | 5 | 2 | `ieee` | 57344 | FP8 gradients (OCP OFP8) |
/// | `float8_e4m3` | 4 | 3 | `finite_nan` | 448 | FP8 weights and activations (OCP OFP8 "E4M3FN") |
/// | `float6_e3m2`, `float6_e2m3` | 3/2 | 2/3 | `finite` | 28, 7.5 | MX FP6 (OCP MX v1.0) |
/// | `float4_e2m1` | 2 | 1 | `finite` | 6 | MX FP4, NVFP4 elements |
///
/// - `ieee`: the all-ones exponent is infinity (mantissa 0) or NaN, as IEEE 754.
/// - `finite_nan`: no infinity; only all-ones exponent *and* mantissa is NaN, so E4M3 reaches 448.
/// - `finite`: no infinity and no NaN; every code is a number.
///
/// Overflow on conversion follows `overflow_policy`: `infinity` (the default for `ieee`: infinity; for
/// `finite_nan`: NaN, OCP's non-saturating mode) or `saturate` (the default otherwise: the largest finite value,
/// which is what FP8 training uses). An 8-bit or narrower format also has `decode_table<F>`, all its values in
/// a table the compiler builds.
///
/// \snippet ml/minifloat_demo.cpp minifloat
///
/// **Relationships.** `ml/quantize.hpp` stores block scales in `float16` and MX elements in these formats.
/// Group: \ref core.
///
/// **Implementations.** One generic conversion; `bfloat16` from `float` takes the usual integer fast path,
/// which gives the same bits.
///
/// **Guarantees.** `constexpr` (through `std::bit_cast`), `noexcept`, trivially copyable, `sizeof` 1 or 2.
/// Decoding is exact. Encoding rounds once from `double`, so a `float` or a `double` converts correctly rounded.
/// Stochastic rounding rounds up with probability equal to the discarded fraction, to 2^-32.
///
/// **Errors.** None. A NaN converts to NaN where the format has one, and to the largest magnitude, with its
/// sign, where it does not. A value below half the smallest subnormal converts to zero, keeping its sign.
///
/// **Performance.** Benchmarked against hand-written bfloat16 conversion (`benchmarks/ml/minifloat_benchmark.cpp`).
///
/// **Tests and demos.** `tests/ml/minifloat_test.cpp` (every code of every 8-bit format, round trips, ties,
/// overflow, stochastic rounding's mean), `demos/ml/minifloat_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<bit>`, `<cstddef>`, `<cstdint>`, `<limits>`, `<type_traits>`.
///
/// **Known issues.** No arithmetic: convert to `float`, compute, convert back, as hardware without native
/// support does. C++23's `std::bfloat16_t` and `std::float16_t` are arithmetic types; these are storage formats
/// that also work in C++20 and on a device.
///
/// **Status.** Experimental, new in 0.6.0.

#ifndef snicholls_core_ml_minifloat_hpp
#define snicholls_core_ml_minifloat_hpp

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace snicholls::ml
{

/// \brief What a format's all-ones exponent means.
/// \ingroup core
enum class minifloat_kind
{
    ieee,         ///< Infinity and NaN, as IEEE 754.
    finite_nan,   ///< No infinity; NaN only when every exponent and mantissa bit is set (OCP E4M3).
    finite,       ///< Neither: every code is a finite number (OCP MX FP6 and FP4).
};

/// \brief What a conversion does with a value beyond the format's largest finite number.
/// \ingroup core
enum class overflow_policy
{
    infinity,   ///< Infinity, or NaN in a `finite_nan` format; a `finite` format saturates.
    saturate,   ///< The largest finite value, with the sign.
};

/// \brief A narrow floating-point number: `E` exponent bits, `M` mantissa bits and a sign, stored in 8 or 16 bits.
/// \ingroup core
/// \tparam E Exponent bits, 1 to 8.
/// \tparam M Mantissa bits, 0 to 10 (at most 15 bits in all).
/// \tparam Kind What the all-ones exponent means.
template <int E, int M, minifloat_kind Kind = minifloat_kind::ieee>
struct minifloat
{
    static_assert(E >= 1 && E <= 8 && M >= 0 && E + M <= 15, "snicholls::ml::minifloat: 1 to 8 exponent bits, at most 15 bits besides the sign");
    static_assert(Kind != minifloat_kind::ieee || M >= 1, "snicholls::ml::minifloat: an IEEE format needs a mantissa bit to tell NaN from infinity");

    /// \brief The word the bits are stored in.
    using storage_type = std::conditional_t<(E + M + 1 <= 8), std::uint8_t, std::uint16_t>;

    static constexpr int exponent_bits = E;                  ///< Exponent bits.
    static constexpr int mantissa_bits = M;                  ///< Mantissa bits.
    static constexpr int total_bits = E + M + 1;             ///< Bits in all, with the sign.
    static constexpr int bias = (1 << (E - 1)) - 1;          ///< The exponent bias.
    static constexpr minifloat_kind kind = Kind;             ///< What the all-ones exponent means.
    /// \brief The overflow policy conversions use unless told otherwise.
    static constexpr overflow_policy default_overflow = Kind == minifloat_kind::ieee ? overflow_policy::infinity : overflow_policy::saturate;

    storage_type bits = 0;   ///< The sign, exponent and mantissa, the sign highest; public, so the type is structural.

    /// \brief A minifloat from its bits.
    /// \param b The bits; only the low `total_bits` are used.
    /// \return The minifloat.
    [[nodiscard]] static constexpr minifloat from_bits(std::uint32_t b) noexcept
    {
        return minifloat{static_cast<storage_type>(b & ((1u << total_bits) - 1))};
    }

    /// \brief Whether the format has infinities.
    static constexpr bool has_infinity = Kind == minifloat_kind::ieee;
    /// \brief Whether the format has NaN.
    static constexpr bool has_nan = Kind != minifloat_kind::finite;
};

/// \brief Brain float: `float`'s 8-bit exponent and 7 mantissa bits.
using bfloat16 = minifloat<8, 7>;
/// \brief IEEE 754 binary16.
using float16 = minifloat<5, 10>;
/// \brief OCP OFP8 E5M2: binary16's exponent, 2 mantissa bits, IEEE infinities and NaN.
using float8_e5m2 = minifloat<5, 2>;
/// \brief OCP OFP8 E4M3 ("E4M3FN"): no infinity, one NaN pattern, largest 448.
using float8_e4m3 = minifloat<4, 3, minifloat_kind::finite_nan>;
/// \brief OCP MX FP6 E3M2.
using float6_e3m2 = minifloat<3, 2, minifloat_kind::finite>;
/// \brief OCP MX FP6 E2M3.
using float6_e2m3 = minifloat<2, 3, minifloat_kind::finite>;
/// \brief OCP MX FP4 E2M1: the values 0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives.
using float4_e2m1 = minifloat<2, 1, minifloat_kind::finite>;

/// \brief Whether `T` is a `minifloat`.
/// \ingroup core
template <typename T>
inline constexpr bool is_minifloat_v = false;
/// \brief Whether `T` is a `minifloat`: true for every specialisation.
template <int E, int M, minifloat_kind K>
inline constexpr bool is_minifloat_v<minifloat<E, M, K>> = true;

namespace detail::minifloat_
{
    // 2^k as a double, for k a normal double exponent
    constexpr double pow2(int k) noexcept
    {
        return std::bit_cast<double>(static_cast<std::uint64_t>(k + 1023) << 52);
    }

    template <typename F>
    constexpr std::uint32_t max_code() noexcept
    {
        constexpr std::uint32_t all = (1u << (F::exponent_bits + F::mantissa_bits)) - 1;
        if constexpr (F::kind == minifloat_kind::ieee)
            return (((1u << F::exponent_bits) - 1) << F::mantissa_bits) - 1;
        else if constexpr (F::kind == minifloat_kind::finite_nan)
            return all - 1;
        else
            return all;
    }

    template <typename F>
    constexpr std::uint32_t nan_code() noexcept
    {
        if constexpr (F::kind == minifloat_kind::ieee)
            return (((1u << F::exponent_bits) - 1) << F::mantissa_bits) | (1u << (F::mantissa_bits - 1));
        else
            return (1u << (F::exponent_bits + F::mantissa_bits)) - 1;
    }

    template <typename F>
    constexpr std::uint32_t overflowed(overflow_policy p) noexcept
    {
        if (p == overflow_policy::saturate || F::kind == minifloat_kind::finite)
            return max_code<F>();
        if constexpr (F::kind == minifloat_kind::ieee)
            return ((1u << F::exponent_bits) - 1) << F::mantissa_bits;   // infinity
        else
            return nan_code<F>();
    }

    // The code of |x| (finite, at least 0) rounded to F: to nearest even, or up with probability = the fraction
    // dropped when `stochastic`, against `random` / 2^32.
    template <typename F>
    constexpr std::uint32_t encode(double x, bool stochastic, std::uint32_t random, overflow_policy p) noexcept
    {
        const std::uint64_t u = std::bit_cast<std::uint64_t>(x) & ~(std::uint64_t{1} << 63);
        const std::uint32_t sign = static_cast<std::uint32_t>(std::bit_cast<std::uint64_t>(x) >> 63) << (F::exponent_bits + F::mantissa_bits);
        const int field = static_cast<int>(u >> 52);
        if (field == 0x7ff) {
            if ((u & ((std::uint64_t{1} << 52) - 1)) != 0)   // NaN
                return F::has_nan ? (sign | nan_code<F>()) : (sign | max_code<F>());
            return sign | overflowed<F>(p);   // infinity
        }
        if (field == 0)   // zero, or a double subnormal: far below every format's smallest value
            return sign;
        constexpr int e_min = 1 - F::bias;
        const int e = field - 1023;
        const int q_exp = (e < e_min ? e_min : e) - F::mantissa_bits;   // the exponent of the format's last place here
        if (q_exp > 1000)
            return sign | overflowed<F>(p);
        const double q = std::bit_cast<double>(u) * pow2(-q_exp);   // exact: a power-of-two scaling, below 2^(M+1)
        const auto whole = static_cast<std::uint64_t>(q);
        const double frac = q - static_cast<double>(whole);         // exact
        std::uint64_t n = whole;
        if (stochastic) {
            if (frac * 4294967296.0 > static_cast<double>(random))
                ++n;
        } else if (frac > 0.5 || (frac == 0.5 && (whole & 1) != 0)) {
            ++n;
        }
        // Continuous codes: a mantissa that rounds up to 2^M carries into the exponent, a subnormal into the first normal
        const std::uint64_t code = e < e_min ? n : (static_cast<std::uint64_t>(e + F::bias - 1) << F::mantissa_bits) + n;
        if (code > max_code<F>())
            return sign | overflowed<F>(p);
        return sign | static_cast<std::uint32_t>(code);
    }
} // namespace detail::minifloat_

/// \brief `x` rounded to the nearest `F`, ties to even.
/// \tparam F A `minifloat`.
/// \param x The value.
/// \param overflow What a value beyond the largest finite becomes.
/// \return The minifloat.
template <typename F>
    requires is_minifloat_v<F>
[[nodiscard]] constexpr F to_minifloat(double x, overflow_policy overflow = F::default_overflow) noexcept
{
    return F::from_bits(detail::minifloat_::encode<F>(x, false, 0, overflow));
}

/// \brief `x` rounded to `F` stochastically: up with probability equal to the fraction of the last place dropped, so the mean is `x`.
///
/// Used where many small updates must not round away (weights in bfloat16, llm.c's AdamW).
/// \tparam F A `minifloat`.
/// \param x The value.
/// \param random 32 uniformly random bits, from the caller's generator or a hash of the element's index: the magnitude
///   rounds up when `random / 2^32` is below the fraction dropped.
/// \param overflow What a value beyond the largest finite becomes.
/// \return The minifloat.
template <typename F>
    requires is_minifloat_v<F>
[[nodiscard]] constexpr F to_minifloat_stochastic(double x, std::uint32_t random, overflow_policy overflow = F::default_overflow) noexcept
{
    return F::from_bits(detail::minifloat_::encode<F>(x, true, random, overflow));
}

/// \brief A `float` rounded to the nearest `bfloat16`: the integer fast path, the same bits as `to_minifloat<bfloat16>`.
/// \param x The value.
/// \return The bfloat16.
[[nodiscard]] constexpr bfloat16 to_bfloat16(float x) noexcept
{
    const std::uint32_t u = std::bit_cast<std::uint32_t>(x);
    if ((u & 0x7fffffffu) > 0x7f800000u)   // NaN: keep the sign, quiet
        return bfloat16::from_bits((u >> 16) | 0x0040u);
    return bfloat16::from_bits((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

/// \brief The exact value of a `minifloat`, as a `double`.
/// \tparam F A `minifloat`.
/// \param f The minifloat.
/// \return Its value: infinity and NaN where the format has them.
template <typename F>
    requires is_minifloat_v<F>
[[nodiscard]] constexpr double to_double(F f) noexcept
{
    constexpr int E = F::exponent_bits, M = F::mantissa_bits;
    const std::uint32_t b = f.bits;
    const bool negative = ((b >> (E + M)) & 1u) != 0;
    const std::uint32_t ef = (b >> M) & ((1u << E) - 1);
    const std::uint32_t mf = b & ((1u << M) - 1);
    double v = 0.0;
    if (F::kind == minifloat_kind::ieee && ef == (1u << E) - 1)
        v = mf == 0 ? std::numeric_limits<double>::infinity() : std::numeric_limits<double>::quiet_NaN();
    else if (F::kind == minifloat_kind::finite_nan && ef == (1u << E) - 1 && mf == (1u << M) - 1)
        v = std::numeric_limits<double>::quiet_NaN();
    else if (ef == 0)
        v = static_cast<double>(mf) * detail::minifloat_::pow2(1 - F::bias - M);
    else
        v = static_cast<double>((1u << M) + mf) * detail::minifloat_::pow2(static_cast<int>(ef) - F::bias - M);
    return negative ? -v : v;
}

/// \brief The exact value of a `minifloat`, as a `float` (every format here fits in `float` exactly).
/// \tparam F A `minifloat`.
/// \param f The minifloat.
/// \return Its value.
template <typename F>
    requires is_minifloat_v<F>
[[nodiscard]] constexpr float to_float(F f) noexcept
{
    if constexpr (std::is_same_v<F, bfloat16>)
        return std::bit_cast<float>(static_cast<std::uint32_t>(f.bits) << 16);
    else
        return static_cast<float>(to_double(f));
}

/// \brief The largest finite value of `F`.
/// \tparam F A `minifloat`.
/// \return The minifloat.
template <typename F>
    requires is_minifloat_v<F>
[[nodiscard]] constexpr F minifloat_max() noexcept
{
    return F::from_bits(detail::minifloat_::max_code<F>());
}

/// \brief Every value of an 8-bit or narrower format, indexed by its bits: a table the compiler builds, for decoding by one load.
/// \tparam F A `minifloat` of at most 8 bits.
template <typename F>
    requires(is_minifloat_v<F> && F::total_bits <= 8)
inline constexpr std::array<float, (std::size_t{1} << F::total_bits)> decode_table = [] {
    std::array<float, (std::size_t{1} << F::total_bits)> t{};
    for (std::size_t i = 0; i < t.size(); ++i)
        t[i] = to_float(F::from_bits(static_cast<std::uint32_t>(i)));
    return t;
}();

} // namespace snicholls::ml

#endif // snicholls_core_ml_minifloat_hpp
