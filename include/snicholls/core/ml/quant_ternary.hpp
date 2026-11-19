//
//  quant_ternary.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` ternary blocks: ggml's TQ1_0 (1.6875 bits per weight, five trits a byte) and TQ2_0 (2.0625 bits, four 2-bit codes a byte), byte for byte as ggml lays them out, with quantisation, dequantisation and the dot products with Q8_K activations, all `constexpr`.
/// \ingroup core
///
/// **Overview.** A ternary model (BitNet b1.58, TriLMs) stores each weight as -1, 0 or +1 times one scale per 256.
/// ggml packs those trits two ways:
///
/// | Block | Elements | Scale | Codes | Bytes | Bits per weight |
/// |---|---|---|---|---|---|
/// | `block_tq1_0` | 256 | `float16` amax | five trits a byte in `qs` (240 weights), four in `qh` (16) | 54 | 1.6875 |
/// | `block_tq2_0` | 256 | `float16` amax | `q + 1` in 2 bits, four a byte | 66 | 2.0625 |
///
/// A trit is `lroundf(x / amax) + 1` in {0, 1, 2}. TQ1_0 writes five of them as the base-3 number `t0 t1 t2 t3 t4`
/// scaled to a byte, `ceil(n · 256 / 243)`, so trit `k` comes back as `((uint8_t)(b · 3^k) · 3) >> 8` with no
/// division; `ternary_pow3` is that table, built by the compiler. The 240 weights of `qs` run in strides: byte
/// `m` of the first 32 holds weights `m, m + 32, ..., m + 128`, byte `32 + m` of the last 16 holds weights
/// `160 + m, 176 + m, ..., 224 + m`, and `qh[j]` holds `240 + j, 244 + j, 248 + j, 252 + j` (four trits, shifted one
/// place up). TQ2_0's byte `32 g + m` holds weights `128 g + m + 32 l` in bits `2l`.
///
/// \snippet ml/quant_ternary_demo.cpp ternary
///
/// - `quantize_row(x, importance, blocks)`: ggml's `quantize_tq1_0` / `quantize_tq2_0` (which ignore the importance
///   matrix, as here);
/// - `dequantize_row(blocks, x)`: `dequantize_row_tq*_0`;
/// - `dot(weights, activations)`: `ggml_vec_dot_tq*_0_q8_K_generic`, an integer sum per block and one
///   multiplication by the scales.
///
/// **Relationships.** `ml/quantize.hpp` (the contract: `quantized_block`, `block_traits`, `quantize`,
/// `dequantize`, `block_q8_k`); `ml/bitnet.hpp` (BitNet's own absmean ternary quantisation and its I2_S, TL1 and
/// TL2 layouts). Group: \ref core.
///
/// **Implementations.** One per block, the scalar reference.
///
/// **Guarantees.** `constexpr`, no allocation, linear in the row. The blocks are trivially copyable, ggml's sizes,
/// no padding (`static_assert`ed). Bytes and dot products equal ggml's on every row of the committed fixture
/// (`tests/fixtures/ml/quant_ternary_fixture.txt`, from `scripts/generate_quant_ternary_fixture.cpp` built against
/// ggml, llama.cpp 39039aa).
///
/// **Errors.** Preconditions, asserted: the row has `256 · blocks.size()` numbers; the rows of a dot product have the
/// same number of blocks. NaN and infinity are not representable; the result is ggml's on x86 (see Known issues).
///
/// **Performance.** Against ggml's scalar loops written by hand (`benchmarks/ml/quant_ternary_benchmark.cpp`); numbers
/// in BENCHMARKS.md.
///
/// **Tests and demos.** `tests/ml/quant_ternary_test.cpp`, `demos/ml/quant_ternary_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cassert>`, `<cstddef>`, `<cstdint>`, `<limits>`, `<span>`, `ml/quantize.hpp`.
///
/// **Known issues.** ggml rounds with `lroundf`, whose result C leaves undefined for NaN and infinity (an infinite
/// inverse scale meets them when the amax is subnormal); core gives what ggml gives where the fixture was made,
/// whose `lroundf` saturates (+inf to LONG_MAX, whose low 32 bits make the trit 0; -inf and NaN to LONG_MIN, trit 1).
/// A C library that returns LONG_MIN for every invalid input gives other bytes for such a block in ggml. ggml's TQ2_0 code 3 (never written by its
/// quantiser) decodes to +2, as here.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_quant_ternary_hpp
#define snicholls_core_ml_quant_ternary_hpp

#include <snicholls/core/ml/quantize.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace snicholls::ml
{

/// \brief ggml's TQ1_0: 256 trits, five a byte, and a `float16` scale: `x ≈ d · (t - 1)`.
/// \ingroup core
struct block_tq1_0
{
    static constexpr ggml_type type = ggml_type::tq1_0;   ///< The ggml id.
    static constexpr std::size_t elements = 256;          ///< Numbers per block.
    using dot_block = block_q8_k;                         ///< ggml's `vec_dot_type`.

    std::array<std::uint8_t, 48> qs;   ///< 240 trits, five a byte.
    std::array<std::uint8_t, 4> qh;    ///< 16 trits, four a byte.
    float16 d;                         ///< The scale: the block's largest magnitude.
};

/// \brief ggml's TQ2_0: 256 two-bit codes and a `float16` scale: `x ≈ d · (q - 1)`.
/// \ingroup core
struct block_tq2_0
{
    static constexpr ggml_type type = ggml_type::tq2_0;   ///< The ggml id.
    static constexpr std::size_t elements = 256;          ///< Numbers per block.
    using dot_block = block_q8_k;                         ///< ggml's `vec_dot_type`.

    std::array<std::uint8_t, 64> qs;   ///< Four codes a byte.
    float16 d;                         ///< The scale: the block's largest magnitude.
};

static_assert(quantized_block<block_tq1_0> && quantized_block<block_tq2_0>, "snicholls::ml: the ternary blocks must have ggml's sizes");

/// \brief 3^k for k = 0 to 5: trit `k` of a TQ1_0 byte `b` is `((uint8_t)(b · 3^k) · 3) >> 8`.
inline constexpr std::array<std::uint8_t, 6> ternary_pow3 = [] {
    std::array<std::uint8_t, 6> p{};
    unsigned v = 1;
    for (auto& e : p) {
        e = static_cast<std::uint8_t>(v);
        v *= 3;
    }
    return p;
}();

/// \brief Every TQ1_0 byte's five trits, decoded once by the compiler: `ternary_trits[b][k]` in {0, 1, 2}.
inline constexpr std::array<std::array<std::uint8_t, 5>, 256> ternary_trits = [] {
    std::array<std::array<std::uint8_t, 5>, 256> t{};
    for (unsigned b = 0; b < 256; ++b)
        for (unsigned k = 0; k < 5; ++k) {
            const auto q = static_cast<std::uint8_t>(b * ternary_pow3[k]);
            t[b][k] = static_cast<std::uint8_t>((unsigned(q) * 3) >> 8);
        }
    return t;
}();

namespace detail::ternary
{
    // (int) lroundf(x) + 1. Out of a long's range (an infinite inverse scale) lroundf saturates, as Apple's libm
    // does where the fixture was made: LONG_MAX, whose low word is -1, for +inf; LONG_MIN, low word 0, for -inf and NaN
    constexpr int trit(float x)
    {
        if (!(x >= -2147483648.0f && x < 2147483648.0f))
            return x > 0.0f ? 0 : 1;
        return quantize::round_to_int(x) + 1;
    }

    // ceil(q · 256 / 243): five trits as a byte
    constexpr std::uint8_t pack5(unsigned q)
    {
        return static_cast<std::uint8_t>((std::uint16_t(std::uint8_t(q)) * 256 + 242) / 243);
    }

    constexpr int trit_of(std::uint8_t b, std::size_t k)
    {
        const auto q = static_cast<std::uint8_t>(b * ternary_pow3[k]);
        return int((unsigned(q) * 3) >> 8);
    }

    constexpr float amax_of(std::span<const float> b)
    {
        float amax = 0.0f;
        for (const float v : b)
            amax = quantize::fmax(amax, quantize::abs(v));
        return amax;
    }
} // namespace detail::ternary

/// \brief Quantises one row to TQ1_0, as ggml's `quantize_tq1_0` (`quantize_row_tq1_0_ref`): trits `lroundf(x / amax) + 1`.
/// \param x The row, `256 · y.size()` floats.
/// \param importance Ignored, as ggml ignores it.
/// \param y Receives the blocks.
constexpr void quantize_row(std::span<const float> x, std::span<const float> importance, std::span<block_tq1_0> y)
{
    using namespace detail::ternary;
    detail::quantize::check(256, y.size(), x.size(), importance.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
        const auto b = x.subspan(i * 256, 256);
        const float d = amax_of(b);
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        y[i].d = detail::quantize::fp16(d);
        const auto code = [&](std::size_t at) {
            const float v = b[at] * id;
            return trit(v);
        };
        for (std::size_t m = 0; m < 32; ++m) {
            unsigned q = 0;
            for (std::size_t n = 0; n < 5; ++n)
                q = std::uint8_t(std::uint8_t(q * 3) + code(m + n * 32));
            y[i].qs[m] = pack5(q);
        }
        for (std::size_t m = 0; m < 16; ++m) {
            unsigned q = 0;
            for (std::size_t n = 0; n < 5; ++n)
                q = std::uint8_t(std::uint8_t(q * 3) + code(160 + m + n * 16));
            y[i].qs[32 + m] = pack5(q);
        }
        for (std::size_t j = 0; j < 4; ++j) {
            unsigned q = 0;
            for (std::size_t m = 0; m < 4; ++m)
                q = std::uint8_t(std::uint8_t(q * 3) + code(240 + j + m * 4));
            y[i].qh[j] = pack5(std::uint8_t(q * 3));
        }
    }
}

/// \brief Quantises one row to TQ2_0, as ggml's `quantize_tq2_0` (`quantize_row_tq2_0_ref`): codes `lroundf(x / amax) + 1`.
/// \param x The row, `256 · y.size()` floats.
/// \param importance Ignored, as ggml ignores it.
/// \param y Receives the blocks.
constexpr void quantize_row(std::span<const float> x, std::span<const float> importance, std::span<block_tq2_0> y)
{
    using namespace detail::ternary;
    detail::quantize::check(256, y.size(), x.size(), importance.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
        const auto b = x.subspan(i * 256, 256);
        const float d = amax_of(b);
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        y[i].d = detail::quantize::fp16(d);
        for (std::size_t g = 0; g < 2; ++g)
            for (std::size_t m = 0; m < 32; ++m) {
                unsigned q = 0;
                for (std::size_t n = 0; n < 4; ++n) {
                    const float v = b[128 * g + m + n * 32] * id;
                    q = std::uint8_t(q + unsigned((trit(v) & 3) << (2 * n)));
                }
                y[i].qs[32 * g + m] = static_cast<std::uint8_t>(q);
            }
    }
}

/// \brief Dequantises TQ1_0 blocks: `x = d · (t - 1)`, as ggml's `dequantize_row_tq1_0`.
/// \param x The blocks.
/// \param y Receives `256 · x.size()` floats.
constexpr void dequantize_row(std::span<const block_tq1_0> x, std::span<float> y)
{
    assert(y.size() == 256 * x.size());
    std::size_t o = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float d = to_float(x[i].d);
        for (std::size_t n = 0; n < 5; ++n)
            for (std::size_t m = 0; m < 32; ++m)
                y[o++] = float(ternary_trits[x[i].qs[m]][n] - 1) * d;
        for (std::size_t n = 0; n < 5; ++n)
            for (std::size_t m = 0; m < 16; ++m)
                y[o++] = float(ternary_trits[x[i].qs[32 + m]][n] - 1) * d;
        for (std::size_t n = 0; n < 4; ++n)
            for (std::size_t j = 0; j < 4; ++j)
                y[o++] = float(ternary_trits[x[i].qh[j]][n] - 1) * d;
    }
}

/// \brief Dequantises TQ2_0 blocks: `x = d · (q - 1)`, as ggml's `dequantize_row_tq2_0`.
/// \param x The blocks.
/// \param y Receives `256 · x.size()` floats.
constexpr void dequantize_row(std::span<const block_tq2_0> x, std::span<float> y)
{
    assert(y.size() == 256 * x.size());
    std::size_t o = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float d = to_float(x[i].d);
        for (std::size_t g = 0; g < 2; ++g)
            for (std::size_t l = 0; l < 4; ++l)
                for (std::size_t m = 0; m < 32; ++m)
                    y[o++] = float(int((x[i].qs[32 * g + m] >> (2 * l)) & 3) - 1) * d;
    }
}

/// \brief The dot product of a TQ1_0 row with a Q8_K row, as ggml's `ggml_vec_dot_tq1_0_q8_K`.
/// \param x The weights.
/// \param y The activations.
/// \return `sum over blocks of (sum (t - 1) · q) · (d_x · d_y)`.
constexpr float dot(std::span<const block_tq1_0> x, std::span<const block_q8_k> y)
{
    assert(x.size() == y.size());
    float sum = 0.0f;
    for (std::size_t i = 0; i < x.size(); ++i) {
        int s = 0;
        for (std::size_t l = 0; l < 5; ++l)
            for (std::size_t m = 0; m < 32; ++m)
                s += (ternary_trits[x[i].qs[m]][l] - 1) * int(y[i].qs[l * 32 + m]);
        for (std::size_t l = 0; l < 5; ++l)
            for (std::size_t m = 0; m < 16; ++m)
                s += (ternary_trits[x[i].qs[32 + m]][l] - 1) * int(y[i].qs[160 + l * 16 + m]);
        for (std::size_t l = 0; l < 4; ++l)
            for (std::size_t j = 0; j < 4; ++j)
                s += (ternary_trits[x[i].qh[j]][l] - 1) * int(y[i].qs[240 + l * 4 + j]);
        const float scale = to_float(x[i].d) * y[i].d;
        const float term = float(s) * scale;
        sum += term;
    }
    return sum;
}

/// \brief The dot product of a TQ2_0 row with a Q8_K row, as ggml's `ggml_vec_dot_tq2_0_q8_K`.
/// \param x The weights.
/// \param y The activations.
/// \return `sum over blocks of (sum (q - 1) · q_y) · (d_y · d_x)`.
constexpr float dot(std::span<const block_tq2_0> x, std::span<const block_q8_k> y)
{
    assert(x.size() == y.size());
    float sum = 0.0f;
    for (std::size_t i = 0; i < x.size(); ++i) {
        int s = 0;
        for (std::size_t g = 0; g < 2; ++g)
            for (std::size_t l = 0; l < 4; ++l)
                for (std::size_t k = 0; k < 32; ++k)
                    s += int(y[i].qs[128 * g + l * 32 + k]) * (int((x[i].qs[32 * g + k] >> (2 * l)) & 3) - 1);
        const float d = y[i].d * to_float(x[i].d);
        const float term = float(s) * d;
        sum += term;
    }
    return sum;
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_quant_ternary_hpp
