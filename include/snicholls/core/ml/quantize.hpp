//
//  quantize.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` quantised weights: ggml's Q8_0 and Q4_0 blocks and the OCP MXFP4 block, byte for byte as ggml lays them out, with quantisation, dequantisation and the integer dot products, plus symmetric int8 rows.
/// \ingroup core
///
/// **Overview.** A quantised model stores its weights in blocks of 32 numbers, each block with one scale. These
/// are the formats llama.cpp and gpt-oss ship, so a GGUF tensor of one of them is an array of these structs, read
/// in place:
///
/// | Block | Scale | Elements | Bytes | Bits per weight |
/// |---|---|---|---|---|
/// | `block_q8_0` | `float16` `d` = amax / 127 | `int8` `round(x / d)` | 34 | 8.5 |
/// | `block_q4_0` | `float16` `d` = (the signed value of largest magnitude) / -8 | 4-bit `x / d + 8`, element `j` in the low nibble of byte `j`, `j + 16` in the high | 18 | 4.5 |
/// | `block_mxfp4` | E8M0 exponent `e` = floor(log2 amax) - 2 + 127 | FP4 E2M1, the nearest of ±{0, 0.5, 1, 1.5, 2, 3, 4, 6} · 2^(e - 127), packed as Q4_0 | 17 | 4.25 |
///
/// - `quantize<Block>(x, blocks)` and `dequantize(blocks, x)`: rows of `32 · blocks.size()` floats.
/// - `dot(x, y)`: the dot product of a Q8_0, Q4_0 or MXFP4 row with a Q8_0 row (the activations, quantised on the
///   fly), in integers within a block and one multiplication by the scales per block, as ggml's matrix-vector
///   products do. Each gives ggml's scalar reference result bit for bit.
/// - `quantize_int8(x, q)` and `dequantize_int8(q, scale, x)`: one symmetric scale for a whole row (per-channel or
///   per-tensor int8), the same rounding as Q8_0.
///
/// \snippet ml/quantize_demo.cpp quantize
///
/// **Relationships.** `ml/minifloat.hpp` (`float16`, and the FP8, FP6 and FP4 element formats of the other MX
/// blocks); `ml/row.hpp`. Group: \ref core.
///
/// **Implementations.** One per block. The blocks are the formats; the dot products are the scalar reference, which
/// a SIMD or GPU kernel reorders.
///
/// **Guarantees.** `constexpr`, no allocation, linear in the row. The blocks are trivially copyable, have ggml's
/// sizes and no padding (`static_assert`ed). Quantised bytes equal ggml's `quantize_row_*_ref` and dot products
/// equal ggml's `ggml_vec_dot_*_generic` on every input of the committed fixture
/// (`tests/fixtures/ml/quant_fixture.txt`, from `scripts/generate_quant_fixture.cpp` built against ggml). Each
/// multiplication and each addition is its own statement, so no compiler fuses them into an FMA and the result is
/// the same on every machine.
///
/// **Errors.** Preconditions, asserted: the row has `32 · blocks.size()` elements; the rows of a dot product have the
/// same number of blocks. NaN and infinity in a block are not representable in these formats; the result is
/// unspecified, as in ggml.
///
/// **Performance.** Against ggml's scalar loops written out by hand (`benchmarks/ml/quantize_benchmark.cpp`), rows of
/// 4,096: the three dot products level, 1.9 to 2.6 billion weights per second; Q8_0 quantisation 2x faster than
/// ggml's loop (590 against 300 million per second), from rounding without `roundf` (Xeon W-3245, Release, Apple clang 17, 2026-11-11,
/// one thread, scalar code; SIMD kernels are the libraries' above).
///
/// **Tests and demos.** `tests/ml/quantize_test.cpp`, `demos/ml/quantize_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<bit>`, `<cassert>`, `<cstddef>`, `<cstdint>`, `<span>`, `<type_traits>`,
/// `snicholls/core/math/constexpr_math.hpp`, `ml/minifloat.hpp`, `ml/row.hpp`.
///
/// **Known issues.** ggml computes MXFP4's exponent as `floorf(log2f(amax))`, which depends on the C library and can
/// round up for an amax a few units in the last place below a power of two; core reads the exponent from the bits,
/// as the OCP specification defines it. An amax below 2^-125 gives exponent 0 (ggml's conversion there is
/// undefined). A dot product over infinities is a NaN whose sign is the hardware's (negative on x86, positive on
/// ARM), so results are bit-identical across platforms except for the sign of a NaN. Each multiply and add is its own
/// statement, which Clang never fuses, but GCC fuses across statements by default in C++ (`-ffp-contract=fast`), so
/// on a target with FMA (every ARM64) the last bit can differ: compile with `-ffp-contract=off` for ggml's bits.
///
/// **Status.** Experimental, new in 0.7.0.

#ifndef snicholls_core_ml_quantize_hpp
#define snicholls_core_ml_quantize_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/minifloat.hpp>
#include <snicholls/core/ml/row.hpp>

#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief The numbers in one quantised block: 32, for every format here.
inline constexpr std::size_t quant_block_size = 32;

/// \brief ggml's Q8_0: a `float16` scale and 32 signed bytes, `x ≈ d · q`.
/// \ingroup core
struct block_q8_0
{
    float16 d;                                     ///< The scale, amax / 127.
    std::array<std::int8_t, quant_block_size> qs;  ///< The quantised numbers.
};

/// \brief ggml's Q4_0: a `float16` scale and 32 four-bit numbers, `x ≈ d · (q - 8)`.
/// \ingroup core
struct block_q4_0
{
    float16 d;                                          ///< The scale: the signed value of largest magnitude / -8.
    std::array<std::uint8_t, quant_block_size / 2> qs;  ///< Element `j` in the low nibble of byte `j`, `j + 16` in the high.
};

/// \brief The OCP MXFP4 block as ggml stores it: an E8M0 exponent and 32 FP4 (E2M1) numbers, `x ≈ 2^(e - 127) · v`.
/// \ingroup core
struct block_mxfp4
{
    std::uint8_t e;                                     ///< The shared exponent, biased by 127.
    std::array<std::uint8_t, quant_block_size / 2> qs;  ///< E2M1 codes, packed as Q4_0's.
};

static_assert(sizeof(block_q8_0) == 34 && sizeof(block_q4_0) == 18 && sizeof(block_mxfp4) == 17,
              "snicholls::ml: the quantised blocks must have ggml's sizes");

namespace detail::quantize
{
    using block = std::span<const float, quant_block_size>;

    // E2M1 values, doubled (ggml's kvalues_mxfp4): code c is kvalues[c] / 2
    inline constexpr std::array<std::int8_t, 16> fp4_doubled{0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

    constexpr float abs(float x) { return x < 0.0f ? -x : x; }

    // roundf for |x| below 2^31, halves away from zero: the fraction x - trunc(x) is exact
    constexpr int round_to_int(float x)
    {
        const int t = static_cast<int>(x);
        const float frac = x - static_cast<float>(t);
        return t + (frac >= 0.5f ? 1 : 0) - (frac <= -0.5f ? 1 : 0);
    }

    // 2^(e - 128): half the MX scale, as ggml_e8m0_to_fp32_half, to pair with the doubled values
    constexpr float e8m0_half(std::uint8_t e)
    {
        return std::bit_cast<float>(e < 2 ? std::uint32_t{0x00200000} << e : std::uint32_t(e - 1) << 23);
    }

    // floor(log2 x) for a positive finite float, from its bits
    constexpr int floor_log2(float x)
    {
        const std::uint32_t b = std::bit_cast<std::uint32_t>(x);
        const int biased = int((b >> 23) & 0xff);
        if (biased != 0)
            return biased - 127;
        return std::bit_width(b & 0x7fffffu) - 1 - 149;   // subnormal: m · 2^-149
    }

    constexpr block_q8_0 quantize(std::type_identity<block_q8_0>, block x)
    {
        float amax = 0.0f;
        for (const float v : x)
            amax = amax < abs(v) ? abs(v) : amax;
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        block_q8_0 b{to_minifloat<float16>(d), {}};
        for (std::size_t j = 0; j < quant_block_size; ++j) {
            const float scaled = x[j] * id;
            b.qs[j] = static_cast<std::int8_t>(round_to_int(scaled));
        }
        return b;
    }

    constexpr block_q4_0 quantize(std::type_identity<block_q4_0>, block x)
    {
        float amax = 0.0f, max = 0.0f;
        for (const float v : x)
            if (amax < abs(v)) {
                amax = abs(v);
                max = v;
            }
        const float d = max / -8.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        block_q4_0 b{to_minifloat<float16>(d), {}};
        const auto code = [id](float v) {
            const float scaled = v * id;
            const float shifted = scaled + 8.5f;
            const int q = static_cast<std::int8_t>(shifted);
            return static_cast<std::uint8_t>(q < 15 ? q : 15);
        };
        for (std::size_t j = 0; j < quant_block_size / 2; ++j)
            b.qs[j] = static_cast<std::uint8_t>(code(x[j]) | (code(x[j + quant_block_size / 2]) << 4));
        return b;
    }

    constexpr std::uint8_t nearest_fp4(float v, float half_scale)
    {
        std::uint8_t best = 0;
        float best_err = abs(float(fp4_doubled[0]) * half_scale - v);
        for (std::uint8_t c = 1; c < 16; ++c) {
            const float err = abs(float(fp4_doubled[c]) * half_scale - v);
            if (err < best_err) {
                best = c;
                best_err = err;
            }
        }
        return best;
    }

    constexpr block_mxfp4 quantize(std::type_identity<block_mxfp4>, block x)
    {
        float amax = 0.0f;
        for (const float v : x)
            amax = amax < abs(v) ? abs(v) : amax;
        const int exponent = amax > 0.0f ? floor_log2(amax) - 2 + 127 : 0;
        block_mxfp4 b{static_cast<std::uint8_t>(exponent < 0 ? 0 : exponent), {}};
        const float half = e8m0_half(b.e);
        for (std::size_t j = 0; j < quant_block_size / 2; ++j)
            b.qs[j] = static_cast<std::uint8_t>(nearest_fp4(x[j], half) | (nearest_fp4(x[j + quant_block_size / 2], half) << 4));
        return b;
    }

    constexpr void dequantize(const block_q8_0& b, std::span<float, quant_block_size> out)
    {
        const float d = to_float(b.d);
        for (std::size_t j = 0; j < quant_block_size; ++j)
            out[j] = float(b.qs[j]) * d;
    }

    constexpr void dequantize(const block_q4_0& b, std::span<float, quant_block_size> out)
    {
        const float d = to_float(b.d);
        for (std::size_t j = 0; j < quant_block_size / 2; ++j) {
            out[j] = float((b.qs[j] & 0x0f) - 8) * d;
            out[j + quant_block_size / 2] = float((b.qs[j] >> 4) - 8) * d;
        }
    }

    constexpr void dequantize(const block_mxfp4& b, std::span<float, quant_block_size> out)
    {
        const float half = e8m0_half(b.e);
        for (std::size_t j = 0; j < quant_block_size / 2; ++j) {
            out[j] = half * float(fp4_doubled[b.qs[j] & 0x0f]);
            out[j + quant_block_size / 2] = half * float(fp4_doubled[b.qs[j] >> 4]);
        }
    }
} // namespace detail::quantize

/// \brief Whether `B` is one of the quantised blocks.
/// \ingroup core
template <typename B>
concept quant_block = std::same_as<B, block_q8_0> || std::same_as<B, block_q4_0> || std::same_as<B, block_mxfp4>;

/// \brief Quantises a row of floats into blocks, as ggml's `quantize_row_*_ref`.
/// \tparam Block `block_q8_0`, `block_q4_0` or `block_mxfp4`.
/// \param x The row, `32 · blocks.size()` floats.
/// \param blocks Receives the blocks.
template <quant_block Block, std::size_t N, row X>
    requires std::same_as<row_value_t<X>, float>
constexpr void quantize(const X& x, std::span<Block, N> blocks)
{
    const auto in = as_row(x);
    assert(in.size() == quant_block_size * blocks.size());
    for (std::size_t i = 0; i < blocks.size(); ++i)
        blocks[i] = detail::quantize::quantize(std::type_identity<Block>{}, in.subspan(i * quant_block_size).template first<quant_block_size>());
}

/// \brief Dequantises blocks into a row of floats.
/// \param blocks The blocks.
/// \param x Receives `32 · blocks.size()` floats.
template <quant_block Block, std::size_t N, mutable_row X>
    requires std::same_as<row_value_t<X>, float>
constexpr void dequantize(std::span<const Block, N> blocks, X&& x)
{
    const auto out = as_mutable_row(x);
    assert(out.size() == quant_block_size * blocks.size());
    for (std::size_t i = 0; i < blocks.size(); ++i)
        detail::quantize::dequantize(blocks[i], out.subspan(i * quant_block_size).template first<quant_block_size>());
}

/// \brief The dot product of two Q8_0 rows, as ggml's `ggml_vec_dot_q8_0_q8_0`.
/// \param x The weights.
/// \param y The activations.
/// \return `sum over blocks of (sum q_x · q_y) · d_x · d_y`.
constexpr float dot(std::span<const block_q8_0> x, std::span<const block_q8_0> y)
{
    assert(x.size() == y.size());
    float sum = 0.0f;
    for (std::size_t i = 0; i < x.size(); ++i) {
        int s = 0;
        for (std::size_t j = 0; j < quant_block_size; ++j)
            s += int(x[i].qs[j]) * int(y[i].qs[j]);
        const float scale = to_float(x[i].d) * to_float(y[i].d);
        const float term = float(s) * scale;
        sum += term;
    }
    return sum;
}

/// \brief The dot product of a Q4_0 row with a Q8_0 row, as ggml's `ggml_vec_dot_q4_0_q8_0`.
/// \param x The weights.
/// \param y The activations.
/// \return `sum over blocks of (sum (q_x - 8) · q_y) · d_x · d_y`.
constexpr float dot(std::span<const block_q4_0> x, std::span<const block_q8_0> y)
{
    assert(x.size() == y.size());
    constexpr std::size_t h = quant_block_size / 2;
    float sum = 0.0f;
    for (std::size_t i = 0; i < x.size(); ++i) {
        int s0 = 0, s1 = 0;
        for (std::size_t j = 0; j < h; ++j) {
            s0 += ((x[i].qs[j] & 0x0f) - 8) * int(y[i].qs[j]);
            s1 += ((x[i].qs[j] >> 4) - 8) * int(y[i].qs[j + h]);
        }
        const float term = float(s0 + s1) * to_float(x[i].d) * to_float(y[i].d);
        sum += term;
    }
    return sum;
}

/// \brief The dot product of an MXFP4 row with a Q8_0 row, as ggml's `ggml_vec_dot_mxfp4_q8_0`.
/// \param x The weights.
/// \param y The activations.
/// \return `sum over blocks of d_y · 2^(e - 128) · (sum 2v_x · q_y)`.
constexpr float dot(std::span<const block_mxfp4> x, std::span<const block_q8_0> y)
{
    assert(x.size() == y.size());
    constexpr std::size_t h = quant_block_size / 2;
    float sum = 0.0f;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float d = to_float(y[i].d) * detail::quantize::e8m0_half(x[i].e);
        int s1 = 0, s2 = 0;
        for (std::size_t j = 0; j < h; ++j) {
            s1 += int(y[i].qs[j]) * detail::quantize::fp4_doubled[x[i].qs[j] & 0x0f];
            s2 += int(y[i].qs[j + h]) * detail::quantize::fp4_doubled[x[i].qs[j] >> 4];
        }
        const float term = d * float(s1 + s2);
        sum += term;
    }
    return sum;
}

/// \brief Symmetric int8 quantisation of a whole row with one scale: `q = round(x / scale)`, `scale = amax / 127`.
/// \param x The row.
/// \param q Receives `x.size()` signed bytes.
/// \return The scale; 0 for a row of zeros.
template <row X>
    requires std::same_as<row_value_t<X>, float>
constexpr float quantize_int8(const X& x, std::span<std::int8_t> q)
{
    const auto in = as_row(x);
    assert(q.size() == in.size());
    float amax = 0.0f;
    for (const float v : in)
        amax = amax < detail::quantize::abs(v) ? detail::quantize::abs(v) : amax;
    const float scale = amax / 127.0f;
    const float inverse = scale != 0.0f ? 1.0f / scale : 0.0f;
    for (std::size_t i = 0; i < in.size(); ++i) {
        const float scaled = in[i] * inverse;
        q[i] = static_cast<std::int8_t>(detail::quantize::round_to_int(scaled));
    }
    return scale;
}

/// \brief Dequantises a symmetric int8 row: `x = q · scale`.
/// \param q The signed bytes.
/// \param scale What `quantize_int8` returned.
/// \param x Receives `q.size()` floats.
template <mutable_row X>
    requires std::same_as<row_value_t<X>, float>
constexpr void dequantize_int8(std::span<const std::int8_t> q, float scale, X&& x)
{
    const auto out = as_mutable_row(x);
    assert(out.size() == q.size());
    for (std::size_t i = 0; i < q.size(); ++i)
        out[i] = float(q[i]) * scale;
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_quantize_hpp
