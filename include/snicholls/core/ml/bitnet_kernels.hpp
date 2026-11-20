//
//  bitnet_kernels.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` BitNet's packed ternary layouts: I2_S (2 bits a weight, multiply-add kernels) and the lookup-table formats TL1 (pairs of weights, 4-bit indices) and TL2 (triples, 4-bit magnitude and a sign bit), byte for byte as bitnet.cpp writes them, with reference kernels, all `constexpr`.
/// \ingroup core
///
/// **Overview.** bitnet.cpp (microsoft/BitNet 0b341e5) runs ternary weights two ways. **I2_S** stores each weight
/// as `t + 1` in two bits and multiplies by int8 activations; **TL1** and **TL2** replace the multiplications with
/// table lookups: for each pair (TL1) or triple (TL2) of activations, the sums of every ternary combination of them
/// are built once, and each weight group's index picks its sum.
///
/// | Layout | Group | Code | Bits per weight | Source |
/// |---|---|---|---|---|
/// | I2_S | 1 | `t + 1` in {0, 1, 2}; block `b` of `Block` weights in `Block / 4` bytes, weight `j` in byte `j % (Block / 4)`, bits `6 - 2 (j / (Block / 4))` | 2 | `quantize_i2_s`, `src/ggml-bitnet-mad.cpp` |
/// | TL1 | 2 | `3 w0 + w1 + 4` in [0, 8] | 2 | `preprocess_weights_tl1`, `utils/convert-hf-to-gguf-bitnet.py` |
/// | TL2 | 3 | `|9 w0 + 3 w1 + w2|` in [0, 13] and a sign bit (1 when negative) | 5/3 | `preprocess_weights_tl2`, same file |
///
/// `Block` is 128 on x86 and 64 on ARM (`QK_I2_S`); the float scale follows the codes at byte `n / 4`, in a 32-byte
/// tail. The TL layouts also tile the matrix for the SIMD kernels (`lut_tiles`: `BM` rows, `BK` columns, `bm` rows
/// a kernel step, from a model's `kernel_config.ini`); `tl1_pack` and `tl2_pack` write those bytes exactly.
///
/// \snippet ml/bitnet_kernels_demo.cpp kernels
///
/// - `quantize_i2_s(w, out)`: bitnet.cpp's quantiser, which takes a matrix already ternary times a scale (as
///   `absmean_quantize` leaves it): `|w| < 1e-6` is 0, otherwise the sign; the scale stored is max |w|.
///   `pack_i2_s(t, scale, out)` packs codes directly. `i2_s_dot_raw` is `ggml_vec_dot_i2_i8_s_1x1`, the sum of
///   `(t + 1) · a`; `i2_s_dot` subtracts `sum a`, giving the ternary dot product.
/// - `tl1_code` / `tl2_code`, `tl1_lut(a0, a1)` / `tl2_lut(a0, a1, a2)`: the codes, and the tables the kernels build
///   (`lut_ctor` in `utils/codegen_tl1.py`, `three_lut_ctor` in the TL2 kernels): entry `c` is the sum of the
///   combination `tl1_weights[c]` / `tl2_weights[c]`, tables the compiler builds.
/// - `tl1_gemv`, `tl2_gemv`: the lookup-table matrix-vector product over unpacked codes, the kernels' arithmetic in
///   plain order: exactly `ternary_dot` row by row.
/// - `lut_activation_quantize(x, q)`: the kernels' per-tensor activations, `s = 127 / max |x|`, round to nearest
///   even (`vcvtnq`, `_mm256_round_ps`), no clamp; the output is `acc / s · weight_scale`.
///
/// **Relationships.** `ml/bitnet.hpp` (absmean weights, per-token activations, BitLinear, `ternary_dot`);
/// `ml/quant_ternary.hpp` (ggml's TQ1_0 and TQ2_0). Group: \ref core.
///
/// **Implementations.** One each: the reference the SIMD and GPU kernels reorder.
///
/// **Guarantees.** `constexpr`, no allocation, linear. Integer sums are exact (int32). I2_S bytes and
/// `i2_s_dot_raw` equal bitnet.cpp's `quantize_i2_s` and `ggml_vec_dot_i2_i8_s_1x1` on every row of
/// `tests/fixtures/ml/bitnet_i2s_fixture.txt` (built from BitNet's C++, x86); TL1 and TL2 bytes equal the converter's
/// on `tests/fixtures/ml/bitnet_lut_fixture.txt` (its Python, transcribed), for real model tilings; the LUT GEMVs
/// equal the plain ternary product (tested).
///
/// **Errors.** Preconditions, asserted: sizes (`n` a multiple of `Block`; the tiling dividing the matrix as the
/// converter requires: `M % BM`, `K % BK`, `BM % bm`, TL1's `BK % (256 / bm)`, TL2's `bm == 32`). bitnet.cpp's
/// quantiser treats NaN as a negative weight, as here.
///
/// **Performance.** Against plain loops (`benchmarks/ml/bitnet_kernels_benchmark.cpp`); numbers in BENCHMARKS.md.
///
/// **Tests and demos.** `tests/ml/bitnet_kernels_test.cpp`, `demos/ml/bitnet_kernels_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<bit>`, `<cassert>`, `<cstddef>`, `<cstdint>`, `<limits>`, `<span>`, `<type_traits>`,
/// `ml/bitnet.hpp`, `ml/row.hpp`.
///
/// **Known issues.** bitnet.cpp's AVX2 I2_S kernel sums up to 32 blocks in 16-bit lanes before widening, which wraps
/// for adversarial activations (|partial sum| above 32767 in a lane); the reference here is exact, so it agrees with
/// the kernel wherever the kernel does not wrap (every fixture row). The x86 non-`ACT_PARALLEL` I2_S form (four rows
/// interleaved a byte, `quantize_i2_s_1x4`) is not here. The I2_S matrix product's correction by `sum a` and its
/// scaling are done in bitnet.cpp's llama.cpp fork, which the clone does not include: `i2_s_dot` and the
/// `acc / s · scale` epilogue are recalled from it and agree with the GPU kernel's epilogue
/// (`gpu/bitnet_kernels/bitnet_kernels.h`). TL2's padded size is the converter's formula, which assumes `K >= 256`.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_bitnet_kernels_hpp
#define snicholls_core_ml_bitnet_kernels_hpp

#include <snicholls/core/ml/bitnet.hpp>
#include <snicholls/core/ml/row.hpp>

#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief bitnet.cpp's I2_S block on x86 (`QK_I2_S` with AVX2): 128 weights in 32 bytes.
inline constexpr std::size_t i2_s_block_x86 = 128;
/// \brief bitnet.cpp's I2_S block on ARM (`QK_I2_S` with NEON): 64 weights in 16 bytes.
inline constexpr std::size_t i2_s_block_arm = 64;

/// \brief The bytes of an I2_S tensor of `n` weights: the codes and a 32-byte tail whose first four hold the scale.
/// \param n The weights.
/// \return `n / 4 + 32`.
[[nodiscard]] constexpr std::size_t i2_s_bytes(std::size_t n) noexcept { return n / 4 + 32; }

namespace detail::bitnet
{
    constexpr double dabs(double x) { return x < 0 ? -x : x; }
    constexpr double dmax(double a, double b) { return a != a ? b : b != b ? a : a < b ? b : a; }   // C's fmax

    template <std::size_t Block>
    constexpr void set_i2(std::span<std::uint8_t> out, std::size_t j, unsigned code)
    {
        constexpr std::size_t g = Block / 4;
        const std::size_t byte = j / Block * g + j % g;
        const unsigned shift = unsigned(6 - 2 * ((j % Block) / g));
        out[byte] = static_cast<std::uint8_t>(out[byte] | (code << shift));
    }

    constexpr void put_scale(std::span<std::uint8_t> out, std::size_t at, float scale)
    {
        const auto b = std::bit_cast<std::array<std::uint8_t, 4>>(scale);
        for (std::size_t k = 0; k < 4; ++k)
            out[at + k] = b[k];
        for (std::size_t k = 4; k < 32; ++k)
            out[at + k] = 0;
    }
} // namespace detail::bitnet

/// \brief bitnet.cpp's `quantize_i2_s` (`ACT_PARALLEL`): a matrix already ternary times a scale, packed to I2_S.
/// \tparam Block `i2_s_block_x86` or `i2_s_block_arm`.
/// \param w The `n` weights, row-major (`n` a multiple of `Block`).
/// \param out Receives `i2_s_bytes(n)` bytes: codes `|w| < 1e-6 ? 1 : w > 0 ? 2 : 0`, then max |w| as a float and zeros.
/// \return The scale written, max |w|.
template <std::size_t Block = i2_s_block_x86>
    requires(Block == i2_s_block_x86 || Block == i2_s_block_arm)
constexpr float quantize_i2_s(std::span<const float> w, std::span<std::uint8_t> out) noexcept
{
    using namespace detail::bitnet;
    assert(w.size() % Block == 0 && out.size() == i2_s_bytes(w.size()));
    double max = 0;
    for (const float v : w)
        max = dmax(max, dabs(double(v)));
    for (std::size_t i = 0; i < w.size() / 4; ++i)
        out[i] = 0;
    for (std::size_t j = 0; j < w.size(); ++j) {
        const double v = double(w[j]);
        const double p = v * max;
        set_i2<Block>(out, j, dabs(v) < 1e-6 ? 1u : p > 0 ? 2u : 0u);
    }
    put_scale(out, w.size() / 4, float(max));
    return float(max);
}

/// \brief Packs ternary codes to I2_S with a given scale (bitnet.cpp's layout, its converter's `quantize_to_i2_s`).
/// \tparam Block `i2_s_block_x86` or `i2_s_block_arm`.
/// \param t The `n` codes in {-1, 0, 1}.
/// \param scale The scale, `γ`.
/// \param out Receives `i2_s_bytes(n)` bytes.
template <std::size_t Block = i2_s_block_x86>
    requires(Block == i2_s_block_x86 || Block == i2_s_block_arm)
constexpr void pack_i2_s(std::span<const std::int8_t> t, float scale, std::span<std::uint8_t> out) noexcept
{
    assert(t.size() % Block == 0 && out.size() == i2_s_bytes(t.size()));
    for (std::size_t i = 0; i < t.size() / 4; ++i)
        out[i] = 0;
    for (std::size_t j = 0; j < t.size(); ++j)
        detail::bitnet::set_i2<Block>(out, j, unsigned(t[j] + 1));
    detail::bitnet::put_scale(out, t.size() / 4, scale);
}

/// \brief Weight `j` of an I2_S tensor, in {-1, 0, 1} (a stored code 3, never written, reads as 2).
/// \tparam Block `i2_s_block_x86` or `i2_s_block_arm`.
/// \param packed The codes.
/// \param j The index.
/// \return `code - 1`.
template <std::size_t Block = i2_s_block_x86>
[[nodiscard]] constexpr int i2_s_at(std::span<const std::uint8_t> packed, std::size_t j) noexcept
{
    constexpr std::size_t g = Block / 4;
    return int((packed[j / Block * g + j % g] >> (6 - 2 * ((j % Block) / g))) & 3) - 1;
}

/// \brief The scale of an I2_S tensor of `n` weights: the float after the codes.
/// \param packed The `i2_s_bytes(n)` bytes.
/// \param n The weights.
/// \return The scale.
[[nodiscard]] constexpr float i2_s_scale(std::span<const std::uint8_t> packed, std::size_t n) noexcept
{
    std::array<std::uint8_t, 4> b{};
    for (std::size_t k = 0; k < 4; ++k)
        b[k] = packed[n / 4 + k];
    return std::bit_cast<float>(b);
}

/// \brief `ggml_vec_dot_i2_i8_s_1x1`: the sum of `(t + 1) · a` over an I2_S row, exact.
/// \tparam Block `i2_s_block_x86` or `i2_s_block_arm`.
/// \param packed The row's codes (`a.size() / 4` bytes at least).
/// \param a The int8 activations.
/// \return The sum.
template <std::size_t Block = i2_s_block_x86>
[[nodiscard]] constexpr std::int32_t i2_s_dot_raw(std::span<const std::uint8_t> packed, std::span<const std::int8_t> a) noexcept
{
    assert(a.size() % Block == 0 && packed.size() >= a.size() / 4);
    constexpr std::size_t g = Block / 4;
    std::int32_t s = 0;
    for (std::size_t b = 0; b < a.size() / Block; ++b)
        for (std::size_t k = 0; k < 4; ++k)
            for (std::size_t j = 0; j < g; ++j)
                s += std::int32_t((packed[b * g + j] >> (6 - 2 * k)) & 3) * std::int32_t(a[b * Block + k * g + j]);
    return s;
}

/// \brief The ternary dot product of an I2_S row and int8 activations: `i2_s_dot_raw - sum a`.
/// \tparam Block `i2_s_block_x86` or `i2_s_block_arm`.
/// \param packed The row's codes.
/// \param a The int8 activations.
/// \return `sum t · a`.
template <std::size_t Block = i2_s_block_x86>
[[nodiscard]] constexpr std::int32_t i2_s_dot(std::span<const std::uint8_t> packed, std::span<const std::int8_t> a) noexcept
{
    std::int32_t sum = 0;
    for (const std::int8_t v : a)
        sum += v;
    return i2_s_dot_raw<Block>(packed, a) - sum;
}

/// \brief TL1's weight pairs: code `c` is the pair `(c / 3 - 1, c % 3 - 1)`, `c = 3 w0 + w1 + 4`.
inline constexpr std::array<std::array<std::int8_t, 2>, 9> tl1_weights = [] {
    std::array<std::array<std::int8_t, 2>, 9> t{};
    for (int c = 0; c < 9; ++c)
        t[std::size_t(c)] = {static_cast<std::int8_t>(c / 3 - 1), static_cast<std::int8_t>(c % 3 - 1)};
    return t;
}();

/// \brief TL2's weight triples: magnitude `m` is the triple with `9 w0 + 3 w1 + w2 = m` (balanced ternary), for m in [0, 13].
inline constexpr std::array<std::array<std::int8_t, 3>, 14> tl2_weights = [] {
    std::array<std::array<std::int8_t, 3>, 14> t{};
    for (int w0 = -1; w0 <= 1; ++w0)
        for (int w1 = -1; w1 <= 1; ++w1)
            for (int w2 = -1; w2 <= 1; ++w2) {
                const int v = 9 * w0 + 3 * w1 + w2;
                if (v >= 0)
                    t[std::size_t(v)] = {static_cast<std::int8_t>(w0), static_cast<std::int8_t>(w1), static_cast<std::int8_t>(w2)};
            }
    return t;
}();

static_assert(tl2_weights[13] == std::array<std::int8_t, 3>{1, 1, 1} && tl2_weights[5] == std::array<std::int8_t, 3>{1, -1, -1}
                  && tl1_weights[0] == std::array<std::int8_t, 2>{-1, -1} && tl1_weights[8] == std::array<std::int8_t, 2>{1, 1},
              "snicholls::ml: the TL tables must be BitNet's (lut_ctor, three_lut_ctor)");

/// \brief TL1's code of a weight pair: `3 w0 + w1 + 4`.
/// \param w0 The first weight, in {-1, 0, 1}.
/// \param w1 The second.
/// \return The code, in [0, 8].
[[nodiscard]] constexpr std::uint8_t tl1_code(int w0, int w1) noexcept { return static_cast<std::uint8_t>(3 * w0 + w1 + 4); }

/// \brief TL2's code of a weight triple: `|9 w0 + 3 w1 + w2|` and whether that sum is negative.
/// \ingroup core
struct tl2_code_t
{
    std::uint8_t magnitude = 0;  ///< In [0, 13].
    bool negative = false;       ///< The sign bit.
};

/// \brief TL2's code of a weight triple.
/// \param w0 The first weight, in {-1, 0, 1}.
/// \param w1 The second.
/// \param w2 The third.
/// \return Its magnitude and sign.
[[nodiscard]] constexpr tl2_code_t tl2_code(int w0, int w1, int w2) noexcept
{
    const int v = 9 * w0 + 3 * w1 + w2;
    return {static_cast<std::uint8_t>(v < 0 ? -v : v), v < 0};
}

/// \brief The TL1 table of an activation pair: entry `c` is `w0 · a0 + w1 · a1` for `tl1_weights[c]`; 9 to 15 are 0.
/// \param a0 The first activation code.
/// \param a1 The second.
/// \return The 16 entries.
[[nodiscard]] constexpr std::array<std::int16_t, 16> tl1_lut(int a0, int a1) noexcept
{
    std::array<std::int16_t, 16> lut{};
    for (std::size_t c = 0; c < 9; ++c)
        lut[c] = static_cast<std::int16_t>(tl1_weights[c][0] * a0 + tl1_weights[c][1] * a1);
    return lut;
}

/// \brief The TL2 table of an activation triple: entry `m` is `w0 · a0 + w1 · a1 + w2 · a2` for `tl2_weights[m]`; 14 and 15 are 0.
/// \param a0 The first activation code.
/// \param a1 The second.
/// \param a2 The third.
/// \return The 16 entries.
[[nodiscard]] constexpr std::array<std::int16_t, 16> tl2_lut(int a0, int a1, int a2) noexcept
{
    std::array<std::int16_t, 16> lut{};
    for (std::size_t m = 0; m < 14; ++m)
        lut[m] = static_cast<std::int16_t>(tl2_weights[m][0] * a0 + tl2_weights[m][1] * a1 + tl2_weights[m][2] * a2);
    return lut;
}

/// \brief The TL kernels' activations: one scale for the tensor, `s = 127 / max |x|`, `q = round(x · s)` to nearest even, no clamp.
/// \param x The activations.
/// \param q Receives `x.size()` codes in [-127, 127].
/// \return `s`; a row of zeros gives infinity and codes 0.
template <row X>
    requires std::same_as<row_value_t<X>, float>
constexpr float lut_activation_quantize(const X& x, std::span<std::int8_t> q) noexcept
{
    const auto in = as_row(x);
    assert(q.size() == in.size());
    float amax = 0.0f;
    for (const float v : in)
        amax = detail::quantize::fmax(detail::quantize::abs(v), amax);
    const float s = amax > 0.0f ? 127.0f / amax : std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < in.size(); ++i) {
        const float v = in[i] * s;
        q[i] = static_cast<std::int8_t>(detail::bitnet::round_clamp(v, -127, 127));
    }
    return s;
}

/// \brief TL1 codes of a row-major ternary matrix: pair `(t[2i], t[2i + 1])` to code `i`.
/// \param t The `rows · cols` codes in {-1, 0, 1} (`cols` even).
/// \param codes Receives `t.size() / 2` codes.
constexpr void tl1_encode(std::span<const std::int8_t> t, std::span<std::uint8_t> codes) noexcept
{
    assert(t.size() % 2 == 0 && codes.size() == t.size() / 2);
    for (std::size_t i = 0; i < codes.size(); ++i)
        codes[i] = tl1_code(t[2 * i], t[2 * i + 1]);
}

/// \brief TL2 codes of a row-major ternary matrix: triple `(t[3i], t[3i + 1], t[3i + 2])` to code `i`.
/// \param t The codes in {-1, 0, 1} (rows of a multiple of 3).
/// \param codes Receives `t.size() / 3` codes.
constexpr void tl2_encode(std::span<const std::int8_t> t, std::span<tl2_code_t> codes) noexcept
{
    assert(t.size() % 3 == 0 && codes.size() == t.size() / 3);
    for (std::size_t i = 0; i < codes.size(); ++i)
        codes[i] = tl2_code(t[3 * i], t[3 * i + 1], t[3 * i + 2]);
}

/// \brief The TL1 lookup-table matrix-vector product: tables for each activation pair, then one lookup per weight pair.
/// \param codes `rows · cols / 2` TL1 codes, row-major.
/// \param rows The outputs.
/// \param cols The inputs (even).
/// \param a `cols` activation codes.
/// \param luts Scratch for `cols / 2` tables.
/// \param y Receives `rows` sums, exactly `ternary_dot` of each row.
constexpr void tl1_gemv(std::span<const std::uint8_t> codes, std::size_t rows, std::size_t cols, std::span<const std::int8_t> a,
                        std::span<std::array<std::int16_t, 16>> luts, std::span<std::int32_t> y) noexcept
{
    assert(cols % 2 == 0 && codes.size() == rows * cols / 2 && a.size() == cols && luts.size() == cols / 2 && y.size() == rows);
    for (std::size_t p = 0; p < cols / 2; ++p)
        luts[p] = tl1_lut(a[2 * p], a[2 * p + 1]);
    for (std::size_t i = 0; i < rows; ++i) {
        std::int32_t s = 0;
        for (std::size_t p = 0; p < cols / 2; ++p)
            s += luts[p][codes[i * cols / 2 + p]];
        y[i] = s;
    }
}

/// \brief The TL2 lookup-table matrix-vector product: tables for each activation triple, one lookup and a sign per weight triple.
/// \param codes `rows · cols / 3` TL2 codes, row-major.
/// \param rows The outputs.
/// \param cols The inputs (a multiple of 3).
/// \param a `cols` activation codes.
/// \param luts Scratch for `cols / 3` tables.
/// \param y Receives `rows` sums, exactly `ternary_dot` of each row.
constexpr void tl2_gemv(std::span<const tl2_code_t> codes, std::size_t rows, std::size_t cols, std::span<const std::int8_t> a,
                        std::span<std::array<std::int16_t, 16>> luts, std::span<std::int32_t> y) noexcept
{
    assert(cols % 3 == 0 && codes.size() == rows * cols / 3 && a.size() == cols && luts.size() == cols / 3 && y.size() == rows);
    for (std::size_t p = 0; p < cols / 3; ++p)
        luts[p] = tl2_lut(a[3 * p], a[3 * p + 1], a[3 * p + 2]);
    for (std::size_t i = 0; i < rows; ++i) {
        std::int32_t s = 0;
        for (std::size_t p = 0; p < cols / 3; ++p) {
            const tl2_code_t c = codes[i * cols / 3 + p];
            const std::int32_t v = luts[p][c.magnitude];
            s += c.negative ? -v : v;
        }
        y[i] = s;
    }
}

/// \brief The tiling of a TL1 or TL2 matrix for bitnet.cpp's kernels, as a model's `kernel_config.ini` gives it.
/// \ingroup core
struct lut_tiles
{
    std::size_t BM = 0;   ///< Rows a tile (`bm` in the ini).
    std::size_t BK = 0;   ///< Columns a tile (`bk`).
    std::size_t bm = 0;   ///< Rows a kernel step (`bmm`).
};

namespace detail::bitnet
{
    // One numpy step, a reshape to `shape` then a transpose by `perm`; source_of maps a flat index of the result of
    // the whole chain back to the flat index it came from
    struct step
    {
        std::size_t n = 0;
        std::array<std::size_t, 8> shape{};
        std::array<std::size_t, 8> perm{};
    };

    template <std::size_t S>
    constexpr std::size_t source_of(std::size_t o, const std::array<step, S>& steps)
    {
        for (std::size_t s = S; s-- > 0;) {
            const step& st = steps[s];
            std::array<std::size_t, 8> idx_out{}, idx_in{};
            for (std::size_t k = st.n; k-- > 0;) {   // decompose o by the transposed shape
                const std::size_t extent = st.shape[st.perm[k]];
                idx_out[k] = o % extent;
                o /= extent;
            }
            for (std::size_t k = 0; k < st.n; ++k)
                idx_in[st.perm[k]] = idx_out[k];
            o = 0;
            for (std::size_t k = 0; k < st.n; ++k)
                o = o * st.shape[k] + idx_in[k];
        }
        return o;
    }

    // The four tiling steps the TL converters share, for groups of g weights (by the kernel's group of columns)
    constexpr std::array<step, 4> tl_tiling(std::size_t M, std::size_t K, std::size_t g, std::size_t BM, std::size_t BY, std::size_t bm,
                                            std::size_t by)
    {
        return {step{3, {M / BM, BM, K / g}, {0, 2, 1}}, step{4, {M / BM, K / BY, BY / g, BM}, {0, 1, 3, 2}},
                step{5, {M / BM, K / BY, BM / bm, bm, BY / g}, {0, 1, 2, 4, 3}},
                step{6, {M / BM, K / BY, BM / bm, BY / by, by / g, bm}, {0, 1, 2, 3, 5, 4}}};
    }

    // TL2's byte rows of bm: groups of 8 bytes 1 and 2 swapped (weight[:, [0, 1, 2, 3], :] = weight[:, [0, 2, 1, 3], :])
    constexpr std::size_t swap_groups(std::size_t o)
    {
        constexpr std::array<std::size_t, 4> perm{0, 2, 1, 3};
        return o / 32 * 32 + perm[o % 32 / 8] * 8 + o % 8;
    }

    // The two-weight (TL1-coded) part of a TL2 matrix: BY = 32, bm = 32, by = 4, as preprocess_two_weights_tl2
    constexpr void tl2_two(std::span<const std::uint8_t> codes, std::size_t M, std::size_t K, std::size_t BM, std::span<std::uint8_t> out)
    {
        const auto steps = tl_tiling(M, K, 2, BM, 32, 32, 4);
        for (std::size_t o = 0; o < out.size(); ++o) {
            const std::size_t b = swap_groups(o);
            out[o] = static_cast<std::uint8_t>((codes[source_of(2 * b, steps)] << 4) + codes[source_of(2 * b + 1, steps)]);
        }
    }
} // namespace detail::bitnet

/// \brief The bytes of a TL1 matrix: two codes a byte.
/// \param M The rows.
/// \param K The columns.
/// \return `M · K / 4`.
[[nodiscard]] constexpr std::size_t tl1_packed_bytes(std::size_t M, std::size_t K) noexcept { return M * K / 4; }

/// \brief TL1's tiled bytes of a ternary matrix, as BitNet's converter writes them (`preprocess_weights_tl1`, `process_tl1`).
/// \param t The `M · K` codes in {-1, 0, 1}, row-major.
/// \param M The rows.
/// \param K The columns.
/// \param tiles The kernel's tiling.
/// \param out Receives `tl1_packed_bytes(M, K)` bytes.
constexpr void tl1_pack(std::span<const std::int8_t> t, std::size_t M, std::size_t K, lut_tiles tiles, std::span<std::uint8_t> out) noexcept
{
    using namespace detail::bitnet;
    const std::size_t BM = tiles.BM, BY = tiles.BK, bm = tiles.bm, by = 256 / bm;
    assert(t.size() == M * K && out.size() == tl1_packed_bytes(M, K) && M % BM == 0 && K % BY == 0 && BM % bm == 0 && BY % by == 0
           && bm % 16 == 0 && by % 4 == 0);
    const auto tiling = tl_tiling(M, K, 2, BM, BY, bm, by);
    const std::array<detail::bitnet::step, 6> steps{tiling[0], tiling[1], tiling[2], tiling[3],
                                    detail::bitnet::step{7, {M / BM, K / BY, BM / bm, BY / by, bm / 16, 16, by / 2}, {0, 1, 2, 3, 4, 6, 5}},
                                    detail::bitnet::step{8, {M / BM, K / BY, BM / bm, BY / by, bm / 16, by / 4, 2, 16}, {0, 1, 2, 3, 4, 5, 7, 6}}};
    const auto code = [&](std::size_t pair) { return tl1_code(t[2 * pair], t[2 * pair + 1]); };
    for (std::size_t o = 0; o < out.size(); ++o)
        out[o] = static_cast<std::uint8_t>((code(source_of(2 * o, steps)) << 4) + code(source_of(2 * o + 1, steps)));
}

/// \brief The bytes of a TL2 matrix as the converter pads it: `(K - 256) · M / 3 · 5 / 8 + 256 · M / 2 · 4 / 8`, in Python's integer order.
/// \param M The rows.
/// \param K The columns, at least 256.
/// \return The size.
[[nodiscard]] constexpr std::size_t tl2_packed_bytes(std::size_t M, std::size_t K) noexcept
{
    assert(K >= 256);
    return (K - 256) * M / 3 * 5 / 8 + 256 * M / 2 * 4 / 8;
}

/// \brief TL2's tiled bytes of a ternary matrix, as BitNet's converter writes them (`preprocess_weights_tl2`): the
/// magnitudes of the first `K - K % BK` columns' triples, their sign bits, then the last `K % BK` columns as TL1
/// pairs (tiled with `BK = 32`, `bm = 32`), then zeros to `tl2_packed_bytes`.
/// \param t The `M · K` codes in {-1, 0, 1}, row-major.
/// \param M The rows.
/// \param K The columns.
/// \param tiles The kernel's tiling (`bm` 32).
/// \param out Receives `tl2_packed_bytes(M, K)` bytes.
/// \param scratch `M · (K % BK) / 2` bytes for the tail's codes (empty when `K % BK == 0`).
constexpr void tl2_pack(std::span<const std::int8_t> t, std::size_t M, std::size_t K, lut_tiles tiles, std::span<std::uint8_t> out,
                        std::span<std::uint8_t> scratch = {}) noexcept
{
    using namespace detail::bitnet;
    const std::size_t BM = tiles.BM, BY = tiles.BK, bm = tiles.bm, by = 192 / bm;
    const std::size_t K3 = K - K % BY, K2 = K % BY;
    assert(t.size() == M * K && out.size() == tl2_packed_bytes(M, K) && bm == 32 && M % BM == 0 && BM % bm == 0 && BY % (4 * by) == 0
           && K2 % 32 == 0 && scratch.size() == M * K2 / 2);
    const auto triple = [&](std::size_t i) {   // triple i of the first K3 columns, row-major
        const std::size_t r = i / (K3 / 3), c = 3 * (i % (K3 / 3));
        const std::int8_t* w = &t[r * K + c];
        return tl2_code(w[0], w[1], w[2]);
    };
    // magnitudes: two triples a byte, rows of bm bytes with groups swapped
    const auto mags = tl_tiling(M, K3, 3, BM, BY, bm, by);
    const std::size_t n_mag = M * K3 / 6;
    for (std::size_t o = 0; o < n_mag; ++o) {
        const std::size_t b = swap_groups(o);
        out[o] = static_cast<std::uint8_t>((triple(source_of(2 * b, mags)).magnitude << 4) + triple(source_of(2 * b + 1, mags)).magnitude);
    }
    // signs: 16 bits a uint16, written little-endian
    const std::array<detail::bitnet::step, 5> signs{mags[0], mags[1], mags[2],
                                    detail::bitnet::step{6, {M / BM, K3 / BY, BM / bm, BY / (by * 4), by / 3 * 4, bm}, {0, 1, 2, 3, 5, 4}},
                                    detail::bitnet::step{6, {M / BM, K3 / BY, BM / bm, BY / (by * 4), bm, by / 3 * 4}, {0, 1, 2, 3, 5, 4}}};
    const std::size_t n_sign = M * K3 / 3 / 8;
    for (std::size_t u = 0; u < n_sign / 2; ++u) {
        const std::size_t outer = u / (bm / 2), c = u % (bm / 2);
        unsigned word = 0;
        for (std::size_t i = 0; i < 16; ++i)
            word |= unsigned(triple(source_of(outer * 16 * (bm / 2) + i * (bm / 2) + c, signs)).negative) << (15 - i);
        out[n_mag + 2 * u] = static_cast<std::uint8_t>(word & 0xff);
        out[n_mag + 2 * u + 1] = static_cast<std::uint8_t>(word >> 8);
    }
    std::size_t end = n_mag + n_sign;
    if (K2 != 0) {
        for (std::size_t r = 0; r < M; ++r)
            for (std::size_t p = 0; p < K2 / 2; ++p)
                scratch[r * K2 / 2 + p] = tl1_code(t[r * K + K3 + 2 * p], t[r * K + K3 + 2 * p + 1]);
        tl2_two(scratch, M, K2, BM, out.subspan(end, M * K2 / 4));
        end += M * K2 / 4;
    }
    assert(end <= out.size());
    for (std::size_t o = end; o < out.size(); ++o)
        out[o] = 0;
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_bitnet_kernels_hpp
