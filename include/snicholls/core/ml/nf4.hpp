//
//  nf4.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` QLoRA's 4-bit NormalFloat weights with double quantisation: quantise, dequantise, and products that dequantise on the fly.
/// \ingroup core
///
/// **Overview.** QLoRA (Dettmers et al. 2023) fine-tunes a model whose frozen base weights are stored in 4 bits and
/// read back to the compute type as they are used, with LoRA adapters trained in full precision beside them
/// (`ml/lora.hpp`; TRAINING.md T11). Its storage, `nf4_weights`, is three levels:
///
/// - **NormalFloat4 codes**, two a byte (the first of a pair in the high nibble). Each code indexes `nf4_values`, the
///   sixteen values of the paper's appendix E: the quantiles of a standard normal, asymmetric so that 0 is exact
///   (its eq. 4), normalised to [-1, 1]. A block of 64 weights shares one scale, its absolute maximum `c2`, and a
///   weight is `nf4_values[code] · c2`. A weight's code is the value nearest `w / c2`, ties to the lower code.
/// - **Double quantisation** of the block scales (the paper's §3): their mean is subtracted, and each group of 256
///   block scales is stored as 8-bit floats (OCP E4M3, `float8_e4m3`) of `(c2 - mean) / s1`, `s1 = max|c2 - mean| /
///   448` the group's float scale. The scales cost 8.5 bits a block of 64 instead of 32, about 0.13 bits a weight.
/// - The mean, one float.
///
/// `linear(nf4, x, y, bias)`, `linear_backward_input` and the sequence forms dequantise each weight as they read it,
/// in `float`, and add the products in the order of a dense `linear`: a product through `nf4_weights` equals `linear`
/// through the dense matrix `nf4_dequantize` gives, to the bit. There is no `linear_backward` into NF4 weights: they are
/// frozen (TRAINING.md T4).
///
/// \snippet ml/nf4_demo.cpp nf4
///
/// **Relationships.** `ml/lora.hpp` (the adapters over a frozen base), `ml/linear.hpp` (the same `prepare_input` /
/// `linear` seam), `ml/quantize.hpp` (ggml's formats, for inference), `ml/minifloat.hpp` (`float8_e4m3`). Group: \ref
/// core.
///
/// **Implementations.** One, on the host. A device library overloads `linear_sequence` and
/// `linear_sequence_backward_input` for its backend (TRAINING.md T9).
///
/// **Guarantees.** `constexpr`, no allocation, `noexcept`. Quantising is linear in the weights (three passes);
/// a product is one multiply and one add a weight, plus one scale a block.
///
/// **Errors.** Preconditions, asserted: `cols` is a multiple of 64 (a block never crosses a row); the spans have the
/// sizes `nf4_code_bytes`, `nf4_block_count` and `nf4_group_count` give.
///
/// **Definitions.** `nf4_values` checked digit for digit against the paper's appendix E (arXiv 2305.14314), and
/// reproduced in the tests from the normal quantiles by bitsandbytes' construction (recalled: offset 0.9677083, 8 and
/// 9 quantiles, normalised). The block of 64, the double quantisation to 8-bit floats in groups of 256 after
/// subtracting the mean: checked against the paper. Not checked, and not bitsandbytes' bytes: bitsandbytes stores the
/// second level in its own dynamic 8-bit code, where core uses E4M3, and its nibble order is recalled.
///
/// **Performance.** `linear` through NF4 against the same loop by hand over the same bytes, after a bit-for-bit check
/// (`benchmarks/ml/nf4_benchmark.cpp`), 1024 x 1024, one row: 2.5 ms against 2.7 ms, level; the dense product of the
/// dequantised matrix takes 1.2 ms, so dequantising on the fly costs about twice the product on the host
/// (Xeon W-3245, -O2, Apple clang 17, 2026-11-19, one thread, CPU time, load average about 100: a record, not a quiet-machine figure). It saves memory, not time; a device dequantises in registers.
///
/// **Tests and demos.** `tests/ml/nf4_test.cpp`, `demos/ml/nf4_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cassert>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<ranges>`, `<span>`, `ml/minifloat.hpp`, `ml/row.hpp`.
///
/// **Known issues.** The products are in `float`, as QLoRA computes in a 16-bit type; a `bfloat16` compute path is the
/// caller's, after `nf4_dequantize`.
///
/// **Status.** Experimental, new in Unreleased.

#ifndef snicholls_core_ml_nf4_hpp
#define snicholls_core_ml_nf4_hpp

#include <snicholls/core/ml/minifloat.hpp>
#include <snicholls/core/ml/row.hpp>

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <span>

namespace snicholls::ml
{

/// \brief The sixteen NormalFloat4 values, by code (QLoRA, appendix E).
inline constexpr std::array<float, 16> nf4_values{-1.0f,
                                                  -0.6961928009986877f,
                                                  -0.5250730514526367f,
                                                  -0.39491748809814453f,
                                                  -0.28444138169288635f,
                                                  -0.18477343022823334f,
                                                  -0.09105003625154495f,
                                                  0.0f,
                                                  0.07958029955625534f,
                                                  0.16093020141124725f,
                                                  0.24611230194568634f,
                                                  0.33791524171829224f,
                                                  0.44070982933044434f,
                                                  0.5626170039176941f,
                                                  0.7229568362236023f,
                                                  1.0f};

/// \brief The weights sharing one NF4 scale.
inline constexpr std::size_t nf4_block_size = 64;
/// \brief The block scales sharing one second-level scale.
inline constexpr std::size_t nf4_group_blocks = 256;

/// \brief The bytes of codes for `n` weights. \param n The weights. \return `n / 2`.
constexpr std::size_t nf4_code_bytes(std::size_t n) noexcept { return n / 2; }
/// \brief The blocks of `n` weights. \param n The weights, a multiple of 64. \return `n / 64`.
constexpr std::size_t nf4_block_count(std::size_t n) noexcept { return n / nf4_block_size; }
/// \brief The second-level groups of `n` weights. \param n The weights. \return `ceil(blocks / 256)`.
constexpr std::size_t nf4_group_count(std::size_t n) noexcept { return (nf4_block_count(n) + nf4_group_blocks - 1) / nf4_group_blocks; }

/// \brief The NF4 code nearest a normalised value: ties to the lower code.
/// \param x A value in [-1, 1] (outside, the nearest end). \return The code, 0 to 15.
constexpr std::uint8_t nf4_code(float x) noexcept
{
    std::uint8_t best = 0;
    float best_d = x - nf4_values[0] < 0 ? nf4_values[0] - x : x - nf4_values[0];
    for (std::uint8_t c = 1; c < 16; ++c) {
        const float d = x - nf4_values[c] < 0 ? nf4_values[c] - x : x - nf4_values[c];
        if (d < best_d) {
            best_d = d;
            best = c;
        }
    }
    return best;
}

/// \brief A frozen weight matrix in NF4 with double-quantised scales, over the caller's memory: `rows x cols`, row by row.
/// \ingroup core
struct nf4_weights
{
    std::span<const std::uint8_t> codes{};          ///< `rows · cols / 2` bytes, the first weight of a pair in the high nibble.
    std::span<const float8_e4m3> block_scales{};    ///< `rows · cols / 64`: `(c2 - mean) / s1` per block.
    std::span<const float> group_scales{};          ///< `s1` per group of 256 blocks.
    float scale_mean = 0.0f;                        ///< The mean of the blocks' absolute maxima.
    std::size_t rows = 0;                           ///< The outputs.
    std::size_t cols = 0;                           ///< The inputs: a multiple of 64.

    /// \brief A block's dequantised scale `c2`. \param block The block, below `rows · cols / 64`. \return The scale.
    [[nodiscard]] constexpr float block_scale(std::size_t block) const noexcept
    {
        return decode_table<float8_e4m3>[block_scales[block].bits] * group_scales[block / nf4_group_blocks] + scale_mean;
    }

    /// \brief A weight's code. \param index The weight's index, row by row. \return 0 to 15.
    [[nodiscard]] constexpr std::uint8_t code(std::size_t index) const noexcept
    {
        const std::uint8_t byte = codes[index / 2];
        return static_cast<std::uint8_t>(index % 2 == 0 ? byte >> 4 : byte & 0x0f);
    }

    /// \brief A weight, dequantised. \param i The row. \param j The column. \return `nf4_values[code] · c2`.
    [[nodiscard]] constexpr float at(std::size_t i, std::size_t j) const noexcept
    {
        const std::size_t index = i * cols + j;
        return nf4_values[code(index)] * block_scale(index / nf4_block_size);
    }
};

/// \brief Quantises a matrix to NF4 with double quantisation: codes from each block's absolute maximum, then the
/// maxima, less their mean, to E4M3 in groups of 256.
/// \param w `n` weights, a multiple of 64. \param codes Receives `n / 2` bytes. \param block_scales Receives `n / 64`.
/// \param group_scales Receives `nf4_group_count(n)`. \return The mean of the maxima, `nf4_weights::scale_mean`.
template <row W>
    requires std::same_as<row_value_t<W>, float>
constexpr float nf4_quantize(const W& w, std::span<std::uint8_t> codes, std::span<float8_e4m3> block_scales,
                             std::span<float> group_scales) noexcept
{
    const auto x = as_row(w);
    const std::size_t n = x.size(), blocks = nf4_block_count(n);
    assert(n % nf4_block_size == 0 && codes.size() == nf4_code_bytes(n) && block_scales.size() == blocks
           && group_scales.size() == nf4_group_count(n));
    const auto absmax = [&](std::size_t b) {
        float m = 0.0f;
        for (std::size_t j = b * nf4_block_size; j < (b + 1) * nf4_block_size; ++j)
            m = (x[j] < 0 ? -x[j] : x[j]) > m ? (x[j] < 0 ? -x[j] : x[j]) : m;
        return m;
    };
    double sum = 0.0;
    for (std::size_t b = 0; b < blocks; ++b) {
        const float c2 = absmax(b);
        sum += static_cast<double>(c2);
        for (std::size_t j = b * nf4_block_size; j < (b + 1) * nf4_block_size; j += 2) {
            const auto hi = c2 == 0.0f ? std::uint8_t{7} : nf4_code(x[j] / c2);
            const auto lo = c2 == 0.0f ? std::uint8_t{7} : nf4_code(x[j + 1] / c2);
            codes[j / 2] = static_cast<std::uint8_t>((hi << 4) | lo);
        }
    }
    const float mean = blocks == 0 ? 0.0f : static_cast<float>(sum / static_cast<double>(blocks));
    for (std::size_t g = 0; g < group_scales.size(); ++g) {
        const std::size_t end = (g + 1) * nf4_group_blocks < blocks ? (g + 1) * nf4_group_blocks : blocks;
        float m = 0.0f;
        for (std::size_t b = g * nf4_group_blocks; b < end; ++b) {
            const float d = absmax(b) - mean;
            m = (d < 0 ? -d : d) > m ? (d < 0 ? -d : d) : m;
        }
        const float s1 = m / 448.0f;
        group_scales[g] = s1;
        for (std::size_t b = g * nf4_group_blocks; b < end; ++b)
            block_scales[b] = s1 == 0.0f ? float8_e4m3::from_bits(0) : to_minifloat<float8_e4m3>(static_cast<double>((absmax(b) - mean) / s1));
    }
    return mean;
}

/// \brief Dequantises NF4 weights into floats. \param w The matrix. \param out Receives `rows · cols`.
constexpr void nf4_dequantize(const nf4_weights& w, std::span<float> out) noexcept
{
    assert(out.size() == w.rows * w.cols);
    for (std::size_t i = 0; i < w.rows; ++i)
        for (std::size_t j = 0; j < w.cols; ++j)
            out[i * w.cols + j] = w.at(i, j);
}

/// \brief The input of an NF4 product: `x` itself, as for dense weights. \tparam N The input's size.
/// \param x The input. \return A view of it.
template <std::size_t N, row X>
    requires std::same_as<row_value_t<X>, float>
constexpr std::span<const float> prepare_input(const nf4_weights&, const X& x) noexcept
{
    assert(std::ranges::size(x) == N);
    return as_row(x);
}

/// \brief `y = W x + b` with NF4 weights, each weight dequantised as it is read: each output starts at its bias and
/// adds `w_ij · x_j` in order, so the result is the dense product of the dequantised matrix to the bit.
/// \param w The matrix. \param x The input, `cols`. \param y The output, `rows`. \param bias `rows`, or empty.
template <row X, mutable_row Y, row B = std::span<const float>>
    requires std::same_as<row_value_t<X>, float> && std::same_as<row_value_t<Y>, float> && std::same_as<row_value_t<B>, float>
constexpr void linear(const nf4_weights& w, const X& x, Y&& y, const B& bias = B{}) noexcept
{
    const auto in = as_row(x);
    const auto out = as_mutable_row(y);
    const auto b = as_row(bias);
    assert(in.size() == w.cols && out.size() == w.rows && w.cols % nf4_block_size == 0 && (b.empty() || b.size() == w.rows));
    for (std::size_t i = 0; i < w.rows; ++i) {
        float acc = b.empty() ? 0.0f : b[i];
        for (std::size_t j0 = 0; j0 < w.cols; j0 += nf4_block_size) {
            const std::size_t base = i * w.cols + j0;
            const float scale = w.block_scale(base / nf4_block_size);
            for (std::size_t j = 0; j < nf4_block_size; ++j) {
                const float wij = nf4_values[w.code(base + j)] * scale;
                const float p = wij * in[j0 + j];
                acc += p;
            }
        }
        out[i] = acc;
    }
}

/// \brief The input's gradient through frozen NF4 weights: `dx += Wᵀ dy`, for each output `i` in order, `dx_j += w_ij · dy_i`.
/// \param w The matrix. \param dy The output's gradient, `rows`. \param dx Added into, `cols`.
constexpr void linear_backward_input(const nf4_weights& w, std::span<const float> dy, std::span<float> dx) noexcept
{
    assert(dy.size() == w.rows && dx.size() == w.cols);
    for (std::size_t i = 0; i < w.rows; ++i)
        for (std::size_t j = 0; j < w.cols; ++j) {
            const float p = w.at(i, j) * dy[i];
            dx[j] += p;
        }
}

/// \brief `linear` through NF4 over `n` rows, on the backend: on the host, row by row. \param w The matrix.
/// \param xs `n · cols`. \param ys Receives `n · rows`. \param bias `rows`, or empty.
template <typename B>
constexpr void linear_sequence(const B&, const nf4_weights& w, std::span<const float> xs, std::span<float> ys, std::span<const float> bias = {}) noexcept
{
    assert(w.cols > 0 && xs.size() % w.cols == 0 && ys.size() == xs.size() / w.cols * w.rows);
    for (std::size_t t = 0; t < xs.size() / w.cols; ++t)
        linear(w, xs.subspan(t * w.cols, w.cols), ys.subspan(t * w.rows, w.rows), bias);
}

/// \brief `linear_backward_input` over `n` rows, on the backend. \param w The matrix. \param dys `n · rows`.
/// \param dxs Added into, `n · cols`.
template <typename B>
constexpr void linear_sequence_backward_input(const B&, const nf4_weights& w, std::span<const float> dys, std::span<float> dxs) noexcept
{
    assert(w.rows > 0 && dys.size() % w.rows == 0 && dxs.size() == dys.size() / w.rows * w.cols);
    for (std::size_t t = 0; t < dys.size() / w.rows; ++t)
        linear_backward_input(w, dys.subspan(t * w.rows, w.rows), dxs.subspan(t * w.cols, w.cols));
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_nf4_hpp
