//
//  linear.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml::linear`: `y = W x + b` over any weight storage: dense rows of float, double or bfloat16, or ggml's quantised blocks.
/// \ingroup core
///
/// **Overview.** The projections of a transformer are matrix-vector products whose weights may be stored densely or
/// quantised. Each storage is a small view (`dense_weights<S>`, `quantized_weights<Block>`) over the caller's
/// memory, laid out as PyTorch's `nn.Linear.weight`: `rows` outputs, each a row of `cols` inputs. A product is two
/// calls:
///
/// - `prepare_input<N>(w, x)`: the input as that storage reads it. A dense matrix reads `x` as it is; a quantised one
///   reads it quantised to Q8_0, once, as ggml does. Projections sharing an input (Q, K and V) share one prepared
///   input.
/// - `linear(w, input, y, bias)`: each output starts at its bias (or 0) and adds the products in input order, as
///   llm.c's `matmul_forward` does, so a dense float product is llm.c's to the bit.
///
/// A storage of the caller's own joins by overloading `prepare_input` and `linear` in its namespace (found by ADL);
/// that is the seam for a BLAS or device backend.
///
/// \snippet ml/linear_demo.cpp linear
///
/// **Relationships.** `ml/quantize.hpp` (the blocks and their dot products), `ml/minifloat.hpp` (`bfloat16`),
/// `ml/attention_layer.hpp` and `ml/mlp.hpp` (the users). Group: \ref core.
///
/// **Implementations.** Dense: one multiply and one add per weight, in the input's type. Quantised: one block dot
/// product per output row against the Q8_0 input (`ml::dot`), plus the bias.
///
/// **Guarantees.** `constexpr`, no allocation (a prepared quantised input is a `std::array` of `N / 32` blocks),
/// linear in `rows · cols`. Each multiply and add is its own statement, so no compiler fuses them under its default
/// contraction; GCC needs `-ffp-contract=off` (see `ml/quantize.hpp`).
///
/// **Errors.** Preconditions, asserted: the input has `cols` elements, `y` has `rows`, a bias is empty or has `rows`,
/// a quantised matrix has `cols` a multiple of 32 and `rows · cols / 32` blocks.
///
/// **Performance.** Benchmarked through `ml/transformer_block.hpp`'s decode step (see there).
///
/// **Tests and demos.** `tests/ml/linear_test.cpp`, `demos/ml/linear_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cassert>`, `<concepts>`, `<cstddef>`, `<span>`, `ml/minifloat.hpp`,
/// `ml/quantize.hpp`, `ml/row.hpp`.
///
/// **Known issues.** A quantised product is in `float` only, as ggml's are.
///
/// **Status.** Experimental, new in 0.8.0.

#ifndef snicholls_core_ml_linear_hpp
#define snicholls_core_ml_linear_hpp

#include <snicholls/core/ml/minifloat.hpp>
#include <snicholls/core/ml/quantize.hpp>
#include <snicholls/core/ml/row.hpp>

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <span>

namespace snicholls::ml
{

/// \brief A dense weight matrix over the caller's memory: `rows` rows of `cols` elements, end to end.
/// \ingroup core
/// \tparam S The stored element: a number type, or a `minifloat` such as `bfloat16`.
template <typename S>
struct dense_weights
{
    std::span<const S> data{};   ///< `rows · cols` elements, row by row.
    std::size_t rows = 0;        ///< The outputs.
    std::size_t cols = 0;        ///< The inputs.

    /// \brief One row of the matrix.
    /// \param i The row, below `rows`.
    /// \return Its `cols` elements.
    [[nodiscard]] constexpr std::span<const S> row(std::size_t i) const { return data.subspan(i * cols, cols); }
};

/// \brief A quantised weight matrix over the caller's memory: `rows` rows of `cols / 32` blocks, end to end.
/// \ingroup core
/// \tparam Block `block_q8_0`, `block_q4_0` or `block_mxfp4`.
template <quant_block Block>
struct quantized_weights
{
    std::span<const Block> blocks{};   ///< `rows · cols / 32` blocks, row by row.
    std::size_t rows = 0;              ///< The outputs.
    std::size_t cols = 0;              ///< The inputs: a multiple of 32.

    /// \brief One row of the matrix.
    /// \param i The row, below `rows`.
    /// \return Its `cols / 32` blocks.
    [[nodiscard]] constexpr std::span<const Block> row(std::size_t i) const
    {
        return blocks.subspan(i * (cols / quant_block_size), cols / quant_block_size);
    }
};

/// \brief A weight matrix: anything with `rows` and `cols` for which `prepare_input` and `linear` are overloaded.
template <typename W>
concept weight_matrix = requires(const W& w) {
    { w.rows } -> std::convertible_to<std::size_t>;
    { w.cols } -> std::convertible_to<std::size_t>;
};

namespace detail::linear
{
    template <typename T, typename S>
        requires math::math_number<S>
    constexpr T widen(S s)
    {
        return static_cast<T>(s);
    }

    template <typename T, typename S>
        requires is_minifloat_v<S>
    constexpr T widen(S s)
    {
        return static_cast<T>(to_float(s));
    }
} // namespace detail::linear

/// \brief The input of a dense product: `x` itself.
/// \tparam N The input's size, as for every storage.
/// \param x The input row.
/// \return A view of it.
template <std::size_t N, typename S, row X>
constexpr std::span<const row_value_t<X>> prepare_input(const dense_weights<S>&, const X& x)
{
    assert(std::ranges::size(x) == N);
    return as_row(x);
}

/// \brief The input of a quantised product: `x` quantised to Q8_0, once for every product that shares it.
/// \tparam N The input's size: a multiple of 32.
/// \param x The input row, `N` floats.
/// \return `N / 32` blocks.
template <std::size_t N, quant_block Block, row X>
    requires(std::same_as<row_value_t<X>, float> && N % quant_block_size == 0)
constexpr std::array<block_q8_0, N / quant_block_size> prepare_input(const quantized_weights<Block>&, const X& x)
{
    std::array<block_q8_0, N / quant_block_size> q{};
    quantize(x, std::span<block_q8_0, N / quant_block_size>(q));
    return q;
}

/// \brief `y = W x + b` with dense weights: each output starts at its bias and adds `w_ij · x_j` for `j` in order.
/// \param w The matrix.
/// \param x The input, from `prepare_input` or any row of `cols` elements.
/// \param y The output, `rows` elements; may not alias `x`.
/// \param bias `rows` biases, or empty for none.
template <typename S, row X, mutable_row Y, row B = std::span<const row_value_t<X>>>
    requires std::same_as<row_value_t<X>, row_value_t<Y>> && std::same_as<row_value_t<X>, row_value_t<B>>
constexpr void linear(const dense_weights<S>& w, const X& x, Y&& y, const B& bias = B{})
{
    using T = row_value_t<X>;
    const auto in = as_row(x);
    const auto out = as_mutable_row(y);
    const auto b = as_row(bias);
    assert(in.size() == w.cols && out.size() == w.rows && w.data.size() == w.rows * w.cols && (b.empty() || b.size() == w.rows));
    for (std::size_t i = 0; i < w.rows; ++i) {
        const auto r = w.row(i);
        T acc = b.empty() ? T(0) : b[i];
        for (std::size_t j = 0; j < w.cols; ++j) {
            const T p = detail::linear::widen<T>(r[j]) * in[j];
            acc += p;
        }
        out[i] = acc;
    }
}

/// \brief `y = W x + b` with quantised weights: each output is the block dot product of its row with the Q8_0 input, plus its bias.
/// \param w The matrix.
/// \param x The input, from `prepare_input`.
/// \param y The output, `rows` floats.
/// \param bias `rows` biases, or empty for none.
template <quant_block Block, std::size_t M, mutable_row Y, row B = std::span<const float>>
    requires std::same_as<row_value_t<Y>, float> && std::same_as<row_value_t<B>, float>
constexpr void linear(const quantized_weights<Block>& w, const std::array<block_q8_0, M>& x, Y&& y, const B& bias = B{})
{
    const auto out = as_mutable_row(y);
    const auto b = as_row(bias);
    assert(M * quant_block_size == w.cols && out.size() == w.rows && w.blocks.size() == w.rows * M && (b.empty() || b.size() == w.rows));
    for (std::size_t i = 0; i < w.rows; ++i) {
        const float d = dot(w.row(i), std::span<const block_q8_0>(x));
        out[i] = b.empty() ? d : b[i] + d;
    }
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_linear_hpp
