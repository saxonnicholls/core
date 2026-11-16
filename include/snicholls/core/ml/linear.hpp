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
/// - `linear_backward(w, x, dy, dx, dw, db)`: for dense (trainable) storage, adds `Wᵀ dy`, `dy xᵀ` and `dy` into the
///   gradients, as llm.c's `matmul_backward`.
/// - `linear_sequence(backend, w, xs, ys, bias)` and `linear_sequence_backward(backend, w, xs, dys, dxs, dw, db)`: the
///   same over a sequence of `n` rows, as one matrix product (and two for the backward) through a `linalg` backend:
///   core's reference, or a BLAS or a GPU backend from another library, the same call. Dense weights in the sequence's type. The result is the backend's:
///   the reference sums each output in input order and adds the bias after, so it can differ from `linear` in the
///   last bit.
///
/// A storage of the caller's own joins by overloading `prepare_input` and `linear` in its namespace (found by ADL);
/// that is the seam for a BLAS or device backend.
///
/// \snippet ml/linear_demo.cpp linear
///
/// \snippet ml/linear_demo.cpp linear_sequence
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
/// **Performance.** Benchmarked through `ml/transformer_block.hpp`'s decode step and training passes (see there), and
/// `linear_backward` through `ml/self_attention.hpp`'s backward; level with llm.c's `matmul_backward` loops.
///
/// `linear_sequence` through `linalg::reference` against the same products as loops over raw pointers, after a
/// bit-for-bit check: 64 tokens of 256 inputs (`benchmarks/ml/linear_benchmark.cpp`; Release, Apple clang 17, Xeon
/// W-3245, macOS 26, 2026-11-16, medians of five, load average about 16):
///
/// | Product | Core | By hand |
/// |---|---|---|
/// | forward, 256 outputs | 3.96 ms | 3.98 ms |
/// | forward, 1024 outputs | 15.9 ms | 16.1 ms |
/// | backward, 256 outputs | 8.43 ms | 8.58 ms |
/// | backward, 1024 outputs | 42.9 ms | 43.9 ms |
///
/// Level: the views, the extent checks and the backend's dispatch cost nothing measurable. The reference backend is
/// no faster than `linear` token by token (3.96 ms and 16.1 ms the same rows); the batched path is for a backend that
/// is (a BLAS, a device).
///
/// **Tests and demos.** `tests/ml/linear_test.cpp`, `demos/ml/linear_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cassert>`, `<concepts>`, `<cstddef>`, `<span>`, `<type_traits>`, `linalg/backend.hpp`, `ml/minifloat.hpp`,
/// `ml/quantize.hpp`, `ml/row.hpp`.
///
/// **Known issues.** A quantised product is in `float` only, as ggml's are.
///
/// **Status.** Experimental, new in 0.8.0.

#ifndef snicholls_core_ml_linear_hpp
#define snicholls_core_ml_linear_hpp

#include <snicholls/core/linalg/backend.hpp>
#include <snicholls/core/ml/minifloat.hpp>
#include <snicholls/core/ml/quantize.hpp>
#include <snicholls/core/ml/row.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>

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

/// \brief The backward pass of a dense `linear`: adds `Wᵀ dy` into `dx`, `dy xᵀ` into `dw` and `dy` into `db`.
///
/// Every gradient is accumulated, so an input shared by several products (Q, K and V) and the positions of a
/// sequence add into one buffer; zero them first. The loops are llm.c's `matmul_backward`: for each output `i`,
/// `dx_j += w_ij · dy_i` and `dw_ij += dy_i · x_j` for `j` in order, then `db_i += dy_i`. Only dense storage has a
/// backward: training quantised weights is a compile error.
/// \param w The matrix, `rows x cols`.
/// \param x The input the forward pass read, `cols`.
/// \param dy The gradient in the output, `rows`.
/// \param dx Added into, `cols`; or empty to skip it (a first layer's input).
/// \param dw Added into, `rows · cols`, row by row as `w`.
/// \param db Added into, `rows`; or empty for a matrix without a bias.
template <typename S, row X, row DY, mutable_row DX, mutable_row DW, mutable_row DB = std::span<row_value_t<X>>>
    requires std::same_as<row_value_t<X>, row_value_t<DY>> && std::same_as<row_value_t<X>, row_value_t<DX>>
             && std::same_as<row_value_t<X>, row_value_t<DW>> && std::same_as<row_value_t<X>, row_value_t<DB>>
constexpr void linear_backward(const dense_weights<S>& w, const X& x, const DY& dy, DX&& dx, DW&& dw, DB&& db = DB{})
{
    using T = row_value_t<X>;
    const auto in = as_row(x);
    const auto g = as_row(dy);
    const auto gx = as_mutable_row(dx);
    const auto gw = as_mutable_row(dw);
    const auto gb = as_mutable_row(db);
    assert(in.size() == w.cols && g.size() == w.rows && w.data.size() == w.rows * w.cols && gw.size() == w.rows * w.cols
           && (gx.empty() || gx.size() == w.cols) && (gb.empty() || gb.size() == w.rows));
    for (std::size_t i = 0; i < w.rows; ++i) {
        const auto r = w.row(i);
        const T d = g[i];
        if (!gx.empty())
            for (std::size_t j = 0; j < w.cols; ++j) {
                const T p = detail::linear::widen<T>(r[j]) * d;
                gx[j] += p;
            }
        for (std::size_t j = 0; j < w.cols; ++j) {
            const T p = d * in[j];
            gw[i * w.cols + j] += p;
        }
        if (!gb.empty())
            gb[i] += d;
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

namespace detail::linear
{
    template <typename T>
    constexpr auto view(std::span<const T> s, std::size_t rows, std::size_t cols)
    {
        return linalg::matrix_view<const T, linalg::layout_right>(s.data(), {rows, cols});
    }

    template <typename T>
    constexpr auto view(std::span<T> s, std::size_t rows, std::size_t cols)
    {
        return linalg::matrix_view<T, linalg::layout_right>(s.data(), {rows, cols});
    }

    // C = alpha op(A) op(B) + beta C. The shapes are right by construction and C never overlaps A or B, so the
    // backend's product always succeeds
    template <typename B, typename T, typename L, typename R, typename C>
    constexpr void product(const B& backend, T alpha, const L& lhs, const R& rhs, T beta, const C& c)
    {
        [[maybe_unused]] const auto done = linalg::gemm(backend, alpha, lhs, rhs, beta, c);
        assert(done.is_ok());
    }
} // namespace detail::linear

/// \name The elementwise steps of the batched passes, on the backend (TRAINING.md T9)
/// Every step of a batched training pass that is not a matrix product is one of these calls, made with the pass's
/// backend, so a device library runs it where the backend keeps its data. The overloads here take any backend and run
/// on the host; a device library adds an overload for its own backend type (found by argument-dependent lookup, and
/// more specialised, so it is chosen) that runs on the device.
/// @{

/// \brief `x = value` everywhere. \param x The rows, end to end. \param value The value.
template <typename B, math::math_number T>
constexpr void fill_rows(const B&, std::span<T> x, std::type_identity_t<T> value)
{
    std::fill(x.begin(), x.end(), value);
}

/// \brief `to = from`. \param from The source. \param to The destination, the same size; may not overlap `from`.
template <typename B, math::math_number T>
constexpr void copy_rows(const B&, std::type_identity_t<std::span<const T>> from, std::span<T> to)
{
    assert(from.size() == to.size());
    std::copy(from.begin(), from.end(), to.begin());
}

/// \brief `x = a + x`, elementwise: a residual connection. \param a The other summand. \param x Updated in place.
template <typename B, math::math_number T>
constexpr void add_rows(const B&, std::type_identity_t<std::span<const T>> a, std::span<T> x)
{
    assert(a.size() == x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = a[i] + x[i];
}

/// \brief `y_t += bias` for every row `t` of `bias.size()` elements. \param ys The rows. \param bias The bias.
template <typename B, math::math_number T>
constexpr void add_bias_rows(const B&, std::span<T> ys, std::type_identity_t<std::span<const T>> bias)
{
    const std::size_t r = bias.size();
    assert(r > 0 && ys.size() % r == 0);
    for (std::size_t t = 0; t < ys.size() / r; ++t)
        for (std::size_t i = 0; i < r; ++i)
            ys[t * r + i] += bias[i];
}

/// \brief `into += Σ_t ys_t`, the rows summed in order: a bias's gradient. \param ys The rows. \param into Added into.
template <typename B, math::math_number T>
constexpr void sum_rows(const B&, std::type_identity_t<std::span<const T>> ys, std::span<T> into)
{
    const std::size_t r = into.size();
    assert(r > 0 && ys.size() % r == 0);
    for (std::size_t t = 0; t < ys.size() / r; ++t)
        for (std::size_t i = 0; i < r; ++i)
            into[i] += ys[t * r + i];
}

/// @}

/// \brief `Y = X Wᵀ + b` for a sequence of `n` rows, as one matrix product through a `linalg` backend.
///
/// The batched form of `linear`, for training and prefill: the backend does the work (core's reference, a BLAS, a
/// device), so its result is the backend's: core's reference backend sums each output in input order and then
/// adds the bias, which can differ from `linear`'s bias-first sum in the last bit.
/// \param backend A `linalg::blas_backend`.
/// \param w The matrix, `rows x cols`, dense in the sequence's own type.
/// \param xs `n` inputs of `cols`, end to end.
/// \param ys Receives `n` outputs of `rows`; may not alias `xs`.
/// \param bias `rows` biases, or empty for none.
template <math::math_number T, typename B>
    requires linalg::blas_backend<B, T>
constexpr void linear_sequence(const B& backend, const dense_weights<T>& w, std::type_identity_t<std::span<const T>> xs,
                               std::type_identity_t<std::span<T>> ys, std::type_identity_t<std::span<const T>> bias = {})
{
    assert(w.cols > 0 && xs.size() % w.cols == 0);
    const std::size_t n = xs.size() / w.cols;
    assert(ys.size() == n * w.rows && w.data.size() == w.rows * w.cols && (bias.empty() || bias.size() == w.rows));
    if (n == 0 || w.rows == 0)
        return;
    using detail::linear::view;
    detail::linear::product(backend, T(1), view(xs, n, w.cols), linalg::transposed(view(w.data, w.rows, w.cols)), T(0),
                            view(ys, n, w.rows));
    if (!bias.empty())
        add_bias_rows(backend, ys, bias);
}

/// \brief The backward pass of `linear_sequence`: `dX += dY W`, `dW += dYᵀ X`, `db += Σ dY`, as two matrix
/// products through the backend.
/// \param backend A `linalg::blas_backend`.
/// \param w The matrix the forward pass used.
/// \param xs Its `n` inputs.
/// \param dys The gradients in its `n` outputs.
/// \param dxs Added into, `n · cols`; or empty to skip it.
/// \param dw Added into, `rows · cols`.
/// \param db Added into, `rows`; or empty.
template <math::math_number T, typename B>
    requires linalg::blas_backend<B, T>
constexpr void linear_sequence_backward(const B& backend, const dense_weights<T>& w, std::type_identity_t<std::span<const T>> xs,
                                        std::type_identity_t<std::span<const T>> dys, std::type_identity_t<std::span<T>> dxs,
                                        std::type_identity_t<std::span<T>> dw, std::type_identity_t<std::span<T>> db = {})
{
    assert(w.cols > 0 && xs.size() % w.cols == 0);
    const std::size_t n = xs.size() / w.cols;
    assert(dys.size() == n * w.rows && (dxs.empty() || dxs.size() == xs.size()) && dw.size() == w.rows * w.cols
           && (db.empty() || db.size() == w.rows));
    if (n == 0 || w.rows == 0)
        return;
    using detail::linear::view;
    const auto dy = view(dys, n, w.rows);
    if (!dxs.empty())
        detail::linear::product(backend, T(1), dy, view(w.data, w.rows, w.cols), T(1), view(dxs, n, w.cols));
    detail::linear::product(backend, T(1), linalg::transposed(dy), view(xs, n, w.cols), T(1), view(dw, w.rows, w.cols));
    if (!db.empty())
        sum_rows(backend, dys, db);
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_linear_hpp
