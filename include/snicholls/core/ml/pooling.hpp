//
//  pooling.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` pooling for embeddings and rerankers: one row per sequence from its token rows by mean, first (CLS) or last token, with backward, and the classification head of a reranker (RANK).
/// \ingroup core
///
/// **Overview.** An embedding model or a reranker ends where a language model would compute logits: the final hidden
/// rows of each sequence become one row (`llm_graph_context::build_pooling`, `src/llama-graph.cpp`, llama.cpp 39039aa):
///
/// | `pooling_kind` | Row of sequence `s` | llama.cpp |
/// |---|---|---|
/// | `none` | every token's row, unchanged | `LLAMA_POOLING_TYPE_NONE` |
/// | `mean` | the mean of its rows (`mean_pool_rows`, `ml/probe.hpp`) | `MEAN`: sentence-transformers, nomic-embed |
/// | `cls` | its first row | `CLS`: BERT, BGE |
/// | `last` | its last row | `LAST`: Qwen3-Embedding, e5-mistral |
/// | `rank` | the CLS (or mean) row through `rank_head` | `RANK`: BGE-reranker, Jina, Qwen3-Reranker |
///
/// - Sequences are packed end to end, as a llama.cpp batch holds them: `starts` gives each sequence's first row and the
///   end of the last (`starts.size() == sequences + 1`).
/// - `pool_rows(kind, rows, width, starts, out)` and `pool_rows_backward(kind, dout, width, starts, drows)`.
/// - `rank_head(w, pooled, scratch, scores)`: RoBERTa's classification head as llama.cpp builds it: `cls` (a dense
///   layer, its bias and an activation, tanh by default, then an optional layer norm), then `cls_out` (the score
///   projection) and a softmax over the scores for Qwen3's reranker; either layer may be absent (`rows == 0`), as in
///   Jina's single-layer head.
///
/// \snippet ml/pooling_demo.cpp pooling
///
/// **Relationships.** `ml/probe.hpp` (`mean_pool_rows`, reused), `ml/linear.hpp` (the head's layers), `ml/norm.hpp`,
/// `ml/softmax.hpp`, `ml/activations.hpp`; `ml/reward_model.hpp` (a reward head over the last token). Group: \ref core.
///
/// **Implementations.** One each.
///
/// **Guarantees.** `constexpr`, no allocation; linear in the rows. The mean sums in order and divides once (`mean_pool_rows`'s
/// order). An empty sequence pools to zeros.
///
/// **Errors.** Preconditions, asserted: `starts` non-decreasing from 0 to the rows; `out` `sequences · width` (`rows`'
/// size for `none`); the head's layers fit each other; a nonempty sequence for `cls` and `last`.
///
/// **Performance.** Not benchmarked: one pass over the rows; the head is two matrix-vector products.
///
/// **Tests and demos.** `tests/ml/pooling_test.cpp`, `demos/ml/pooling_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<cassert>`, `<cstddef>`, `<span>`, `<type_traits>`, `ml/activations.hpp`, `ml/linear.hpp`,
/// `ml/norm.hpp`, `ml/probe.hpp`, `ml/softmax.hpp`.
///
/// **Known issues.** ggml takes the mean as a matrix product with `1/n` weights (`build_inp_mean`), so each term is
/// scaled before the sum: the last bits can differ from core's sum-then-divide. A reranker whose model pools by mean
/// before its head (`pooling_type_cls`) uses `pool_rows(mean, ...)` and then `rank_head`.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_pooling_hpp
#define snicholls_core_ml_pooling_hpp

#include <snicholls/core/ml/activations.hpp>
#include <snicholls/core/ml/linear.hpp>
#include <snicholls/core/ml/norm.hpp>
#include <snicholls/core/ml/probe.hpp>
#include <snicholls/core/ml/softmax.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief How the token rows of a sequence become one row; the order is llama.cpp's `llama_pooling_type`.
/// \ingroup core
enum class pooling_kind
{
    none,   ///< Every token's row, unchanged.
    mean,   ///< The mean of the sequence's rows.
    cls,    ///< The first row (a [CLS] token).
    last,   ///< The last row.
    rank,   ///< The first row, for `rank_head` (a reranker's score).
};

namespace detail::pooling
{
    struct host
    {
    };

    constexpr bool packed(std::span<const std::size_t> starts, std::size_t rows) noexcept
    {
        if (starts.size() < 2 || starts.front() != 0 || starts.back() != rows)
            return false;
        for (std::size_t s = 1; s < starts.size(); ++s)
            if (starts[s] < starts[s - 1])
                return false;
        return true;
    }
} // namespace detail::pooling

/// \brief Pools each sequence's token rows into one row.
/// \tparam T A `math::math_number`.
/// \param kind The pooling.
/// \param rows The token rows of every sequence, end to end, `width` each.
/// \param width The values of a row.
/// \param starts Each sequence's first row, then the end of the last: `sequences + 1` offsets from 0 to the rows.
/// \param out Receives `sequences` rows (every row for `none`).
template <math::math_number T>
constexpr void pool_rows(pooling_kind kind, std::type_identity_t<std::span<const T>> rows, std::size_t width,
                         std::span<const std::size_t> starts, std::span<T> out)
{
    assert(width > 0 && rows.size() % width == 0 && detail::pooling::packed(starts, rows.size() / width));
    const std::size_t sequences = starts.size() - 1;
    if (kind == pooling_kind::none) {
        assert(out.size() == rows.size());
        // bounded by both sizes, which the assert makes equal: GCC's -Warray-bounds cannot see that under NDEBUG
        const std::size_t n = rows.size() < out.size() ? rows.size() : out.size();
        for (std::size_t i = 0; i < n; ++i)
            out[i] = rows[i];
        return;
    }
    assert(out.size() == sequences * width);
    for (std::size_t s = 0; s < sequences; ++s) {
        const std::size_t first = starts[s], n = starts[s + 1] - first;
        const auto dst = out.subspan(s * width, width);
        if (kind == pooling_kind::mean) {
            mean_pool_rows<detail::pooling::host, T>(detail::pooling::host{}, rows.subspan(first * width, n * width), width, dst);
            continue;
        }
        assert(n > 0);
        const std::size_t r = kind == pooling_kind::last ? first + n - 1 : first;
        for (std::size_t c = 0; c < width; ++c)
            dst[c] = rows[r * width + c];
    }
}

/// \brief The backward pass of `pool_rows`: adds each pooled row's gradient into the token rows it came from.
/// \tparam T A `math::math_number`.
/// \param kind The pooling.
/// \param dout The gradients in the pooled rows.
/// \param width The values of a row.
/// \param starts The forward pass's offsets.
/// \param drows Added into: the gradients in the token rows.
template <math::math_number T>
constexpr void pool_rows_backward(pooling_kind kind, std::type_identity_t<std::span<const T>> dout, std::size_t width,
                                  std::span<const std::size_t> starts, std::span<T> drows)
{
    assert(width > 0 && drows.size() % width == 0 && detail::pooling::packed(starts, drows.size() / width));
    const std::size_t sequences = starts.size() - 1;
    if (kind == pooling_kind::none) {
        assert(dout.size() == drows.size());
        for (std::size_t i = 0; i < drows.size(); ++i)
            drows[i] += dout[i];
        return;
    }
    assert(dout.size() == sequences * width);
    for (std::size_t s = 0; s < sequences; ++s) {
        const std::size_t first = starts[s], n = starts[s + 1] - first;
        const auto g = dout.subspan(s * width, width);
        if (kind == pooling_kind::mean) {
            mean_pool_rows_backward<detail::pooling::host, T>(detail::pooling::host{}, g, width, drows.subspan(first * width, n * width));
            continue;
        }
        const std::size_t r = kind == pooling_kind::last ? first + n - 1 : first;
        for (std::size_t c = 0; c < width; ++c)
            drows[r * width + c] += g[c];
    }
}

/// \brief The activation of a reranker's classification layer (llama.cpp's `act_cls`, tanh by default).
/// \ingroup core
enum class rank_activation
{
    tanh,       ///< RoBERTa's and BERT's pooler.
    gelu,       ///< The exact GELU.
    relu,       ///< ReLU.
    identity,   ///< None.
};

/// \brief The weights of a reranker's head: views the caller fills; a layer with `rows == 0` is absent.
/// \ingroup core
/// \tparam S The weights' stored element.
/// \tparam T The number type.
template <typename S, math::math_number T>
struct rank_head_weights
{
    dense_weights<S> cls{};                          ///< `hidden x width`, or absent.
    std::span<const T> cls_bias{};                   ///< `hidden`, or empty.
    rank_activation activation = rank_activation::tanh;   ///< After `cls`.
    std::span<const T> norm_weight{};                ///< A layer norm after the activation (`cls_norm`), or empty for none.
    std::span<const T> norm_bias{};                  ///< Its shift, or empty.
    T norm_eps = T(1e-12);                           ///< Its epsilon (the model's `f_norm_eps`; BERT's 1e-12).
    dense_weights<S> out{};                          ///< `scores x hidden` (the score projection), or absent.
    std::span<const T> out_bias{};                   ///< `scores`, or empty.
    bool softmax = false;                            ///< A softmax over the scores (Qwen3's reranker: yes/no).
};

/// \brief A reranker's head over one pooled row: `scores = [softmax](out · norm(act(cls · x + b)) + b)`, as
/// llama.cpp's `RANK` pooling builds it.
/// \tparam S The weights' stored element.
/// \tparam T A `math::math_number`.
/// \param w The head.
/// \param pooled The pooled row (`pool_rows(rank, ...)`).
/// \param scratch `cls.rows` values of working space (empty without `cls`).
/// \param scores Receives `out.rows` scores (or `cls.rows`, or the pooled row's size, when layers are absent).
template <typename S, math::math_number T>
constexpr void rank_head(const rank_head_weights<S, T>& w, std::type_identity_t<std::span<const T>> pooled,
                         std::type_identity_t<std::span<T>> scratch, std::type_identity_t<std::span<T>> scores)
{
    std::span<const T> x = pooled;
    if (w.cls.rows > 0) {
        assert(w.cls.cols == pooled.size() && scratch.size() == w.cls.rows);
        linear(w.cls, pooled, scratch, w.cls_bias);
        for (T& v : scratch)
            switch (w.activation) {
            case rank_activation::tanh: v = math::tanh(v); break;
            case rank_activation::gelu: v = gelu(v); break;
            case rank_activation::relu: v = relu(v); break;
            case rank_activation::identity: break;
            }
        if (!w.norm_weight.empty())
            layer_norm(std::span<const T>(scratch), w.norm_weight, w.norm_bias, scratch, w.norm_eps);
        x = scratch;
    }
    if (w.out.rows > 0) {
        assert(w.out.cols == x.size() && scores.size() == w.out.rows);
        linear(w.out, x, scores, w.out_bias);
    } else {
        assert(scores.size() == x.size());
        std::copy(x.begin(), x.end(), scores.begin());
    }
    if (w.softmax)
        softmax(std::span<const T>(scores), scores);
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_pooling_hpp
