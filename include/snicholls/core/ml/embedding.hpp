//
//  embedding.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` embeddings: a token's row of a table, GPT-2's token plus position, and the backward pass that adds a gradient into the token's row.
/// \ingroup core
///
/// **Overview.** The first layer of a language model: token `id` becomes row `id` of a table of `V` rows of `C`
/// channels, laid out row after row in one contiguous row (`table[id · C + c]`), as llm.c's `wte` and `wpe` and
/// nanochat's `nn.Embedding`.
///
/// - `embedding_row(table, id, channels)`: the row, as a span: no copy.
/// - `embed(table, id, out)`: copies the row into `out` (`C = out.size()`).
/// - `embed_with_position(tokens, id, positions, pos, out)`: `out = tokens[id] + positions[pos]`, GPT-2's learned
///   position embedding (llm.c's `encoder_forward`). A model with RoPE has no position table: use `embed`.
/// - `embedding_backward(dout, id, dtable)`: adds `dout` into row `id` of the gradient table (llm.c's
///   `encoder_backward`). Rows of the same token in a batch add up, so workers that share a table either own
///   disjoint tokens or keep their own gradient tables and add them.
///
/// \snippet ml/embedding_demo.cpp embedding
///
/// `embed_rows` and `embed_rows_backward` are a whole sequence's embeddings and their gradients on a backend (TRAINING.md
/// T9), 32-bit token ids, the backward adding in token order; a device library overloads them.
///
/// \snippet ml/embedding_demo.cpp embedding_rows
///
/// **Relationships.** `ml/row.hpp`; `ml/norm.hpp` and `ml/rope.hpp` come next in the model. Group: \ref core.
///
/// **Implementations.** One each.
///
/// **Guarantees.** `constexpr`, no allocation, linear in `C`. `embedding_row` refers to the caller's table and
/// lives as long as it does.
///
/// **Errors.** Preconditions, asserted: the row `id` (and `pos`) lies inside its table, and `dout` has `C`
/// channels.
///
/// **Performance.** Not benchmarked: one row copied or added.
///
/// **Tests and demos.** `tests/ml/embedding_test.cpp`, `demos/ml/embedding_demo.cpp`.
///
/// **Dependencies.** `<cassert>`, `<concepts>`, `<cstddef>`, `<span>`, `ml/row.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** Experimental, new in 0.7.0.

#ifndef snicholls_core_ml_embedding_hpp
#define snicholls_core_ml_embedding_hpp

#include <snicholls/core/ml/row.hpp>

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief Row `id` of a table of rows of `channels`, as a span over the table.
/// \param table The rows, end to end.
/// \param id The row.
/// \param channels The channels of a row.
/// \return A span of `channels` elements.
template <row Table>
constexpr std::span<const row_value_t<Table>> embedding_row(const Table& table, std::size_t id, std::size_t channels)
{
    const auto t = as_row(table);
    assert((id + 1) * channels <= t.size());
    return t.subspan(id * channels, channels);
}

/// \brief Copies row `id` of a table into `out`, whose size is the channels.
/// \param table The rows, end to end.
/// \param id The token.
/// \param out The embedding.
template <row Table, mutable_row Out>
    requires std::same_as<row_value_t<Table>, row_value_t<Out>>
constexpr void embed(const Table& table, std::size_t id, Out&& out)
{
    const auto y = as_mutable_row(out);
    const auto r = embedding_row(table, id, y.size());
    for (std::size_t c = 0; c < y.size(); ++c)
        y[c] = r[c];
}

/// \brief GPT-2's input: `out = tokens[id] + positions[pos]`.
/// \param tokens The token table.
/// \param id The token.
/// \param positions The position table, with the same channels.
/// \param pos The position.
/// \param out The embedding.
template <row Tokens, row Positions, mutable_row Out>
    requires std::same_as<row_value_t<Tokens>, row_value_t<Out>> && std::same_as<row_value_t<Positions>, row_value_t<Out>>
constexpr void embed_with_position(const Tokens& tokens, std::size_t id, const Positions& positions, std::size_t pos, Out&& out)
{
    const auto y = as_mutable_row(out);
    const auto t = embedding_row(tokens, id, y.size());
    const auto p = embedding_row(positions, pos, y.size());
    for (std::size_t c = 0; c < y.size(); ++c)
        y[c] = t[c] + p[c];
}

/// \brief The backward pass of an embedding: adds `dout` into row `id` of `dtable`.
/// \param dout The gradient in the embedding, `C` channels.
/// \param id The token (or the position, for a position table).
/// \param dtable The gradient of the table: accumulated into.
template <row D, mutable_row DTable>
    requires std::same_as<row_value_t<D>, row_value_t<DTable>>
constexpr void embedding_backward(const D& dout, std::size_t id, DTable&& dtable)
{
    const auto d = as_row(dout);
    const auto t = as_mutable_row(dtable);
    assert((id + 1) * d.size() <= t.size());
    for (std::size_t c = 0; c < d.size(); ++c)
        t[id * d.size() + c] += d[c];
}

/// \brief A sequence's embeddings on the backend (TRAINING.md T9): `out_t = table[ids_t] + positions_t`.
/// \param table The token table, `V` rows of `d`.
/// \param positions The position rows, one per token (`n · d`), or empty.
/// \param ids `n` token ids.
/// \param out Receives `n` rows of `d`.
template <typename B, math::math_number T>
constexpr void embed_rows(const B&, std::type_identity_t<std::span<const T>> table, std::type_identity_t<std::span<const T>> positions,
                          std::span<const std::uint32_t> ids, std::span<T> out)
{
    const std::size_t n = ids.size();
    assert(n == 0 || (out.size() % n == 0 && (positions.empty() || positions.size() == out.size())));
    const std::size_t d = n == 0 ? 0 : out.size() / n;
    for (std::size_t t = 0; t < n; ++t)
        for (std::size_t c = 0; c < d; ++c)
            out[t * d + c] = positions.empty() ? table[ids[t] * d + c] : table[ids[t] * d + c] + positions[t * d + c];
}

/// \brief The backward pass of `embed_rows`, on the backend: each row of `dout` added into its token's row of
/// `dtable`, in token order (as `embedding_backward`), and into its position's row of `dpositions`.
/// \param dout `n` rows of `d`. \param ids `n` token ids. \param dtable Added into. \param dpositions Added into, or empty.
template <typename B, math::math_number T>
constexpr void embed_rows_backward(const B&, std::type_identity_t<std::span<const T>> dout, std::span<const std::uint32_t> ids,
                                   std::span<T> dtable, std::type_identity_t<std::span<T>> dpositions)
{
    const std::size_t n = ids.size();
    assert(n == 0 || (dout.size() % n == 0 && (dpositions.empty() || dpositions.size() == dout.size())));
    const std::size_t d = n == 0 ? 0 : dout.size() / n;
    for (std::size_t t = 0; t < n; ++t)
        for (std::size_t c = 0; c < d; ++c) {
            dtable[ids[t] * d + c] += dout[t * d + c];
            if (!dpositions.empty())
                dpositions[t * d + c] += dout[t * d + c];
        }
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_embedding_hpp
