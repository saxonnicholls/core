//
//  kv_cells.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml::kv_cells`: which position and which sequences each slot of a KV cache holds, llama.cpp's sequence operations on them (remove a range, copy, keep, add to and divide positions), the K-shift that re-rotates cached keys after a position change, and attention over the cells a query sees.
/// \ingroup core
///
/// **Overview.** `kv_cache_view` (`ml/kv_cache.hpp`) stores keys and values by slot. A server holding several
/// sequences in one cache, sharing a prompt between them, or dropping the oldest tokens to keep generating (context
/// shift) needs to know what each slot holds. `kv_cells<Capacity>` is that table, as llama.cpp's `llama_kv_cells` keeps it
/// (`src/llama-kv-cells.h` and `src/llama-kv-cache.cpp`, llama.cpp 39039aa): per slot a position (-1 when empty), the set of
/// sequences that share it (a 64-bit mask: sequences 0 to 63), and the shift its position has accumulated since its key
/// was rotated.
///
/// - `place(slot, pos, seq)` records a token written with `kv_cache_view::append(layer, slot, k, v)`; `find_free(from)`
///   finds a slot; `share(slot, seq)` adds a sequence to a cell (a shared prompt).
/// - The sequence operations, over the positions `[p0, p1)` (a negative `p0` is 0, a negative `p1` is unbounded, as
///   llama.cpp's): `seq_rm(seq, p0, p1)` (`seq` -1: every sequence), `seq_cp(src, dst, p0, p1)`, `seq_keep(seq)`,
///   `seq_add(seq, p0, p1, delta)` (cells moved below position 0 are freed), `seq_div(seq, p0, p1, d)` (self-extend's
///   grouping); `seq_pos_min` and `seq_pos_max`.
/// - `kv_shift_keys<C>(keys, layers, cells)`: rotates every cached key whose position moved by its shift (RoPE's
///   rotations compose: a key rotated at `p` and then by `δ` is the key rotated at `p + δ`, with the configuration's
///   base, layout and rescaling), then clears the shifts; llama.cpp's `build_graph_shift`.
/// - `visible(slot, seq, query_pos, window)`: the causal mask of a query of `seq` at `query_pos`, with a sliding window
///   as `llama_hparams::is_masked_swa` (standard: a key is outside when `query_pos - pos >= window`).
/// - `attend_cells<C>(cache, cells, layer, kv_head, q, out, seq, query_pos, slope)`: one query head over the cells it
///   sees, in slot order, with the configuration's scale and soft cap and an optional ALiBi slope (`ml/alibi.hpp`) read
///   from each cell's position.
///
/// \snippet ml/kv_cells_demo.cpp kv_cells
///
/// **Relationships.** `ml/kv_cache.hpp` (the storage, by slot), `ml/rope.hpp` (the rotation), `ml/alibi.hpp`,
/// `ml/attention.hpp` (the recurrence). `ml/kv_cache_quant.hpp` holds the keys and values as quantised rows. Group:
/// \ref core.
///
/// **Implementations.** One, of fixed capacity (arrays: no allocation, synthesisable).
///
/// **Guarantees.** `constexpr`. Every operation is linear in the capacity; llama.cpp's per-sequence ordered sets of
/// positions are replaced by scans, which cost the same order as the attention that follows. Trivially copyable. A
/// cell is empty exactly when its sequence mask is empty.
///
/// **Errors.** Preconditions, asserted: slots below the capacity; sequences from 0 to 63 (-1 where an operation
/// allows it); `place` on an empty slot; `d > 0` for `seq_div`; the key storage is `kv_cache_view`'s size.
///
/// **Performance.** Not benchmarked: the table is a few bytes a slot; `attend_cells` is `attend`'s loop with a test per
/// slot.
///
/// **Tests and demos.** `tests/ml/kv_cells_test.cpp`, `demos/ml/kv_cells_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<array>`, `<cassert>`, `<cstddef>`, `<cstdint>`, `<limits>`, `<span>`, `ml/alibi.hpp`,
/// `ml/activations.hpp`, `ml/attention.hpp`, `ml/kv_cache.hpp`, `ml/rope.hpp`, `ml/transformer_config.hpp`.
///
/// **Known issues.** llama.cpp's `pos_div` (`src/llama-kv-cells.h`, 39039aa) accumulates `p_old - p_new` into the
/// shift, the opposite sign of `pos_add`'s `p_new - p_old`, and its K-shift rotates by the shift as stored, so a divided
/// key is rotated away from its new position; core records `p_new - p_old` in both, so the K-shift always moves a key to
/// its new position (tested). A position belongs to the cell, so moving a shared cell moves it for every sequence
/// that shares it, as in llama.cpp. Multiple streams and the per-cell extra data of llama.cpp's two-dimensional positions are
/// not modelled. A latent attention's cache (MLA) is not shifted here (`kv_shift_keys` rejects it).
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_kv_cells_hpp
#define snicholls_core_ml_kv_cells_hpp

#include <snicholls/core/ml/activations.hpp>
#include <snicholls/core/ml/alibi.hpp>
#include <snicholls/core/ml/attention.hpp>
#include <snicholls/core/ml/kv_cache.hpp>
#include <snicholls/core/ml/rope.hpp>
#include <snicholls/core/ml/transformer_config.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace snicholls::ml
{

/// \brief A position in a KV cache, -1 for none (llama.cpp's `llama_pos`).
using kv_pos = std::int32_t;

/// \brief A sequence of a KV cache, 0 to 63 (llama.cpp's `llama_seq_id`; -1 means every sequence where allowed).
using kv_seq = std::int32_t;

/// \brief The sequences a `kv_cells` mask can hold.
inline constexpr kv_seq kv_max_sequences = 64;

/// \brief What each slot of a KV cache holds: a position, the sequences sharing it, and the shift since its key was
/// rotated.
/// \ingroup core
/// \tparam Capacity The slots.
template <std::size_t Capacity>
    requires(Capacity > 0)
class kv_cells
{
public:
    /// \brief An empty table: every slot at position -1.
    constexpr kv_cells() noexcept { pos_.fill(-1); }

    /// \brief The slots.
    /// \return `Capacity`.
    [[nodiscard]] static constexpr std::size_t size() noexcept { return Capacity; }

    /// \brief Whether a slot holds nothing.
    /// \param i The slot.
    /// \return Whether its sequence mask is empty.
    [[nodiscard]] constexpr bool is_empty(std::size_t i) const noexcept { return seq_[i] == 0; }

    /// \brief A slot's position.
    /// \param i The slot.
    /// \return The position, -1 if empty.
    [[nodiscard]] constexpr kv_pos pos(std::size_t i) const noexcept { return pos_[i]; }

    /// \brief How far a slot's position has moved since its key was rotated.
    /// \param i The slot.
    /// \return The shift, 0 if none.
    [[nodiscard]] constexpr kv_pos shift(std::size_t i) const noexcept { return shift_[i]; }

    /// \brief Whether any position moved since the last `reset_shift`.
    /// \return Whether a K-shift is due.
    [[nodiscard]] constexpr bool has_shift() const noexcept { return has_shift_; }

    /// \brief Whether a slot belongs to a sequence.
    /// \param i The slot.
    /// \param s The sequence, 0 to 63.
    /// \return Whether it does.
    [[nodiscard]] constexpr bool seq_has(std::size_t i, kv_seq s) const noexcept { return (seq_[i] & bit(s)) != 0; }

    /// \brief The slots in use.
    /// \return Their number.
    [[nodiscard]] constexpr std::size_t used() const noexcept
    {
        std::size_t n = 0;
        for (const auto m : seq_)
            n += m != 0 ? 1 : 0;
        return n;
    }

    /// \brief The first empty slot at or after `from`, wrapping around.
    /// \param from Where to start.
    /// \return The slot, or `Capacity` if every slot is in use.
    [[nodiscard]] constexpr std::size_t find_free(std::size_t from = 0) const noexcept
    {
        for (std::size_t k = 0; k < Capacity; ++k)
            if (const std::size_t i = (from + k) % Capacity; is_empty(i))
                return i;
        return Capacity;
    }

    /// \brief Records a token written into an empty slot.
    /// \param i The slot.
    /// \param p Its position, at least 0.
    /// \param s Its sequence.
    constexpr void place(std::size_t i, kv_pos p, kv_seq s) noexcept
    {
        assert(i < Capacity && is_empty(i) && p >= 0);
        pos_[i] = p;
        seq_[i] = bit(s);
        shift_[i] = 0;
    }

    /// \brief Adds a sequence to a slot in use (a prompt shared by several sequences).
    /// \param i The slot.
    /// \param s The sequence.
    constexpr void share(std::size_t i, kv_seq s) noexcept
    {
        assert(i < Capacity && !is_empty(i));
        seq_[i] |= bit(s);
    }

    /// \brief Removes a sequence's cells at positions `[p0, p1)`; with `s` -1, every cell there.
    /// \param s The sequence, or -1.
    /// \param p0 The first position (negative: 0).
    /// \param p1 One past the last (negative: unbounded).
    /// \return The first slot freed, or `Capacity`.
    constexpr std::size_t seq_rm(kv_seq s, kv_pos p0, kv_pos p1) noexcept
    {
        assert(s >= -1 && s < kv_max_sequences);
        std::size_t first = Capacity;
        for (std::size_t i = 0; i < Capacity; ++i) {
            if (is_empty(i) || !in(i, p0, p1) || (s >= 0 && !seq_has(i, s)))
                continue;
            seq_[i] = s < 0 ? 0 : seq_[i] & ~bit(s);
            if (is_empty(i)) {
                clear(i);
                first = first < i ? first : i;
            }
        }
        return first;
    }

    /// \brief Adds sequence `dst` to every cell of `src` at positions `[p0, p1)`: the two then share those keys.
    /// \param src The source sequence.
    /// \param dst The destination sequence.
    /// \param p0 The first position (negative: 0).
    /// \param p1 One past the last (negative: unbounded).
    constexpr void seq_cp(kv_seq src, kv_seq dst, kv_pos p0, kv_pos p1) noexcept
    {
        if (src == dst)
            return;
        for (std::size_t i = 0; i < Capacity; ++i)
            if (!is_empty(i) && in(i, p0, p1) && seq_has(i, src))
                seq_[i] |= bit(dst);
    }

    /// \brief Keeps only sequence `s`: every other cell is freed, and shared cells keep `s` alone.
    /// \param s The sequence.
    /// \return The first slot freed, or `Capacity`.
    constexpr std::size_t seq_keep(kv_seq s) noexcept
    {
        std::size_t first = Capacity;
        for (std::size_t i = 0; i < Capacity; ++i) {
            if (is_empty(i))
                continue;
            if (seq_has(i, s)) {
                seq_[i] = bit(s);
                continue;
            }
            seq_[i] = 0;
            clear(i);
            first = first < i ? first : i;
        }
        return first;
    }

    /// \brief Adds `delta` to the positions of sequence `s` in `[p0, p1)` (llama.cpp's `seq_add`: a context shift);
    /// a cell moved below 0 is freed. The keys are re-rotated by `kv_shift_keys`.
    /// \param s The sequence.
    /// \param p0 The first position (negative: 0).
    /// \param p1 One past the last (negative: unbounded).
    /// \param delta The change.
    constexpr void seq_add(kv_seq s, kv_pos p0, kv_pos p1, kv_pos delta) noexcept
    {
        if (delta == 0)
            return;
        for (std::size_t i = 0; i < Capacity; ++i) {
            if (is_empty(i) || !in(i, p0, p1) || !seq_has(i, s))
                continue;
            pos_[i] += delta;
            shift_[i] += delta;
            has_shift_ = true;
            if (pos_[i] < 0) {
                seq_[i] = 0;
                clear(i);
            }
        }
    }

    /// \brief Divides the positions of sequence `s` in `[p0, p1)` by `d`, rounding down (llama.cpp's `seq_div`, for
    /// self-extend); the shift records the change, `p / d - p`.
    /// \param s The sequence.
    /// \param p0 The first position (negative: 0).
    /// \param p1 One past the last (negative: unbounded).
    /// \param d The divisor, at least 1.
    constexpr void seq_div(kv_seq s, kv_pos p0, kv_pos p1, kv_pos d) noexcept
    {
        assert(d > 0);
        if (d == 1)
            return;
        for (std::size_t i = 0; i < Capacity; ++i) {
            if (is_empty(i) || !in(i, p0, p1) || !seq_has(i, s))
                continue;
            const kv_pos old = pos_[i];
            pos_[i] = old / d;
            shift_[i] += pos_[i] - old;
            has_shift_ = true;
        }
    }

    /// \brief The smallest position of a sequence.
    /// \param s The sequence.
    /// \return It, or -1 if the sequence has no cells.
    [[nodiscard]] constexpr kv_pos seq_pos_min(kv_seq s) const noexcept
    {
        kv_pos m = -1;
        for (std::size_t i = 0; i < Capacity; ++i)
            if (!is_empty(i) && seq_has(i, s) && (m < 0 || pos_[i] < m))
                m = pos_[i];
        return m;
    }

    /// \brief The largest position of a sequence.
    /// \param s The sequence.
    /// \return It, or -1 if the sequence has no cells.
    [[nodiscard]] constexpr kv_pos seq_pos_max(kv_seq s) const noexcept
    {
        kv_pos m = -1;
        for (std::size_t i = 0; i < Capacity; ++i)
            if (!is_empty(i) && seq_has(i, s) && pos_[i] > m)
                m = pos_[i];
        return m;
    }

    /// \brief Clears every shift, once the keys have been re-rotated.
    constexpr void reset_shift() noexcept
    {
        shift_.fill(0);
        has_shift_ = false;
    }

    /// \brief Whether a query of sequence `s` at `query_pos` sees slot `i`: the slot is `s`'s, at or before the query,
    /// and within the window.
    /// \param i The slot.
    /// \param s The query's sequence.
    /// \param query_pos The query's position.
    /// \param window The sliding window, counting the query's own position; 0 for none.
    /// \return Whether it sees it.
    [[nodiscard]] constexpr bool visible(std::size_t i, kv_seq s, kv_pos query_pos, std::size_t window = 0) const noexcept
    {
        return !is_empty(i) && seq_has(i, s) && pos_[i] <= query_pos
               && (window == 0 || static_cast<std::size_t>(query_pos - pos_[i]) < window);
    }

private:
    static constexpr std::uint64_t bit(kv_seq s) noexcept
    {
        assert(s >= 0 && s < kv_max_sequences);
        return std::uint64_t{1} << s;
    }

    constexpr bool in(std::size_t i, kv_pos p0, kv_pos p1) const noexcept
    {
        const kv_pos lo = p0 < 0 ? 0 : p0, hi = p1 < 0 ? std::numeric_limits<kv_pos>::max() : p1;
        return pos_[i] >= lo && pos_[i] < hi;
    }

    constexpr void clear(std::size_t i) noexcept
    {
        pos_[i] = -1;
        shift_[i] = 0;
    }

    std::array<kv_pos, Capacity> pos_{};
    std::array<std::uint64_t, Capacity> seq_{};
    std::array<kv_pos, Capacity> shift_{};
    bool has_shift_ = false;
};

/// \brief The K-shift: rotates each cached key whose position moved by the cell's shift, in every layer and key-value
/// head, with the configuration's base, layout and rescaling (amplitude 1), then clears the shifts.
/// \tparam C A configuration with RoPE and standard attention.
/// \tparam T The cache's number type.
/// \tparam Capacity The cells' capacity, the cache's.
/// \param keys The key storage of a `kv_cache_view<C, T>` of `layers` and `Capacity`: `[layer][kv_head][slot][head_dim]`.
/// \param layers The layers.
/// \param cells The cells, their shifts cleared.
template <transformer_config C, math::math_number T, std::size_t Capacity>
constexpr void kv_shift_keys(std::span<T> keys, std::size_t layers, kv_cells<Capacity>& cells)
{
    static_assert(valid(C) && C.rope != rope_kind::none && !C.latent(),
                  "snicholls::ml::kv_shift_keys: the configuration rotates no keys, or its keys are latent");
    assert((keys.size() == kv_cache_view<C, T>::required_size(layers, Capacity)));
    constexpr std::size_t d = C.head_dim, h = d / 2;
    if (!cells.has_shift())
        return;
    for (std::size_t i = 0; i < Capacity; ++i) {
        const kv_pos delta = cells.shift(i);
        if (cells.is_empty(i) || delta == 0)
            continue;
        std::array<T, h> c{}, s{};
        rope_angles(static_cast<std::size_t>(delta < 0 ? -delta : delta), c, s, rope_base_v<C, T>, rope_scaling_v<C, T>);
        for (std::size_t layer = 0; layer < layers; ++layer)
            for (std::size_t kv = 0; kv < C.n_kv_heads; ++kv) {
                const auto row = keys.subspan(((layer * C.n_kv_heads + kv) * Capacity + i) * d, d);
                if (delta > 0)
                    rope(row, c, s, C.layout());
                else
                    rope_inverse(row, c, s, C.layout());
            }
    }
    cells.reset_shift();
}

/// \brief One query head over the cells its sequence sees, in slot order: `softmax(cap(scale · q·k) + bias) · v`, with
/// the configuration's scale `1/sqrt(head_dim)` and soft cap, and `alibi_bias(slope, query_pos, cell position)`.
/// \tparam C The configuration.
/// \tparam T The number type.
/// \tparam Capacity The cells' capacity, the cache's.
/// \param cache The keys and values, by slot.
/// \param cells What each slot holds.
/// \param layer The layer.
/// \param kv_head The key-value head the query head reads (GQA: `h / group`).
/// \param q The query head, `head_dim`.
/// \param out Receives the head's output, `head_dim`.
/// \param seq The query's sequence.
/// \param query_pos The query's position.
/// \param slope The head's ALiBi slope; 0 for none.
/// \return The log-sum-exp of the scores, -infinity if no cell is visible.
template <transformer_config C, math::math_number T, std::size_t Capacity, row Q, mutable_row Out>
    requires std::same_as<row_value_t<Q>, T> && std::same_as<row_value_t<Out>, T>
constexpr T attend_cells(const kv_cache_view<C, T>& cache, const kv_cells<Capacity>& cells, std::size_t layer, std::size_t kv_head,
                         const Q& q, Out&& out, kv_seq seq, kv_pos query_pos, std::type_identity_t<T> slope = 0)
{
    const auto qs = as_row(q);
    const auto y = as_mutable_row(out);
    assert(cache.capacity() == Capacity && qs.size() == C.head_dim && y.size() == C.head_dim);
    const T scale = T(1) / math::sqrt(T(C.head_dim));
    std::fill(y.begin(), y.end(), T(0));
    T max = -std::numeric_limits<T>::infinity(), sum = T(0);
    for (std::size_t i = 0; i < Capacity; ++i) {
        if (!cells.visible(i, seq, query_pos, C.window))
            continue;
        const auto run = cache.visible(layer, kv_head, key_range{i, i + 1});
        T s = scale * detail::attention::dot(qs, run.keys[0]);
        if constexpr (C.attention_softcap > 0)
            s = softcap(s, attention_softcap_v<C, T>);
        if (slope != T(0))
            s = s + alibi_bias(slope, query_pos, cells.pos(i));
        detail::attention::fold(max, sum, y, s, run.values[0]);
    }
    for (T& v : y)
        v = sum == T(0) ? T(0) : v / sum;
    return sum == T(0) ? -std::numeric_limits<T>::infinity() : max + math::log(sum);
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_kv_cells_hpp
