//
//  kv_cache.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml::kv_cache_view` and `kv_cache`: the keys and values of every layer and key-value head, laid out so one head's keys are contiguous, with a sliding window as a ring.
/// \ingroup core
///
/// **Overview.** Decoding a token attends to the keys and values of every earlier position, so they are kept, per
/// layer and per key-value head, in `[layer][kv_head][slot][head_dim]`: the keys one head reads are contiguous rows,
/// which is what `attend` and `attention_accumulator` take. Position `p` is in slot `p % capacity`, so one layout
/// serves both uses:
///
/// - a capacity of the longest sequence: every position has its own slot (a flat cache);
/// - a capacity of at least the sliding window: older slots are overwritten (a ring), and the keys a query may see are
///   at most two contiguous runs, visited in position order.
///
/// `kv_cache_view<C, T>` works over two spans the caller owns (keys and values), sized by `required_size`;
/// `kv_cache<C, T, Layers, Capacity>` owns fixed-size arrays and hands out a view.
///
/// \snippet ml/kv_cache_demo.cpp kv_cache
///
/// **Relationships.** `ml/transformer_config.hpp`, `ml/attention.hpp` (`key_range`, `attention_window`),
/// `ml/attention_layer.hpp` (the user). Group: \ref core.
///
/// **Implementations.** One layout; the view over the caller's memory and the owning form.
///
/// **Guarantees.** `constexpr`, no allocation, O(`kv_size`) per append, O(1) to find the visible runs. A view does
/// not own its spans and copying it shares them; the owning form builds its view per call, so a copy of it is a
/// cache of its own. The cache records no length: the caller says which positions it appends and reads.
///
/// **Errors.** Preconditions, asserted: the spans have `required_size(layers, capacity)` elements; a layer is below
/// `layers`; a visible range is at most `capacity` long and ends at or before the last appended position plus one.
///
/// **Performance.** O(`kv_size`) copies per token per layer; benchmarked through `ml/transformer_block.hpp`.
///
/// **Tests and demos.** `tests/ml/kv_cache_test.cpp`, `demos/ml/kv_cache_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cassert>`, `<cstddef>`, `<span>`, `ml/attention.hpp`, `ml/row.hpp`,
/// `ml/transformer_config.hpp`.
///
/// **Known issues.** A `kv_cache` of a real model is large (Llama 3 8B: 128 KB per position across its 32 layers in
/// float): give it static storage, or use the view over memory of your own.
///
/// **Status.** Experimental, new in 0.8.0.

#ifndef snicholls_core_ml_kv_cache_hpp
#define snicholls_core_ml_kv_cache_hpp

#include <snicholls/core/ml/attention.hpp>
#include <snicholls/core/ml/row.hpp>
#include <snicholls/core/ml/transformer_config.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <span>

namespace snicholls::ml
{

/// \brief The visible keys and values of one head, as at most two contiguous runs in position order.
/// \tparam T The number type.
template <math::math_number T>
struct kv_runs
{
    std::array<std::span<const T>, 2> keys{};     ///< The keys, `head_dim` per position, end to end; the second may be empty.
    std::array<std::span<const T>, 2> values{};   ///< The values, laid out as the keys.
};

/// \brief A KV cache over the caller's memory: `[layer][kv_head][slot][head_dim]`, position `p` in slot `p % capacity`.
/// \ingroup core
/// \tparam C The configuration.
/// \tparam T The number type.
template <transformer_config C, math::math_number T = float>
class kv_cache_view
{
    static_assert(valid(C), "snicholls::ml::kv_cache_view: the transformer_config is not valid");

public:
    /// \brief The elements each of the two spans needs.
    /// \param layers The layers.
    /// \param capacity The slots per head.
    /// \return `layers · n_kv_heads · capacity · head_dim`.
    [[nodiscard]] static constexpr std::size_t required_size(std::size_t layers, std::size_t capacity) noexcept
    {
        return layers * C.n_kv_heads * capacity * C.head_dim;
    }

    /// \brief An empty view: no layers, no slots.
    constexpr kv_cache_view() noexcept = default;

    /// \brief A view over the caller's keys and values.
    /// \param keys `required_size(layers, capacity)` elements.
    /// \param values The same.
    /// \param layers The layers.
    /// \param capacity The slots per head: the longest sequence, or at least the configuration's window.
    constexpr kv_cache_view(std::span<T> keys, std::span<T> values, std::size_t layers, std::size_t capacity) noexcept
        : keys_(keys), values_(values), layers_(layers), capacity_(capacity)
    {
        assert(keys.size() == required_size(layers, capacity) && values.size() == keys.size());
        assert(C.window == 0 || capacity == 0 || capacity >= C.window);
    }

    /// \brief Stores the keys and values of one position: every key-value head of one layer.
    /// \param layer The layer.
    /// \param pos The position; it goes to slot `pos % capacity`.
    /// \param k `kv_size` keys, head by head.
    /// \param v `kv_size` values, head by head.
    template <row K, row V>
        requires std::same_as<row_value_t<K>, T> && std::same_as<row_value_t<V>, T>
    constexpr void append(std::size_t layer, std::size_t pos, const K& k, const V& v) const
    {
        const auto ks = as_row(k);
        const auto vs = as_row(v);
        assert(layer < layers_ && capacity_ > 0 && ks.size() == C.kv_size() && vs.size() == C.kv_size());
        for (std::size_t h = 0; h < C.n_kv_heads; ++h) {
            const std::size_t at = offset(layer, h, pos % capacity_);
            for (std::size_t i = 0; i < C.head_dim; ++i) {
                keys_[at + i] = ks[h * C.head_dim + i];
                values_[at + i] = vs[h * C.head_dim + i];
            }
        }
    }

    /// \brief The keys and values one head may see, as at most two contiguous runs in position order.
    /// \param layer The layer.
    /// \param kv_head The key-value head.
    /// \param r The visible positions, at most `capacity` of them (from `attention_window`).
    /// \return The runs.
    [[nodiscard]] constexpr kv_runs<T> visible(std::size_t layer, std::size_t kv_head, key_range r) const
    {
        assert(layer < layers_ && kv_head < C.n_kv_heads && r.size() <= capacity_);
        kv_runs<T> out{};
        if (r.size() == 0)
            return out;
        const std::size_t first = r.first % capacity_;
        const std::size_t n1 = first + r.size() <= capacity_ ? r.size() : capacity_ - first;
        const std::size_t base = offset(layer, kv_head, 0);
        out.keys[0] = std::span<const T>(keys_).subspan(base + first * C.head_dim, n1 * C.head_dim);
        out.values[0] = std::span<const T>(values_).subspan(base + first * C.head_dim, n1 * C.head_dim);
        out.keys[1] = std::span<const T>(keys_).subspan(base, (r.size() - n1) * C.head_dim);
        out.values[1] = std::span<const T>(values_).subspan(base, (r.size() - n1) * C.head_dim);
        return out;
    }

    /// \brief The layers.
    /// \return The number given at construction.
    [[nodiscard]] constexpr std::size_t layers() const noexcept { return layers_; }
    /// \brief The slots per head.
    /// \return The number given at construction.
    [[nodiscard]] constexpr std::size_t capacity() const noexcept { return capacity_; }

private:
    constexpr std::size_t offset(std::size_t layer, std::size_t h, std::size_t slot) const noexcept
    {
        return ((layer * C.n_kv_heads + h) * capacity_ + slot) * C.head_dim;
    }

    std::span<T> keys_{};
    std::span<T> values_{};
    std::size_t layers_ = 0;
    std::size_t capacity_ = 0;
};

/// \brief A KV cache that owns fixed-size storage: `Layers` layers of `Capacity` slots per key-value head.
/// \ingroup core
/// \tparam C The configuration.
/// \tparam T The number type.
/// \tparam Layers The layers.
/// \tparam Capacity The slots per head.
template <transformer_config C, math::math_number T, std::size_t Layers, std::size_t Capacity>
class kv_cache
{
    static constexpr std::size_t size = kv_cache_view<C, T>::required_size(Layers, Capacity);

public:
    /// \brief A view of this cache, for the attention functions; made per call, so a copy of the cache is independent.
    /// \return The view.
    [[nodiscard]] constexpr kv_cache_view<C, T> view() noexcept { return {keys_, values_, Layers, Capacity}; }

private:
    std::array<T, size> keys_{};
    std::array<T, size> values_{};
};

} // namespace snicholls::ml

#endif // snicholls_core_ml_kv_cache_hpp
