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
/// `paged_kv_cache_view<C, T, MaxPages>` is the form for serving: a pool of fixed pages and a page table per sequence,
/// so sequences that share a prefix share its pages, and `kv_page_hash` gives each page a content address (its
/// tokens and the address of the page before), so a request can find the pages an earlier one already computed. A
/// visible range is one run per page, still in position order, so a paged cache gives the flat cache's bits. The pool's
/// allocation, sharing and eviction are the serving layer's, above core.
///
/// \snippet ml/kv_cache_demo.cpp kv_cache
///
/// **Relationships.** `ml/transformer_config.hpp`, `ml/attention.hpp` (`key_range`, `attention_window`),
/// `ml/attention_layer.hpp` (the user). Group: \ref core.
///
/// **Implementations.** The flat or ring layout, over the caller's memory or owned; the paged layout over the
/// caller's pool.
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
/// **Dependencies.** `<algorithm>`, `<array>`, `<cassert>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<span>`,
/// `<type_traits>`, `hash/wyhash.hpp`, `ml/attention.hpp`, `ml/row.hpp`, `ml/transformer_config.hpp`.
///
/// **Known issues.** A `kv_cache` of a real model is large (Llama 3 8B: 128 KB per position across its 32 layers in
/// float): give it static storage, or use the view over memory of your own.
///
/// **Status.** Experimental, new in 0.8.0.

#ifndef snicholls_core_ml_kv_cache_hpp
#define snicholls_core_ml_kv_cache_hpp

#include <snicholls/core/hash/wyhash.hpp>
#include <snicholls/core/ml/attention.hpp>
#include <snicholls/core/ml/row.hpp>
#include <snicholls/core/ml/transformer_config.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief The visible keys and values of one head, as at most `N` contiguous runs in position order.
/// \tparam T The number type.
/// \tparam N The most runs: 2 for `kv_cache_view` (a ring wraps once), the pages a range can span for
/// `paged_kv_cache_view`.
template <math::math_number T, std::size_t N = 2>
struct kv_runs
{
    std::array<std::span<const T>, N> keys{};     ///< The keys, `head_dim` per position, end to end; later runs may be empty.
    std::array<std::span<const T>, N> values{};   ///< The values, laid out as the keys.
};

/// \brief A KV cache an attention layer can use: `append` one position's keys and values to a layer, and
/// `visible(layer, kv_head, range)` as `kv_runs` in position order.
template <typename S, typename T>
concept kv_store = requires(const S& s, std::size_t n, std::span<const T> row, key_range r) {
    s.append(n, n, row, row);
    s.visible(n, n, r);
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

/// \brief A paged KV cache over the caller's pool: fixed pages of `page_size` positions, a sequence's pages named by
/// its page table, so sequences that share a prefix can share its pages.
/// \ingroup core
///
/// The pool is `[page][layer][kv_head][slot][head_dim]`; logical page `i` of the sequence is physical page
/// `table[i]`, so position `p` is in page `table[p / page_size]`, slot `p % page_size`. Which pages are free, shared or
/// evicted is the caller's (the serving layer above core); this view only reads and writes them.
/// \tparam C The configuration.
/// \tparam T The number type.
/// \tparam MaxPages The most pages one visible range may span: `visible` returns `kv_runs<T, MaxPages>`.
template <transformer_config C, math::math_number T = float, std::size_t MaxPages = 64>
class paged_kv_cache_view
{
    static_assert(valid(C), "snicholls::ml::paged_kv_cache_view: the transformer_config is not valid");

public:
    /// \brief The elements each of the two pool spans needs.
    /// \param pages The pages in the pool.
    /// \param layers The layers.
    /// \param page_size The positions per page.
    /// \return `pages · layers · n_kv_heads · page_size · head_dim`.
    [[nodiscard]] static constexpr std::size_t required_size(std::size_t pages, std::size_t layers, std::size_t page_size) noexcept
    {
        return pages * layers * C.n_kv_heads * page_size * C.head_dim;
    }

    /// \brief A view of one sequence over the pool.
    /// \param keys The pool's keys, `required_size(pages, layers, page_size)` elements.
    /// \param values The pool's values, the same.
    /// \param layers The layers.
    /// \param page_size The positions per page.
    /// \param table The sequence's pages: logical page `i` is pool page `table[i]`.
    constexpr paged_kv_cache_view(std::span<T> keys, std::span<T> values, std::size_t layers, std::size_t page_size,
                                  std::span<const std::uint32_t> table) noexcept
        : keys_(keys), values_(values), layers_(layers), page_size_(page_size), table_(table)
    {
        assert(page_size > 0 && values.size() == keys.size() && keys.size() % required_size(1, layers, page_size) == 0);
    }

    /// \brief Stores the keys and values of one position: every key-value head of one layer.
    /// \param layer The layer.
    /// \param pos The position: its page must be in the table.
    /// \param k `kv_size` keys, head by head.
    /// \param v `kv_size` values, head by head.
    template <row K, row V>
        requires std::same_as<row_value_t<K>, T> && std::same_as<row_value_t<V>, T>
    constexpr void append(std::size_t layer, std::size_t pos, const K& k, const V& v) const
    {
        const auto ks = as_row(k);
        const auto vs = as_row(v);
        assert(layer < layers_ && pos / page_size_ < table_.size() && ks.size() == C.kv_size() && vs.size() == C.kv_size());
        for (std::size_t h = 0; h < C.n_kv_heads; ++h) {
            const std::size_t at = offset(table_[pos / page_size_], layer, h, pos % page_size_);
            for (std::size_t i = 0; i < C.head_dim; ++i) {
                keys_[at + i] = ks[h * C.head_dim + i];
                values_[at + i] = vs[h * C.head_dim + i];
            }
        }
    }

    /// \brief The keys and values one head may see, one run per page in position order.
    /// \param layer The layer.
    /// \param kv_head The key-value head.
    /// \param r The visible positions: their pages in the table, at most `MaxPages` of them.
    /// \return The runs.
    [[nodiscard]] constexpr kv_runs<T, MaxPages> visible(std::size_t layer, std::size_t kv_head, key_range r) const
    {
        assert(layer < layers_ && kv_head < C.n_kv_heads);
        kv_runs<T, MaxPages> out{};
        std::size_t run = 0;
        for (std::size_t p = r.first; p < r.last; ++run) {
            assert(run < MaxPages && p / page_size_ < table_.size());
            const std::size_t slot = p % page_size_, n = std::min(page_size_ - slot, r.last - p);
            const std::size_t at = offset(table_[p / page_size_], layer, kv_head, slot);
            out.keys[run] = std::span<const T>(keys_).subspan(at, n * C.head_dim);
            out.values[run] = std::span<const T>(values_).subspan(at, n * C.head_dim);
            p += n;
        }
        return out;
    }

    /// \brief The positions per page.
    /// \return The number given at construction.
    [[nodiscard]] constexpr std::size_t page_size() const noexcept { return page_size_; }

private:
    constexpr std::size_t offset(std::size_t page, std::size_t layer, std::size_t h, std::size_t slot) const noexcept
    {
        return (((page * layers_ + layer) * C.n_kv_heads + h) * page_size_ + slot) * C.head_dim;
    }

    std::span<T> keys_;
    std::span<T> values_;
    std::size_t layers_;
    std::size_t page_size_;
    std::span<const std::uint32_t> table_;
};

/// \brief The content address of a page of tokens: a hash of its tokens and of the page before it, so equal
/// addresses mean equal prefixes (up to a 64-bit collision; check the tokens on a hit).
///
/// A page's keys and values depend on every token before it, so a cache of pages is keyed by the whole prefix: the
/// address of page `i` is `kv_page_hash(address of page i - 1, its tokens)`, from `kv_page_hash_root` for the first.
/// Two requests whose prompts start alike reach the same addresses and can share those pages. Machine-independent:
/// the tokens are mixed as integers, never as bytes in memory.
/// \param parent The address of the page before, or `kv_page_hash_root`.
/// \param tokens The page's tokens.
/// \return The address.
template <typename Token>
    requires std::integral<Token>
[[nodiscard]] constexpr std::uint64_t kv_page_hash(std::uint64_t parent, std::span<const Token> tokens) noexcept
{
    std::uint64_t h = wymix(parent ^ 0xa0761d6478bd642full, tokens.size() ^ 0xe7037ed1a0b428dbull);
    for (const Token t : tokens)
        h = wymix(h ^ static_cast<std::uint64_t>(static_cast<std::make_unsigned_t<Token>>(t)), 0x8ebc6af09c88c6e3ull);
    return h;
}

/// \brief The parent of a sequence's first page in `kv_page_hash`.
inline constexpr std::uint64_t kv_page_hash_root = 0x589965cc75374cc3ull;

} // namespace snicholls::ml

#endif // snicholls_core_ml_kv_cache_hpp
