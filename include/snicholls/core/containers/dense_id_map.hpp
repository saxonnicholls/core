//
//  dense_id_map.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief An id map for (mostly) sequential ids: a growing direct-indexed window, a hash map for outliers.
/// \ingroup core

#ifndef snicholls_core_containers_dense_id_map_hpp
#define snicholls_core_containers_dense_id_map_hpp

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include <snicholls/core/containers/flat_id_map.hpp>

namespace snicholls::containers {

/// \brief Maps ids to pointers with one array lookup when ids are dense and increasing (exchange order refs).
/// \ingroup core
///
/// \overview
/// Feeds such as NASDAQ ITCH number orders sequentially per day, so an order id is almost an array index. This map
/// keeps a window `[base, base + size)` of slots indexed by `id - base`; a lookup is one subtraction, one compare and
/// one load - no hashing, no probing, no growth rehash of the whole table (the window grows by doubling and copies
/// slots, never re-inserts). Ids outside the window, or too far ahead of it (`max_window`), go to a
/// `snicholls::containers::flat_id_map`, so arbitrary ids still work.
///
/// Profiling limit order books showed id-map growth at about 20% of a large book's add path; this map is the fix for
/// sequential feeds. For random ids (client order ids, UUID-derived) use
/// `flat_id_map` directly.
///
/// \relations
/// - Same interface as `snicholls::containers::flat_id_map` (find, insert, erase, contains, clear, size), so the order
///   books take either (`id_index` in the book policy, lob/book_core.hpp).
/// - Module: \ref core.
///
/// \impls
/// | Map | Choose it when |
/// |---|---|
/// | `flat_id_map<V>` | ids arbitrary |
/// | `dense_id_map<V>` | ids mostly increasing and dense (exchange order refs) |
///
/// \contracts
/// - **Thread safety:** none internal.
/// - **Exception safety:** `insert` is strong (growth allocates before changing anything); the rest is `noexcept`.
/// - **Complexity:** O(1) in the window; expected O(1) in the fallback; growth amortised O(1).
/// - **Memory:** one pointer per id in the window, including ids already erased below the highest live id; the window
///   start moves up (`compact`) when its first half is empty.
/// - **Invariants:** a null slot means absent, so `Value` must be a pointer type; `empty_key` is never stored.
///
/// \errors
/// `std::bad_alloc` from growth. Inserting an existing key returns `false`.
///
/// \examples
/// \snippet dense_id_map_demo.cpp dense
///
/// \perf
/// Inside a ladder book on sequential-id workloads: 1.3x-2.3x the `flat_id_map` book (Ubuntu Release, 2026-10-18).
/// In core, benchmarks/containers/dense_id_map_benchmark.cpp compares it like for like with `std::unordered_map`, after a fairness
/// check; not yet measured in Release.
///
/// \testing
/// - Tests: tests/containers/dense_id_map_test.cpp (model-checked against `std::unordered_map`).
/// - Demo: demos/containers/dense_id_map_demo.cpp.
/// - Benchmark: benchmarks/containers/dense_id_map_benchmark.cpp.
///
/// \deps
/// CMake target `snicholls::core`; standard library only.
///
/// \gotchas
/// - Pointers returned by `find` are invalidated by `insert` (growth) and `erase`/`compact`.
///
/// \status
/// Experimental.
///
/// \tparam Value a pointer type.
template <class Value>
class dense_id_map
{
    static_assert(std::is_pointer_v<Value>, "snicholls: dense_id_map values must be pointers (null marks an empty slot)");

public:
    using key_type = std::uint64_t;   ///< The key type.
    using mapped_type = Value;        ///< The value type.

    /// \brief The reserved key (as in `flat_id_map`).
    static constexpr key_type empty_key = flat_id_map<Value>::empty_key;

    /// \brief An empty map.
    /// \param expected expected live entries (sizes the first window).
    /// \param max_window the widest window, in ids; ids further ahead go to the fallback.
    explicit dense_id_map(std::size_t expected = 64, std::size_t max_window = std::size_t{1} << 26)
        : fallback_(16), initial_(std::max<std::size_t>(expected, 64)), max_window_(max_window)
    {
    }

    /// \brief Entries in the map.
    /// \return the count.
    [[nodiscard]] std::size_t size() const noexcept { return in_window_ + fallback_.size(); }
    /// \brief Whether the map is empty.
    /// \return `true` when empty.
    [[nodiscard]] bool empty() const noexcept { return size() == 0; }
    /// \brief Slots in the window.
    /// \return the window size.
    [[nodiscard]] std::size_t window() const noexcept { return slots_.size(); }
    /// \brief Entries held by the fallback hash map.
    /// \return the count.
    [[nodiscard]] std::size_t outliers() const noexcept { return fallback_.size(); }

    /// \brief The value for `key`, or null.
    /// \param key the key.
    /// \return a pointer to the value, or null.
    [[nodiscard]] Value* find(key_type key) noexcept
    {
        const key_type i = key - base_;
        if (i < slots_.size()) return slots_[i] ? &slots_[i] : nullptr;
        return fallback_.find(key);
    }
    /// \brief The value for `key`, or null (const).
    /// \param key the key.
    /// \return a pointer to the value, or null.
    [[nodiscard]] const Value* find(key_type key) const noexcept { return const_cast<dense_id_map*>(this)->find(key); }
    /// \brief Whether `key` is present.
    /// \param key the key.
    /// \return `true` when present.
    [[nodiscard]] bool contains(key_type key) const noexcept { return find(key) != nullptr; }

    /// \brief Insert `{key, value}` unless present.
    /// \param key the key.
    /// \param value a non-null pointer.
    /// \return `true` when inserted; `false` when the key was already present, or is the reserved `empty_key`
    ///         (never stored; nothing changes).
    /// \throws std::bad_alloc on growth failure (map unchanged).
    bool insert(key_type key, Value value)
    {
        if (key == empty_key) [[unlikely]]
            return false;   // reserved, as in flat_id_map
        if (slots_.empty() && fallback_.empty()) {
            base_ = key;
            slots_.assign(initial_, nullptr);
        }
        key_type i = key - base_;
        if (i >= slots_.size() && key >= base_ && i < max_window_ && !fallback_.contains(key)) {
            grow(static_cast<std::size_t>(i) + 1);
            i = key - base_;
        }
        if (i < slots_.size()) {
            if (slots_[i]) return false;
            slots_[i] = value;
            ++in_window_;
            if (i < low_) low_ = static_cast<std::size_t>(i);   // a re-used id below the lowest live one
            return true;
        }
        return fallback_.insert(key, value);
    }

    /// \brief Remove `key`.
    /// \param key the key.
    /// \return `true` when it was present.
    bool erase(key_type key) noexcept
    {
        const key_type i = key - base_;
        if (i < slots_.size()) {
            if (!slots_[i]) return false;
            slots_[i] = nullptr;
            --in_window_;
            if (i == low_) advance_low();
            return true;
        }
        return fallback_.erase(key);
    }

    /// \brief Remove every entry (keeps the window's capacity).
    void clear() noexcept
    {
        std::fill(slots_.begin(), slots_.end(), nullptr);
        slots_.clear();
        fallback_.clear();
        in_window_ = 0;
        low_ = 0;
        base_ = 0;
    }

    /// \brief Make room for `n` entries (sizes the first window).
    /// \param n the number of entries.
    void reserve(std::size_t n)
    {
        initial_ = std::max(initial_, std::min(n, max_window_));
    }

private:
    // Lowest possibly-occupied slot, advanced lazily as the front empties.
    void advance_low() noexcept
    {
        while (low_ < slots_.size() && !slots_[low_]) ++low_;
    }

    void grow(std::size_t needed)
    {
        // Strong guarantee: everything that can throw (the new window, the list of fallback keys it will cover) is
        // allocated first; the map changes only after that, in steps that cannot throw.
        // Slide the window up first if its front half is empty (ids below the lowest live one are never reused).
        advance_low();   // moves a lazy cursor over empty slots only: no observable change
        const std::size_t drop = low_;
        std::size_t size = std::max<std::size_t>(slots_.size(), 64);
        while (size < needed - drop) size *= 2;
        const bool slide = size == slots_.size() && drop > 0;
        std::vector<Value> next;
        if (!slide) next.assign(std::max(size, needed - drop), nullptr);
        const key_type new_base = base_ + drop;
        const std::size_t new_size = slide ? slots_.size() : next.size();
        // Keep the invariant "no fallback key lies inside the window": find the keys the new window covers.
        std::vector<std::pair<key_type, Value>> covered;
        fallback_.for_each([&](key_type k, Value v) {
            if (k - new_base < new_size) covered.emplace_back(k, v);
        });
        // Commit: nothing below allocates or throws.
        if (slide) {
            std::move(slots_.begin() + static_cast<std::ptrdiff_t>(drop), slots_.end(), slots_.begin());
            std::fill(slots_.end() - static_cast<std::ptrdiff_t>(drop), slots_.end(), nullptr);
        } else {
            std::copy(slots_.begin() + static_cast<std::ptrdiff_t>(drop), slots_.end(), next.begin());
            slots_.swap(next);
        }
        base_ = new_base;
        low_ = 0;
        for (const auto& [k, v] : covered) {
            fallback_.erase(k);
            slots_[k - base_] = v;
            ++in_window_;
        }
    }

    std::vector<Value> slots_;
    flat_id_map<Value> fallback_;
    key_type base_ = 0;
    std::size_t in_window_ = 0;
    std::size_t low_ = 0;
    std::size_t initial_;
    std::size_t max_window_;
};

} // namespace snicholls::containers

#endif /* snicholls_core_containers_dense_id_map_hpp */
