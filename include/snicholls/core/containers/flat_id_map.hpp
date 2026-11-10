//
//  flat_id_map.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief An open-addressing map from 64-bit ids to small values: linear probing, no tombstones.
/// \ingroup core

#ifndef snicholls_core_containers_flat_id_map_hpp
#define snicholls_core_containers_flat_id_map_hpp

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace snicholls::containers {

/// \brief A hash map from `std::uint64_t` ids to trivially copyable values, stored in one flat array.
/// \ingroup core
///
/// \overview
/// Order books look orders up by exchange id on every cancel, execute and replace. `std::unordered_map` allocates
/// a node per insert (the reference implementations pay a heap allocation per order for it). This
/// map keeps `{key, value}` pairs in one power-of-two array, finds them by linear probing from a Fibonacci hash
/// (which spreads the sequential ids exchanges assign), and deletes by backward shifting, so there are no
/// tombstones and lookups never degrade after many erases. The load factor stays at or below one half.
///
/// Use it for integer ids mapped to pointers or indices. Not for string keys, or values that are not trivially
/// copyable (they are moved by bitwise assignment during probing and growth).
///
/// \relations
/// - Used by limit order books (order id -> order node).
/// - Module: \ref core.
///
/// \impls
/// None: a single implementation.
///
/// \contracts
/// - **Thread safety:** none internal; one writer, or many readers with no writer.
/// - **Exception safety:** `insert` and `reserve` are strong (growth allocates a new array before touching the old
///   one); `find`, `erase`, `clear` are `noexcept`.
/// - **Complexity:** expected O(1) for find, insert and erase; growth O(n), amortised O(1).
/// - **Ownership and lifetime:** pointers returned by `find` are invalidated by any insert (growth) or erase
///   (backward shift).
/// - **Invariants:** `size() <= capacity() / 2`; the key `empty_key` (`UINT64_MAX`) is never stored.
/// The reserved key `empty_key` is never stored: `insert` of it returns `false`, and `find`, `contains` and `erase`
/// of it report "absent".
///
/// \errors
/// `std::bad_alloc` from growth. Inserting an existing key returns `false` and changes nothing.
///
/// \examples
/// \snippet flat_id_map_demo.cpp map
///
/// \perf
/// Measured inside limit order books.
/// In core, benchmarks/containers/flat_id_map_benchmark.cpp compares it like for like with `std::unordered_map`, after a fairness
/// check; not yet measured in Release.
///
/// \testing
/// - Tests: tests/containers/flat_id_map_test.cpp (model-checked against `std::unordered_map`).
/// - Demo: demos/containers/flat_id_map_demo.cpp.
/// - Benchmark: benchmarks/containers/flat_id_map_benchmark.cpp.
///
/// \deps
/// CMake target `snicholls::core`; standard library only.
///
/// \gotchas
/// - `UINT64_MAX` is reserved; a feed that uses it as an id must remap it.
///
/// \status
/// Experimental.
///
/// \tparam Value a trivially copyable value type.
template <class Value>
class flat_id_map
{
    static_assert(std::is_trivially_copyable_v<Value>, "snicholls: flat_id_map values must be trivially copyable");

public:
    using key_type = std::uint64_t;   ///< The key type.
    using mapped_type = Value;        ///< The value type.

    /// \brief The reserved key that marks an empty slot.
    static constexpr key_type empty_key = std::numeric_limits<key_type>::max();

    /// \brief An empty map with room for `expected` entries before growing.
    /// \param expected expected number of entries.
    /// \throws std::bad_alloc on allocation failure; std::length_error when `expected` exceeds a quarter of the
    ///         address space.
    explicit flat_id_map(std::size_t expected = 64) { allocate(capacity_for(expected)); }

    flat_id_map(const flat_id_map&) = delete;              ///< Not copyable.
    flat_id_map& operator=(const flat_id_map&) = delete;   ///< Not copyable.
    /// \brief Move; the source may only be destroyed or assigned to afterwards.
    flat_id_map(flat_id_map&&) noexcept = default;
    /// \brief Move; the source may only be destroyed or assigned to afterwards.
    /// \return `*this`.
    flat_id_map& operator=(flat_id_map&&) noexcept = default;

    /// \brief Number of entries.
    /// \return the size.
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    /// \brief Whether the map is empty.
    /// \return `true` when empty.
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    /// \brief Slots in the array.
    /// \return the capacity.
    [[nodiscard]] std::size_t capacity() const noexcept { return mask_ + 1; }

    /// \brief The value for `key`, or null.
    /// \param key the key (`empty_key` is never present: null).
    /// \return a pointer to the value (invalidated by insert and erase), or null.
    [[nodiscard]] Value* find(key_type key) noexcept
    {
        if (key == empty_key) [[unlikely]]
            return nullptr;   // it would match the first empty slot
        for (std::size_t i = home(key);; i = (i + 1) & mask_) {
            slot& s = slots_[i];
            if (s.key == key) return &s.value;
            if (s.key == empty_key) return nullptr;
        }
    }

    /// \brief The value for `key`, or null (const).
    /// \param key the key.
    /// \return a pointer to the value, or null.
    [[nodiscard]] const Value* find(key_type key) const noexcept { return const_cast<flat_id_map*>(this)->find(key); }

    /// \brief Whether `key` is present.
    /// \param key the key.
    /// \return `true` when present.
    [[nodiscard]] bool contains(key_type key) const noexcept { return find(key) != nullptr; }

    /// \brief Insert `{key, value}` unless `key` is present.
    /// \param key the key.
    /// \param value the value.
    /// \return `true` when inserted; `false` when the key was already present, or is the reserved `empty_key`
    ///         (never stored; nothing changes).
    /// \throws std::bad_alloc when growth fails (map unchanged).
    bool insert(key_type key, Value value)
    {
        if (key == empty_key) [[unlikely]]
            return false;   // it marks an empty slot: storing it would count an entry that no lookup can find
        if ((size_ + 1) * 2 > capacity()) grow();
        for (std::size_t i = home(key);; i = (i + 1) & mask_) {
            slot& s = slots_[i];
            if (s.key == empty_key) {
                s.key = key;
                s.value = value;
                ++size_;
                return true;
            }
            if (s.key == key) return false;
        }
    }

    /// \brief Remove `key`.
    /// \param key the key (`empty_key`: nothing to remove).
    /// \return `true` when it was present.
    bool erase(key_type key) noexcept
    {
        if (key == empty_key) [[unlikely]]
            return false;
        std::size_t i = home(key);
        for (;; i = (i + 1) & mask_) {
            if (slots_[i].key == key) break;
            if (slots_[i].key == empty_key) return false;
        }
        // Backward-shift deletion: pull later members of the probe run into the hole.
        for (std::size_t j = (i + 1) & mask_;; j = (j + 1) & mask_) {
            if (slots_[j].key == empty_key) break;
            const std::size_t k = home(slots_[j].key);
            // Move j into i unless j's home lies cyclically in (i, j].
            const bool in_range = (i <= j) ? (i < k && k <= j) : (i < k || k <= j);
            if (!in_range) {
                slots_[i] = slots_[j];
                i = j;
            }
        }
        slots_[i].key = empty_key;
        --size_;
        return true;
    }

    /// \brief Remove every entry (keeps the capacity).
    void clear() noexcept
    {
        for (std::size_t i = 0; i <= mask_; ++i) slots_[i].key = empty_key;
        size_ = 0;
    }

    /// \brief Make room for `n` entries without growing.
    /// \param n the number of entries.
    /// \throws std::bad_alloc on allocation failure; std::length_error when `n` exceeds a quarter of the address
    ///         space (map unchanged).
    void reserve(std::size_t n)
    {
        if (capacity_for(n) > capacity()) rehash(capacity_for(n));
    }

    /// \brief Call `f(key, value)` for every entry (unspecified order).
    /// \param f the visitor.
    template <class F>
    void for_each(F&& f) const
    {
        for (std::size_t i = 0; i <= mask_; ++i)
            if (slots_[i].key != empty_key) f(slots_[i].key, slots_[i].value);
    }

private:
    struct slot
    {
        key_type key;
        Value value;
    };

    static std::size_t capacity_for(std::size_t n)
    {
        // Twice n, rounded up to a power of two: beyond max/4 that overflows (and the loop never ended).
        if (n > (std::numeric_limits<std::size_t>::max() >> 2)) [[unlikely]]
            throw std::length_error("snicholls::containers::flat_id_map: too many entries");
        std::size_t c = 16;
        while (c < n * 2) c *= 2;
        return c;
    }

    [[nodiscard]] std::size_t home(key_type key) const noexcept
    {
        return static_cast<std::size_t>((key * 0x9E3779B97F4A7C15ull) >> shift_);
    }

    void allocate(std::size_t cap)
    {
        slots_ = std::make_unique_for_overwrite<slot[]>(cap);
        for (std::size_t i = 0; i < cap; ++i) slots_[i].key = empty_key;
        mask_ = cap - 1;
        shift_ = 64 - static_cast<unsigned>(std::countr_zero(cap));
    }

    void grow() { rehash(capacity() * 2); }

    void rehash(std::size_t cap)
    {
        auto old = std::move(slots_);
        const std::size_t old_cap = mask_ + 1;
        const auto old_mask = mask_;
        const auto old_shift = shift_;
        try {
            allocate(cap);
        } catch (...) {
            slots_ = std::move(old);
            mask_ = old_mask;
            shift_ = old_shift;
            throw;
        }
        for (std::size_t i = 0; i < old_cap; ++i) {
            if (old[i].key == empty_key) continue;
            for (std::size_t j = home(old[i].key);; j = (j + 1) & mask_) {
                if (slots_[j].key == empty_key) {
                    slots_[j] = old[i];
                    break;
                }
            }
        }
    }

    std::unique_ptr<slot[]> slots_;
    std::size_t mask_ = 0;
    unsigned shift_ = 64;
    std::size_t size_ = 0;
};

} // namespace snicholls::containers

#endif /* snicholls_core_containers_flat_id_map_hpp */
