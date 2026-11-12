//
//  index_pool.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief A pool of trivially copyable slots addressed by index inside a caller's byte region, which may grow and move.
/// \ingroup core

#ifndef snicholls_core_containers_index_pool_hpp
#define snicholls_core_containers_index_pool_hpp

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <span>
#include <type_traits>

#include <snicholls/core/result.hpp>

namespace snicholls::containers {

/// \brief Why an `index_pool` refused a region or an allocation.
/// \ingroup core
enum class index_pool_error : std::uint8_t
{
    misaligned,        ///< the region does not start on `index_pool::region_alignment`
    too_small,         ///< the region cannot hold the pool's state (`attach`, `format`) or its used slots (`rebase`)
    full,              ///< every slot the region holds is in use: grow the region and `rebase`
    not_formatted,     ///< the region does not start with a pool's state (wrong magic)
    layout_mismatch,   ///< the region was formatted for another slot size or index width
    corrupt,           ///< the stored counters contradict each other or the region's size
};

/// \brief A pool of `T` slots addressed by `Index` from the start of a caller's region, with a free list of indices.
/// \ingroup core
///
/// \overview
/// `object_pool` hands out pointers, which mean nothing once the memory is mapped at another address. `index_pool`
/// hands out indices: slot `i` is at a fixed offset from the region's start, the free list is threaded through freed
/// slots as indices, and the pool's counters live in the first bytes of the region itself. So the region can be a
/// memory-mapped file that is closed and reopened elsewhere, or a buffer that grows by being copied: call `rebase`
/// with the new span and every index still names the same object. Indices are 32 bits by default, half the size of a
/// pointer, which also makes them the cheaper link in structures inside the region.
///
/// Use it for many fixed-size records that must persist, move or be shared between processes. Not for types with
/// pointers, virtual functions or owned resources (`T` must be trivially copyable), and not when addresses suffice:
/// `object_pool` is simpler and does not need a region.
///
/// Region layout, in native byte order:
///
///     [ state: 64 bytes, rounded up to region_alignment ][ slot 0 ][ slot 1 ] ... [ slot capacity()-1 ]
///
/// The state holds a magic number, the slot size and index width it was formatted for, the high-water mark, the
/// head of the free list and the live count. Slots past the high-water mark have never been used.
///
/// \relations
/// - `object_pool`, the pointer-addressed pool on chunks from a `std::pmr::memory_resource`.
/// - `mapped_header` (containers/mapped_header.hpp), the checked file header to put in front of a persisted region.
/// - ts-moveables' mapped storage, which keeps an `index_pool` in a memory-mapped file.
/// - Module: \ref core.
///
/// \impls
/// None: a single implementation, over `Index` = `std::uint32_t` (default) or `std::uint64_t`.
///
/// \contracts
/// - **Thread safety:** none internal; one writer. Readers of slots that no one destroys may run concurrently.
/// - **Exception safety:** nothing throws: every operation is `noexcept`, and refusals are `result` values.
/// - **Complexity:** `allocate` / `create` / `destroy` / access O(1); `format` O(1) (slots are not touched);
///   `attach` and `rebase` O(1) (the free list is not walked).
/// - **Ownership and lifetime:** the caller owns the region; the pool is a view of it, trivially copyable. The region
///   holds the pool's whole state, so two `index_pool` objects over one region are the same pool.
/// - **Invariants:** `live() <= high_water() <= capacity()`; the free list holds exactly `high_water() - live()`
///   indices, each below `high_water()`.
///
/// \errors
/// `format`, `attach` and `rebase` return `misaligned`, `too_small`, `not_formatted`, `layout_mismatch` or `corrupt`
/// (see `index_pool_error`); `allocate` and `create` return `full` when every slot is in use.
///
/// \examples
/// \snippet index_pool_demo.cpp pool
/// Growing the region by copying it, and carrying on with the same indices:
/// \snippet index_pool_demo.cpp rebase
///
/// \perf
/// benchmarks/containers/index_pool_benchmark.cpp compares it like for like with a hand-written index free list over
/// a `std::vector` and with `object_pool`, after a fairness check: 4,096 creates, 2,048 destroys and re-creates,
/// every record read, everything destroyed. Release, Apple Clang 17, Intel Xeon W-3245 (macOS 26), 2026-11-12, median
/// of 5 on a loaded machine: `index_pool` 23.7 µs (173 M items/s), the vector free list 32.5 µs, `object_pool`
/// 37.0 µs. The pool wins because its free list lives in the freed slots, not in a second vector.
///
/// \testing
/// - Tests: tests/containers/index_pool_test.cpp.
/// - Demo: demos/containers/index_pool_demo.cpp.
/// - Benchmark: benchmarks/containers/index_pool_benchmark.cpp.
///
/// \deps
/// `result.hpp`; standard library only.
///
/// \gotchas
/// - A double `destroy(i)` is not detected and links `i` into the free list twice, as `object_pool` does; and
///   `attach` trusts the free list it finds (it checks the counters, not each link), so a region written by a
///   different program must be trusted, or validated by its owner (`mapped_header`'s checksum covers only the header).
/// - The layout is native: a region is readable only by a build with the same `T` layout, index width and byte
///   order. `layout_mismatch` catches a different slot size or index width, not a reordered `T`.
/// - Slot memory past `high_water()` is never written by `format` or `rebase`, so a sparse file stays sparse.
/// - `allocate` returns a slot whose bytes are whatever was there; `create` writes a value.
///
/// \status
/// Experimental.
///
/// \tparam T the slot type, trivially copyable.
/// \tparam Index `std::uint32_t` or `std::uint64_t`.
template <class T, class Index = std::uint32_t>
class index_pool
{
    static_assert(std::is_trivially_copyable_v<T>, "snicholls: index_pool needs a trivially copyable T (it may be "
                                                   "copied, mapped or reopened byte for byte)");
    static_assert(std::is_same_v<Index, std::uint32_t> || std::is_same_v<Index, std::uint64_t>,
                  "snicholls: index_pool indices are std::uint32_t or std::uint64_t");

    struct state
    {
        std::uint64_t magic;
        std::uint64_t slot_size;
        std::uint64_t index_size;
        std::uint64_t high_water;
        std::uint64_t free_head;
        std::uint64_t live;
        std::uint64_t reserved[2];
    };
    static_assert(sizeof(state) == 64 && std::is_trivially_copyable_v<state>);   // read from a region with memcpy

    static constexpr std::size_t round_up(std::size_t n, std::size_t a) noexcept { return (n + a - 1) / a * a; }

public:
    using value_type = T;       ///< The slot type.
    using index_type = Index;   ///< The index type.

    /// \brief The index that names no slot: the end of the free list.
    static constexpr Index npos = std::numeric_limits<Index>::max();
    /// \brief The magic number at the start of a formatted region ("SNIDXPL1" read as little-endian).
    static constexpr std::uint64_t magic = 0x314c505844494e53ull;
    /// \brief The alignment the region must start on.
    static constexpr std::size_t region_alignment = std::max({alignof(T), alignof(Index), alignof(state)});
    /// \brief Bytes between consecutive slots: `T` or an index, whichever is larger, rounded to the alignment.
    static constexpr std::size_t slot_size = round_up(std::max(sizeof(T), sizeof(Index)),
                                                      std::max(alignof(T), alignof(Index)));
    /// \brief Bytes before slot 0: the pool's state.
    static constexpr std::size_t header_size = round_up(sizeof(state), region_alignment);

    /// \brief The region size that holds `slots` slots.
    /// \param slots the capacity wanted.
    /// \return `header_size + slots * slot_size`.
    [[nodiscard]] static constexpr std::size_t bytes_for(std::size_t slots) noexcept
    {
        return header_size + slots * slot_size;
    }

    /// \brief A pool over no region; every operation but `format` / `attach` is a precondition violation.
    index_pool() noexcept = default;

    /// \brief Start an empty pool in `region`, overwriting its first `header_size` bytes.
    /// \param region the caller's bytes, aligned to `region_alignment`, at least `header_size` long.
    /// \return the pool, or `misaligned` / `too_small`.
    [[nodiscard]] static result<index_pool, index_pool_error> format(std::span<std::byte> region) noexcept
    {
        if (reinterpret_cast<std::uintptr_t>(region.data()) % region_alignment != 0) return err(index_pool_error::misaligned);
        if (region.size() < header_size) return err(index_pool_error::too_small);
        index_pool p;
        p.base_ = region.data();
        p.capacity_ = capacity_of(region.size());
        p.st_ = ::new (static_cast<void*>(region.data())) state{magic, slot_size, sizeof(Index), 0, npos, 0, {0, 0}};
        return ok(p);
    }

    /// \brief Take up the pool already in `region` (written by `format` and used, here or in an earlier run).
    /// \param region the bytes, at the same or another address, at least as long as the used slots.
    /// \return the pool, or `misaligned`, `too_small`, `not_formatted`, `layout_mismatch` or `corrupt`.
    [[nodiscard]] static result<index_pool, index_pool_error> attach(std::span<std::byte> region) noexcept
    {
        index_pool p;
        if (auto r = p.rebase(region); r.is_err()) return err(r.unwrap_err_unchecked());
        return ok(p);
    }

    /// \brief Point the pool at its region after the region grew, moved or was remapped; indices keep their meaning.
    /// \param region the same bytes (a copy of them, or a longer mapping of them), wherever they now are.
    /// \return `ok()`, or the error of `attach`: the pool is unchanged then.
    [[nodiscard]] result<void, index_pool_error> rebase(std::span<std::byte> region) noexcept
    {
        if (reinterpret_cast<std::uintptr_t>(region.data()) % region_alignment != 0) return err(index_pool_error::misaligned);
        if (region.size() < header_size) return err(index_pool_error::too_small);
        // The bytes may hold another pool's state (another T or Index), or no object at all (a file mapped in): read
        // them as bytes, never through this pool's state type, which would be undefined (GCC at -O3 then reads stale
        // bytes). Only a valid header becomes this pool's state, constructed in place with the same value.
        state seen;
        std::memcpy(&seen, region.data(), sizeof(state));
        if (seen.magic != magic) return err(index_pool_error::not_formatted);
        if (seen.slot_size != slot_size || seen.index_size != sizeof(Index)) return err(index_pool_error::layout_mismatch);
        const std::size_t cap = capacity_of(region.size());
        if (seen.live > seen.high_water || (seen.free_head != npos && seen.free_head >= seen.high_water)
            || (seen.free_head == npos && seen.live != seen.high_water))
            return err(index_pool_error::corrupt);
        if (seen.high_water > cap) return err(index_pool_error::too_small);
        base_ = region.data();
        capacity_ = cap;
        st_ = ::new (static_cast<void*>(region.data())) state(seen);
        return ok();
    }

    /// \brief Take a free slot; its bytes are unspecified (see `create`).
    /// \return its index, or `full`.
    [[nodiscard]] result<Index, index_pool_error> allocate() noexcept
    {
        if (st_->free_head != npos) {
            const auto i = static_cast<Index>(st_->free_head);
            Index next;
            std::memcpy(&next, slot_bytes(i), sizeof next);
            st_->free_head = next;
            ++st_->live;
            return ok(i);
        }
        if (st_->high_water == capacity_) return err(index_pool_error::full);
        const auto i = static_cast<Index>(st_->high_water++);
        ++st_->live;
        return ok(i);
    }

    /// \brief Take a free slot and copy `value` into it.
    /// \param value the slot's value.
    /// \return its index, or `full`.
    [[nodiscard]] result<Index, index_pool_error> create(const T& value) noexcept
    {
        auto r = allocate();
        if (r.is_ok()) ::new (static_cast<void*>(slot_bytes(*r))) T(value);
        return r;
    }

    /// \brief Put slot `i` on the free list.
    /// \param i an index this pool handed out and has not been given back.
    void destroy(Index i) noexcept
    {
        const auto next = static_cast<Index>(st_->free_head);
        std::memcpy(slot_bytes(i), &next, sizeof next);
        st_->free_head = i;
        --st_->live;
    }

    /// \brief The object in slot `i`.
    /// \param i a live index.
    /// \return the object.
    [[nodiscard]] T& operator[](Index i) noexcept { return *std::launder(reinterpret_cast<T*>(slot_bytes(i))); }
    /// \brief The object in slot `i`.
    /// \param i a live index.
    /// \return the object.
    [[nodiscard]] const T& operator[](Index i) const noexcept
    {
        return *std::launder(reinterpret_cast<const T*>(slot_bytes(i)));
    }

    /// \brief Slots in use.
    /// \return the count.
    [[nodiscard]] std::size_t live() const noexcept { return static_cast<std::size_t>(st_->live); }
    /// \brief Slots ever used: every live or freed index is below this.
    /// \return the high-water mark.
    [[nodiscard]] std::size_t high_water() const noexcept { return static_cast<std::size_t>(st_->high_water); }
    /// \brief Slots the region holds (at most `npos`).
    /// \return the capacity.
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    /// \brief Whether the pool has a region.
    /// \return true after a successful `format`, `attach` or `rebase`.
    [[nodiscard]] bool attached() const noexcept { return st_ != nullptr; }
    /// \brief The start of the region.
    /// \return the address `rebase` was last given.
    [[nodiscard]] std::byte* data() const noexcept { return base_; }

private:
    static constexpr std::size_t capacity_of(std::size_t bytes) noexcept
    {
        return std::min<std::size_t>((bytes - header_size) / slot_size, static_cast<std::size_t>(npos));
    }
    std::byte* slot_bytes(Index i) const noexcept { return base_ + header_size + static_cast<std::size_t>(i) * slot_size; }

    std::byte* base_ = nullptr;
    state* st_ = nullptr;
    std::size_t capacity_ = 0;
};

} // namespace snicholls::containers

#endif /* snicholls_core_containers_index_pool_hpp */
