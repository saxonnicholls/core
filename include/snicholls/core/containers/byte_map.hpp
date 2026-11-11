//
//  byte_map.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `containers::byte_map_view` and `containers::byte_map`: byte strings to 32-bit values in one flat table of 16-byte slots, with a lookup split in two so a batch of them overlaps its cache misses.
/// \ingroup core
///
/// **Overview.** A hash table from byte strings (`std::string_view`) to `std::uint32_t` values, built once and then
/// read, laid out for the inner loop of a tokenizer: one flat array of 16-byte slots, linear probing, at most half
/// full. Each slot holds the first eight bytes of its key as a word (the *head*), its value and its length, so a key
/// of eight bytes or fewer, which is most of a BPE vocabulary, is compared as one word and a length and the slot is
/// the only memory a lookup touches. Longer keys compare their remaining bytes against an arena. An optional
/// reverse table gives the key of a value, which is what decoding needs.
///
/// A lookup is split in two: `probe(key)` hashes the key and prefetches its slot, `find(key, probe)` finishes it.
/// Probing a batch of keys and then finding them lets the cache misses of a table larger than L2 overlap instead of
/// queueing one behind the other. `operator()` does both and returns `byte_map_no_value` for a miss, so a map is an
/// `ml::bpe_vocabulary` as it stands.
///
/// Use it for a vocabulary or any dictionary of short byte strings that is built once and read often, from many
/// threads. Do not use it for a map that changes after it is built (there is no erase), for values that are not
/// 32-bit, or for a table known when the program is written (`perfect_map` probes nothing).
///
/// \snippet containers/byte_map_demo.cpp byte_map
///
/// **Relationships.** `ml/bpe.hpp` (`operator()` is its vocabulary), `containers/perfect_map.hpp` and
/// `containers/static_map.hpp` (tables known at compile time), `hash/wyhash.hpp` (the hash), `result.hpp` (the
/// errors). Group: \ref core.
///
/// **Implementations.** Two, with one lookup. `byte_map_view` works over three spans the caller owns (slots, key
/// arena, reverse table), so a table of any size lives where the caller puts it: the heap, a memory map, static
/// storage. `byte_map<Slots, ArenaBytes, Values>` owns fixed-size arrays of the same three, so a small table is built
/// by the compiler (its `consteval` constructor) or placed in static storage with no allocation at all.
///
/// **Guarantees.** No allocation, no exceptions at run time. `insert` and `build` are `constexpr` and `noexcept`,
/// with the strong guarantee: a failed insert leaves the map as it was. Lookups are `constexpr`, `noexcept` and
/// `const`, so a built map is read from any number of threads at once with no synchronisation; inserting while
/// another thread reads is a data race. A lookup is expected O(1): at most half the slots are full. A view does not
/// own its spans; they must outlive it, and it clears them when it is constructed. Keys are copied into the arena;
/// the caller's key strings need not outlive the map. The hash is machine-independent (wyhash over the bytes in
/// order), so a table built on one machine probes the same slots on another.
///
/// **Errors.** `insert` and `build` return `result<void, byte_map_error>`: `too_full` (the insert would make the
/// table more than half full), `duplicate` (the key, or with a reverse table the value, is already in the map),
/// `key_too_long` (more than 255 bytes), `arena_full` (the key's bytes do not fit the arena, or the arena would pass
/// 16 MB), `bad_value` (the value is `byte_map_no_value`, or with a reverse table is not below its size). `build`
/// stops at the first failure, keeping the entries before it. The `consteval` constructor of `byte_map` makes a
/// failure a compile error. Lookups do not fail: a miss is `byte_map_no_value`.
///
/// **Performance.** `benchmarks/containers/byte_map_benchmark.cpp`: 65,536 lookups, half hits and half misses (a key
/// with a byte added, or two keys glued, as a BPE merge asks), against `std::unordered_map<std::string_view,
/// std::uint32_t>` and an open-addressing table written by hand (`{string_view, value}` slots at the same load, the
/// whole key hashed with wyhash); a fairness check refuses to run unless all four agree. Release, Apple clang 17,
/// Xeon W-3245, 2026-11-12, median of 3:
///
/// | | 512 keys (the BPE fixture) | 100,000 keys (4 MB of slots) |
/// |---|---:|---:|
/// | `byte_map`, `operator()` | 921 us | 2.06 ms |
/// | `byte_map`, `probe` 32 then `find` | 990 us | 1.68 ms |
/// | open addressing by hand | 1.24 ms | 2.05 ms |
/// | `std::unordered_map` | 2.04 ms | 6.81 ms |
///
/// A table in L1 is 1.35x faster than the hand-written one (the head word answers most keys without touching the
/// arena). A table past L2 is memory-bound and level with it one lookup at a time; the batched probe-then-find is
/// what wins there, 1.2x the hand-written table and 4x `std::unordered_map`. In ts-moveables' BPE encoder (cl100k,
/// 100,256 tokens) the batched lookup took serial encoding of 12.3 MB from 253 to 179 ms.
///
/// **Tests and demos.** `tests/containers/byte_map_test.cpp`, `demos/containers/byte_map_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<bit>`, `<cstddef>`, `<cstdint>`, `<span>`, `<stdexcept>`, `<string_view>`,
/// `<type_traits>`, `<utility>`, `hash/wyhash.hpp`, `result.hpp`.
///
/// **Known issues.** Keys are at most 255 bytes and the arena at most 16 MB (the slot packs a 24-bit offset and an
/// 8-bit length). `byte_map_no_value` cannot be stored. The prefetch is GCC's and Clang's `__builtin_prefetch`; on
/// MSVC `probe` only hashes, since core includes no platform header. A `byte_map` of a large vocabulary is megabytes:
/// give it static storage, never the stack. Clang's default constant-evaluation budget limits a `consteval` build to
/// a few thousand short keys; `-fconstexpr-steps=` raises it.
///
/// **Status.** Experimental, new in 0.7.0.

#ifndef snicholls_core_containers_byte_map_hpp
#define snicholls_core_containers_byte_map_hpp

#include <snicholls/core/hash/wyhash.hpp>
#include <snicholls/core/result.hpp>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

namespace snicholls::containers {

    /// \brief The value a lookup returns for a key that is not in the map; it cannot be stored. Equal to
    /// `ml::bpe_no_rank`.
    inline constexpr std::uint32_t byte_map_no_value = 0xffffffffu;

    /// \brief Why an insert failed.
    enum class byte_map_error : std::uint8_t {
        too_full,       ///< The insert would make the table more than half full.
        duplicate,      ///< The key, or with a reverse table the value, is already in the map.
        key_too_long,   ///< The key is longer than 255 bytes.
        arena_full,     ///< The key's bytes do not fit the arena, or the arena would pass 16 MB.
        bad_value,      ///< The value is `byte_map_no_value`, or with a reverse table not below its size.
    };

    /// \brief One slot of the table: the key's first eight bytes as a little-endian word, its value, and its arena
    /// offset and length packed as `offset << 8 | length`. An empty slot has the value `byte_map_no_value`.
    struct byte_map_slot {
        std::uint64_t head = 0;                      ///< The first min(n, 8) bytes of the key, zero above them.
        std::uint32_t value = byte_map_no_value;     ///< The value, or `byte_map_no_value` when empty.
        std::uint32_t off_len = 0;                   ///< Arena offset << 8 | key length.
    };
    static_assert(sizeof(byte_map_slot) == 16 && std::is_trivially_copyable_v<byte_map_slot>);

    /// \brief An entry of the reverse table: where the key of a value is in the arena. Unused entries have the length
    /// `byte_map_no_value`.
    using byte_map_key_ref = std::pair<std::uint32_t, std::uint32_t>;

    /// \brief The result of `probe`: the key's head and hash, to hand to `find`.
    struct byte_map_probe {
        std::uint64_t head;   ///< The key's first min(n, 8) bytes as a word.
        std::uint64_t hash;   ///< The key's hash.
    };

    namespace detail {
        // Byte i of p as an unsigned value; constexpr, unlike a load through memcpy or reinterpret_cast
        constexpr std::uint64_t byte_map_byte(const char* p, std::size_t i) noexcept
        {
            return static_cast<unsigned char>(p[i]);
        }

        // Little-endian loads assembled with shifts: the same word on every machine, and folded to one load by GCC
        // and Clang on a little-endian target
        constexpr std::uint64_t byte_map_load4(const char* p) noexcept
        {
            return byte_map_byte(p, 0) | byte_map_byte(p, 1) << 8 | byte_map_byte(p, 2) << 16 | byte_map_byte(p, 3) << 24;
        }
        constexpr std::uint64_t byte_map_load8(const char* p) noexcept
        {
            return byte_map_load4(p) | byte_map_load4(p + 4) << 32;
        }

        // The first min(n, 8) bytes as a word, zero above them, reading only [p, p + n): from 4 to 7 bytes two
        // overlapping loads agree on the bytes they share, so OR is exact; below 4, the first, middle and last byte
        constexpr std::uint64_t byte_map_head(const char* p, std::size_t n) noexcept
        {
            if (n >= 8)
                return byte_map_load8(p);
            if (n >= 4)
                return byte_map_load4(p) | byte_map_load4(p + n - 4) << (8 * (n - 4));
            if (n == 0)
                return 0;
            return byte_map_byte(p, 0) | byte_map_byte(p, n >> 1) << (8 * (n >> 1)) | byte_map_byte(p, n - 1) << (8 * (n - 1));
        }

        // Keys of up to 16 bytes hash with one or two wymix rounds over words already loaded; longer keys by wyhash
        constexpr std::uint64_t byte_map_hash(const char* p, std::size_t n, std::uint64_t head) noexcept
        {
            if (n <= 8)
                return wymix(head ^ 0xa0761d6478bd642full, n ^ 0xe7037ed1a0b428dbull);
            if (n <= 16)
                return wymix(head ^ 0x8ebc6af09c88c6e3ull, byte_map_load8(p + n - 8) ^ (n * 0x589965cc75374cc3ull));
            return wyhash(std::string_view(p, n));
        }

        // The length of a key, or of the key of an entry (anything with a `first`): the constrained overload wins
        template <typename K>
        constexpr std::size_t byte_map_key_size(const K& k) noexcept
        {
            return std::string_view(k).size();
        }
        template <typename E>
            requires requires(const E& e) { e.first; }
        constexpr std::size_t byte_map_key_size(const E& e) noexcept
        {
            return std::string_view(e.first).size();
        }

        // The three spans of a table, as plain pointers: the view and the owning map share every operation
        struct byte_map_parts {
            byte_map_slot* slots;
            std::size_t capacity;   // a power of two, or 0
            char* arena;
            std::size_t arena_capacity;
            byte_map_key_ref* by_value;
            std::size_t values;
        };

        constexpr void byte_map_clear(const byte_map_parts& t) noexcept
        {
            for (std::size_t i = 0; i < t.capacity; ++i)
                t.slots[i] = byte_map_slot{};
            for (std::size_t i = 0; i < t.values; ++i)
                t.by_value[i] = byte_map_key_ref{0, byte_map_no_value};
        }

        constexpr byte_map_probe byte_map_probe_of(std::string_view key) noexcept
        {
            const std::uint64_t head = byte_map_head(key.data(), key.size());
            return {head, byte_map_hash(key.data(), key.size(), head)};
        }

        // The index of key's slot if it is there, else of the empty slot that ends its probe; capacity > 0
        constexpr std::size_t byte_map_slot_of(const byte_map_slot* slots, std::size_t capacity, const char* arena,
                                               std::string_view key, byte_map_probe pr) noexcept
        {
            const std::size_t mask = capacity - 1, n = key.size();
            for (std::size_t i = pr.hash & mask;; i = (i + 1) & mask) {
                const byte_map_slot& e = slots[i];
                if (e.value == byte_map_no_value)
                    return i;
                if (e.head == pr.head && (e.off_len & 0xffu) == n &&
                    (n <= 8 || std::string_view(arena + (e.off_len >> 8) + 8, n - 8) == std::string_view(key.data() + 8, n - 8)))
                    return i;
            }
        }

        constexpr result<void, byte_map_error> byte_map_insert(const byte_map_parts& t, std::size_t& size, std::size_t& arena_used,
                                                               std::string_view key, std::uint32_t value) noexcept
        {
            if (key.size() > 255)
                return err(byte_map_error::key_too_long);
            if (value == byte_map_no_value || (t.values != 0 && value >= t.values))
                return err(byte_map_error::bad_value);
            if (size + 1 > t.capacity / 2)
                return err(byte_map_error::too_full);
            const byte_map_probe pr = byte_map_probe_of(key);
            const std::size_t i = byte_map_slot_of(t.slots, t.capacity, t.arena, key, pr);
            if (t.slots[i].value != byte_map_no_value || (t.values != 0 && t.by_value[value].second != byte_map_no_value))
                return err(byte_map_error::duplicate);
            if (key.size() > t.arena_capacity - arena_used || arena_used + key.size() > (std::size_t{1} << 24))
                return err(byte_map_error::arena_full);
            const auto off = static_cast<std::uint32_t>(arena_used);
            for (std::size_t k = 0; k < key.size(); ++k)
                t.arena[arena_used + k] = key[k];
            arena_used += key.size();
            t.slots[i] = byte_map_slot{pr.head, value, off << 8 | static_cast<std::uint32_t>(key.size())};
            if (t.values != 0)
                t.by_value[value] = byte_map_key_ref{off, static_cast<std::uint32_t>(key.size())};
            ++size;
            return ok();
        }

        // The lookups, written once for both kinds of map: Derived gives parts() and the counts
        template <typename Derived>
        class byte_map_lookup {
        public:
            /// \brief The first half of a lookup: hash `key` and prefetch its slot. Probe a batch, then `find` each.
            /// \param key The key's bytes.
            /// \return The key's head and hash, for `find`.
            [[nodiscard]] constexpr byte_map_probe probe(std::string_view key) const noexcept
            {
                const byte_map_probe pr = byte_map_probe_of(key);
#if defined(__GNUC__) || defined(__clang__)
                const detail::byte_map_parts t = self().parts();
                if (!std::is_constant_evaluated() && t.capacity != 0)
                    __builtin_prefetch(t.slots + (pr.hash & (t.capacity - 1)));
#endif
                return pr;
            }

            /// \brief The second half of a lookup: the value of `key`, or `byte_map_no_value`.
            /// \param key The key's bytes, the same as given to `probe`.
            /// \param pr What `probe(key)` returned.
            /// \return The value, or `byte_map_no_value` if `key` is not in the map.
            [[nodiscard]] constexpr std::uint32_t find(std::string_view key, byte_map_probe pr) const noexcept
            {
                const detail::byte_map_parts t = self().parts();
                if (t.capacity == 0)
                    return byte_map_no_value;
                return t.slots[byte_map_slot_of(t.slots, t.capacity, t.arena, key, pr)].value;
            }

            /// \brief The value of `key`, or `byte_map_no_value`: `probe` and `find` in one. This is the
            /// `ml::bpe_vocabulary` call.
            /// \param key The key's bytes.
            /// \return The value, or `byte_map_no_value`.
            [[nodiscard]] constexpr std::uint32_t operator()(std::string_view key) const noexcept
            {
                return find(key, byte_map_probe_of(key));
            }

            /// \brief Whether `key` is in the map.
            /// \param key The key's bytes.
            /// \return Whether it is.
            [[nodiscard]] constexpr bool contains(std::string_view key) const noexcept { return (*this)(key) != byte_map_no_value; }

            /// \brief The key whose value is `value`, from the reverse table; empty if there is none or no table.
            /// \param value A value.
            /// \return A view into the map's arena.
            [[nodiscard]] constexpr std::string_view key_of(std::uint32_t value) const noexcept
            {
                const detail::byte_map_parts t = self().parts();
                if (value >= t.values || t.by_value[value].second == byte_map_no_value)
                    return {};
                return std::string_view(t.arena + t.by_value[value].first, t.by_value[value].second);
            }

            /// \brief The number of keys.
            /// \return The number of successful inserts.
            [[nodiscard]] constexpr std::size_t size() const noexcept { return self().size_; }

            /// \brief The number of slots; at most half of them are ever full.
            /// \return A power of two, or 0.
            [[nodiscard]] constexpr std::size_t capacity() const noexcept { return self().parts().capacity; }

            /// \brief The bytes of the arena in use: the sum of the keys' lengths.
            /// \return The bytes used.
            [[nodiscard]] constexpr std::size_t arena_used() const noexcept { return self().arena_used_; }

        private:
            constexpr const Derived& self() const noexcept { return static_cast<const Derived&>(*this); }
        };
    } // namespace detail

    /// \brief A byte-string map over three spans the caller owns: the slots, the key arena and, optionally, the
    /// reverse table.
    /// \ingroup core
    ///
    /// Size the spans with `required_slots` and `required_arena`. The view clears them when it is constructed and
    /// writes only inside them. Copying a view copies the pointers, not the table: the copies share it.
    class byte_map_view : public detail::byte_map_lookup<byte_map_view> {
        friend class detail::byte_map_lookup<byte_map_view>;

    public:
        /// \brief The slots a table of `n` keys needs: the smallest power of two of at least `2n`, at least 16.
        /// \param n The number of keys.
        /// \return The number of slots.
        [[nodiscard]] static constexpr std::size_t required_slots(std::size_t n) noexcept
        {
            return std::bit_ceil(n * 2 < 16 ? std::size_t{16} : n * 2);
        }

        /// \brief The arena bytes a set of keys needs: the sum of their lengths.
        /// \tparam R A range of keys, or of pairs whose `first` is a key.
        /// \param keys The keys.
        /// \return The bytes.
        template <typename R>
        [[nodiscard]] static constexpr std::size_t required_arena(const R& keys) noexcept
        {
            std::size_t n = 0;
            for (const auto& k : keys)
                n += detail::byte_map_key_size(k);
            return n;
        }

        /// \brief An empty view with no storage: every lookup misses and every insert is `too_full`.
        constexpr byte_map_view() noexcept = default;

        /// \brief A view over the caller's storage, cleared.
        /// \param slots The slots; only the largest power of two that fits is used.
        /// \param arena Where the keys' bytes go.
        /// \param by_value The reverse table, one entry per value `0 .. by_value.size() - 1`; empty for none, and
        /// then values need not be distinct or small.
        constexpr byte_map_view(std::span<byte_map_slot> slots, std::span<char> arena, std::span<byte_map_key_ref> by_value = {}) noexcept
            : parts_{slots.data(), slots.empty() ? 0 : std::bit_floor(slots.size()), arena.data(), arena.size(), by_value.data(), by_value.size()}
        {
            detail::byte_map_clear(parts_);
        }

        /// \brief Add `key` with `value`. Strong guarantee: on failure the map is unchanged.
        /// \param key The key's bytes, copied into the arena.
        /// \param value Its value: not `byte_map_no_value`, and below `by_value.size()` when there is a reverse table.
        /// \return Nothing, or why it failed.
        constexpr result<void, byte_map_error> insert(std::string_view key, std::uint32_t value) noexcept
        {
            return detail::byte_map_insert(parts_, size_, arena_used_, key, value);
        }

        /// \brief Insert every entry, in order, stopping at the first failure (the entries before it stay).
        /// \tparam R A range of pairs: a key convertible to `std::string_view`, a value convertible to `std::uint32_t`.
        /// \param entries The entries.
        /// \return Nothing, or why the first failing insert failed.
        template <typename R>
        constexpr result<void, byte_map_error> build(const R& entries) noexcept
        {
            for (const auto& [k, v] : entries)
                if (auto r = insert(std::string_view(k), static_cast<std::uint32_t>(v)); r.is_err())
                    return r;
            return ok();
        }

    private:
        constexpr detail::byte_map_parts parts() const noexcept { return parts_; }

        detail::byte_map_parts parts_{nullptr, 0, nullptr, 0, nullptr, 0};
        std::size_t size_ = 0;
        std::size_t arena_used_ = 0;
    };

    /// \brief A byte-string map that owns fixed-size storage, so it can be built by the compiler or live in static
    /// storage with no allocation.
    /// \ingroup core
    /// \tparam Slots      The number of slots: a power of two; holds `Slots / 2` keys.
    /// \tparam ArenaBytes The bytes for the keys.
    /// \tparam Values     The size of the reverse table (values `0 .. Values - 1`); 0 for none.
    template <std::size_t Slots, std::size_t ArenaBytes, std::size_t Values = 0>
    class byte_map : public detail::byte_map_lookup<byte_map<Slots, ArenaBytes, Values>> {
        static_assert(std::has_single_bit(Slots), "snicholls::containers::byte_map: Slots must be a power of two");
        friend class detail::byte_map_lookup<byte_map>;

    public:
        /// \brief An empty map.
        constexpr byte_map() noexcept { detail::byte_map_clear(parts()); }

        /// \brief A map of `entries`, built by the compiler; a failing insert is a compile error.
        /// \tparam N The number of entries.
        /// \param entries Pairs of key and value.
        template <std::size_t N>
        consteval explicit byte_map(const std::array<std::pair<std::string_view, std::uint32_t>, N>& entries) : byte_map()
        {
            if (build(entries).is_err())
                throw std::invalid_argument("snicholls::containers::byte_map: an entry does not fit (too full, duplicate, too long or a bad value)");
        }

        /// \brief Add `key` with `value`. Strong guarantee: on failure the map is unchanged.
        /// \param key The key's bytes, copied into the arena.
        /// \param value Its value: not `byte_map_no_value`, and below `Values` when there is a reverse table.
        /// \return Nothing, or why it failed.
        constexpr result<void, byte_map_error> insert(std::string_view key, std::uint32_t value) noexcept
        {
            return detail::byte_map_insert(parts(), size_, arena_used_, key, value);
        }

        /// \brief Insert every entry, in order, stopping at the first failure (the entries before it stay).
        /// \tparam R A range of pairs: a key convertible to `std::string_view`, a value convertible to `std::uint32_t`.
        /// \param entries The entries.
        /// \return Nothing, or why the first failing insert failed.
        template <typename R>
        constexpr result<void, byte_map_error> build(const R& entries) noexcept
        {
            for (const auto& [k, v] : entries)
                if (auto r = insert(std::string_view(k), static_cast<std::uint32_t>(v)); r.is_err())
                    return r;
            return ok();
        }

    private:
        // The pointers are made on each call, never stored, so a copy of the map is a map of its own
        constexpr detail::byte_map_parts parts() const noexcept
        {
            auto& self = const_cast<byte_map&>(*this);
            return {self.slots_.data(), Slots, self.arena_.data(), ArenaBytes, self.by_value_.data(), Values};
        }

        std::array<byte_map_slot, Slots> slots_{};
        std::array<char, ArenaBytes> arena_{};
        std::array<byte_map_key_ref, Values> by_value_{};
        std::size_t size_ = 0;
        std::size_t arena_used_ = 0;
    };

} // namespace snicholls::containers

#endif /* snicholls_core_containers_byte_map_hpp */
