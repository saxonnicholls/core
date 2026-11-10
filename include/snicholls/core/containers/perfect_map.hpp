//
//  perfect_map.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `containers::perfect_map`: a map built by the compiler with a perfect hash: two hashes, one slot, one compare.
/// \ingroup core
///
/// **Overview.** For a table known when the program is written, a perfect hash places every key in its own
/// slot, so a lookup never probes. Hash-and-displace: hash keys into buckets; for each bucket, largest first,
/// find a displacement that sends all its keys to free slots; keep one displacement per bucket. A lookup is
/// `slot = h(key, displacement[h(key, 0) % buckets]) % slots` and one key comparison. The constructor is
/// `consteval`: the search always runs in the compiler, never at run time.
///
/// \snippet containers/perfect_map_demo.cpp perfect_map
///
/// **Relationships.** `containers/static_map.hpp` (binary search, ordered iteration), `constexpr_map.hpp`
/// (linear, two-way); `hash/wyhash.hpp`. Group: \ref core.
///
/// **Guarantees.** `find` is O(1): two hashes and one comparison, no branches on collisions. Keys are
/// integers, enumerations or anything convertible to `std::string_view`. Clang's default constant-evaluation
/// budget holds 1,024 text keys and 1,600 integer keys (GCC 13 more); `-fconstexpr-steps=` raises it.
///
/// **Errors.** A duplicate key is a compile error.
///
/// **Performance.** `benchmarks/containers/perfect_map_benchmark.cpp`: 32,768 lookups of 512 four-letter
/// codes, Release, Apple clang 17, Xeon W-3245, 2026-11-08: 211 us, against 685 us for `std::unordered_map`,
/// 2.81 ms for `static_map` and 2.90 ms for a binary search written by hand.
///
/// **Tests and demos.** `tests/containers/perfect_map_test.cpp`, `demos/containers/perfect_map_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<array>`, `<bit>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<stdexcept>`,
/// `<string_view>`, `<type_traits>`, `<utility>`, `hash/wyhash.hpp`.
///
/// **Known issues.** Space is about 1.5 slots per key, plus a 32-bit displacement per two keys.
///
/// **Status.** New in 0.3.0.

#ifndef snicholls_core_containers_perfect_map_hpp
#define snicholls_core_containers_perfect_map_hpp

#include <snicholls/core/hash/wyhash.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

namespace snicholls::containers {

    namespace detail {
        // A key's hash under a seed: integers and enumerations by splitmix64, text by wyhash.
        template <typename K>
            requires std::integral<K> || std::is_enum_v<K>
        constexpr std::uint64_t perfect_hash(const K& k, std::uint64_t seed) noexcept
        {
            return splitmix64(static_cast<std::uint64_t>(k) ^ (seed * 0x9e3779b97f4a7c15ull));
        }
        template <typename K>
            requires std::convertible_to<const K&, std::string_view>
        constexpr std::uint64_t perfect_hash(const K& k, std::uint64_t seed) noexcept
        {
            return wyhash(std::string_view(k), seed);
        }
    } // namespace detail

    /// \brief A map of `N` entries placed by a perfect hash at construction.
    /// \ingroup core
    /// \tparam Key   An integer, an enumeration, or a type convertible to `std::string_view`.
    /// \tparam Value The value type, default-constructible.
    /// \tparam N     The number of entries.
    template <typename Key, typename Value, std::size_t N>
    class perfect_map {
        static constexpr std::size_t buckets = std::max<std::size_t>(1, std::bit_ceil(N) / 2);
        static constexpr std::size_t slots = std::bit_ceil(N + N / 2 + 1);

    public:
        /// \brief Place every entry; refuse a duplicate key.
        /// \param entries The entries, in any order.
        /// A duplicate key is a compile error.
        consteval explicit perfect_map(const std::array<std::pair<Key, Value>, N>& entries)
        {
            std::array<std::size_t, N> order{}, home{};   // entries grouped by bucket, the fullest first; each key hashed once
            std::array<std::size_t, buckets> fill{};
            for (std::size_t i = 0; i < N; ++i) {
                order[i] = i;
                home[i] = bucket(entries[i].first);
                ++fill[home[i]];
            }
            std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
                return fill[home[a]] != fill[home[b]] ? fill[home[a]] > fill[home[b]] : home[a] < home[b];
            });
            std::array<std::size_t, N> at{};   // the trial slots of the bucket being placed
            for (std::size_t start = 0; start < N;) {
                const std::size_t b = home[order[start]], end = start + fill[b];
                for (std::size_t i = start; i < end; ++i)   // equal keys always share a bucket
                    for (std::size_t j = start; j < i; ++j)
                        if (entries[order[i]].first == entries[order[j]].first)
                            throw std::invalid_argument("snicholls::containers::perfect_map: a key appears twice");
                for (std::uint32_t d = 1;; ++d) {   // the first displacement that puts the whole bucket in free slots
                    bool fits = true;
                    for (std::size_t i = start; i < end && fits; ++i) {
                        at[i] = slot(entries[order[i]].first, d);
                        fits = !used_[at[i]];
                        for (std::size_t j = start; j < i && fits; ++j)
                            fits = at[j] != at[i];
                    }
                    if (!fits)
                        continue;
                    displacement_[b] = d;
                    for (std::size_t i = start; i < end; ++i) {
                        used_[at[i]] = true;
                        table_[at[i]] = entries[order[i]];
                    }
                    break;
                }
                start = end;
            }
        }

        /// \brief The value of `key`, or null: two hashes and one comparison.
        /// \tparam K A key, or a type comparable with keys (a `std::string_view`).
        /// \param key The key.
        /// \return A pointer into the map.
        template <typename K>
        [[nodiscard]] constexpr const Value* find(const K& key) const noexcept
        {
            const std::size_t s = slot(key, displacement_[bucket(key)]);
            return used_[s] && table_[s].first == key ? &table_[s].second : nullptr;
        }

        /// \brief Whether `key` is in the map.
        /// \tparam K A key, or a type comparable with keys.
        /// \param key The key.
        /// \return Whether it is.
        template <typename K>
        [[nodiscard]] constexpr bool contains(const K& key) const noexcept
        {
            return find(key) != nullptr;
        }

        /// \brief The number of entries.
        /// \return `N`.
        [[nodiscard]] static constexpr std::size_t size() noexcept { return N; }

    private:
        template <typename K>
        static constexpr std::size_t bucket(const K& k) noexcept
        {
            return static_cast<std::size_t>(detail::perfect_hash(k, 0) & (buckets - 1));
        }
        template <typename K>
        static constexpr std::size_t slot(const K& k, std::uint32_t d) noexcept
        {
            return static_cast<std::size_t>(detail::perfect_hash(k, d) & (slots - 1));
        }

        std::array<std::uint32_t, buckets> displacement_{};
        std::array<std::pair<Key, Value>, slots> table_{};
        std::array<bool, slots> used_{};
    };

} // namespace snicholls::containers

#endif /* snicholls_core_containers_perfect_map_hpp */
