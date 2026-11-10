//
//  static_map.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `containers::static_map` and `static_set`: tables sorted at compile time, looked up in O(log n), duplicates refused.
/// \ingroup core
///
/// **Overview.** A table known when the program is written (instrument codes to tick sizes, venue
/// identifiers to names, currency codes to minor units) should cost nothing to build and little to search.
/// `static_map` takes its entries in any order, sorts them at compile time, and finds a key by binary search;
/// two entries with one key are a compile error when the map is a constant. `static_set` is the same for keys
/// alone. The constructors are `consteval`: a map is always sorted and checked by the compiler. `constexpr_map` is the linear, two-way table for tens of entries; this is for hundreds or
/// thousands, or when the order of declaration should not matter.
///
/// \snippet containers/static_map_demo.cpp static_map
///
/// **Relationships.** `constexpr_map.hpp`; `meta/enums.hpp` for enumeration names. Group: \ref core.
///
/// **Guarantees.** `constexpr` throughout; `find` is O(log n) and `noexcept` when comparing keys is;
/// iteration is in key order; no allocation.
///
/// **Errors.** `at` throws `std::out_of_range` for a missing key; a duplicate key is a compile error.
///
/// **Performance.** `benchmarks/containers/perfect_map_benchmark.cpp`: 32,768 lookups of 512 four-letter
/// codes, Release, Apple clang 17, Xeon W-3245, 2026-11-08: 2.81 ms, against 2.90 ms for a binary search
/// written by hand; `perfect_map` takes 211 us for the same lookups and is the choice when order is not needed.
///
/// **Tests and demos.** `tests/containers/static_map_test.cpp`, `demos/containers/static_map_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<array>`, `<cstddef>`, `<functional>`, `<stdexcept>`, `<utility>`.
///
/// **Known issues.** Keys and values must be default-constructible and copyable (the entries live in a
/// `std::array`). The comparison is `std::less<>` by default.
///
/// **Status.** New in 0.5.0.

#ifndef snicholls_core_containers_static_map_hpp
#define snicholls_core_containers_static_map_hpp

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <utility>

namespace snicholls::containers {

    /// \brief A map of `N` entries, sorted by key at construction.
    /// \ingroup core
    /// \tparam Key   The key type.
    /// \tparam Value The value type.
    /// \tparam N     The number of entries.
    /// \tparam Less  The ordering of keys.
    template <typename Key, typename Value, std::size_t N, typename Less = std::less<>>
    class static_map {
    public:
        using value_type = std::pair<Key, Value>;                              ///< An entry.
        using const_iterator = typename std::array<value_type, N>::const_iterator;   ///< An iterator, in key order.

        /// \brief Sort the entries; refuse a duplicate key.
        /// \param entries The entries, in any order.
        /// A duplicate key is a compile error.
        consteval explicit static_map(std::array<value_type, N> entries) : entries_(std::move(entries))
        {
            std::sort(entries_.begin(), entries_.end(), [](const value_type& a, const value_type& b) { return Less{}(a.first, b.first); });
            for (std::size_t i = 1; i < N; ++i)
                if (!Less{}(entries_[i - 1].first, entries_[i].first))
                    throw std::invalid_argument("snicholls::containers::static_map: a key appears twice");
        }

        /// \brief The value of `key`, or null.
        /// \tparam K A type comparable with the keys (a `std::string_view` for `std::string_view` keys).
        /// \param key The key.
        /// \return A pointer into the map.
        template <typename K>
        [[nodiscard]] constexpr const Value* find(const K& key) const noexcept(noexcept(Less{}(key, key)))
        {
            const auto it = std::lower_bound(entries_.begin(), entries_.end(), key,
                                             [](const value_type& e, const K& k) { return Less{}(e.first, k); });
            return it != entries_.end() && !Less{}(key, it->first) ? &it->second : nullptr;
        }

        /// \brief Whether `key` is in the map.
        /// \tparam K A type comparable with the keys.
        /// \param key The key.
        /// \return Whether it is.
        template <typename K>
        [[nodiscard]] constexpr bool contains(const K& key) const noexcept(noexcept(find(key)))
        {
            return find(key) != nullptr;
        }

        /// \brief The value of `key`.
        /// \tparam K A type comparable with the keys.
        /// \param key The key.
        /// \return The value.
        /// \throws std::out_of_range when it is missing.
        template <typename K>
        [[nodiscard]] constexpr const Value& at(const K& key) const
        {
            const Value* v = find(key);
            if (v == nullptr)
                throw std::out_of_range("snicholls::containers::static_map::at: no such key");
            return *v;
        }

        /// \brief The number of entries.
        /// \return `N`.
        [[nodiscard]] static constexpr std::size_t size() noexcept { return N; }
        /// \brief The first entry, in key order.
        /// \return An iterator.
        [[nodiscard]] constexpr const_iterator begin() const noexcept { return entries_.begin(); }
        /// \brief One past the last entry.
        /// \return An iterator.
        [[nodiscard]] constexpr const_iterator end() const noexcept { return entries_.end(); }

    private:
        std::array<value_type, N> entries_;
    };

    /// \brief A set of `N` keys, sorted at construction.
    /// \ingroup core
    /// \tparam Key  The key type.
    /// \tparam N    The number of keys.
    /// \tparam Less The ordering.
    template <typename Key, std::size_t N, typename Less = std::less<>>
    class static_set {
    public:
        using const_iterator = typename std::array<Key, N>::const_iterator;   ///< An iterator, in order.

        /// \brief Sort the keys; refuse a duplicate.
        /// \param keys The keys, in any order.
        /// A duplicate is a compile error.
        consteval explicit static_set(std::array<Key, N> keys) : keys_(std::move(keys))
        {
            std::sort(keys_.begin(), keys_.end(), Less{});
            for (std::size_t i = 1; i < N; ++i)
                if (!Less{}(keys_[i - 1], keys_[i]))
                    throw std::invalid_argument("snicholls::containers::static_set: a key appears twice");
        }

        /// \brief Whether `key` is in the set.
        /// \tparam K A type comparable with the keys.
        /// \param key The key.
        /// \return Whether it is.
        template <typename K>
        [[nodiscard]] constexpr bool contains(const K& key) const noexcept(noexcept(Less{}(key, key)))
        {
            return std::binary_search(keys_.begin(), keys_.end(), key, Less{});
        }

        /// \brief The number of keys.
        /// \return `N`.
        [[nodiscard]] static constexpr std::size_t size() noexcept { return N; }
        /// \brief The first key.
        /// \return An iterator.
        [[nodiscard]] constexpr const_iterator begin() const noexcept { return keys_.begin(); }
        /// \brief One past the last key.
        /// \return An iterator.
        [[nodiscard]] constexpr const_iterator end() const noexcept { return keys_.end(); }

    private:
        std::array<Key, N> keys_;
    };

    /// \brief A `static_map` from pairs: `make_static_map<std::string_view, int>({{"usd", 2}, {"jpy", 0}})`.
    /// \ingroup core
    /// \tparam Key   The key type.
    /// \tparam Value The value type.
    /// \tparam N     The number of entries.
    /// \param entries The entries.
    /// \return The map.
    template <typename Key, typename Value, std::size_t N>
    [[nodiscard]] consteval static_map<Key, Value, N> make_static_map(const std::pair<Key, Value> (&entries)[N])
    {
        std::array<std::pair<Key, Value>, N> a{};
        for (std::size_t i = 0; i < N; ++i)
            a[i] = entries[i];
        return static_map<Key, Value, N>(a);
    }

} // namespace snicholls::containers

#endif /* snicholls_core_containers_static_map_hpp */
