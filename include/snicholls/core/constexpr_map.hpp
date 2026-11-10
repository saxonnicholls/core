//
//  constexpr_map.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The shape is the same, an aggregate over std::array<std::pair<Key, Value>, N>; core adds contains,
//  find_key, keys_unique, iteration, a factory and a throwing at.
//

/// \file
/// \brief `snicholls::constexpr_map`: a fixed table of key-value pairs, looked up at compile time or run time.
/// \ingroup core
///
/// **Overview.** A map whose contents are known when the program is written: an enumeration's
/// names, a protocol's codes, a unit's scale factors. It is an aggregate over a
/// `std::array` of pairs, so it is a literal type: declare it `constexpr`, look it up in a
/// `static_assert` or a `case` label, and nothing is built at run time:
///
/// \snippet constexpr_map_demo.cpp table
///
/// Lookups are linear, which for the tables this is for (tens of entries) beats a hash or a
/// tree at compile time and at run time. `find` returns a pointer or null; `at` throws, which
/// in a constant expression is a compile error; `find_key` looks the other way, for the
/// enumeration-to-name-and-back pair of tables that would otherwise be two.
///
/// **Relationships.** `snicholls::fixed_string` makes a compile-time key; `basic_type_registry`
/// binds names to types where the value is a type rather than a value. Group: \ref core.
///
/// **Guarantees.** An aggregate and a literal type: `constexpr` construction and every lookup
/// when `Key` and `Value` are literal types with a `constexpr operator==`. `find`,
/// `find_key`, `contains` and `keys_unique` are `noexcept`. No allocation, no run-time
/// initialisation. Keys need not be unique; `find` returns the first match, and
/// `keys_unique()` says whether that matters.
///
/// **Errors.** `at` throws `std::out_of_range` when the key is absent, at run time; in a
/// constant expression the throw makes the expression not constant, so the program does not
/// compile.
///
/// **Performance.** O(N) comparisons per lookup. Not benchmarked: a table of tens of entries
/// compared by `==` is a handful of instructions, and a constant-expression lookup is none.
///
/// **Tests and demos.** `tests/constexpr_map_test.cpp`, `demos/constexpr_map_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<stdexcept>`, `<utility>`.
///
/// **Known issues.** A key type with no `constexpr operator==` (a `std::string`) still works
/// at run time but not in a constant expression; use `std::string_view` or
/// `snicholls::fixed_string` for text keys.
///
/// **Status.** New in 0.5.0.

#ifndef snicholls_core_constexpr_map_hpp
#define snicholls_core_constexpr_map_hpp

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace snicholls {

    /// \brief A fixed table of `N` key-value pairs, an aggregate over `std::array`, looked up by linear search.
    /// \ingroup core
    /// \tparam Key   The key type; needs `operator==`.
    /// \tparam Value The value type; `find_key` needs its `operator==`.
    /// \tparam N     The number of entries.
    template <typename Key, typename Value, std::size_t N>
    struct constexpr_map {
        /// \brief The key type.
        using key_type = Key;
        /// \brief The value type.
        using mapped_type = Value;
        /// \brief One entry.
        using value_type = std::pair<Key, Value>;
        /// \brief The iterator over entries.
        using const_iterator = typename std::array<value_type, N>::const_iterator;

        /// \brief The entries, in the order written.
        std::array<value_type, N> entries;

        /// \brief The number of entries.
        /// \return `N`.
        [[nodiscard]] static constexpr std::size_t size() noexcept { return N; }

        /// \brief Whether there are no entries.
        /// \return `N == 0`.
        [[nodiscard]] static constexpr bool empty() noexcept { return N == 0; }

        /// \brief The value for `key`, or null.
        /// \param key The key.
        /// \return A pointer to the first entry's value whose key equals `key`, or null.
        [[nodiscard]] constexpr const Value* find(const Key& key) const noexcept
        {
            for (const auto& entry : entries)
                if (entry.first == key)
                    return &entry.second;
            return nullptr;
        }

        /// \brief The key for `value`, or null: the reverse lookup.
        /// \param value The value.
        /// \return A pointer to the first entry's key whose value equals `value`, or null.
        [[nodiscard]] constexpr const Key* find_key(const Value& value) const noexcept
        {
            for (const auto& entry : entries)
                if (entry.second == value)
                    return &entry.first;
            return nullptr;
        }

        /// \brief Whether some entry has this key.
        /// \param key The key.
        /// \return true when `find(key)` is not null.
        [[nodiscard]] constexpr bool contains(const Key& key) const noexcept { return find(key) != nullptr; }

        /// \brief The value for `key`.
        /// \param key The key.
        /// \return The first entry's value whose key equals `key`.
        /// \throws std::out_of_range when no entry has the key; in a constant expression, a compile error.
        [[nodiscard]] constexpr const Value& at(const Key& key) const
        {
            if (const Value* found = find(key))
                return *found;
            throw std::out_of_range("snicholls::constexpr_map::at: key not found");
        }

        /// \brief Whether every key appears once.
        /// \return true when no two entries share a key.
        [[nodiscard]] constexpr bool keys_unique() const noexcept
        {
            for (std::size_t i = 0; i < N; ++i)
                for (std::size_t j = i + 1; j < N; ++j)
                    if (entries[i].first == entries[j].first)
                        return false;
            return true;
        }

        /// \brief The first entry.
        /// \return An iterator to it.
        [[nodiscard]] constexpr const_iterator begin() const noexcept { return entries.begin(); }

        /// \brief One past the last entry.
        /// \return An iterator to it.
        [[nodiscard]] constexpr const_iterator end() const noexcept { return entries.end(); }
    };

    /// \brief Deduction guide: `constexpr_map{std::array{std::pair{k, v}, ...}}`.
    /// \tparam Key   The key type.
    /// \tparam Value The value type.
    /// \tparam N     The number of entries.
    template <typename Key, typename Value, std::size_t N>
    constexpr_map(std::array<std::pair<Key, Value>, N>) -> constexpr_map<Key, Value, N>;

    /// \brief Build a `constexpr_map` from its pairs: `make_constexpr_map(std::pair{k1, v1}, std::pair{k2, v2})`.
    /// \ingroup core
    /// \tparam Key   The key type, from the first pair.
    /// \tparam Value The value type, from the first pair.
    /// \tparam Rest  The remaining pairs, each convertible to `std::pair<Key, Value>`.
    /// \param first The first entry.
    /// \param rest  The other entries, in order.
    /// \return The map.
    template <typename Key, typename Value, typename... Rest>
    [[nodiscard]] constexpr constexpr_map<Key, Value, 1 + sizeof...(Rest)> make_constexpr_map(std::pair<Key, Value> first,
                                                                                            Rest... rest)
    {
        return constexpr_map<Key, Value, 1 + sizeof...(Rest)>{
            std::array<std::pair<Key, Value>, 1 + sizeof...(Rest)>{std::move(first), std::pair<Key, Value>(std::move(rest))...}};
    }
}

#endif /* snicholls_core_constexpr_map_hpp */
