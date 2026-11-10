//
//  attributes.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `attribute_set`, `attrs` and `attributes_of`: compile-time attributes declared beside a type, any number, keyed by type.
/// \ingroup core
///
/// **Overview.** A currency's colour, a P&L's format, an index's tooltip should be written once, beside the type,
/// and every printer (acout, a pivot table, imgui, a React front end) should read the same values. A type declares
/// its attributes with a `constexpr` function `attributes` in its own namespace, found by argument-dependent lookup;
/// it returns an `attribute_set` built with `attrs(...)`, one value per attribute type, as many as wanted. An
/// attribute is any literal type: core's display attributes (`display.hpp`) or the caller's own (a font, a column
/// width). An enumeration gives each enumerator its own set with `per_enumerator`, a table the compiler builds
/// once and checks covers every enumerator.
///
/// \snippet meta/attributes_demo.cpp attributes
///
/// **Relationships.** `display.hpp` (colour, sign colours, label, number format, tooltip, JSON export), `acout.hpp`
/// (prints a value with its display attributes), `stats/pivot.hpp` (keys and cells), `meta/enums.hpp`. Group: \ref core.
///
/// **Guarantees.** Everything is `constexpr`; a set is its attributes and nothing else: an aggregate,
/// trivially copyable when they are, empty when there are none. For a value
/// known at compile time the lookup folds away; for an enumerator known at run time it is one table load. A type
/// without `attributes` has the empty set, so asking costs nothing and changes nothing.
///
/// **Errors.** Two attributes of one type in a set, or a `per_enumerator` table that does not cover every
/// enumerator, are compile errors with a message.
///
/// **Performance.** `benchmarks/meta/attributes_benchmark.cpp`: the colours of 1,048,576 greeks known at run time,
/// Release, Apple clang 17, Xeon W-3245, 2026-11-08: 674 us through `attributes_of`, against 806 us for the same
/// table written by hand (a colour and a label per greek, guarded) and 281 us for a bare colour table with no guard.
///
/// **Tests and demos.** `tests/meta/attributes_test.cpp`, `demos/meta/attributes_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<concepts>`, `<cstddef>`, `<type_traits>`, `<utility>`, `meta/enums.hpp`.
///
/// **Known issues.** `attributes` must be declared in the type's namespace (or as a friend), before the point
/// where it is printed. `snicholls` itself declares nothing called `attributes`, so the name is the caller's.
///
/// **Status.** New in 0.5.0.

#ifndef snicholls_core_meta_attributes_hpp
#define snicholls_core_meta_attributes_hpp

#include <snicholls/core/meta/enums.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace snicholls {

    namespace detail::attribute_lookup {
        // One attribute, held as a base so that the set is an aggregate, trivially copyable when its attributes are,
        // and empty when it has none.
        template <typename A>
        struct holder {
            A value;
        };
    } // namespace detail::attribute_lookup

    /// \brief Attributes, one value per type: what a type's `attributes` returns.
    /// \ingroup core
    /// \tparam As The attribute types, distinct.
    template <typename... As>
    struct attribute_set : detail::attribute_lookup::holder<As>... {
        /// \brief Whether the set has an attribute of type `A`.
        template <typename A>
        static constexpr bool has = (std::same_as<A, As> || ...);

        /// \brief The attribute of type `A`.
        /// \tparam A One of `As`.
        /// \return It.
        template <typename A>
            requires has<A>
        [[nodiscard]] constexpr const A& get() const noexcept
        {
            return static_cast<const detail::attribute_lookup::holder<A>&>(*this).value;
        }
    };

    namespace detail::attribute_lookup {
        template <typename A, typename... As>
        inline constexpr std::size_t count_of = (std::size_t{std::same_as<A, As>} + ... + 0);
    } // namespace detail::attribute_lookup

    /// \brief An `attribute_set` of these attributes: `attrs(colours::green, label{"USD"})`.
    /// \ingroup core
    /// \tparam As The attribute types, distinct.
    /// \param as The attributes.
    /// \return The set.
    template <typename... As>
    [[nodiscard]] constexpr attribute_set<As...> attrs(As... as)
    {
        static_assert(((detail::attribute_lookup::count_of<As, As...> == 1) && ...), "snicholls::attrs: two attributes of one type");
        return {detail::attribute_lookup::holder<As>{std::move(as)}...};
    }

    namespace detail::attribute_lookup {
        // The table a per_enumerator lambda builds: once, by the compiler, one constant per lambda.
        template <typename Make>
        inline constexpr auto table = Make{}();

        template <typename E>
        inline constexpr long long lowest = static_cast<long long>(static_cast<std::underlying_type_t<E>>(enum_values_v<E>.front()));
        template <typename E>
        inline constexpr std::size_t value_span =
            static_cast<std::size_t>(static_cast<long long>(static_cast<std::underlying_type_t<E>>(enum_values_v<E>.back())) - lowest<E>) + 1;

        // Values close together (gaps at most three times the enumerators): the table is laid out by value.
        template <typename E>
        concept dense = value_span<E> <= 3 * enum_count_v<E> + 16;

        template <typename E, typename Make, std::size_t... Is>
        constexpr auto spread(std::index_sequence<Is...>)
        {
            return std::array{table<Make>[enum_index(static_cast<E>(lowest<E> + static_cast<long long>(Is))).value_or(0)]...};
        }
        template <typename E, typename Make>
        inline constexpr auto by_value = spread<E, Make>(std::make_index_sequence<value_span<E>>{});

        // Dense: subtract, compare, one load. A value that is not an enumerator has the first set.
        template <dense E, typename Make>
        constexpr const auto& lookup(E e) noexcept
        {
            const auto at = static_cast<std::size_t>(static_cast<long long>(static_cast<std::underlying_type_t<E>>(e)) - lowest<E>);
            return by_value<E, Make>[at < value_span<E> ? at : 0];
        }
        template <typename E, typename Make>
        constexpr const auto& lookup(E e) noexcept
        {
            return table<Make>[enum_index(e).value_or(0)];
        }
    } // namespace detail::attribute_lookup

    /// \brief An enumerator's set from a table with one set per enumerator, in the order of `enum_values_v<E>`.
    ///
    /// `make` is a lambda without captures that returns the table; the compiler builds it once, as a constant, so a
    /// lookup at run time is a subtraction, a compare and one load (two for an enumeration with wide gaps):
    /// `per_enumerator(c, [] { return std::array{attrs(...), attrs(...)}; })`.
    /// \ingroup core
    /// \tparam E    A `reflectable_enum`.
    /// \tparam Make The lambda's type.
    /// \param e    The enumerator.
    /// \param make The lambda; its table needs one set per enumerator, or a compile error.
    /// \return `e`'s set; the first set for a value that is not an enumerator.
    template <reflectable_enum E, typename Make>
    [[nodiscard]] constexpr const auto& per_enumerator(E e, Make make) noexcept
    {
        static_cast<void>(make);
        constexpr const auto& table = detail::attribute_lookup::table<Make>;
        static_assert(std::tuple_size_v<std::remove_cvref_t<decltype(table)>> == enum_count_v<E>,
                      "snicholls::per_enumerator: the table needs one set per enumerator");
        return detail::attribute_lookup::lookup<E, Make>(e);
    }

    namespace detail::attribute_lookup {
        void attributes() = delete;   // ordinary lookup stops here; only the type's own namespace is searched

        template <typename T>
        concept declares = requires(const T& t) { attributes(t); };

        template <typename T>
            requires declares<T>
        constexpr auto of(const T& v) noexcept(noexcept(attributes(v)))
        {
            return attributes(v);
        }
        template <typename T>
        constexpr attribute_set<> of(const T&) noexcept
        {
            return {};
        }

        template <typename A, typename Set>
            requires Set::template has<A>
        constexpr A pick(const Set& s, A) { return s.template get<A>(); }
        template <typename A, typename Set>
        constexpr A pick(const Set&, A fallback) { return fallback; }
    } // namespace detail::attribute_lookup

    /// \brief Whether `T` declares `attributes`.
    /// \ingroup core
    /// \tparam T The type asked about.
    template <typename T>
    concept has_attributes = detail::attribute_lookup::declares<std::remove_cvref_t<T>>;

    /// \brief A value's attributes: what its type's `attributes` returns, or the empty set.
    /// \ingroup core
    /// \tparam T Any type.
    /// \param v The value.
    /// \return The set.
    template <typename T>
    [[nodiscard]] constexpr auto attributes_of(const T& v) noexcept(noexcept(detail::attribute_lookup::of(v)))
    {
        return detail::attribute_lookup::of(v);
    }

    /// \brief Whether values of type `T` carry an attribute of type `A`.
    /// \ingroup core
    template <typename A, typename T>
    inline constexpr bool has_attribute_v = decltype(attributes_of(std::declval<const T&>()))::template has<A>;

    /// \brief A value's attribute of type `A`, or `fallback` when its type has none.
    /// \ingroup core
    /// \tparam A The attribute type.
    /// \tparam T The value's type.
    /// \param v        The value.
    /// \param fallback The attribute to use when there is none.
    /// \return It.
    template <typename A, typename T>
    [[nodiscard]] constexpr A attribute_or(const T& v, A fallback)
    {
        return detail::attribute_lookup::pick<A>(attributes_of(v), std::move(fallback));
    }

} // namespace snicholls

#endif /* snicholls_core_meta_attributes_hpp */
