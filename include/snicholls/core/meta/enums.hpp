//
//  enums.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Enum reflection at compile time: names, values and parsing of an enumeration's enumerators.
/// \ingroup core
///
/// **Overview.** `enum_name(side::buy)` is `"buy"`, `enum_cast<side>("sell")` is `side::sell`, and
/// `enum_values_v<side>` is every enumerator in order of value, all `constexpr`, from nothing but the
/// enumeration's declaration. It replaces magic_enum: the compiler spells a non-type template argument in
/// its function signature, and spells it as a name exactly when the value is an enumerator.
///
/// \snippet meta/enums_demo.cpp enums
///
/// The values from `enum_range<E>` are tried, -128 to 127 by default, cut to the underlying type.
/// Specialise `enum_range` for an enumeration with values outside it:
/// `template <> struct snicholls::enum_range<venue> { static constexpr long long min = 0, max = 1000; };`.
///
/// **Relationships.** `meta/type_name.hpp`, the same technique for types; `acout.hpp` prints a
/// reflectable enumeration by its name; `constexpr_map.hpp` for names chosen by hand. Group: \ref core.
///
/// **Guarantees.** Everything is `constexpr` and `noexcept`; the tables are computed once per enumeration
/// at compile time and live in static storage. Lookup by value is O(1), by name O(n).
///
/// **Errors.** `enum_cast` and `enum_index` return `std::nullopt` for a name or value that is not an
/// enumerator; `enum_name` returns an empty view. An enumeration that cannot be reflected is a compile
/// error with a message.
///
/// **Performance.** Compile time: one instantiation per value tried (256 by default) per enumeration.
/// Run time: `enum_index` and `enum_name` are one table load (two bytes per value tried), `enum_cast` from a
/// name a linear search.
///
/// **Tests and demos.** `tests/meta/enums_test.cpp`, `demos/meta/enums_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<cstdint>`, `<limits>`, `<optional>`, `<string_view>`, `<type_traits>`,
/// `<utility>`.
///
/// **Known issues.**
/// - An unscoped enumeration without a fixed underlying type (`enum e { a, b };`) is refused: forming one
///   of its out-of-range values in a constant expression is undefined, and Clang rejects it. Give it one:
///   `enum e : int { a, b };`. Every scoped enumeration qualifies.
/// - Two enumerators with one value (aliases) reflect as one, under the name the compiler prints.
/// - Flag enumerations (combinations of bits) are reflected as their individual enumerators only.
/// - An enumeration local to a function cannot be reflected on every compiler.
///
/// **Status.** New in 0.5.0.

#ifndef snicholls_core_meta_enums_hpp
#define snicholls_core_meta_enums_hpp

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief An enumeration whose values can be reflected: scoped, or unscoped with a fixed underlying type.
    ///
    /// List-initialisation from an integer compiles exactly for an enumeration with a fixed underlying type.
    /// \ingroup core
    template <typename E>
    concept reflectable_enum = std::is_enum_v<E> && requires { E{std::underlying_type_t<E>{}}; };

    /// \brief The values tried when reflecting `E`: specialise it for enumerators outside -128 .. 127.
    /// \ingroup core
    /// \tparam E The enumeration.
    template <typename E>
    struct enum_range {
        static constexpr long long min = -128;   ///< The smallest value tried.
        static constexpr long long max = 127;    ///< The largest value tried.
    };

    namespace detail {

        template <auto V>
        constexpr std::string_view enum_signature() noexcept
        {
#if defined(__clang__) || defined(__GNUC__)
            return __PRETTY_FUNCTION__;
#elif defined(_MSC_VER)
            return __FUNCSIG__;
#else
            return {};
#endif
        }

        // The part of the signature that is V: a name if V is an enumerator, a cast of a number if not.
        template <auto V>
        constexpr std::string_view enum_spelling() noexcept
        {
            constexpr std::string_view sig = enum_signature<V>();
#if defined(__clang__)
            constexpr std::string_view head = "[V = ";
            constexpr char tail = ']';
#elif defined(__GNUC__)
            constexpr std::string_view head = "[with auto V = ";
            constexpr char tail = ';';
#elif defined(_MSC_VER)
            constexpr std::string_view head = "enum_signature<";
            constexpr char tail = '>';
#else
            constexpr std::string_view head = "";
            constexpr char tail = ']';
#endif
            const std::size_t start = sig.find(head);
            if (start == std::string_view::npos)
                return {};
            const std::size_t from = start + head.size();
            std::size_t end = sig.find(tail, from);
            if (end == std::string_view::npos)
                end = sig.find(']', from);
            if (end == std::string_view::npos)
                return {};
            return sig.substr(from, end - from);
        }

        // The enumerator's own name, or empty when V is not an enumerator
        template <auto V>
        constexpr std::string_view enum_value_name() noexcept
        {
            // The last component: an identifier for an enumerator ("(anonymous namespace)::side::sell"),
            // a number for anything else ("(side)9", "(enum side)0x9", "((anonymous namespace)::side)-1").
            constexpr std::string_view spelling = enum_spelling<V>();
            const std::size_t colon = spelling.rfind("::");
            const std::size_t paren = spelling.rfind(')');
            std::size_t from = 0;
            if (colon != std::string_view::npos)
                from = colon + 2;
            if (paren != std::string_view::npos && paren + 1 > from)
                from = paren + 1;
            const std::string_view last = spelling.substr(from);
            const bool identifier = !last.empty() && (last.front() == '_' || (last.front() >= 'a' && last.front() <= 'z') ||
                                                      (last.front() >= 'A' && last.front() <= 'Z'));
            return identifier ? last : std::string_view{};
        }

        template <typename E>
        consteval long long enum_scan_min() noexcept
        {
            using U = std::underlying_type_t<E>;
            const long long lowest = std::is_signed_v<U> ? static_cast<long long>(std::numeric_limits<U>::min()) : 0;
            return enum_range<E>::min > lowest ? enum_range<E>::min : lowest;
        }

        template <typename E>
        consteval long long enum_scan_max() noexcept
        {
            using U = std::underlying_type_t<E>;
            const auto highest = static_cast<unsigned long long>(std::numeric_limits<U>::max());
            const unsigned long long wanted = enum_range<E>::max < 0 ? 0 : static_cast<unsigned long long>(enum_range<E>::max);
            return static_cast<long long>(wanted < highest ? wanted : highest);
        }

        template <typename E, long long Min, std::size_t... Is>
        consteval std::array<bool, sizeof...(Is)> enum_valid(std::index_sequence<Is...>) noexcept
        {
            return {!enum_value_name<static_cast<E>(Min + static_cast<long long>(Is))>().empty()...};
        }

        template <typename E, long long Min, std::size_t... Is>
        consteval std::array<std::string_view, sizeof...(Is)> enum_scanned_names(std::index_sequence<Is...>) noexcept
        {
            return {enum_value_name<static_cast<E>(Min + static_cast<long long>(Is))>()...};
        }

        template <typename E>
        struct enum_table {
            static_assert(reflectable_enum<E>, "snicholls::enum reflection: give the enumeration a fixed underlying type "
                                               "(enum e : int { ... }); without one its out-of-range values are undefined");
            static constexpr long long min = enum_scan_min<E>();
            static constexpr long long max = enum_scan_max<E>();
            static_assert(max >= min && max - min < 65536, "snicholls::enum reflection: enum_range spans more than 65536 values");
            static constexpr auto span = std::make_index_sequence<static_cast<std::size_t>(max - min + 1)>{};
            static constexpr auto valid = enum_valid<E, min>(span);
            static constexpr auto scanned = enum_scanned_names<E, min>(span);

            static consteval std::size_t count() noexcept
            {
                std::size_t n = 0;
                for (const bool v : valid)
                    n += v ? 1 : 0;
                return n;
            }

            static consteval std::array<E, count()> values() noexcept
            {
                std::array<E, count()> out{};
                std::size_t k = 0;
                for (std::size_t i = 0; i < valid.size(); ++i)
                    if (valid[i])
                        out[k++] = static_cast<E>(min + static_cast<long long>(i));
                return out;
            }

            // The position of each scanned value in values(), or none: lookup by value is one load.
            static consteval std::array<std::uint16_t, valid.size()> positions() noexcept
            {
                std::array<std::uint16_t, valid.size()> out{};
                std::uint16_t k = 0;
                for (std::size_t i = 0; i < valid.size(); ++i)
                    out[i] = valid[i] ? k++ : std::uint16_t{0xFFFF};
                return out;
            }

            static consteval std::array<std::string_view, count()> names() noexcept
            {
                std::array<std::string_view, count()> out{};
                std::size_t k = 0;
                for (std::size_t i = 0; i < valid.size(); ++i)
                    if (valid[i])
                        out[k++] = scanned[i];
                return out;
            }
        };

    } // namespace detail

    /// \brief Every enumerator of `E`, in increasing order of value.
    /// \ingroup core
    template <reflectable_enum E>
    inline constexpr auto enum_values_v = detail::enum_table<E>::values();

    /// \brief Every enumerator's name, in the order of `enum_values_v`.
    /// \ingroup core
    template <reflectable_enum E>
    inline constexpr auto enum_names_v = detail::enum_table<E>::names();

    /// \brief How many enumerators `E` has (aliases count once).
    /// \ingroup core
    template <reflectable_enum E>
    inline constexpr std::size_t enum_count_v = enum_values_v<E>.size();

    namespace detail {
        template <reflectable_enum E>
        inline constexpr auto enum_positions_v = enum_table<E>::positions();
    } // namespace detail

    /// \brief The position of `value` in `enum_values_v<E>`, or none if it is not an enumerator.
    /// \ingroup core
    /// \tparam E The enumeration.
    /// \param value The value.
    /// \return The index.
    template <reflectable_enum E>
    [[nodiscard]] constexpr std::optional<std::size_t> enum_index(E value) noexcept
    {
        using table = detail::enum_table<E>;
        const auto v = static_cast<long long>(static_cast<std::underlying_type_t<E>>(value));
        if (v < table::min || v > table::max)
            return std::nullopt;
        const std::uint16_t position = detail::enum_positions_v<E>[static_cast<std::size_t>(v - table::min)];
        return position == 0xFFFF ? std::nullopt : std::optional<std::size_t>(position);
    }

    /// \brief The enumerator's name: `enum_name(side::buy)` is `"buy"`; empty if `value` is not an enumerator.
    /// \ingroup core
    /// \tparam E The enumeration.
    /// \param value The value.
    /// \return A view into static storage.
    template <reflectable_enum E>
    [[nodiscard]] constexpr std::string_view enum_name(E value) noexcept
    {
        const auto i = enum_index(value);
        return i ? enum_names_v<E>[*i] : std::string_view{};
    }

    /// \brief The name of one enumerator, as a constant: `enum_name<side::buy>()`.
    /// \ingroup core
    /// \tparam V The enumerator.
    /// \return Its name; empty if `V` is not an enumerator.
    template <auto V>
        requires std::is_enum_v<decltype(V)>
    [[nodiscard]] constexpr std::string_view enum_name() noexcept
    {
        return detail::enum_value_name<V>();
    }

    /// \brief Whether `value` is an enumerator of `E`.
    /// \ingroup core
    /// \tparam E The enumeration.
    /// \param value The value.
    /// \return Whether it is.
    template <reflectable_enum E>
    [[nodiscard]] constexpr bool enum_contains(E value) noexcept
    {
        return enum_index(value).has_value();
    }

    /// \brief The enumerator named `name`, exactly; none if there is no such enumerator.
    /// \ingroup core
    /// \tparam E The enumeration.
    /// \param name The name, as in the declaration.
    /// \return The enumerator.
    template <reflectable_enum E>
    [[nodiscard]] constexpr std::optional<E> enum_cast(std::string_view name) noexcept
    {
        for (std::size_t i = 0; i < enum_count_v<E>; ++i)
            if (enum_names_v<E>[i] == name)
                return enum_values_v<E>[i];
        return std::nullopt;
    }

    /// \brief The enumerator with this underlying value; none if no enumerator has it.
    /// \ingroup core
    /// \tparam E The enumeration.
    /// \param value The underlying value.
    /// \return The enumerator.
    template <reflectable_enum E>
    [[nodiscard]] constexpr std::optional<E> enum_cast(std::underlying_type_t<E> value) noexcept
    {
        const auto e = static_cast<E>(value);
        return enum_contains(e) ? std::optional<E>(e) : std::nullopt;
    }

} // namespace snicholls

#endif /* snicholls_core_meta_enums_hpp */
