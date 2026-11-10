//
//  concepts.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief The small concepts the family keeps re-declaring: `streamable`, `printable`, `enumeration`, `integer`, `map_like` and friends.
/// \ingroup core
///
/// **Overview.** A `first_match` chain, a `to_text` function or a serialiser asks the same
/// questions of a type over and over: can it be streamed, is it an enumeration, is it a
/// number rather than a `bool` or a `char`, is it a map. The standard library answers some
/// (`std::integral`, `std::floating_point`, `std::ranges::range`) and not others, and
/// `std::integral` says yes to `bool` and to every character type, which is rarely what a
/// branch means. These concepts are the answers, declared once:
///
/// \snippet concepts_demo.cpp chain
///
/// `always_false_v<T>` is the dependent `false` for a `static_assert` in a template branch
/// that must fail only when it is instantiated.
///
/// **Relationships.** `snicholls::first_match` is where constrained lambdas meet; the string
/// concepts (`string_like`, `byte_sequence` ...) live in `reflect/string_traits.hpp` and
/// `tuple_like` in `reflect/struct_tuple.hpp`, and are not repeated here. Group: \ref core.
///
/// **Guarantees.** Compile time only; nothing here has a run-time footprint. Every concept
/// strips `const`, `volatile` and references before asking, so `const int&` is an `integer`.
///
/// **Errors.** None.
///
/// **Performance.** Not applicable.
///
/// **Tests and demos.** `tests/meta/concepts_test.cpp`, `demos/meta/concepts_demo.cpp`.
///
/// **Dependencies.** `<concepts>`, `<iosfwd>`, `<ranges>`, `<type_traits>`.
///
/// **Known issues.** `streamable` only requires that `os << t` is an expression; a stream
/// operator that returns something other than the stream still counts. `map_like` and
/// `set_like` are duck-typed on `key_type` and `mapped_type`, which every standard associative
/// container has and a `std::vector<std::pair<K, V>>` does not.
///
/// **Status.** New in 0.5.0.

#ifndef snicholls_core_meta_concepts_hpp
#define snicholls_core_meta_concepts_hpp

#include <array>
#include <concepts>
#include <cstddef>
#include <functional>
#include <iosfwd>
#include <ranges>
#include <string_view>
#include <type_traits>

namespace snicholls {

    /// \brief The dependent `false`: `static_assert(always_false_v<T>, "...")` fails only when that branch is instantiated.
    /// \ingroup core
    /// \tparam T Any type; only its dependence matters.
    template <typename T>
    inline constexpr bool always_false_v = false;

    namespace detail {
        template <typename T, template <typename...> class Template>
        struct is_specialisation_of : std::false_type {};

        template <template <typename...> class Template, typename... Args>
        struct is_specialisation_of<Template<Args...>, Template> : std::true_type {};
    } // namespace detail

    /// \brief `T` is `Template<Args...>` for some `Args`, cv and references removed: `specialisation_of<std::optional<int>, std::optional>`.
    /// \ingroup core
    /// \tparam T        The type asked about.
    /// \tparam Template A class template with type parameters only.
    template <typename T, template <typename...> class Template>
    concept specialisation_of = detail::is_specialisation_of<std::remove_cvref_t<T>, Template>::value;

    /// \brief `s << t` is an expression for a `Stream&` and a `const T&`.
    /// \ingroup core
    /// \tparam T      The type streamed.
    /// \tparam Stream The stream type.
    template <typename T, typename Stream>
    concept streamable_to = requires(Stream& s, const std::remove_cvref_t<T>& t) { s << t; };

    /// \brief `os << t` is an expression for a `std::ostream`.
    /// \ingroup core
    /// \tparam T The type streamed.
    template <typename T>
    concept streamable = streamable_to<T, std::ostream>;

    class acout;   // acout.hpp: the message that is written in one call

    /// \brief Has `print(std::ostream&) const`: a type that writes itself to a stream, the classic form of the house pattern.
    /// \ingroup core
    /// \tparam T The type asked about.
    template <typename T>
    concept printable_to_stream = requires(const std::remove_cvref_t<T>& t, std::ostream& os) { t.print(os); };

    /// \brief Has `print(snicholls::acout&) const`: a type that writes itself into a message, the form with format strings.
    /// \ingroup core
    ///
    /// Inside such a `print`, `out.print("{} @ {:.2f}", symbol, price)` and `out << ...` both work,
    /// and the whole message is written in one call. Take the `acout` by reference.
    /// \tparam T The type asked about.
    template <typename T>
    concept printable_to_acout = requires(const std::remove_cvref_t<T>& t, acout& out) { t.print(out); };

    /// \brief A type that prints itself, in either form: the member is the whole contract.
    /// \ingroup core
    ///
    /// `snicholls::acout` streams a printable type through its member (the `acout` form first
    /// when a type has both), and `snicholls::print_operators::operator<<` lends it to any
    /// `std::ostream`. The concept replaces the `-> decltype(obj.print(os), os)` comma-operator
    /// SFINAE the pattern used to need, and says `const`: a `print` that is not `const` does
    /// not qualify.
    /// \tparam T The type asked about.
    template <typename T>
    concept printable = printable_to_stream<T> || printable_to_acout<T>;

    /// \brief An enumeration type, scoped or not.
    /// \ingroup core
    /// \tparam T The type asked about.
    template <typename T>
    concept enumeration = std::is_enum_v<std::remove_cvref_t<T>>;

    /// \brief An `enum class`: an enumeration that does not convert to its underlying type.
    /// \ingroup core
    /// \tparam T The type asked about.
    template <typename T>
    concept scoped_enumeration =
        enumeration<T> &&
        !std::is_convertible_v<std::remove_cvref_t<T>, std::underlying_type_t<std::remove_cvref_t<T>>>;

    /// \brief A plain `enum`: an enumeration that converts to its underlying type.
    /// \ingroup core
    /// \tparam T The type asked about.
    template <typename T>
    concept unscoped_enumeration = enumeration<T> && !scoped_enumeration<T>;

    /// \brief One of the character types: `char`, `wchar_t`, `char8_t`, `char16_t`, `char32_t`.
    ///
    /// `signed char` and `unsigned char` are not characters here: they are the one-byte
    /// numbers (`std::int8_t`, `std::uint8_t`), as `reflect/string_traits.hpp` treats them.
    /// \ingroup core
    /// \tparam T The type asked about.
    template <typename T>
    concept character = std::same_as<std::remove_cvref_t<T>, char> || std::same_as<std::remove_cvref_t<T>, wchar_t> ||
                        std::same_as<std::remove_cvref_t<T>, char8_t> || std::same_as<std::remove_cvref_t<T>, char16_t> ||
                        std::same_as<std::remove_cvref_t<T>, char32_t>;

    /// \brief An integral type that is a number: not `bool`, not a character type.
    /// \ingroup core
    /// \tparam T The type asked about.
    template <typename T>
    concept integer = std::integral<std::remove_cvref_t<T>> && !std::same_as<std::remove_cvref_t<T>, bool> && !character<T>;

    /// \brief An `integer` or a floating-point type: what arithmetic means in a branch.
    /// \ingroup core
    /// \tparam T The type asked about.
    template <typename T>
    concept number = integer<T> || std::floating_point<std::remove_cvref_t<T>>;

    /// \brief An input range with `key_type` and `mapped_type`: `std::map`, `std::unordered_map` and their multi forms.
    /// \ingroup core
    /// \tparam T The type asked about.
    template <typename T>
    concept map_like = std::ranges::input_range<std::remove_cvref_t<T>> && requires {
        typename std::remove_cvref_t<T>::key_type;
        typename std::remove_cvref_t<T>::mapped_type;
    };

    /// \brief An input range with `key_type` and no `mapped_type`: `std::set`, `std::unordered_set` and their multi forms.
    /// \ingroup core
    /// \tparam T The type asked about.
    template <typename T>
    concept set_like = std::ranges::input_range<std::remove_cvref_t<T>> && requires {
        typename std::remove_cvref_t<T>::key_type;
    } && !map_like<T>;

    /// \brief A range whose elements are exactly `E`: `range_of<std::vector<int>, int>`.
    /// \ingroup core
    template <typename R, typename E>
    concept range_of = std::ranges::range<R> && std::same_as<std::ranges::range_value_t<R>, E>;

    /// \brief `std::hash<T>` exists and gives a `std::size_t`.
    /// \ingroup core
    template <typename T>
    concept std_hashable = requires(const T& v) {
        { std::hash<T>{}(v) } -> std::convertible_to<std::size_t>;
    };

    /// \brief `std::hash<T>` exists and does not throw.
    /// \ingroup core
    template <typename T>
    concept nothrow_std_hashable = std_hashable<T> && requires(const T& v) {
        { std::hash<T>{}(v) } noexcept;
    };

    /// \brief A container with `reserve(n)`, so that a known size can be allocated up front.
    /// \ingroup core
    template <typename T>
    concept reservable = requires(std::remove_cvref_t<T>& c, std::size_t n) { c.reserve(n); };

    /// \brief Whether `T` is a `std::array`, which `specialisation_of` cannot say: its size is a value.
    /// \ingroup core
    template <typename T>
    struct is_std_array : std::false_type {};
    /// \brief `std::array<T, N>` is one.
    /// \ingroup core
    template <typename T, std::size_t N>
    struct is_std_array<std::array<T, N>> : std::true_type {};
    /// \brief `is_std_array<T>::value`, for any cv and reference.
    /// \ingroup core
    template <typename T>
    inline constexpr bool is_std_array_v = is_std_array<std::remove_cvref_t<T>>::value;

    /// \brief A type with a static `name()` usable as a `std::string_view`.
    /// \ingroup core
    template <typename T>
    concept has_static_name = requires {
        { std::remove_cvref_t<T>::name() } -> std::convertible_to<std::string_view>;
    };

    namespace detail {
        template <typename... Ts>
        consteval bool static_names_distinct() noexcept
        {
            const std::array<std::string_view, sizeof...(Ts)> names{std::string_view(Ts::name())...};
            for (std::size_t i = 0; i < names.size(); ++i)
                for (std::size_t j = i + 1; j < names.size(); ++j)
                    if (names[i] == names[j])
                        return false;
            return true;
        }
    } // namespace detail

    /// \brief Whether the static names of `Ts...` are all different, at compile time: a registry's duplicate check.
    /// \ingroup core
    template <has_static_name... Ts>
    inline constexpr bool distinct_static_names_v = detail::static_names_distinct<std::remove_cvref_t<Ts>...>();
} // namespace snicholls

#endif /* snicholls_core_meta_concepts_hpp */
