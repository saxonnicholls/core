//
//  string_traits.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Classification of every string and byte-sequence type: code units, ownership, termination, length.
/// \ingroup core
///
/// String / byte-sequence classification: separate, composable compile-time properties of every "string-ish"
/// type the snicholls libraries meet (field names, keys, values and serialisation), plus concepts built from them.
/// Zero-copy access (to_string_view, as_bytes_view, length_of, c_str_of) is in \<snicholls/core/reflect/string_access.hpp>.
///
/// A *code-unit sequence* is a contiguous run of code units carried by one of these types (any cv, any reference):
///
/// \code
///   carrier                                  owning  NUL-terminated   length                      units
///   C*                                       no      by contract      O(n) (strlen; null -> 0)    text only
///   C[N]  (text: a string literal)           yes     by contract      N-1, compile time           text
///   B[N]  (bytes)                            yes     no               N, compile time             bytes
///   std::basic_string<C, T, A> (incl. pmr)   yes     guaranteed       O(1)                        text only
///   std::basic_string_view<C, T>             no      no               O(1)                        text only
///   std::vector<U, A> (incl. pmr)            yes     no               O(1)                        text or bytes
///   std::array<U, N>                         yes     no               N, compile time             text or bytes
///   std::span<U, dynamic_extent>             no      no               O(1)                        text or bytes
///   std::span<U, N>                          no      no               N, compile time             text or bytes
///   basic_fixed_string<C, N> (and the        yes     guaranteed       N, compile time             text only
///     fixed_string<N>, u8/u16/u32/w wrappers)
/// \endcode
///
/// Classification rules (see CLAUDE.md "Strings"):
/// - Text code units are the character types char, wchar_t, char8_t, char16_t, char32_t. Byte code units are
///   std::byte and unsigned char (std::uint8_t is unsigned char on every supported platform). signed char /
///   std::int8_t is neither: it is a small *number* (canonical int8), not a character type distinct from char,
///   and not one of the types the standard lets inspect object representation.
///   So std::vector\<std::uint8_t> is a byte sequence here (same object representation as std::vector\<std::byte>,
///   viewing it as bytes is lossless); this says nothing about how a consumer maps it to its own types.
/// - A character array is a string literal: its length is N-1 (every element but the last, embedded NULs
///   included, never strlen), and its last element must be NUL (checked by the access functions). For a
///   fixed-width, possibly unterminated buffer use std::array or std::span (length N, not NUL-terminated).
/// - A byte array has no terminator: its length is N.
/// - Ownership describes the type as given: a reference never owns (std::string& is a view of someone else's
///   string; a string literal is a const char(&)[N]). Owning containers propagate top-level const to their
///   code units; views (span, pointer, string_view) do not.
/// - Volatile carriers or code units, unbounded arrays, pointers to bytes (no length) and non-contiguous
///   containers (std::deque, std::list) are not code-unit sequences.

#ifndef snicholls_core_reflect_string_traits_hpp
#define snicholls_core_reflect_string_traits_hpp

#include <array>
#include <ranges>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <snicholls/core/fixed_string.hpp>

namespace snicholls {

// ---- code units ----------------------------------------------------------------------------------------

// char, wchar_t, char8_t, char16_t, char32_t (cv-unqualified): units of text.
template <typename C>
struct is_text_code_unit
    : std::bool_constant<std::is_same_v<C, char> || std::is_same_v<C, wchar_t> || std::is_same_v<C, char8_t>
                         || std::is_same_v<C, char16_t> || std::is_same_v<C, char32_t>> {};

/// \brief Whether `C` is a text code unit (char, wchar_t, char8_t, char16_t, char32_t).
template <typename C>
inline constexpr bool is_text_code_unit_v = is_text_code_unit<C>::value;

// std::byte, unsigned char, std::uint8_t (cv-unqualified): units of raw bytes. Never signed char.
template <typename C>
struct is_byte_code_unit
    : std::bool_constant<std::is_same_v<C, std::byte> || std::is_same_v<C, unsigned char>
                         || std::is_same_v<C, std::uint8_t>> {};

/// \brief Whether `C` is a byte code unit (`std::byte`, `unsigned char`).
template <typename C>
inline constexpr bool is_byte_code_unit_v = is_byte_code_unit<C>::value;

// ---- property vocabulary -------------------------------------------------------------------------------

/// \brief The code unit category.
enum class code_unit_category : unsigned char { text, bytes };
/// \var snicholls::code_unit_category::text
/// Characters (char, wchar_t, char8_t, char16_t, char32_t).
/// \var snicholls::code_unit_category::bytes
/// Octets (`std::byte`, `unsigned char`).

/// \brief The null termination.
enum class null_termination : unsigned char {
    none,   ///< no terminator may be assumed (string_view, span, vector, std::array, byte arrays)
    by_contract,   ///< the type cannot prove it; the carrier's contract requires it (C*, character arrays)
    guaranteed,   ///< the type maintains it (std::basic_string, basic_fixed_string)
};

/// \brief The length kind.
enum class length_kind : unsigned char {
    compile_time,   ///< part of the type: static_length_v
    constant_time,   ///< stored: .size()
    linear_time,   ///< scanned: char_traits::length (C* only)
};

/// \brief Lower-case enumerator spelling, for tables and diagnostics.
/// \ingroup core
/// \param c the c.
/// \return the result.
constexpr std::string_view enumerator_name(code_unit_category c) noexcept
{
    return c == code_unit_category::text ? "text" : "bytes";
}

/// \brief The enumerator name.
/// \param t the t.
/// \return the enumerator name.
constexpr std::string_view enumerator_name(null_termination t) noexcept
{
    switch (t) {
    case null_termination::none: return "none";
    case null_termination::by_contract: return "by_contract";
    case null_termination::guaranteed: return "guaranteed";
    }
    return "?";
}

/// \brief The enumerator name.
/// \param k the kind.
/// \return the enumerator name.
constexpr std::string_view enumerator_name(length_kind k) noexcept
{
    switch (k) {
    case length_kind::compile_time: return "compile_time";
    case length_kind::constant_time: return "constant_time";
    case length_kind::linear_time: return "linear_time";
    }
    return "?";
}

namespace detail {

enum class carrier : unsigned char { none, pointer, array, string, string_view, vector, std_array, span, fixed_string };

// Detects basic_fixed_string<C, N> and every class derived from it (fixed_string<N>, u8fixed_string<N>, ...).
template <typename C, std::size_t N>
basic_fixed_string<C, N> fixed_string_base(const basic_fixed_string<C, N>*);

template <typename B>
concept derived_from_basic_fixed_string = std::is_class_v<B> && requires(const B* p) { detail::fixed_string_base(p); };

// The carrier shape of a cv/ref-stripped type. element keeps its own const (span<const char> -> const char).
template <typename B>
struct shape
{
    static constexpr carrier kind = carrier::none;
};

template <typename E>
struct shape<E*>
{
    static constexpr carrier kind = carrier::pointer;
    using element = E;
    static constexpr bool owning = false;
    static constexpr std::size_t extent = std::dynamic_extent;
};

template <typename E, std::size_t N>
struct shape<E[N]>
{
    static constexpr carrier kind = carrier::array;
    using element = E;
    static constexpr bool owning = true;
    static constexpr std::size_t extent = N;
};

template <typename C, typename T, typename A>
struct shape<std::basic_string<C, T, A>>
{
    static constexpr carrier kind = carrier::string;
    using element = C;
    using traits = T;
    static constexpr bool owning = true;
    static constexpr std::size_t extent = std::dynamic_extent;
};

template <typename C, typename T>
struct shape<std::basic_string_view<C, T>>
{
    static constexpr carrier kind = carrier::string_view;
    using element = const C;
    using traits = T;
    static constexpr bool owning = false;
    static constexpr std::size_t extent = std::dynamic_extent;
};

template <typename E, typename A>
struct shape<std::vector<E, A>>
{
    static constexpr carrier kind = carrier::vector;
    using element = E;
    static constexpr bool owning = true;
    static constexpr std::size_t extent = std::dynamic_extent;
};

template <typename E, std::size_t N>
struct shape<std::array<E, N>>
{
    static constexpr carrier kind = carrier::std_array;
    using element = E;
    static constexpr bool owning = true;
    static constexpr std::size_t extent = N;
};

template <typename E, std::size_t X>
struct shape<std::span<E, X>>
{
    static constexpr carrier kind = carrier::span;
    using element = E;
    static constexpr bool owning = false;
    static constexpr std::size_t extent = X;
};

// Checked only when no standard carrier matched, so naming e.g. std::basic_string<std::byte> (invalid on libc++)
// never instantiates it.
template <typename B>
struct shape_of : shape<B> {};

template <typename B>
    requires(shape<B>::kind == carrier::none && derived_from_basic_fixed_string<B>)
struct shape_of<B>
{
    using base = decltype(detail::fixed_string_base(static_cast<const B*>(nullptr)));
    static constexpr carrier kind = carrier::fixed_string;
    using element = const typename base::value_type; // read-only API: data() is const
    static constexpr bool owning = true;
    static constexpr std::size_t extent = base::size();
};

template <typename S>
using bare_t = std::remove_cv_t<std::remove_reference_t<S>>;

template <typename Shape>
concept has_traits = requires { typename Shape::traits; };

// Every property of S, or valid == false (the defaults keep the boolean traits well-defined for any type).
template <typename S, typename Shape = shape_of<bare_t<S>>>
struct classify
{
    static constexpr carrier kind = carrier::none;
    static constexpr bool valid = false;
    static constexpr bool text = false;
    static constexpr bool bytes = false;
    static constexpr bool owning = false;
    static constexpr bool mutable_units = false;
};

template <typename S, typename Shape>
    requires(Shape::kind != carrier::none)
struct classify<S, Shape>
{
    using element = typename Shape::element;
    using unit = std::remove_const_t<element>;

    static constexpr carrier kind = Shape::kind;
    static constexpr bool text = is_text_code_unit_v<unit>;
    static constexpr bool bytes = is_byte_code_unit_v<unit>;
    static constexpr bool text_only = kind == carrier::pointer || kind == carrier::string
                                      || kind == carrier::string_view || kind == carrier::fixed_string;

    static constexpr bool valid = (text || (bytes && !text_only)) && !std::is_volatile_v<element>
                                  && !std::is_volatile_v<std::remove_reference_t<S>>;

    static constexpr bool owning = Shape::owning && !std::is_reference_v<S>;
    static constexpr bool mutable_units =
        !std::is_const_v<element> && !(Shape::owning && std::is_const_v<std::remove_reference_t<S>>);

    static constexpr null_termination termination =
        (kind == carrier::string || kind == carrier::fixed_string)  ? null_termination::guaranteed
        : kind == carrier::pointer || (kind == carrier::array && text) ? null_termination::by_contract
                                                                       : null_termination::none;

    static constexpr std::size_t static_length = kind == carrier::pointer ? std::dynamic_extent
                                                 : kind == carrier::array && text ? Shape::extent - 1
                                                                                  : Shape::extent;

    static constexpr length_kind length = kind == carrier::pointer             ? length_kind::linear_time
                                          : static_length == std::dynamic_extent ? length_kind::constant_time
                                                                                 : length_kind::compile_time;

    static constexpr auto traits_tag() noexcept
    {
        if constexpr (has_traits<Shape>)
            return std::type_identity<typename Shape::traits>{};
        else if constexpr (is_text_code_unit_v<unit>)
            return std::type_identity<std::char_traits<unit>>{};
        else
            return std::type_identity<void>{}; // never name char_traits of a byte unit (deprecated in libc++)
    }
};

} // namespace detail

// ---- traits (well-defined for any type; false for anything that is not a code-unit sequence) ----------

// S is one of the carriers above over text or byte code units, contiguous.
template <typename S>
struct is_code_unit_sequence : std::bool_constant<detail::classify<S>::valid> {};

/// \brief Whether `S` is a code-unit sequence (text or bytes).
template <typename S>
inline constexpr bool is_code_unit_sequence_v = is_code_unit_sequence<S>::value;

template <typename S>
concept code_unit_sequence = is_code_unit_sequence_v<S>;

template <typename S>
struct is_text_sequence : std::bool_constant<code_unit_sequence<S> && detail::classify<S>::text> {};

/// \brief Whether `S` is a text sequence.
template <typename S>
inline constexpr bool is_text_sequence_v = is_text_sequence<S>::value;

template <typename S>
struct is_byte_sequence : std::bool_constant<code_unit_sequence<S> && detail::classify<S>::bytes> {};

/// \brief Whether `S` is a byte sequence.
template <typename S>
inline constexpr bool is_byte_sequence_v = is_byte_sequence<S>::value;

// The type itself owns its code units (string, vector, std::array, fixed_string, C array object). References,
// pointers, string_view and span do not.
template <typename S>
struct is_owning_sequence : std::bool_constant<code_unit_sequence<S> && detail::classify<S>::owning> {};

/// \brief Whether `S` owns its code units.
template <typename S>
inline constexpr bool is_owning_sequence_v = is_owning_sequence<S>::value;

// The code units can be modified through S (std::string, char*, span<char>; not const std::string,
// string_view, span<const char>, fixed_string).
template <typename S>
struct has_mutable_code_units : std::bool_constant<code_unit_sequence<S> && detail::classify<S>::mutable_units> {};

/// \brief Whether `S` gives mutable access to its code units.
template <typename S>
inline constexpr bool has_mutable_code_units_v = has_mutable_code_units<S>::value;

// Every code-unit sequence is contiguous by construction; spelled out so generic code can require it.
template <typename S>
struct is_contiguous_sequence : std::bool_constant<code_unit_sequence<S>> {};

/// \brief Whether `S` is contiguous (always true for sequences).
template <typename S>
inline constexpr bool is_contiguous_sequence_v = is_contiguous_sequence<S>::value;

// ---- value traits (code-unit sequences only) -----------------------------------------------------------

template <typename S>
struct code_unit {};

/// \brief The code unit<.
template <code_unit_sequence S>
struct code_unit<S>
{
    /// \brief The type.
    using type = typename detail::classify<S>::unit;
};

/// \brief The cv-unqualified code unit: char for const char*, std::byte for span\<const std::byte>.
/// \ingroup core
template <typename S>
using code_unit_t = typename code_unit<S>::type;

/// \brief The category of `S`'s code units (text or bytes).
template <code_unit_sequence S>
inline constexpr code_unit_category code_unit_category_v =
    detail::classify<S>::text ? code_unit_category::text : code_unit_category::bytes;

/// \brief Whether `S` is null-terminated: none, by contract or guaranteed.
template <code_unit_sequence S>
inline constexpr null_termination null_termination_v = detail::classify<S>::termination;

/// \brief How `S`'s length is known: compile time, constant time or linear time.
template <code_unit_sequence S>
inline constexpr length_kind length_kind_v = detail::classify<S>::length;

/// \brief The length in code units when it is part of the type, else std::dynamic_extent.
///
/// (like std::span::extent).
/// \ingroup core
template <code_unit_sequence S>
inline constexpr std::size_t static_length_v = detail::classify<S>::static_length;

// ---- concepts ------------------------------------------------------------------------------------------

// Text of any character type, owning or not.
template <typename S>
concept string_like = is_text_sequence_v<S>;

/// \brief A range that is not text: a container of elements, never a string, a string view or a character array.
///
/// What a serialiser or a printer means by "a list": `std::vector<int>` and `std::set<std::string>` are,
/// `std::string` and `"abc"` are not.
/// \ingroup core
template <typename R>
concept sequence_like = std::ranges::range<std::remove_cvref_t<R>> && !string_like<R>;

// Raw bytes (std::byte / unsigned char / std::uint8_t), owning or not.
template <typename S>
concept byte_sequence = is_byte_sequence_v<S>;

template <typename S>
concept owning_string = string_like<S> && is_owning_sequence_v<S>;

// Text this type does not own: string_view, span, C*, references (including string literals).
template <typename S>
concept string_view_like = string_like<S> && !is_owning_sequence_v<S>;

// Text whose data() may be handed to a C API (guaranteed, or by the carrier's contract).
template <typename S>
concept null_terminated_string = string_like<S> && null_termination_v<S> != null_termination::none;

// Text whose length is part of the type (fixed_string, std::array, static-extent span, string literal).
template <typename S>
concept fixed_length_string = string_like<S> && length_kind_v<S> == length_kind::compile_time;

// A snicholls::basic_fixed_string (or a wrapper derived from it): the one string type usable as a template
// argument, so its value is a compile-time constant - field names and keys (template <snicholls::fixed_string Name>).
template <typename S>
concept compile_time_string = string_like<S> && detail::classify<S>::kind == detail::carrier::fixed_string;

/// \brief The character traits of a text sequence.
///
/// T of basic_string\<C, T, A> / basic_string_view\<C, T>, std::char_traits\<C> for every other carrier.
///
/// Views built by to_string_view keep them.
/// \ingroup core
template <string_like S>
using char_traits_t = typename decltype(detail::classify<S>::traits_tag())::type;

// ---- all properties in one place -----------------------------------------------------------------------

/// \brief Every classification property of a string or byte-sequence type `S`, bundled.
/// \ingroup core
///
/// \overview
/// Separate, composable compile-time properties of every "string-ish" type the snicholls libraries meet (field names, keys, values, serialisation): code unit and its category, owning, mutable, contiguous, null termination, length kind and static length, plus the concepts built from them. The carriers and the rules are in the file description above and in CLAUDE.md, "Strings".
///
/// \relations
/// - Access: `snicholls::to_string_view`, `snicholls::as_bytes_view`, `snicholls::length_of`, `snicholls::c_str_of` (`reflect/string_access.hpp`).
/// - Concepts: `snicholls::code_unit_sequence`, `snicholls::string_like`, `snicholls::byte_sequence`, `snicholls::owning_string`, `snicholls::compile_time_string`, ...
/// - Module: \ref core.
///
/// \impls
/// Each property is also its own trait (`code_unit_t`, `code_unit_category_v`, `is_owning_sequence_v`, ...).
///
/// \contracts
/// - Compile time only; any cv/ref qualification of `S`.
///
/// \errors
/// A type that is not a code-unit sequence has no traits (the concepts are false).
///
/// \examples
/// \snippet string_traits_demo.cpp demo
///
/// \perf
/// Not benchmarked.
///
/// \testing
/// - Tests: tests/reflect/string_traits_test.cpp.
/// - Demo: demos/reflect/string_traits_demo.cpp.
///
/// \deps
/// CMake target `snicholls::core`; standard library only.
///
/// \gotchas
/// None known.
///
/// \status
/// Stable. Rules: CLAUDE.md, "Strings".
///
/// \tparam S the string or byte-sequence type.
template <code_unit_sequence S>
struct string_traits
{
    /// \brief The code unit type of `S`.
    using code_unit_type = code_unit_t<S>;
    /// \brief The category.
    static constexpr code_unit_category category = code_unit_category_v<S>;
    /// \brief The owning.
    static constexpr bool owning = is_owning_sequence_v<S>;
    /// \brief The mutable code units.
    static constexpr bool mutable_code_units = has_mutable_code_units_v<S>;
    /// \brief The contiguous.
    static constexpr bool contiguous = is_contiguous_sequence_v<S>;
    /// \brief The termination.
    static constexpr null_termination termination = null_termination_v<S>;
    /// \brief The length.
    static constexpr length_kind length = length_kind_v<S>;
    /// \brief The static length.
    static constexpr std::size_t static_length = static_length_v<S>;
};

} // namespace snicholls

#endif /* snicholls_core_reflect_string_traits_hpp */
