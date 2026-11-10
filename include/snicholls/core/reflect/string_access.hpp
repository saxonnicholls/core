//
//  string_access.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Zero-copy, allocation-free access to strings and byte sequences.
/// \ingroup core
///
/// Zero-copy, allocation-free uniform access to every code-unit sequence classified by \<snicholls/core/reflect/string_traits.hpp>.
///
/// \code
///   snicholls::to_string_view(s)  text  -> std::basic_string_view<code_unit_t<S>, char_traits_t<S>>
///   snicholls::as_bytes_view(s)   bytes, or single-byte text (char, char8_t) -> std::span<const std::byte>
///   snicholls::length_of(s)       any   -> length in code units
///   snicholls::c_str_of(s)        NUL-terminated text -> const code_unit_t<S>* (never null)
/// \endcode
///
/// Cost: O(1) for every carrier except C*, which is one char_traits::length scan (O(n)) - visible in the type
/// as length_kind::linear_time. Nothing allocates or throws (all noexcept); all are constexpr except
/// as_bytes_view over non-std::byte units, which needs reinterpret_cast.
///
/// Null C* (char* p = nullptr): an empty string - to_string_view gives an empty view, length_of gives 0,
/// c_str_of gives "", as_bytes_view an empty span. One branch, never UB (std::string_view(p) would be).
///
/// Character arrays are string literals: length N-1 (embedded NULs kept), and the last element must be NUL.
/// Violating that is a compile error in constant evaluation and an assert at run time (debug builds).

#ifndef snicholls_core_reflect_string_access_hpp
#define snicholls_core_reflect_string_access_hpp

#include <cassert>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>

#include <snicholls/core/reflect/string_traits.hpp>

namespace snicholls {

namespace detail {

// Deliberately not constexpr: reaching it during constant evaluation makes the expression non-constant.
inline void snicholls_character_array_is_not_nul_terminated() noexcept {}

template <typename C, std::size_t N>
constexpr void check_string_literal(const C (&a)[N]) noexcept
{
    if (std::is_constant_evaluated()) {
        if (a[N - 1] != C{}) snicholls_character_array_is_not_nul_terminated();
    }
    else {
        assert(a[N - 1] == C{} && "snicholls: a character array is a string literal; its last element must be NUL");
    }
}

template <typename C>
inline constexpr C empty_c_string[1]{};

// Pointer to the first code unit (null only for a null C*).
template <code_unit_sequence S>
constexpr const code_unit_t<S>* data_of(const S& s) noexcept
{
    using unit = code_unit_t<S>;
    constexpr carrier kind = classify<S>::kind;
    if constexpr (kind == carrier::pointer) {
        return s;
    }
    else if constexpr (kind == carrier::array) {
        if constexpr (is_text_code_unit_v<unit>) check_string_literal(s);
        return s;
    }
    else {
        return s.data();
    }
}

} // namespace detail

/// \brief The number of code units.
///
/// Static_length_v\<S> when it is part of the type, .size() when stored, char_traits::length for C* (0 for a null pointer).
/// \ingroup core
/// \param s the string.
/// \return the number of code units.
template <code_unit_sequence S>
constexpr std::size_t length_of(const S& s) noexcept
{
    constexpr detail::carrier kind = detail::classify<S>::kind;
    if constexpr (kind == detail::carrier::pointer) {
        return s == nullptr ? 0 : std::char_traits<code_unit_t<S>>::length(s);
    }
    else if constexpr (length_kind_v<S> == length_kind::compile_time) {
        if constexpr (kind == detail::carrier::array && string_like<S>) detail::check_string_literal(s);
        return static_length_v<S>;
    }
    else {
        return s.size();
    }
}

/// \brief The string view type of `S` (keeps its character traits).
template <string_like S>
using string_view_t = std::basic_string_view<code_unit_t<S>, char_traits_t<S>>;

/// \brief Zero-copy view of any text sequence, keeping its character traits.
/// \ingroup core
///
/// \overview
/// Uniform, zero-copy, allocation-free access to every code-unit sequence classified by `reflect/string_traits.hpp`. The function table is in the file description above.
///
/// \relations
/// - Siblings: `snicholls::length_of`, `snicholls::c_str_of`, `snicholls::as_bytes_view`; types: `snicholls::string_view_t`.
/// - Module: \ref core.
///
/// \impls
/// None: a single implementation.
///
/// \contracts
/// - `noexcept`, no allocation, O(1) except a `C*` (one strlen). A null `C*` is the empty string (never UB).
///
/// \errors
/// None.
///
/// \examples
/// \snippet string_access_demo.cpp demo
///
/// \perf
/// Not benchmarked.
///
/// \testing
/// - Tests: tests/reflect/string_access_test.cpp; compile-fail case (unterminated character array in constant
///   evaluation) in tests/compile_fail/string_access_unterminated_array.cpp.
/// - Demo: demos/reflect/string_access_demo.cpp.
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
/// \tparam S a `snicholls::string_like` type.
/// \param s the string.
/// \return a view of its code units.
template <string_like S>
constexpr string_view_t<S> to_string_view(const S& s) noexcept
{
    const auto* data = detail::data_of(s);
    if constexpr (detail::classify<S>::kind == detail::carrier::pointer) {
        if (data == nullptr) return string_view_t<S>{};
    }
    return string_view_t<S>(data, length_of(s));
}

/// \brief A pointer suitable for a C API: s.c_str() / the array / the pointer itself; "" for a null C*.
/// \ingroup core
/// \param s the string.
/// \return a pointer suitable for a C API: s.c_str() / the array / the pointer itself; "" for a null C*.
template <null_terminated_string S>
constexpr const code_unit_t<S>* c_str_of(const S& s) noexcept
{
    const auto* data = detail::data_of(s);
    if constexpr (detail::classify<S>::kind == detail::carrier::pointer) {
        if (data == nullptr) return detail::empty_c_string<code_unit_t<S>>;
    }
    return data;
}

// Sequences whose bytes are well defined and portable: byte sequences, and text of one-byte code units
// (char, char8_t), whose bytes are its encoding (UTF-8). Wider text (wchar_t, char16_t, char32_t) is
// rejected: its bytes depend on platform endianness and wchar_t's width - use std::as_bytes explicitly.
template <typename S>
concept byte_viewable = byte_sequence<S> || (string_like<S> && sizeof(code_unit_t<S>) == 1);

/// \brief The code units as read-only bytes, zero-copy (no terminator included).
/// \ingroup core
/// \param s the string.
/// \return the code units as read-only bytes, zero-copy (no terminator included).
template <byte_viewable S>
constexpr std::span<const std::byte> as_bytes_view(const S& s) noexcept
{
    const auto* data = detail::data_of(s);
    if constexpr (detail::classify<S>::kind == detail::carrier::pointer) {
        if (data == nullptr) return {};
    }
    const std::size_t n = length_of(s);
    if constexpr (std::is_same_v<code_unit_t<S>, std::byte>)
        return std::span<const std::byte>(data, n);
    else
        return std::span<const std::byte>(reinterpret_cast<const std::byte*>(data), n);
}

} // namespace snicholls

#endif /* snicholls_core_reflect_string_access_hpp */
