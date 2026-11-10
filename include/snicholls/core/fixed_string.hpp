//
//  fixed_string.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Moved from base-encode-decode (BaseEncodeDecode/utils/fixed_string.hpp at
//  0ca9892). The definitions are unchanged but for the fixed-to-fixed comparison
//  operators, added in core; the guard and documentation are core's.
//

/// \file
/// \brief `snicholls::basic_fixed_string` and `snicholls::fixed_string`: compile-time strings usable as template parameters.
/// \ingroup core
///
/// **Overview.** A fixed-capacity string, usable as a non-type template parameter,
/// modelled on `std::basic_string` and `std::string_view` conventions and on
/// https://github.com/unterumarmung/fixed_string, but written fresh to be small and
/// dependency-free:
///
/// \snippet fixed_string_demo.cpp basics
///
/// As a template parameter, the reason the type exists:
///
/// \snippet fixed_string_demo.cpp nttp
///
/// `N` is the logical length, excluding the trailing `'\0'`, matching `std::string`:
/// `fixed_string("abc").size() == 3`. Deduction from a string literal drops the
/// literal's terminator, so `fixed_string("abc")` is a string of 3 characters.
///
/// **Implementations / variants.** `basic_fixed_string<CharT, N>` is the type to name
/// as a template parameter (`template <basic_fixed_string Name>`). The convenience
/// wrappers `fixed_string`, `wfixed_string`, `u8fixed_string`, `u16fixed_string` and
/// `u32fixed_string` mirror the standard character-type family; each derives from
/// `basic_fixed_string` and carries its own deduction guide. They are class templates
/// rather than alias templates because alias-template CTAD (P1814) is not supported
/// by every Clang release, so `fixed_string("abc")` must deduce through a real class
/// template.
///
/// **Guarantees.** Every operation is `constexpr`; every operation but construction
/// from a literal is `noexcept`. The character storage `elems` is a public member
/// because a non-type template parameter type must be structural: treat it as
/// read-only. Comparisons go through `std::basic_string_view`, so a fixed string
/// compares with another fixed string of any length, a `string_view`, a
/// `std::string` or a literal. `std::hash` is
/// specialised, so a fixed string can key an unordered container.
///
/// **Errors.** `substr<Pos, Count>()` out of range is a compile error.
/// `operator[]`, `front()` and `back()` are unchecked, as for `std::string_view`;
/// `front()` and `back()` on an empty string are undefined.
///
/// **Known issues.** `std::hash` is specialised for `basic_fixed_string`, not for the
/// wrappers (`fixed_string<N>`, ...): key an unordered container with
/// `basic_fixed_string<CharT, N>`, or pass the base's hash. (As moved from
/// base-encode-decode, comparing two fixed strings with each other was ambiguous
/// under C++20 and rejected by MSVC; the fixed-to-fixed operators fixed that.)
///
/// **Tests and demos.** `tests/fixed_string_test.cpp`, `demos/fixed_string_demo.cpp`.
///
/// **Dependencies.** `<compare>`, `<cstddef>`, `<functional>`, `<string_view>`.
///
/// **Status.** Stable.

#ifndef snicholls_core_fixed_string_hpp
#define snicholls_core_fixed_string_hpp

#include <compare>
#include <cstddef>
#include <functional>
#include <string_view>

namespace snicholls {

    /// \brief A compile-time string of exactly `N` characters, usable as a non-type template parameter.
    /// \tparam CharT The character type.
    /// \tparam N     The number of characters, excluding the trailing `'\0'`.
    /// \ingroup core
    template<typename CharT, std::size_t N>
    struct basic_fixed_string {
        using value_type = CharT;                         ///< The character type.
        using size_type = std::size_t;                    ///< The size type.
        using view_type = std::basic_string_view<CharT>;  ///< The view the comparisons go through.

        /// \brief `N` characters plus a trailing `'\0'`. Public for structural (NTTP) use; treat as read-only.
        CharT elems[N + 1]{}; // N characters plus a trailing '\0'; public for NTTP use

        /// \brief A string of `N` null characters.
        constexpr basic_fixed_string() = default;
        /// \brief Copy a string literal, terminator included.
        /// \param literal A character array of `N + 1` elements, normally a string literal.
        constexpr basic_fixed_string(const CharT (&literal)[N + 1]) {
            for (size_type i = 0; i <= N; ++i) {
                elems[i] = literal[i];
            }
        }

        // Capacity
        /// \brief The number of characters.
        /// \return `N`.
        static constexpr size_type size() noexcept { return N; }
        /// \brief The number of characters, the same as size().
        /// \return `N`.
        static constexpr size_type length() noexcept { return N; }
        /// \brief Whether the string has no characters.
        /// \return `N == 0`.
        [[nodiscard]] static constexpr bool empty() noexcept { return N == 0; }

        // Element access (read-only)
        /// \brief The characters, null-terminated.
        /// \return A pointer to the first character.
        constexpr const CharT* data() const noexcept { return elems; }
        /// \brief The characters, null-terminated, the same as data().
        /// \return A pointer to the first character.
        constexpr const CharT* c_str() const noexcept { return elems; }
        /// \brief The character at index `i`, unchecked.
        /// \param i An index, at most `N` (index `N` is the terminator).
        /// \return The character.
        constexpr const CharT& operator[](size_type i) const noexcept { return elems[i]; }
        /// \brief The first character. \pre `!empty()`.
        /// \return The first character.
        constexpr const CharT& front() const noexcept { return elems[0]; }
        /// \brief The last character. \pre `!empty()`.
        /// \return The last character.
        constexpr const CharT& back() const noexcept { return elems[N - 1]; }

        // Iterators
        /// \brief Iterator to the first character.
        /// \return A pointer to the first character.
        constexpr const CharT* begin() const noexcept { return elems; }
        /// \brief Iterator one past the last character, not past the terminator.
        /// \return A pointer to the terminator.
        constexpr const CharT* end() const noexcept { return elems + N; }
        /// \brief Iterator to the first character, the same as begin().
        /// \return A pointer to the first character.
        constexpr const CharT* cbegin() const noexcept { return elems; }
        /// \brief Iterator one past the last character, the same as end().
        /// \return A pointer to the terminator.
        constexpr const CharT* cend() const noexcept { return elems + N; }

        // Views / conversion
        /// \brief The characters as a string view, without the terminator.
        /// \return A view of `N` characters.
        constexpr view_type view() const noexcept { return view_type(elems, N); }
        /// \brief Implicit conversion to a string view, the same as view().
        /// \return A view of `N` characters.
        constexpr operator view_type() const noexcept { return view(); }

        // All comparisons go through string_view, so a fixed string compares
        // naturally against another fixed string of ANY length, a string_view,
        // a std::string, or a literal - with no ambiguity from the implicit
        // literal constructor. (No defaulted operator== is needed: a class with
        // a public character array is already a structural type, so it works as
        // a non-type template parameter regardless.)
        /// \brief Equality with anything that converts to a string view.
        /// \param other The string to compare with.
        /// \return Whether the characters are equal.
        constexpr bool operator==(view_type other) const noexcept { return view() == other; }
        /// \brief Three-way comparison with anything that converts to a string view.
        /// \param other The string to compare with.
        /// \return The ordering of the two strings, as `std::basic_string_view` orders them.
        constexpr auto operator<=>(view_type other) const noexcept { return view() <=> other; }

        // Fixed string against fixed string, any two lengths. Without these, a == b
        // converts each side to the other's view_type, so the normal and the
        // reversed (rewritten) member candidates tie: ambiguous on Clang for
        // different lengths, a warning on Clang and GCC, an error on MSVC even for
        // equal lengths (and so for std::equal_to). An exact match on both
        // arguments beats both, and the non-reversed form beats the reversed one.
        /// \brief Equality of two fixed strings of any lengths.
        /// \tparam M The length of the right-hand string.
        /// \param a  The left-hand string.
        /// \param b  The right-hand string.
        /// \return Whether the characters are equal (always false when `N != M`).
        template<size_type M>
        friend constexpr bool operator==(const basic_fixed_string& a, const basic_fixed_string<CharT, M>& b) noexcept {
            return a.view() == b.view();
        }
        /// \brief Three-way comparison of two fixed strings of any lengths.
        /// \tparam M The length of the right-hand string.
        /// \param a  The left-hand string.
        /// \param b  The right-hand string.
        /// \return The ordering of the two strings, as `std::basic_string_view::compare` orders them.
        template<size_type M>
        friend constexpr std::strong_ordering operator<=>(const basic_fixed_string& a,
                                                          const basic_fixed_string<CharT, M>& b) noexcept {
            // compare(), not view() <=> view(): libc++ before LLVM 16 (Xcode 14) has no
            // operator<=> for basic_string_view, and compare() is everywhere
            const int c = a.view().compare(b.view());
            return c < 0 ? std::strong_ordering::less
                 : c > 0 ? std::strong_ordering::greater
                         : std::strong_ordering::equal;
        }

        // Concatenation: fixed_string<N> + fixed_string<M> -> fixed_string<N+M>
        /// \brief Concatenate at compile time.
        /// \tparam M   The length of the right-hand string.
        /// \param rhs  The string to append.
        /// \return A string of `N + M` characters.
        template<size_type M>
        constexpr basic_fixed_string<CharT, N + M>
        operator+(const basic_fixed_string<CharT, M>& rhs) const noexcept {
            basic_fixed_string<CharT, N + M> out;
            for (size_type i = 0; i < N; ++i) out.elems[i] = elems[i];
            for (size_type i = 0; i < M; ++i) out.elems[N + i] = rhs.elems[i];
            out.elems[N + M] = CharT{};
            return out;
        }

        // Compile-time substring
        /// \brief A compile-time substring; out of range is a compile error.
        /// \tparam Pos   The first character to take.
        /// \tparam Count The number of characters to take.
        /// \return A string of `Count` characters.
        template<size_type Pos, size_type Count>
        constexpr basic_fixed_string<CharT, Count> substr() const noexcept {
            static_assert(Pos + Count <= N, "basic_fixed_string::substr: range out of bounds");
            basic_fixed_string<CharT, Count> out;
            for (size_type i = 0; i < Count; ++i) out.elems[i] = elems[Pos + i];
            out.elems[Count] = CharT{};
            return out;
        }
    };

    // Deduce the length from a string literal, dropping its terminator.
    /// \cond
    template<typename CharT, std::size_t N>
    basic_fixed_string(const CharT (&)[N]) -> basic_fixed_string<CharT, N - 1>;
    /// \endcond

    // Convenience wrappers mirroring the standard character-type family. These
    // are class templates (each deriving from basic_fixed_string and carrying its
    // own deduction guide) rather than alias templates, because alias-template
    // CTAD (P1814) is not supported by every clang release - so `fixed_string("abc")`
    // must deduce through a real class template. basic_fixed_string above remains
    // the type to name for non-type template parameters (Tag<"object1">).
    /// \brief Defines one convenience wrapper over basic_fixed_string; undefined again after use.
    /// \param Name  The wrapper's name.
    /// \param CharT Its character type.
#define SNICHOLLS_FIXED_STRING_WRAPPER(Name, CharT)                             \
    template<std::size_t N>                                                     \
    struct Name : basic_fixed_string<CharT, N> {                               \
        using basic_fixed_string<CharT, N>::basic_fixed_string;                \
        constexpr Name(const basic_fixed_string<CharT, N>& base) noexcept      \
            : basic_fixed_string<CharT, N>(base) {}                            \
    };                                                                          \
    template<std::size_t N> Name(const CharT (&)[N]) -> Name<N - 1>;

    SNICHOLLS_FIXED_STRING_WRAPPER(fixed_string, char)
    SNICHOLLS_FIXED_STRING_WRAPPER(wfixed_string, wchar_t)
    SNICHOLLS_FIXED_STRING_WRAPPER(u8fixed_string, char8_t)
    SNICHOLLS_FIXED_STRING_WRAPPER(u16fixed_string, char16_t)
    SNICHOLLS_FIXED_STRING_WRAPPER(u32fixed_string, char32_t)
#undef SNICHOLLS_FIXED_STRING_WRAPPER
}

// Hashing, so a fixed string can key an unordered container.
// (Documented in the file overview: Doxygen cannot place a member of std.)
/// \cond
template<typename CharT, std::size_t N>
struct std::hash<snicholls::basic_fixed_string<CharT, N>> {
    constexpr std::size_t operator()(const snicholls::basic_fixed_string<CharT, N>& s) const noexcept {
        return std::hash<std::basic_string_view<CharT>>{}(s.view());
    }
};
/// \endcond

#endif /* snicholls_core_fixed_string_hpp */
