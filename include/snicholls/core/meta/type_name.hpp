//
//  type_name.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  New in core, 2026-11-07, for printing type lists: the name of a type as the compiler
//  spells it in a function signature, cut out of __PRETTY_FUNCTION__ / __FUNCSIG__ at
//  compile time. The classic trick, with the compilers' prefixes normalised.
//

/// \file
/// \brief `snicholls::type_name<T>()`: a type's name as a compile-time string, for messages and printed type lists.
/// \ingroup core
///
/// **Overview.** `type_name<int>()` is `"int"`, `type_name<std::vector<int>>()` is the
/// compiler's spelling of that type, as a `std::string_view` that is a constant expression:
/// usable in a `static_assert` message, a log line, or `acout`'s printing of a
/// `meta::type_list`. There is no `typeid`, no RTTI and no demangler: the name is cut out of
/// the compiler's own pretty-printed signature of this function.
///
/// \snippet meta/type_name_demo.cpp names
///
/// **Relationships.** `acout.hpp` prints a `type_list` with it; `basic_type_registry` binds
/// names chosen by hand, which is what to use where the spelling must be the same on every
/// compiler. Group: \ref core.
///
/// **Guarantees.** `constexpr` and `noexcept`; the view refers to static storage and is valid
/// for the program. Fundamental types and types declared in a user namespace spell the same on
/// GCC, Clang and MSVC (the class-keys MSVC writes, `struct`, `class` and `enum`, and the standard
/// libraries' inline namespaces are stripped, as whole words only). Standard templates spell as
/// the compiler writes them and differ: `std::string` is `std::basic_string<char>` on GCC and
/// Clang and shows its traits and allocator on MSVC.
///
/// **Errors.** None. A compiler that provides neither `__PRETTY_FUNCTION__` nor `__FUNCSIG__`
/// gets `"?"`.
///
/// **Performance.** Compile time only.
///
/// **Tests and demos.** `tests/meta/type_name_test.cpp`, `demos/meta/type_name_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<string_view>`.
///
/// **Known issues.** The spelling of a template specialisation is the compiler's, with the
/// library's inline namespaces removed but default template arguments shown where the
/// compiler shows them (`std::vector<int, std::allocator<int>>` on GCC and MSVC, `std::vector<int>`
/// on Clang). Compare against a fixed string only for the portable cases above.
///
/// **Status.** New in 0.1.0.

#ifndef snicholls_core_meta_type_name_hpp
#define snicholls_core_meta_type_name_hpp

#include <array>
#include <cstddef>
#include <string_view>

namespace snicholls {

    namespace detail {

        // The raw signature of type_name<T>(), with T spelled out by the compiler.
        template <typename T>
        constexpr std::string_view raw_type_signature() noexcept
        {
#if defined(__clang__) || defined(__GNUC__)
            return __PRETTY_FUNCTION__;
#elif defined(_MSC_VER)
            return __FUNCSIG__;
#else
            return "?";
#endif
        }

        constexpr bool is_identifier_character(char c) noexcept
        {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
        }

        // How much to drop at position i: a class-key MSVC writes or a library's inline namespace, as a
        // whole word only, so that foo::my_struct keeps its name and std:: stays on std::__cxx11::string.
        constexpr std::size_t type_name_noise_at(std::string_view raw, std::size_t i) noexcept
        {
            if (i > 0 && is_identifier_character(raw[i - 1]))
                return 0;
            constexpr std::string_view noise[] = {"struct ", "class ", "enum ", "union ", "__cxx11::", "__1::"};
            for (const std::string_view token : noise)
                if (raw.substr(i, token.size()) == token)
                    return token.size();
            return 0;
        }

        // The part of the signature that is T.
        template <typename T>
        constexpr std::string_view cut_type_name() noexcept
        {
            constexpr std::string_view sig = raw_type_signature<T>();
#if defined(__clang__)
            constexpr std::string_view head = "[T = ";
            constexpr std::string_view tail = "]";
#elif defined(__GNUC__)
            constexpr std::string_view head = "[with T = ";
            constexpr std::string_view tail = ";";
#elif defined(_MSC_VER)
            constexpr std::string_view head = "raw_type_signature<";
            constexpr std::string_view tail = ">(void)";
#else
            return sig;
#endif
            const std::size_t start = sig.find(head);
            if (start == std::string_view::npos)
                return sig;
            const std::size_t from = start + head.size();
            std::size_t end = sig.find(tail, from);
            if (end == std::string_view::npos)
                end = sig.size();
            while (end > from && sig[end - 1] == ' ')   // MSVC writes "> >" and leaves one before the tail
                --end;
            return sig.substr(from, end - from);
        }

        constexpr std::size_t normalised_size(std::string_view raw) noexcept
        {
            std::size_t size = 0;
            for (std::size_t i = 0; i < raw.size();) {
                const std::size_t noise = type_name_noise_at(raw, i);
                size += noise == 0 ? 1 : 0;
                i += noise == 0 ? 1 : noise;
            }
            return size;
        }

        // The normalised name, exactly as long as it needs to be: no limit, and no bytes wasted per type.
        template <typename T>
        constexpr auto make_type_name_storage() noexcept
        {
            constexpr std::string_view raw = cut_type_name<T>();
            std::array<char, normalised_size(raw) + 1> out{};
            std::size_t size = 0;
            for (std::size_t i = 0; i < raw.size();) {
                const std::size_t noise = type_name_noise_at(raw, i);
                if (noise == 0)
                    out[size++] = raw[i];
                i += noise == 0 ? 1 : noise;
            }
            return out;
        }

        template <typename T>
        inline constexpr auto type_name_storage = make_type_name_storage<T>();

    } // namespace detail

    /// \brief The name of `T` as the compiler spells it, normalised, as a constant expression.
    /// \ingroup core
    /// \tparam T Any type; cv-qualifiers and references are part of the name.
    /// \return A view into static storage, valid for the program.
    template <typename T>
    [[nodiscard]] constexpr std::string_view type_name() noexcept
    {
        return {detail::type_name_storage<T>.data(), detail::type_name_storage<T>.size() - 1};
    }

    /// \brief The name of the type of `value`, as `type_name<T>()`.
    /// \ingroup core
    /// \tparam T Deduced from the argument; references and top-level cv do not survive deduction.
    /// \param value The value; only its type is used.
    /// \return `type_name<T>()`.
    template <typename T>
    [[nodiscard]] constexpr std::string_view type_name(const T& value) noexcept
    {
        static_cast<void>(value);
        return type_name<T>();
    }

} // namespace snicholls

#endif /* snicholls_core_meta_type_name_hpp */
