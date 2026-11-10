//
//  string_traits_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/reflect/string_traits.hpp>.
//

// Demo: reflect/string_traits.hpp - classification of every string / byte-sequence type the snicholls libraries meet.
// Prints the classification table and checks it against the documented one; exits non-zero on any difference.

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <iostream>
#include <memory_resource>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <snicholls/core/overloaded.hpp>
#include <snicholls/core/reflect/string_traits.hpp>

namespace demo {

// A code unit's name: an overload set over std::type_identity, with a catch-all last; no if constexpr chain.
constexpr auto unit_names = snicholls::overloaded{
    [](std::type_identity<char>) { return std::string_view{"char"}; },
    [](std::type_identity<wchar_t>) { return std::string_view{"wchar_t"}; },
    [](std::type_identity<char8_t>) { return std::string_view{"char8_t"}; },
    [](std::type_identity<char16_t>) { return std::string_view{"char16_t"}; },
    [](std::type_identity<char32_t>) { return std::string_view{"char32_t"}; },
    [](std::type_identity<std::byte>) { return std::string_view{"std::byte"}; },
    [](std::type_identity<unsigned char>) { return std::string_view{"unsigned char"}; },
    [](auto) { return std::string_view{"?"}; },
};

template <typename U>
constexpr std::string_view unit_name()
{
    return unit_names(std::type_identity<U>{});
}

// The properties of S as one table row (without the type's name; optionally without the unit column).
template <typename S>
std::string properties(bool with_unit = true)
{
    using tr = snicholls::string_traits<S>;
    std::ostringstream out;
    out << std::left << std::setw(6) << snicholls::enumerator_name(tr::category);
    if (with_unit) out << std::setw(14) << unit_name<typename tr::code_unit_type>();
    out << std::setw(5) << (tr::owning ? "yes" : "no") << std::setw(5) << (tr::mutable_code_units ? "yes" : "no")
        << std::setw(5) << (tr::contiguous ? "yes" : "no") << std::setw(12) << snicholls::enumerator_name(tr::termination)
        << std::setw(14) << snicholls::enumerator_name(tr::length);
    if (tr::static_length != std::dynamic_extent) out << tr::static_length;
    else out << "-";
    return out.str();
}

std::ostringstream table;

template <typename S>
void row(std::string_view name)
{
    table << std::left << std::setw(40) << name << properties<S>() << '\n';
}

// Every carrier over C, without type names or units: proves each character type classifies like char.
template <typename C>
std::vector<std::string> carriers()
{
    return {properties<C*>(false), properties<const C*>(false), properties<C[4]>(false),
            properties<const C[4]>(false), properties<const C (&)[4]>(false), properties<std::basic_string<C>>(false),
            properties<std::pmr::basic_string<C>>(false), properties<std::basic_string_view<C>>(false),
            properties<std::span<C>>(false), properties<std::span<const C>>(false),
            properties<std::span<const C, 4>>(false), properties<std::vector<C>>(false),
            properties<std::array<C, 4>>(false), properties<snicholls::basic_fixed_string<C, 4>>(false)};
}

template <typename C>
bool classifies_like_char()
{
    const bool same = carriers<C>() == carriers<char>() && std::is_same_v<snicholls::code_unit_t<std::span<const C>>, C>;
    std::cout << "  " << std::right << std::setw(9) << unit_name<C>()
              << ": all 14 carriers classify exactly as for char -> " << (same ? "yes" : "NO") << '\n';
    return same;
}

template <typename S>
void rejected(std::string_view name, std::string_view why)
{
    std::cout << "  " << std::left << std::setw(28) << name << (snicholls::code_unit_sequence<S> ? "ACCEPTED (wrong)" : why)
              << '\n';
}

} // namespace demo

//! [demo]
int main()
{
    using namespace demo;
    const std::string header = "type                                    kind  unit          own  mut  cont NUL         length        static\n";

    row<char*>("char*");
    row<const char*>("const char*");
    row<char[4]>("char[4]");
    row<const char[4]>("const char[4]");
    row<const char (&)[4]>("const char(&)[4]  (\"abc\")");
    row<std::string>("std::string");
    row<const std::string&>("const std::string&");
    row<std::pmr::string>("std::pmr::string");
    row<std::string_view>("std::string_view");
    row<std::span<char>>("std::span<char>");
    row<std::span<const char>>("std::span<const char>");
    row<std::span<const char, 4>>("std::span<const char, 4>");
    row<std::vector<char>>("std::vector<char>");
    row<std::array<char, 4>>("std::array<char, 4>");
    row<snicholls::fixed_string<4>>("snicholls::fixed_string<4>");
    row<snicholls::basic_fixed_string<char16_t, 4>>("basic_fixed_string<char16_t, 4>");
    row<std::u8string>("std::u8string");
    row<const char16_t*>("const char16_t*");
    row<std::u32string_view>("std::u32string_view");
    row<const wchar_t (&)[4]>("const wchar_t(&)[4]  (L\"abc\")");
    row<std::vector<std::uint8_t>>("std::vector<std::uint8_t>");
    row<std::vector<unsigned char>>("std::vector<unsigned char>");
    row<std::vector<std::byte>>("std::vector<std::byte>");
    row<std::span<const std::byte>>("std::span<const std::byte>");
    row<std::span<const std::uint8_t>>("std::span<const std::uint8_t>");
    row<std::span<const std::byte, 16>>("std::span<const std::byte, 16>");
    row<std::array<std::byte, 4>>("std::array<std::byte, 4>");
    row<unsigned char[4]>("unsigned char[4]");

    constexpr std::string_view documented =
        "char*                                   text  char          no   yes  yes  by_contract linear_time   -\n"
        "const char*                             text  char          no   no   yes  by_contract linear_time   -\n"
        "char[4]                                 text  char          yes  yes  yes  by_contract compile_time  3\n"
        "const char[4]                           text  char          yes  no   yes  by_contract compile_time  3\n"
        "const char(&)[4]  (\"abc\")               text  char          no   no   yes  by_contract compile_time  3\n"
        "std::string                             text  char          yes  yes  yes  guaranteed  constant_time -\n"
        "const std::string&                      text  char          no   no   yes  guaranteed  constant_time -\n"
        "std::pmr::string                        text  char          yes  yes  yes  guaranteed  constant_time -\n"
        "std::string_view                        text  char          no   no   yes  none        constant_time -\n"
        "std::span<char>                         text  char          no   yes  yes  none        constant_time -\n"
        "std::span<const char>                   text  char          no   no   yes  none        constant_time -\n"
        "std::span<const char, 4>                text  char          no   no   yes  none        compile_time  4\n"
        "std::vector<char>                       text  char          yes  yes  yes  none        constant_time -\n"
        "std::array<char, 4>                     text  char          yes  yes  yes  none        compile_time  4\n"
        "snicholls::fixed_string<4>              text  char          yes  no   yes  guaranteed  compile_time  4\n"
        "basic_fixed_string<char16_t, 4>         text  char16_t      yes  no   yes  guaranteed  compile_time  4\n"
        "std::u8string                           text  char8_t       yes  yes  yes  guaranteed  constant_time -\n"
        "const char16_t*                         text  char16_t      no   no   yes  by_contract linear_time   -\n"
        "std::u32string_view                     text  char32_t      no   no   yes  none        constant_time -\n"
        "const wchar_t(&)[4]  (L\"abc\")           text  wchar_t       no   no   yes  by_contract compile_time  3\n"
        "std::vector<std::uint8_t>               bytes unsigned char yes  yes  yes  none        constant_time -\n"
        "std::vector<unsigned char>              bytes unsigned char yes  yes  yes  none        constant_time -\n"
        "std::vector<std::byte>                  bytes std::byte     yes  yes  yes  none        constant_time -\n"
        "std::span<const std::byte>              bytes std::byte     no   no   yes  none        constant_time -\n"
        "std::span<const std::uint8_t>           bytes unsigned char no   no   yes  none        constant_time -\n"
        "std::span<const std::byte, 16>          bytes std::byte     no   no   yes  none        compile_time  16\n"
        "std::array<std::byte, 4>                bytes std::byte     yes  yes  yes  none        compile_time  4\n"
        "unsigned char[4]                        bytes unsigned char yes  yes  yes  none        compile_time  4\n";

    std::cout << "Classification (own = owns its code units, mut = code units mutable through the type,\n"
                 "cont = contiguous, static = length when it is part of the type):\n\n"
              << header << table.str() << '\n';
    bool ok = table.str() == documented;
    if (!ok) std::cout << "TABLE DIFFERS FROM THE DOCUMENTED ONE:\n" << documented << '\n';

    std::cout << "Every character type has the same carriers with the same properties:\n";
    ok = classifies_like_char<wchar_t>() && ok;
    ok = classifies_like_char<char8_t>() && ok;
    ok = classifies_like_char<char16_t>() && ok;
    ok = classifies_like_char<char32_t>() && ok;

    std::cout << "\nNot code-unit sequences:\n";
    rejected<char>("char", "a single code unit, not a sequence");
    rejected<bool>("bool", "not a code unit");
    rejected<std::nullptr_t>("std::nullptr_t", "not a pointer to text (a null char* is fine: empty)");
    rejected<std::vector<int>>("std::vector<int>", "int is not a code unit");
    rejected<std::span<int>>("std::span<int>", "int is not a code unit");
    rejected<std::vector<signed char>>("std::vector<signed char>", "signed char / int8_t are numbers");
    rejected<const signed char*>("const signed char*", "signed char / int8_t are numbers");
    rejected<const std::byte*>("const std::byte*", "bytes have no terminator, so no length");
    rejected<std::deque<char>>("std::deque<char>", "not contiguous");
    rejected<volatile char*>("volatile char*", "volatile code units");
    ok = ok && !snicholls::code_unit_sequence<char> && !snicholls::code_unit_sequence<std::vector<signed char>>
         && !snicholls::code_unit_sequence<const std::byte*> && !snicholls::code_unit_sequence<std::nullptr_t>;

    std::cout << "\nConcepts:\n" << std::boolalpha;
    std::cout << "  string_like<std::vector<char>>               " << snicholls::string_like<std::vector<char>> << '\n';
    std::cout << "  byte_sequence<std::vector<std::uint8_t>>     " << snicholls::byte_sequence<std::vector<std::uint8_t>> << '\n';
    std::cout << "  owning_string<std::pmr::string>              " << snicholls::owning_string<std::pmr::string> << '\n';
    std::cout << "  string_view_like<const std::string&>         " << snicholls::string_view_like<const std::string&> << '\n';
    std::cout << "  null_terminated_string<std::string_view>     " << snicholls::null_terminated_string<std::string_view> << '\n';
    std::cout << "  fixed_length_string<std::array<char, 8>>     " << snicholls::fixed_length_string<std::array<char, 8>> << '\n';
    std::cout << "  compile_time_string<snicholls::fixed_string<5>>   " << snicholls::compile_time_string<snicholls::fixed_string<5>> << '\n';
    std::cout << "  compile_time_string<std::array<char, 8>>     " << snicholls::compile_time_string<std::array<char, 8>> << '\n';
    ok = ok && snicholls::string_like<std::vector<char>> && snicholls::byte_sequence<std::vector<std::uint8_t>>
         && snicholls::owning_string<std::pmr::string> && snicholls::string_view_like<const std::string&>
         && !snicholls::null_terminated_string<std::string_view> && snicholls::fixed_length_string<std::array<char, 8>>
         && snicholls::compile_time_string<snicholls::fixed_string<5>> && !snicholls::compile_time_string<std::array<char, 8>>;

    std::cout << "\nchar_traits_t<std::u16string> is std::char_traits<char16_t>: "
              << std::is_same_v<snicholls::char_traits_t<std::u16string>, std::char_traits<char16_t>> << '\n';
    std::cout << "enumerator_name of an out-of-range value: '"
              << snicholls::enumerator_name(static_cast<snicholls::null_termination>(9)) << "' / '"
              << snicholls::enumerator_name(static_cast<snicholls::length_kind>(9)) << "'\n";
    ok = ok && snicholls::enumerator_name(static_cast<snicholls::null_termination>(9)) == "?"
         && snicholls::enumerator_name(static_cast<snicholls::length_kind>(9)) == "?"
         && snicholls::is_text_code_unit_v<char8_t> && snicholls::is_byte_code_unit_v<std::uint8_t>
         && !snicholls::is_byte_code_unit_v<signed char>;

    std::cout << (ok ? "\nOK\n" : "\nFAILED\n");
    return ok ? 0 : 1;
}
//! [demo]
