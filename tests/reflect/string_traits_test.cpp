//
//  string_traits_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/reflect/string_traits.hpp>.
//

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <initializer_list>
#include <list>
#include <memory_resource>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include <snicholls/core/reflect/string_traits.hpp>

namespace {

using snicholls::code_unit_category;
using snicholls::length_kind;
using snicholls::null_termination;

constexpr auto text = code_unit_category::text;
constexpr auto bytes = code_unit_category::bytes;
constexpr auto no_nul = null_termination::none;
constexpr auto contract = null_termination::by_contract;
constexpr auto guaranteed = null_termination::guaranteed;
constexpr auto ct = length_kind::compile_time;
constexpr auto o1 = length_kind::constant_time;
constexpr auto on = length_kind::linear_time;
constexpr std::size_t dyn = std::dynamic_extent;

// One row of the classification table: every property, and every concept derived from them.
template <typename S, code_unit_category Category, typename Unit, bool Owning, bool Mutable, null_termination Term,
          length_kind Length, std::size_t StaticLength>
constexpr bool row()
{
    using tr = snicholls::string_traits<S>;
    static_assert(snicholls::code_unit_sequence<S>);
    static_assert(snicholls::is_code_unit_sequence_v<S>);
    static_assert(std::is_same_v<snicholls::code_unit_t<S>, Unit>);
    static_assert(std::is_same_v<typename tr::code_unit_type, Unit>);
    static_assert(snicholls::code_unit_category_v<S> == Category && tr::category == Category);
    static_assert(snicholls::is_text_sequence_v<S> == (Category == text));
    static_assert(snicholls::is_byte_sequence_v<S> == (Category == bytes));
    static_assert(snicholls::is_owning_sequence_v<S> == Owning && tr::owning == Owning);
    static_assert(snicholls::has_mutable_code_units_v<S> == Mutable && tr::mutable_code_units == Mutable);
    static_assert(snicholls::is_contiguous_sequence_v<S> && tr::contiguous);
    static_assert(snicholls::null_termination_v<S> == Term && tr::termination == Term);
    static_assert(snicholls::length_kind_v<S> == Length && tr::length == Length);
    static_assert(snicholls::static_length_v<S> == StaticLength && tr::static_length == StaticLength);
    static_assert((Length == ct) == (StaticLength != dyn));

    constexpr bool is_text = Category == text;
    static_assert(snicholls::string_like<S> == is_text);
    static_assert(snicholls::byte_sequence<S> == !is_text);
    static_assert(snicholls::owning_string<S> == (is_text && Owning));
    static_assert(snicholls::string_view_like<S> == (is_text && !Owning));
    static_assert(snicholls::null_terminated_string<S> == (is_text && Term != no_nul));
    static_assert(snicholls::fixed_length_string<S> == (is_text && Length == ct));
    return true;
}

template <typename S>
constexpr bool rejected()
{
    static_assert(!snicholls::code_unit_sequence<S>);
    static_assert(!snicholls::is_text_sequence_v<S> && !snicholls::is_byte_sequence_v<S>);
    static_assert(!snicholls::is_owning_sequence_v<S> && !snicholls::has_mutable_code_units_v<S>);
    static_assert(!snicholls::is_contiguous_sequence_v<S>);
    static_assert(!snicholls::string_like<S> && !snicholls::byte_sequence<S> && !snicholls::null_terminated_string<S>);
    static_assert(!snicholls::compile_time_string<S>);
    return true;
}

// A fixed-string wrapper for each character type (the wrappers other than fixed_string<N> are covered through
// basic_fixed_string and the derivation rule below).
template <typename C, std::size_t N>
struct derived_fixed : snicholls::basic_fixed_string<C, N>
{
    using snicholls::basic_fixed_string<C, N>::basic_fixed_string;
};

// Every carrier over text code unit C.
template <typename C>
constexpr bool text_table()
{
    using str = std::basic_string<C>;
    using pmr_str = std::pmr::basic_string<C>;
    using sv = std::basic_string_view<C>;
    using fs = snicholls::basic_fixed_string<C, 3>;

    //                 type                       category unit own    mut    NUL         length  static
    static_assert(row<C*,                         text,    C,   false, true,  contract,   on,     dyn>());
    static_assert(row<const C*,                   text,    C,   false, false, contract,   on,     dyn>());
    static_assert(row<C* const,                   text,    C,   false, true,  contract,   on,     dyn>());
    static_assert(row<const C* const&,            text,    C,   false, false, contract,   on,     dyn>());
    static_assert(row<C[4],                       text,    C,   true,  true,  contract,   ct,     3>());
    static_assert(row<const C[4],                 text,    C,   true,  false, contract,   ct,     3>());
    static_assert(row<const C (&)[4],             text,    C,   false, false, contract,   ct,     3>());   // literal
    static_assert(row<C (&)[4],                   text,    C,   false, true,  contract,   ct,     3>());
    static_assert(row<const C (&)[1],             text,    C,   false, false, contract,   ct,     0>());   // ""
    static_assert(row<str,                        text,    C,   true,  true,  guaranteed, o1,     dyn>());
    static_assert(row<const str,                  text,    C,   true,  false, guaranteed, o1,     dyn>());
    static_assert(row<str&,                       text,    C,   false, true,  guaranteed, o1,     dyn>());
    static_assert(row<const str&,                 text,    C,   false, false, guaranteed, o1,     dyn>());
    static_assert(row<str&&,                      text,    C,   false, true,  guaranteed, o1,     dyn>());
    static_assert(row<pmr_str,                    text,    C,   true,  true,  guaranteed, o1,     dyn>());
    static_assert(row<sv,                         text,    C,   false, false, no_nul,     o1,     dyn>());
    static_assert(row<const sv&,                  text,    C,   false, false, no_nul,     o1,     dyn>());
    static_assert(row<std::span<C>,               text,    C,   false, true,  no_nul,     o1,     dyn>());
    static_assert(row<const std::span<C>,         text,    C,   false, true,  no_nul,     o1,     dyn>());   // shallow const
    static_assert(row<std::span<const C>,         text,    C,   false, false, no_nul,     o1,     dyn>());
    static_assert(row<std::span<C, 5>,            text,    C,   false, true,  no_nul,     ct,     5>());
    static_assert(row<std::span<const C, 5>,      text,    C,   false, false, no_nul,     ct,     5>());
    static_assert(row<std::vector<C>,             text,    C,   true,  true,  no_nul,     o1,     dyn>());
    static_assert(row<const std::vector<C>,       text,    C,   true,  false, no_nul,     o1,     dyn>());
    static_assert(row<const std::vector<C>&,      text,    C,   false, false, no_nul,     o1,     dyn>());
    static_assert(row<std::pmr::vector<C>,        text,    C,   true,  true,  no_nul,     o1,     dyn>());
    static_assert(row<std::array<C, 6>,           text,    C,   true,  true,  no_nul,     ct,     6>());
    static_assert(row<const std::array<C, 6>,     text,    C,   true,  false, no_nul,     ct,     6>());
    static_assert(row<std::array<const C, 6>,     text,    C,   true,  false, no_nul,     ct,     6>());
    static_assert(row<std::array<C, 0>,           text,    C,   true,  true,  no_nul,     ct,     0>());
    static_assert(row<fs,                         text,    C,   true,  false, guaranteed, ct,     3>());
    static_assert(row<const fs&,                  text,    C,   false, false, guaranteed, ct,     3>());
    static_assert(row<derived_fixed<C, 2>,        text,    C,   true,  false, guaranteed, ct,     2>());

    static_assert(std::is_same_v<snicholls::char_traits_t<str>, std::char_traits<C>>);
    static_assert(std::is_same_v<snicholls::char_traits_t<const C*>, std::char_traits<C>>);
    static_assert(snicholls::compile_time_string<fs> && snicholls::compile_time_string<const fs&>);
    static_assert(snicholls::compile_time_string<derived_fixed<C, 2>>);
    static_assert(!snicholls::compile_time_string<std::array<C, 6>> && !snicholls::compile_time_string<const C (&)[4]>);

    // Not text sequences of C: volatile, unbounded arrays, non-contiguous containers, wrappers, the unit alone.
    static_assert(rejected<C>());
    static_assert(rejected<const C&>());
    static_assert(rejected<volatile C*>());
    static_assert(rejected<volatile str>());
    static_assert(rejected<std::span<volatile C>>());
    static_assert(rejected<C[]>());
    static_assert(rejected<C**>());
    static_assert(rejected<str*>());
    static_assert(rejected<std::deque<C>>());
    static_assert(rejected<std::list<C>>());
    static_assert(rejected<std::optional<str>>());
    static_assert(rejected<std::initializer_list<C>>());
    return true;
}

static_assert(text_table<char>());
static_assert(text_table<wchar_t>());
static_assert(text_table<char8_t>());
static_assert(text_table<char16_t>());
static_assert(text_table<char32_t>());

// Stand-in traits: naming std::basic_string<B> would name std::char_traits<B>, which libc++ deprecates for
// non-character types. The classification never looks at the traits, so any type shows the rule.
struct not_char_traits {};

// Every carrier over byte code unit B.
template <typename B>
constexpr bool byte_table()
{
    //                 type                          category unit own    mut    NUL     length static
    static_assert(row<std::vector<B>,                bytes,   B,   true,  true,  no_nul, o1,    dyn>());
    static_assert(row<const std::vector<B>&,         bytes,   B,   false, false, no_nul, o1,    dyn>());
    static_assert(row<std::pmr::vector<B>,           bytes,   B,   true,  true,  no_nul, o1,    dyn>());
    static_assert(row<std::span<B>,                  bytes,   B,   false, true,  no_nul, o1,    dyn>());
    static_assert(row<std::span<const B>,            bytes,   B,   false, false, no_nul, o1,    dyn>());
    static_assert(row<std::span<const B, 4>,         bytes,   B,   false, false, no_nul, ct,    4>());
    static_assert(row<std::array<B, 4>,              bytes,   B,   true,  true,  no_nul, ct,    4>());
    static_assert(row<const std::array<B, 4>,        bytes,   B,   true,  false, no_nul, ct,    4>());
    static_assert(row<B[4],                          bytes,   B,   true,  true,  no_nul, ct,    4>());   // no terminator: N
    static_assert(row<const B (&)[4],                bytes,   B,   false, false, no_nul, ct,    4>());

    // Bytes have no terminator, so no length for a pointer; and they are never a basic_string/string_view.
    static_assert(rejected<B>());
    static_assert(rejected<B*>());
    static_assert(rejected<const B*>());
    static_assert(rejected<std::basic_string<B, not_char_traits>>());
    static_assert(rejected<std::basic_string_view<B, not_char_traits>>());
    static_assert(rejected<std::deque<B>>());
    return true;
}

static_assert(byte_table<std::byte>());
static_assert(byte_table<unsigned char>());
static_assert(byte_table<std::uint8_t>());

TEST(StringTraitsCodeUnits, TextAndByteUnitsAreDisjointAndExhaustive)
{
    static_assert(snicholls::is_text_code_unit_v<char> && snicholls::is_text_code_unit_v<wchar_t>);
    static_assert(snicholls::is_text_code_unit_v<char8_t> && snicholls::is_text_code_unit_v<char16_t>);
    static_assert(snicholls::is_text_code_unit_v<char32_t>);
    static_assert(snicholls::is_byte_code_unit_v<std::byte> && snicholls::is_byte_code_unit_v<unsigned char>);
    static_assert(snicholls::is_byte_code_unit_v<std::uint8_t>);
    static_assert(!snicholls::is_text_code_unit_v<unsigned char> && !snicholls::is_byte_code_unit_v<char>);
    static_assert(!snicholls::is_text_code_unit_v<signed char> && !snicholls::is_byte_code_unit_v<signed char>);
    static_assert(!snicholls::is_text_code_unit_v<const char>);   // cv-unqualified units only
    static_assert(!snicholls::is_byte_code_unit_v<int> && !snicholls::is_text_code_unit_v<bool>);
}

TEST(StringTraitsCodeUnits, Uint8IsUnsignedCharSoUint8VectorsAreBytes)
{
    // The classification relies on this (true on every supported platform): std::vector<std::uint8_t> and
    // std::vector<unsigned char> are the same type, and both are byte sequences here.
    static_assert(std::is_same_v<std::uint8_t, unsigned char>);
    static_assert(snicholls::byte_sequence<std::vector<std::uint8_t>>);
    static_assert(std::is_same_v<snicholls::code_unit_t<std::span<const std::uint8_t>>, unsigned char>);
}

TEST(StringTraitsRejects, NonStringTypes)
{
    static_assert(rejected<int>());
    static_assert(rejected<bool>());
    static_assert(rejected<std::nullptr_t>());
    static_assert(rejected<void>());
    static_assert(rejected<void*>());
    static_assert(rejected<int*>());
    static_assert(rejected<int[4]>());
    static_assert(rejected<std::vector<int>>());
    static_assert(rejected<std::vector<bool>>());
    static_assert(rejected<std::span<int>>());
    static_assert(rejected<std::array<int, 3>>());
    static_assert(rejected<std::optional<std::string>>());
    static_assert(rejected<void (*)()>());
    struct incomplete;
    static_assert(rejected<incomplete*>());
}

TEST(StringTraitsRejects, SignedCharSequencesAreNumbersNotTextOrBytes)
{
    static_assert(rejected<signed char*>());
    static_assert(rejected<const signed char (&)[4]>());
    static_assert(rejected<std::vector<signed char>>());
    static_assert(rejected<std::span<const signed char>>());
    static_assert(rejected<std::array<std::int8_t, 4>>());
    static_assert(rejected<std::basic_string<signed char, not_char_traits>>());   // named only, never instantiated
}

TEST(StringTraitsFixedString, FixedStringIsACompileTimeString)
{
    using name = decltype(snicholls::fixed_string("price"));
    static_assert(row<name, text, char, true, false, guaranteed, ct, 5>());
    static_assert(snicholls::compile_time_string<name>);
    static_assert(snicholls::compile_time_string<snicholls::basic_fixed_string<char, 0>>);
}

struct ci_traits : std::char_traits<char> {};

TEST(StringTraitsCharTraits, CustomTraitsAreKept)
{
    using ci_string = std::basic_string<char, ci_traits>;
    using ci_view = std::basic_string_view<char, ci_traits>;
    static_assert(std::is_same_v<snicholls::char_traits_t<ci_string>, ci_traits>);
    static_assert(std::is_same_v<snicholls::char_traits_t<const ci_view&>, ci_traits>);
    static_assert(std::is_same_v<snicholls::char_traits_t<std::vector<char>>, std::char_traits<char>>);
    static_assert(row<ci_string, text, char, true, true, guaranteed, o1, dyn>());
}

TEST(StringTraitsEnumerators, NamesForTables)
{
    static_assert(snicholls::enumerator_name(code_unit_category::text) == "text");
    static_assert(snicholls::enumerator_name(code_unit_category::bytes) == "bytes");
    static_assert(snicholls::enumerator_name(null_termination::none) == "none");
    static_assert(snicholls::enumerator_name(null_termination::by_contract) == "by_contract");
    static_assert(snicholls::enumerator_name(null_termination::guaranteed) == "guaranteed");
    static_assert(snicholls::enumerator_name(length_kind::compile_time) == "compile_time");
    static_assert(snicholls::enumerator_name(length_kind::constant_time) == "constant_time");
    static_assert(snicholls::enumerator_name(length_kind::linear_time) == "linear_time");
    // Out-of-range values (valid for a fixed underlying type) are named "?", never UB.
    EXPECT_EQ(snicholls::enumerator_name(static_cast<null_termination>(99)), "?");
    EXPECT_EQ(snicholls::enumerator_name(static_cast<length_kind>(99)), "?");
    static_assert(noexcept(snicholls::enumerator_name(length_kind::compile_time)));
}

// sequence_like: a range that is not text
static_assert(snicholls::sequence_like<std::vector<int>> && snicholls::sequence_like<std::array<std::string, 2>>);
static_assert(!snicholls::sequence_like<std::string> && !snicholls::sequence_like<std::string_view> && !snicholls::sequence_like<const char (&)[4]>);
static_assert(!snicholls::sequence_like<int>);

} // namespace
