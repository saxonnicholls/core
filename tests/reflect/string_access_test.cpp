//
//  string_access_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/reflect/string_access.hpp>.
//

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <snicholls/core/reflect/string_access.hpp>

namespace {

using namespace std::string_view_literals;
using namespace std::string_literals;

template <typename S>
concept viewable = requires(const S& s) { snicholls::to_string_view(s); };

template <typename S>
concept c_string_viewable = requires(const S& s) { snicholls::c_str_of(s); };

template <typename S>
constexpr bool all_noexcept()
{
    if constexpr (snicholls::string_like<S>)
        static_assert(noexcept(snicholls::to_string_view(std::declval<const S&>())));
    if constexpr (snicholls::null_terminated_string<S>)
        static_assert(noexcept(snicholls::c_str_of(std::declval<const S&>())));
    if constexpr (snicholls::byte_viewable<S>)
        static_assert(noexcept(snicholls::as_bytes_view(std::declval<const S&>())));
    static_assert(noexcept(snicholls::length_of(std::declval<const S&>())));
    return true;
}

TEST(StringAccessNoexcept, EveryCarrierEveryFunction)
{
    static_assert(all_noexcept<const char*>());
    static_assert(all_noexcept<char[4]>());
    static_assert(all_noexcept<std::string>());
    static_assert(all_noexcept<std::pmr::string>());
    static_assert(all_noexcept<std::u16string_view>());
    static_assert(all_noexcept<std::span<const char, 3>>());
    static_assert(all_noexcept<std::vector<char>>());
    static_assert(all_noexcept<std::array<char8_t, 2>>());
    static_assert(all_noexcept<snicholls::fixed_string<3>>());
    static_assert(all_noexcept<std::vector<std::byte>>());
    static_assert(all_noexcept<std::span<const std::uint8_t>>());
    static_assert(all_noexcept<std::byte[2]>());
}

TEST(StringAccessToStringView, EveryTextCarrier)
{
    const char* ptr = "ptr";
    char buffer[] = "buf";
    std::string str = "str";
    std::pmr::string pmr = "pmr";
    std::string_view sv = "view";
    std::vector<char> vec{'v', 'e', 'c'};
    std::array<char, 3> arr{'a', 'r', 'r'};
    std::span<const char> dyn(vec);
    std::span<const char, 3> fixed(arr);
    constexpr snicholls::fixed_string name("name");

    EXPECT_EQ(snicholls::to_string_view(ptr), "ptr");
    EXPECT_EQ(snicholls::to_string_view(buffer), "buf");
    EXPECT_EQ(snicholls::to_string_view("literal"), "literal");
    EXPECT_EQ(snicholls::to_string_view(str), "str");
    EXPECT_EQ(snicholls::to_string_view(pmr), "pmr");
    EXPECT_EQ(snicholls::to_string_view(sv), "view");
    EXPECT_EQ(snicholls::to_string_view(vec), "vec");
    EXPECT_EQ(snicholls::to_string_view(arr), "arr");
    EXPECT_EQ(snicholls::to_string_view(dyn), "vec");
    EXPECT_EQ(snicholls::to_string_view(fixed), "arr");
    EXPECT_EQ(snicholls::to_string_view(name), "name");
    static_assert(std::is_same_v<decltype(snicholls::to_string_view(str)), std::string_view>);
}

TEST(StringAccessToStringView, IsZeroCopy)
{
    std::string str = "zero copy, longer than any small-string buffer";
    std::vector<char> vec(str.begin(), str.end());
    EXPECT_EQ(snicholls::to_string_view(str).data(), str.data());
    EXPECT_EQ(snicholls::to_string_view(vec).data(), vec.data());
    EXPECT_EQ(snicholls::as_bytes_view(str).data(), reinterpret_cast<const std::byte*>(str.data()));
}

TEST(StringAccessEmbeddedNul, CountedCarriersKeepThemPointersStopAtThem)
{
    const std::string str = "a\0b"s;
    EXPECT_EQ(snicholls::length_of(str), 3u);
    EXPECT_EQ(snicholls::to_string_view(str), "a\0b"sv);
    EXPECT_EQ(snicholls::length_of("a\0b"), 3u);              // a literal is N-1 code units, not strlen
    EXPECT_EQ(snicholls::to_string_view("a\0b"), "a\0b"sv);
    const char* ptr = "a\0b";
    EXPECT_EQ(snicholls::length_of(ptr), 1u);                 // a pointer only knows its NUL
    EXPECT_EQ(snicholls::to_string_view(ptr), "a");
}

TEST(StringAccessEmpty, EveryCarrierCanBeEmpty)
{
    EXPECT_TRUE(snicholls::to_string_view("").empty());
    EXPECT_TRUE(snicholls::to_string_view(std::string{}).empty());
    EXPECT_TRUE(snicholls::to_string_view(std::string_view{}).empty());
    EXPECT_TRUE(snicholls::to_string_view(std::span<const char>{}).empty());
    EXPECT_TRUE(snicholls::to_string_view(std::vector<char>{}).empty());
    EXPECT_TRUE(snicholls::to_string_view(std::array<char, 0>{}).empty());
    EXPECT_TRUE(snicholls::to_string_view(snicholls::basic_fixed_string<char, 0>{}).empty());
    EXPECT_TRUE(snicholls::as_bytes_view(std::vector<std::byte>{}).empty());
    EXPECT_EQ(*snicholls::c_str_of(""), '\0');
}

TEST(StringAccessNullPointer, IsAnEmptyStringNeverUb)
{
    const char* null = nullptr;
    char16_t* null16 = nullptr;
    EXPECT_TRUE(snicholls::to_string_view(null).empty());
    EXPECT_EQ(snicholls::to_string_view(null).data(), nullptr);
    EXPECT_EQ(snicholls::length_of(null), 0u);
    EXPECT_EQ(snicholls::length_of(null16), 0u);
    ASSERT_NE(snicholls::c_str_of(null), nullptr);
    EXPECT_EQ(*snicholls::c_str_of(null), '\0');
    EXPECT_EQ(*snicholls::c_str_of(null16), u'\0');
    EXPECT_TRUE(snicholls::as_bytes_view(null).empty());
    static_assert(snicholls::length_of(static_cast<const char*>(nullptr)) == 0);
    // nullptr itself is not a string: snicholls::to_string_view(nullptr) does not compile.
    static_assert(!viewable<std::nullptr_t>);
}

TEST(StringAccessUnicode, Utf8LengthsAreCodeUnits)
{
    // U+00E9 is 2 UTF-8 code units, U+20AC is 3.
    const std::u8string u8 = u8"é€";
    EXPECT_EQ(snicholls::length_of(u8), 5u);
    EXPECT_EQ(snicholls::length_of(u8"é€"), 5u);
    EXPECT_EQ(snicholls::to_string_view(u8), std::u8string_view(u8"é€"));
    const std::string narrow = "\xC3\xA9\xE2\x82\xAC";
    const auto b = snicholls::as_bytes_view(narrow);
    ASSERT_EQ(b.size(), 5u);
    EXPECT_EQ(b[0], std::byte{0xC3});
    EXPECT_EQ(b[4], std::byte{0xAC});
    const auto b8 = snicholls::as_bytes_view(u8);
    EXPECT_TRUE(std::equal(b.begin(), b.end(), b8.begin(), b8.end()));
}

TEST(StringAccessUnicode, Char16AndWideText)
{
    const char16_t* ptr = u"é€";
    std::u16string u16 = u"é€";
    EXPECT_EQ(snicholls::length_of(ptr), 2u);
    EXPECT_EQ(snicholls::length_of(u16), 2u);
    EXPECT_EQ(snicholls::to_string_view(ptr), u"é€");
    EXPECT_EQ(snicholls::to_string_view(u"é€"), std::u16string_view(u16));
    EXPECT_EQ(snicholls::to_string_view(L"wide"), L"wide");
    EXPECT_EQ(snicholls::length_of(U"\U0001F600"), 1u);
    // Bytes of wide text are endian- and platform-dependent: no as_bytes_view.
    static_assert(!snicholls::byte_viewable<std::u16string>);
    static_assert(!snicholls::byte_viewable<const wchar_t*>);
    static_assert(!snicholls::byte_viewable<std::u32string_view>);
    static_assert(snicholls::byte_viewable<std::u8string> && snicholls::byte_viewable<const char*>);
}

TEST(StringAccessFixedString, ViewsLengthsAndCString)
{
    static constexpr snicholls::fixed_string name("price");
    static_assert(snicholls::to_string_view(name) == "price");
    static_assert(snicholls::length_of(name) == 5);
    static_assert(snicholls::c_str_of(name)[5] == '\0');
    EXPECT_EQ(snicholls::c_str_of(name), name.c_str());
    EXPECT_EQ(snicholls::to_string_view(name).data(), name.data());
    EXPECT_EQ(snicholls::as_bytes_view(name).size(), 5u);
    static constexpr snicholls::basic_fixed_string<char16_t, 2> wide(u"px");
    static_assert(snicholls::to_string_view(wide) == u"px");
}

TEST(StringAccessStaticExtent, SpansAndArrays)
{
    static constexpr std::array<char, 4> arr{'a', 'b', 'c', 'd'};
    constexpr std::span<const char, 4> all(arr);
    constexpr std::span<const char, 2> first = all.first<2>();
    static_assert(snicholls::length_of(all) == 4 && snicholls::to_string_view(first) == "ab");
    const std::array<std::byte, 3> raw{std::byte{1}, std::byte{2}, std::byte{3}};
    const std::span<const std::byte, 3> raw_span(raw);
    EXPECT_EQ(snicholls::as_bytes_view(raw_span).size(), 3u);
    EXPECT_EQ(snicholls::as_bytes_view(raw_span).data(), raw.data());
}

TEST(StringAccessBytes, EveryByteCarrier)
{
    const std::vector<std::uint8_t> u8{1, 2, 3};
    const std::vector<unsigned char> uc{4, 5};
    const std::vector<std::byte> by{std::byte{6}};
    const std::array<std::byte, 2> arr{std::byte{7}, std::byte{8}};
    const unsigned char raw[3] = {9, 0, 10};   // a byte array has no terminator: length 3
    const std::span<const std::uint8_t> sp(u8);

    EXPECT_EQ(snicholls::as_bytes_view(u8).size(), 3u);
    EXPECT_EQ(snicholls::as_bytes_view(u8)[2], std::byte{3});
    EXPECT_EQ(snicholls::as_bytes_view(uc).size(), 2u);
    EXPECT_EQ(snicholls::as_bytes_view(by)[0], std::byte{6});
    EXPECT_EQ(snicholls::as_bytes_view(arr)[1], std::byte{8});
    EXPECT_EQ(snicholls::as_bytes_view(raw).size(), 3u);
    EXPECT_EQ(snicholls::as_bytes_view(raw)[2], std::byte{10});
    EXPECT_EQ(snicholls::as_bytes_view(sp).data(), reinterpret_cast<const std::byte*>(u8.data()));
    EXPECT_EQ(snicholls::length_of(raw), 3u);
    // Bytes are not text.
    static_assert(!viewable<std::vector<std::uint8_t>>);
    static_assert(!c_string_viewable<std::vector<std::byte>>);
}

TEST(StringAccessCString, EveryNulTerminatedCarrier)
{
    const std::string str = "str";
    char buffer[] = "buf";
    const char* ptr = "ptr";
    EXPECT_EQ(snicholls::c_str_of(str), str.c_str());
    EXPECT_EQ(snicholls::c_str_of(buffer), buffer);
    EXPECT_EQ(snicholls::c_str_of(ptr), ptr);
    EXPECT_STREQ(snicholls::c_str_of("lit"), "lit");
    static_assert(!c_string_viewable<std::string_view>);   // a view may not be terminated
    static_assert(!c_string_viewable<std::vector<char>>);
    static_assert(c_string_viewable<std::string> && viewable<std::string_view>);
}

struct ci_traits : std::char_traits<char> {};

TEST(StringAccessCharTraits, ViewKeepsTheStringsTraits)
{
    const std::basic_string<char, ci_traits> s = "ci";
    static_assert(std::is_same_v<decltype(snicholls::to_string_view(s)), std::basic_string_view<char, ci_traits>>);
    EXPECT_EQ(snicholls::to_string_view(s).size(), 2u);
}

// Everything is usable in constant expressions (as_bytes_view only over std::byte units).
constexpr std::size_t constexpr_vector_length()
{
    std::vector<char> v{'a', 'b', 'c'};
    return snicholls::length_of(v) + snicholls::to_string_view(v).size();
}

constexpr bool constexpr_string()
{
    std::string s = "abc";
    return snicholls::to_string_view(s) == "abc" && *snicholls::c_str_of(s) == 'a';
}

TEST(StringAccessConstexpr, EvaluatesAtCompileTime)
{
    static_assert(snicholls::to_string_view("abc") == "abc");
    static_assert(snicholls::length_of("abc") == 3);
    static_assert(snicholls::to_string_view(static_cast<const char*>("abc")) == "abc");
    static_assert(snicholls::length_of(static_cast<const char*>("abc")) == 3);
    static_assert(snicholls::to_string_view(std::string_view("sv")) == "sv");
    static_assert(*snicholls::c_str_of(static_cast<const char*>(nullptr)) == '\0');
    static_assert(constexpr_vector_length() == 6);
    static_assert(constexpr_string());
    static constexpr std::array<std::byte, 2> raw{std::byte{1}, std::byte{2}};
    static_assert(snicholls::as_bytes_view(raw).size() == 2);
}

// A character array that is not NUL-terminated violates the string-literal contract.
static constexpr char unterminated[3] = {'a', 'b', 'c'};
static constexpr char terminated[3] = {'a', 'b', '\0'};

template <const auto& A>
concept constant_length = requires { typename std::integral_constant<std::size_t, snicholls::length_of(A)>; };

TEST(StringAccessContract, UnterminatedCharacterArrayIsACompileErrorInConstantEvaluation)
{
    static_assert(constant_length<terminated>);
    static_assert(!constant_length<unterminated>);
}

#ifndef NDEBUG
TEST(StringAccessContractDeathTest, UnterminatedCharacterArrayAssertsAtRunTime)
{
    char padded[4] = {'A', 'B', 'C', 'D'};   // e.g. a space-padded protocol field: use std::array/span instead
    EXPECT_DEATH((void)snicholls::to_string_view(padded), "last element must be NUL");
    EXPECT_EQ(snicholls::to_string_view(std::span<const char, 4>(padded)), "ABCD");
}
#endif

} // namespace
