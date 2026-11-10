//
//  fixed_string_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/fixed_string.hpp>. Almost everything here is a
//  compile-time claim, so almost everything is a static_assert; the run-time
//  tests cover the pieces that only exist at run time (hashing into a container,
//  comparison with std::string).
//

#include <snicholls/core/fixed_string.hpp>

#include <gtest/gtest.h>

#include <compare>
#include <functional>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace {

using snicholls::basic_fixed_string;
using snicholls::fixed_string;

// Length excludes the terminator, as for std::string
static_assert(fixed_string("abc").size() == 3);
static_assert(fixed_string("abc").length() == 3);
static_assert(!fixed_string("abc").empty());
static_assert(fixed_string("").empty());
static_assert(std::is_same_v<decltype(basic_fixed_string("abc")), basic_fixed_string<char, 3>>);
static_assert(std::is_same_v<decltype(fixed_string("abc")), fixed_string<3>>);

// Element access
static_assert(fixed_string("abc")[0] == 'a');
static_assert(fixed_string("abc").front() == 'a');
static_assert(fixed_string("abc").back() == 'c');
static_assert(fixed_string("abc").c_str()[3] == '\0');
constexpr fixed_string abc = "abc";
static_assert(abc.end() - abc.begin() == 3);

// Comparison with fixed strings of any length, views and literals
static_assert(fixed_string("abc") == "abc");
static_assert(fixed_string("abc") != "abd");
static_assert(fixed_string("abc") == std::string_view{"abc"});
static_assert(fixed_string("abc") < "abd");
static_assert(fixed_string("b") > "abc");
// Through one side's view(), as before the fixed-to-fixed operators existed
static_assert(fixed_string("abc").view() != fixed_string("ab"));
static_assert(fixed_string("abc").view() < fixed_string("abd"));
// Literals and views on the left: the reversed (rewritten) candidates
static_assert("abc" == fixed_string("abc"));
static_assert(std::string_view{"abd"} > fixed_string("abc"));

// Fixed string against fixed string, directly: unambiguous on every compiler
// Different lengths
static_assert(fixed_string("abc") != fixed_string("ab"));
static_assert(!(fixed_string("abc") == fixed_string("ab")));
static_assert(fixed_string("ab") < fixed_string("abc"));
static_assert(fixed_string("b") > fixed_string("abc"));
static_assert((fixed_string("ab") <=> fixed_string("abc")) == std::strong_ordering::less);
// Equal lengths
static_assert(fixed_string("abc") == fixed_string("abc"));
static_assert(fixed_string("abc") != fixed_string("abd"));
static_assert(fixed_string("abc") < fixed_string("abd"));
static_assert(fixed_string("abd") >= fixed_string("abc"));
static_assert((fixed_string("abc") <=> fixed_string("abc")) == std::strong_ordering::equal);
// The base type, the wrappers mixed with it, and the other character types
static_assert(basic_fixed_string("abc") == fixed_string("abc"));
static_assert(fixed_string("ab") == basic_fixed_string("ab"));
static_assert(basic_fixed_string("ab") != basic_fixed_string("abc"));
static_assert(snicholls::u16fixed_string(u"ab") < snicholls::u16fixed_string(u"b"));
static_assert(basic_fixed_string(L"x") == snicholls::wfixed_string(L"x"));
// The result types: bool and the view's ordering, both noexcept
static_assert(std::is_same_v<decltype(fixed_string("a") == fixed_string("ab")), bool>);
static_assert(std::is_same_v<decltype(fixed_string("a") <=> fixed_string("ab")), std::strong_ordering>);
static_assert(noexcept(std::declval<const fixed_string<1>&>() == std::declval<const fixed_string<2>&>()));
static_assert(noexcept(std::declval<const fixed_string<1>&>() < std::declval<const fixed_string<2>&>()));
// std::equal_to and std::less, as the standard containers use them
static_assert(std::equal_to<basic_fixed_string<char, 3>>{}(basic_fixed_string("abc"), basic_fixed_string("abc")));
static_assert(std::equal_to<>{}(fixed_string("abc"), fixed_string("abcd")) == false);
static_assert(std::less<>{}(fixed_string("abc"), fixed_string("abd")));

// Concatenation and substring
static_assert((fixed_string("ab") + fixed_string("cd")) == "abcd");
static_assert((fixed_string("ab") + fixed_string("cd")).size() == 4);
static_assert((fixed_string("") + fixed_string("x")) == "x");
static_assert(fixed_string("hello").substr<1, 3>() == "ell");
static_assert(fixed_string("hello").substr<0, 0>().empty());
static_assert(fixed_string("hello").substr<5, 0>().empty());

// The other character types
static_assert(snicholls::wfixed_string(L"ab").size() == 2);
static_assert(snicholls::u8fixed_string(u8"ab").size() == 2);
static_assert(snicholls::u16fixed_string(u"ab") == std::u16string_view{u"ab"});
static_assert(snicholls::u32fixed_string(U"abc").back() == U'c');

// The reason it exists: a string as a non-type template parameter
template <basic_fixed_string Name>
struct tag {
    static constexpr std::string_view name = Name;
};
static_assert(tag<"object1">::name == "object1");
static_assert(!std::is_same_v<tag<"a">, tag<"b">>);
static_assert(std::is_same_v<tag<"a">, tag<"a">>);

// Structural, as a non-type template parameter type must be
static_assert(std::is_trivially_copyable_v<basic_fixed_string<char, 4>>);
static_assert(noexcept(abc.view()));

TEST(FixedStringCompare, AgainstStdString)
{
    const std::string s = "hello";
    EXPECT_TRUE(fixed_string("hello") == s);
    EXPECT_FALSE(fixed_string("hellO") == s);
}

TEST(FixedStringView, ConvertsImplicitlyToAStringView)
{
    constexpr auto f = fixed_string("view me");
    const std::string_view v = f;
    EXPECT_EQ(v, "view me");
    EXPECT_EQ(v.size(), 7u);
}

TEST(FixedStringHash, HashesAsItsViewAndKeysAnUnorderedSet)
{
    constexpr basic_fixed_string<char, 3> a("abc");
    EXPECT_EQ((std::hash<basic_fixed_string<char, 3>>{}(a)), std::hash<std::string_view>{}("abc"));

    // The default hash and the default std::equal_to
    std::unordered_set<basic_fixed_string<char, 3>> set;
    set.insert(a);
    set.insert(basic_fixed_string("abc"));
    set.insert(basic_fixed_string("xyz"));
    EXPECT_EQ(set.size(), 2u);
    EXPECT_TRUE(set.contains(basic_fixed_string("xyz")));
    EXPECT_FALSE(set.contains(basic_fixed_string("abd")));
}

TEST(FixedStringCompare, AgainstStdStringAndRunTimeViews)
{
    const std::string s = "abc";
    const std::string_view v = s;
    EXPECT_TRUE(fixed_string("abc") == s);
    EXPECT_TRUE(s == fixed_string("abc"));
    EXPECT_TRUE(fixed_string("abc") == v);
    EXPECT_TRUE(fixed_string("ab") < s);
    EXPECT_TRUE(fixed_string("abd") > v);
    const char* p = "abc";
    EXPECT_TRUE(fixed_string("abc") == p);
}

TEST(FixedStringCompare, OrdersASortedSetOfMixedLengths)
{
    std::set<basic_fixed_string<char, 3>, std::less<>> three{basic_fixed_string("xyz"), basic_fixed_string("abc")};
    EXPECT_EQ(three.begin()->view(), "abc");
    EXPECT_TRUE(three.contains(basic_fixed_string("xyz")));
    EXPECT_FALSE(three.contains(basic_fixed_string("ab")));   // heterogeneous lookup, another length
}

TEST(FixedStringIterate, RangeForVisitsEveryCharacterButNotTheTerminator)
{
    std::string out;
    for (char c : fixed_string("loop"))
        out += c;
    EXPECT_EQ(out, "loop");
}

TEST(FixedStringConstruct, DefaultIsAllNulls)
{
    constexpr basic_fixed_string<char, 3> empty3{};
    EXPECT_EQ(empty3.size(), 3u);
    EXPECT_EQ(empty3[0], '\0');
    EXPECT_EQ(empty3[3], '\0');
}

} // namespace
