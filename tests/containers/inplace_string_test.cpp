//
//  inplace_string_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/containers/inplace_string.hpp>: the layout contract, refusal rather than truncation,
//  comparison across capacities, hashing, and use as a template argument.
//

#include <snicholls/core/containers/inplace_string.hpp>
#include <snicholls/core/hash/fnv.hpp>

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>

namespace {

using snicholls::inplace_string;
using snicholls::inplace_string_errc;

// layout: the characters, then the smallest length that holds N
static_assert(sizeof(inplace_string<255>) == 256 && alignof(inplace_string<255>) == 1);
static_assert(sizeof(inplace_string<128>) == 129 && sizeof(inplace_string<64>) == 65);
static_assert(sizeof(inplace_string<256>) == 258 && alignof(inplace_string<256>) == 2);
static_assert(std::is_same_v<inplace_string<255>::size_type, std::uint8_t>);
static_assert(std::is_same_v<inplace_string<70000>::size_type, std::uint32_t>);
static_assert(std::is_trivially_copyable_v<inplace_string<64>> && std::is_standard_layout_v<inplace_string<64>>);
static_assert(std::is_nothrow_default_constructible_v<inplace_string<64>>);

// constexpr throughout
constexpr inplace_string<8> perp{"ETH-PERP"};
static_assert(perp.size() == 8 && perp.view() == "ETH-PERP" && perp.capacity() == 8 && !perp.empty());
static_assert(inplace_string<8>{}.empty() && inplace_string<8>{}.view().empty());
static_assert(inplace_string<4>::make("ABCDE").unwrap_err_unchecked() == inplace_string_errc::too_long);
static_assert(inplace_string<4>::make("ABCD")->view() == "ABCD");
static_assert(inplace_string<4>::make("")->empty());
static_assert(perp == std::string_view{"ETH-PERP"} && perp != std::string_view{"ETH-PERP2"});
static_assert(perp == inplace_string<64>{"ETH-PERP"});                // across capacities
static_assert(inplace_string<64>{"ETH-PERP"} == perp);
static_assert(inplace_string<4>{"AB"} < inplace_string<8>{"ABC"});
static_assert((inplace_string<4>{"B"} <=> std::string_view{"A"}) == std::strong_ordering::greater);
static_assert(perp.hash() == snicholls::fnv1a_64("ETH-PERP"));
static_assert(inplace_string<4>{"AB"}.chars_[2] == '\0' && inplace_string<4>::make("AB")->chars_[3] == '\0');

// structural: a template argument
template <inplace_string<16> Id>
constexpr std::string_view id_of() { return Id.view(); }
static_assert(id_of<"order-7">() == "order-7");
static_assert(&id_of<"a"> == &id_of<inplace_string<16>{"a"}>);   // one specialisation: equal values are one argument

TEST(ContainersInplaceString, RefusesTextTooLongInsteadOfTruncating)
{
    const std::string id(65, 'x');
    const auto r = inplace_string<64>::make(id);
    ASSERT_FALSE(r.is_ok());
    EXPECT_EQ(r.unwrap_err_unchecked(), inplace_string_errc::too_long);
    const auto fits = inplace_string<64>::make(std::string_view(id).substr(0, 64));
    ASSERT_TRUE(fits.is_ok());
    EXPECT_EQ(fits->size(), 64U);
    EXPECT_EQ(fits->view(), std::string(64, 'x'));
    EXPECT_EQ(fits->data(), fits->chars_);
}

TEST(ContainersInplaceString, CapacitiesUsedByContractIdsHoldTheirMaximum)
{
    const std::string a(255, 'a'), b(128, 'b');
    EXPECT_EQ(inplace_string<255>::make(a)->view(), a);
    EXPECT_FALSE(inplace_string<255>::make(a + "a").is_ok());
    EXPECT_EQ(inplace_string<128>::make(b)->view(), b);
    EXPECT_FALSE(inplace_string<128>::make(b + "b").is_ok());
}

TEST(ContainersInplaceString, EqualStringsAreEqualByteForByte)
{
    // the tail is zero, so a record holding one can be compared and hashed as bytes
    auto x = *inplace_string<32>::make("first-very-long-identifier");
    x = *inplace_string<32>::make("abc");
    const auto y = *inplace_string<32>::make("abc");
    EXPECT_EQ(x, y);
    EXPECT_EQ(std::memcmp(&x, &y, sizeof x), 0);
}

TEST(ContainersInplaceString, KeysAnUnorderedSet)
{
    std::unordered_set<inplace_string<64>> ids;
    ids.insert(*inplace_string<64>::make("BTC-PERP"));
    ids.insert(*inplace_string<64>::make("ETH-PERP"));
    ids.insert(*inplace_string<64>::make("BTC-PERP"));
    EXPECT_EQ(ids.size(), 2U);
    EXPECT_TRUE(ids.contains(inplace_string<64>{"ETH-PERP"}));
    EXPECT_EQ(std::hash<inplace_string<64>>{}(inplace_string<64>{"ETH-PERP"}),
              static_cast<std::size_t>(snicholls::fnv1a_64("ETH-PERP")));
}

TEST(ContainersInplaceString, OrdersAsItsText)
{
    const auto a = *inplace_string<16>::make("apple"), b = *inplace_string<16>::make("apples");
    EXPECT_LT(a, b);
    EXPECT_GT(b, std::string_view{"apple"});
    EXPECT_NE(a, b);
}

} // namespace
