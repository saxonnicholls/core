//
//  http_header_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/http_header.hpp>: ASCII case-insensitive equality and the first-match lookup.
//

#include <snicholls/core/http_header.hpp>

#include <gtest/gtest.h>

#include <optional>
#include <string_view>
#include <vector>

using namespace snicholls;

namespace {

static_assert(ascii_iequals("Content-Type", "content-type"));
static_assert(!ascii_iequals("Content-Type", "Content-Typ"));
static_assert(!ascii_iequals("a", "b"));
static_assert(ascii_iequals("", ""));
static_assert(!ascii_iequals("\xC4", "\xE4"), "ASCII only: no locale folds other bytes");
static_assert(noexcept(ascii_iequals(std::string_view{}, std::string_view{})));

TEST(HttpHeader, FindHeaderIgnoresCaseAndTakesTheFirst)
{
    const std::vector<http_header> h{{"Stripe-Signature", "t=1,v1=ab"}, {"X-Two", "first"}, {"x-two", "second"}};
    EXPECT_EQ(find_header(h, "stripe-signature"), "t=1,v1=ab");
    EXPECT_EQ(find_header(h, "X-TWO"), "first");
    EXPECT_EQ(find_header(h, "missing"), std::nullopt);
    EXPECT_EQ(find_header({}, "x"), std::nullopt);
    EXPECT_EQ(h[0], (http_header{"Stripe-Signature", "t=1,v1=ab"}));
    EXPECT_NE(h[1], h[2]) << "equality is case-sensitive";
}

} // namespace
