//
//  strong_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/strong.hpp>: arithmetic within a kind, no mixing between kinds, the
//  layout contract, hashing and printing.
//

#include <snicholls/core/strong.hpp>

#include <gtest/gtest.h>

#include <concepts>
#include <sstream>
#include <type_traits>
#include <unordered_set>

namespace {

using price = snicholls::strong<double, struct price_tag>;
using quantity = snicholls::strong<double, struct quantity_tag>;

static_assert(price(2.0) + price(3.0) == price(5.0) && price(2.0) * 3.0 == price(6.0) && price(6.0) / price(2.0) == 3.0);
static_assert(-price(1.0) < price(0.0) && price(4.0) / 2.0 == price(2.0));
static_assert(!std::is_convertible_v<double, price>);                     // explicit
static_assert(!std::is_convertible_v<quantity, price>);                   // never mixed
template <typename A, typename B>
concept addable = requires(A a, B b) { a + b; };
static_assert(addable<price, price> && !addable<price, quantity> && !addable<price, double>);
static_assert(sizeof(price) == sizeof(double) && std::is_trivially_copyable_v<price>);

TEST(Strong, HashesAndPrints)
{
    std::unordered_set<price> seen{price(1.5), price(1.5), price(2.0)};
    EXPECT_EQ(seen.size(), 2u);
    std::ostringstream os;
    price(1.25).print(os);
    EXPECT_EQ(os.str(), "1.25");
    price p(1.0);
    p += price(0.5);
    p -= price(0.25);
    EXPECT_EQ(p, price(1.25));
}

} // namespace
