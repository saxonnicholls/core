//
//  attributes_test.cpp
//  snicholls-core
//
//  Tests for <snicholls/core/meta/attributes.hpp>: sets keyed by type, lookup by ADL, the empty set for a
//  type without attributes, per-enumerator tables, fallbacks, layout; all at compile time.
//

#include <snicholls/core/meta/attributes.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string_view>
#include <tuple>

namespace attributes_test {

struct width {
    int characters;
};
struct name {
    std::string_view text;
};

enum class side : std::uint8_t { buy, sell };
constexpr auto attributes(side s)
{
    return snicholls::per_enumerator(s, [] { return std::array{snicholls::attrs(width{3}, name{"Buy"}), snicholls::attrs(width{4}, name{"Sell"})}; });
}

struct order {
    int quantity;
};
constexpr auto attributes(const order&) { return snicholls::attrs(width{10}); }

struct plain {
    int x;
};

} // namespace attributes_test

namespace {

using namespace attributes_test;
using snicholls::attribute_set;

// A set is its attributes and nothing more.
static_assert(sizeof(attribute_set<width, name>) == sizeof(std::tuple<width, name>));
static_assert(std::is_aggregate_v<attribute_set<width, name>>);
static_assert(std::is_trivially_copyable_v<attribute_set<width, name>>);
static_assert(std::is_empty_v<attribute_set<>>);

// Keyed by type.
constexpr auto set = snicholls::attrs(width{7}, name{"x"});
static_assert(set.has<width> && set.has<name> && !set.has<int>);
static_assert(set.get<width>().characters == 7 && set.get<name>().text == "x");

// Found by argument-dependent lookup; a type without attributes has the empty set.
static_assert(snicholls::has_attributes<order> && snicholls::has_attributes<side> && !snicholls::has_attributes<plain>);
static_assert(!snicholls::has_attributes<int> && !snicholls::has_attributes<double>);
static_assert(snicholls::attributes_of(order{1}).get<width>().characters == 10);
static_assert(std::is_same_v<decltype(snicholls::attributes_of(plain{})), attribute_set<>>);

// Per enumerator, and asked by type without a value.
static_assert(snicholls::attributes_of(side::sell).get<name>().text == "Sell");
static_assert(snicholls::attributes_of(side::buy).get<width>().characters == 3);
static_assert(snicholls::has_attribute_v<name, side> && !snicholls::has_attribute_v<name, order>);
static_assert(snicholls::has_attribute_v<width, const order&>);

// A fallback where there is none.
static_assert(snicholls::attribute_or(order{2}, width{1}).characters == 10);
static_assert(snicholls::attribute_or(plain{}, width{1}).characters == 1);
static_assert(snicholls::attribute_or(side::buy, name{"?"}).text == "Buy");

// Not an enumerator: the first set.
static_assert(snicholls::attributes_of(static_cast<side>(9)).get<name>().text == "Buy");

TEST(Attributes, ReadAtRunTimeForAnEnumeratorKnownOnlyAtRunTime)
{
    volatile std::uint8_t raw = 1;
    const auto s = static_cast<side>(raw);
    EXPECT_EQ(snicholls::attributes_of(s).get<name>().text, "Sell");
    EXPECT_EQ(snicholls::attributes_of(s).get<width>().characters, 4);
}

TEST(Attributes, LookupIsNoexceptForLiteralAttributes)
{
    static_assert(noexcept(snicholls::attributes_of(plain{})));
    EXPECT_EQ(snicholls::attribute_or(plain{}, width{5}).characters, 5);
}

} // namespace
