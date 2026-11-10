//
//  static_map_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/containers/static_map.hpp>: built and sorted by the compiler,
//  looked up by binary search, in key order.
//

#include <snicholls/core/containers/static_map.hpp>

#include <gtest/gtest.h>

#include <array>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace {

using namespace std::string_view_literals;
using snicholls::containers::static_map;
using snicholls::containers::static_set;

constexpr static_map<std::string_view, int, 4> minor_units{
    std::array<std::pair<std::string_view, int>, 4>{{{"usd", 2}, {"jpy", 0}, {"bhd", 3}, {"eur", 2}}}};
static_assert(minor_units.at("jpy"sv) == 0 && minor_units.at("bhd"sv) == 3);
static_assert(minor_units.find("gbp"sv) == nullptr && minor_units.contains("eur"sv));
static_assert(minor_units.begin()->first == "bhd");   // in key order
constexpr static_set<int, 5> holidays{std::array{25, 1, 26, 7, 4}};
static_assert(holidays.contains(26) && !holidays.contains(2) && *holidays.begin() == 1);
constexpr auto ticks = snicholls::containers::make_static_map<int, double>({{3, 0.25}, {1, 0.01}});
static_assert(ticks.at(1) == 0.01);

TEST(StaticMap, AtThrowsForAMissingKey)
{
    EXPECT_THROW(static_cast<void>(minor_units.at("xxx"sv)), std::out_of_range);
}

} // namespace
