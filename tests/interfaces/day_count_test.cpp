//
//  day_count_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/day_count.hpp>: the adapter forwards, the interface models the concept,
//  and a convention chosen at run time answers as its type does.
//

#include <snicholls/core/finance/day_count.hpp>
#include <snicholls/core/interfaces/day_count.hpp>

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <memory>
#include <string_view>
#include <vector>

namespace {

using namespace std::chrono;
using namespace snicholls;

constexpr sys_days start = sys_days{2024y / January / 31d}, end = sys_days{2025y / March / 31d};

// usable in constant expressions
static_assert(day_count_model<finance::thirty_360>{}.day_count(start, end) == finance::thirty_360::day_count(start, end));
static_assert(day_count_model<finance::act_act_isda>{}.name() == "ACT/ACT ISDA");
static_assert(noexcept(std::declval<const day_count_interface&>().year_fraction(start, end)));

template <day_count_convention C>
finance::year_fraction through_concept(const C& c)
{
    return c.year_fraction(start, end);
}

TEST(DayCountInterface, AModelAnswersAsItsType)
{
    std::vector<std::unique_ptr<day_count_interface>> all;
    all.push_back(std::make_unique<day_count_model<finance::act_360>>());
    all.push_back(std::make_unique<day_count_model<finance::act_365_fixed>>());
    all.push_back(std::make_unique<day_count_model<finance::thirty_360>>());
    all.push_back(std::make_unique<day_count_model<finance::thirty_e_360>>(finance::thirty_e_360{}));
    all.push_back(std::make_unique<day_count_model<finance::act_act_isda>>());
    const std::array<finance::year_fraction, 5> expected{
        finance::act_360::year_fraction(start, end), finance::act_365_fixed::year_fraction(start, end),
        finance::thirty_360::year_fraction(start, end), finance::thirty_e_360::year_fraction(start, end),
        finance::act_act_isda::year_fraction(start, end)};
    const std::array<std::string_view, 5> names{"ACT/360", "ACT/365F", "30/360", "30E/360", "ACT/ACT ISDA"};
    for (std::size_t i = 0; i < all.size(); ++i) {
        EXPECT_EQ(all[i]->name(), names[i]);
        EXPECT_EQ(all[i]->year_fraction(start, end), expected[i]);
        EXPECT_EQ(through_concept(*all[i]), expected[i]);   // the interface models the concept
    }
    EXPECT_EQ(all[2]->day_count(start, end), 420);   // 30/360: 360 + 60 + (30 - 30)
}

} // namespace
