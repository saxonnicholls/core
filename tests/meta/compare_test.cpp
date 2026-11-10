//
//  compare_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/meta/compare.hpp>: the native and synthesised orderings, their result types, and noexcept.
//

#include <snicholls/core/meta/compare.hpp>

#include <gtest/gtest.h>

#include <compare>
#include <string>
#include <type_traits>

namespace {

struct only_less {
    int v;
    friend constexpr bool operator<(const only_less& a, const only_less& b) noexcept { return a.v < b.v; }
};
struct incomparable {};
struct throwing_less {
    int v;
    friend bool operator<(const throwing_less& a, const throwing_less& b) { return a.v < b.v; }
};

static_assert(snicholls::boolean_testable<bool> && snicholls::boolean_testable<int> && !snicholls::boolean_testable<incomparable>);
static_assert(snicholls::synth_three_way_comparable<int> && snicholls::synth_three_way_comparable<only_less>);
static_assert(!snicholls::synth_three_way_comparable<incomparable>);

// Native <=> keeps its own ordering; only < gives a weak ordering
static_assert(std::is_same_v<snicholls::synth_three_way_result_t<int>, std::strong_ordering>);
static_assert(std::is_same_v<snicholls::synth_three_way_result_t<double>, std::partial_ordering>);
static_assert(std::is_same_v<snicholls::synth_three_way_result_t<only_less>, std::weak_ordering>);
static_assert(snicholls::synth_three_way(1, 2) < 0 && snicholls::synth_three_way(2, 2) == 0);
static_assert(snicholls::synth_three_way(only_less{3}, only_less{1}) > 0);
static_assert(snicholls::synth_three_way(only_less{2}, only_less{2}) == 0);

// noexcept follows the comparison it makes
static_assert(noexcept(snicholls::synth_three_way(only_less{}, only_less{})));
static_assert(!noexcept(snicholls::synth_three_way(throwing_less{}, throwing_less{})));

TEST(SynthThreeWay, OrdersStringsByTheirOwnOperator)
{
    EXPECT_TRUE(snicholls::synth_three_way(std::string("a"), std::string("b")) < 0);
    EXPECT_EQ(snicholls::synth_three_way(throwing_less{1}, throwing_less{1}), std::weak_ordering::equivalent);
}

} // namespace
