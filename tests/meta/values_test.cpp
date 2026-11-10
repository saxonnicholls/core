//
//  values_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/meta/values.hpp>: distinct and monotone packs, mixed types, the edges.
//

#include <snicholls/core/meta/values.hpp>

#include <gtest/gtest.h>

namespace {

enum class tenor { m1 = 1, m3 = 3, y1 = 12 };

static_assert(snicholls::all_distinct_v<1, 2, 3> && !snicholls::all_distinct_v<1, 2, 1>);
static_assert(snicholls::all_distinct_v<> && snicholls::all_distinct_v<7>);
static_assert(snicholls::all_distinct_v<'a', 'b'> && snicholls::all_distinct_v<tenor::m1, tenor::y1>);
static_assert(snicholls::strictly_increasing_v<1, 2, 3> && !snicholls::strictly_increasing_v<1, 1, 3>);
static_assert(snicholls::increasing_v<1, 1, 3> && !snicholls::increasing_v<2, 1>);
static_assert(snicholls::strictly_decreasing_v<3, 2, 1> && !snicholls::strictly_decreasing_v<3, 3>);
static_assert(snicholls::decreasing_v<3, 3, 1> && !snicholls::decreasing_v<1, 2>);
static_assert(snicholls::strictly_increasing_v<> && snicholls::strictly_increasing_v<5>);
static_assert(snicholls::strictly_increasing_v<1, 2L, 3ULL>);   // a common type
static_assert(snicholls::strictly_increasing_v<tenor::m1, tenor::m3, tenor::y1>);

TEST(MetaValues, AreCompileTimeFacts)
{
    SUCCEED();
}

} // namespace
