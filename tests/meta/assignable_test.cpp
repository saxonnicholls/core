//
//  assignable_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/meta/assignable.hpp>: narrowing refused, numbers never into strings, everything else as std::is_assignable.
//

#include <snicholls/core/meta/assignable.hpp>

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <string_view>

namespace {

using snicholls::non_narrowing_assignable;

static_assert(non_narrowing_assignable<int, int> && non_narrowing_assignable<double, float> && non_narrowing_assignable<long long, int>);
static_assert(!non_narrowing_assignable<int, double>);    // truncation
static_assert(!non_narrowing_assignable<int, long long>); // may not fit
static_assert(!non_narrowing_assignable<bool, int>);
static_assert(!non_narrowing_assignable<double, int>);    // brace rules: an int may not be exact as a double... in general
static_assert(!non_narrowing_assignable<std::string, int>);   // would be the character '\x03'
static_assert(!non_narrowing_assignable<std::string, char>);  // documented: a character is arithmetic
static_assert(non_narrowing_assignable<std::string, const char*> && non_narrowing_assignable<std::string, std::string_view>);
static_assert(non_narrowing_assignable<std::optional<int>, int>);
static_assert(!non_narrowing_assignable<const int, int>);    // not assignable at all
static_assert(!non_narrowing_assignable<std::wstring, int>);  // every basic_string

TEST(NonNarrowingAssignable, IsACompileTimeConcept)
{
    SUCCEED();
}

} // namespace
