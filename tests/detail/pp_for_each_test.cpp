//
//  pp_for_each_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/detail/pp_for_each.hpp>.
//

#include <string_view>

#include <gtest/gtest.h>

#include <snicholls/core/detail/pp_for_each.hpp>

namespace {

#define SNICHOLLS_TEST_SUM(data, x) +(data * (x))
#define SNICHOLLS_TEST_NAME(data, x) #x,

static_assert(0 SNICHOLLS_PP_FOR_EACH(SNICHOLLS_TEST_SUM, 1) == 0);   // no elements
static_assert(0 SNICHOLLS_PP_FOR_EACH(SNICHOLLS_TEST_SUM, 1, 7) == 7);
static_assert(0 SNICHOLLS_PP_FOR_EACH(SNICHOLLS_TEST_SUM, 2, 1, 2, 3) == 12);
constexpr std::string_view paren[] = {SNICHOLLS_PP_FOR_EACH(SNICHOLLS_TEST_NAME, 0, (1, 2), x)};
static_assert(std::size(paren) == 2 && paren[0] == "(1, 2)");   // parenthesised elements stay one element

// 300 elements: beyond the 256 a uint8 enum can hold.
#define SNICHOLLS_TEST_10(p) p##0, p##1, p##2, p##3, p##4, p##5, p##6, p##7, p##8, p##9
#define SNICHOLLS_TEST_100(p) SNICHOLLS_TEST_10(p##0), SNICHOLLS_TEST_10(p##1), SNICHOLLS_TEST_10(p##2), SNICHOLLS_TEST_10(p##3), SNICHOLLS_TEST_10(p##4), \
    SNICHOLLS_TEST_10(p##5), SNICHOLLS_TEST_10(p##6), SNICHOLLS_TEST_10(p##7), SNICHOLLS_TEST_10(p##8), SNICHOLLS_TEST_10(p##9)
#define SNICHOLLS_TEST_FOR_EACH(...) SNICHOLLS_PP_FOR_EACH(SNICHOLLS_TEST_NAME, 0, __VA_ARGS__)
constexpr std::string_view names[] = {SNICHOLLS_TEST_FOR_EACH(SNICHOLLS_TEST_100(a), SNICHOLLS_TEST_100(b), SNICHOLLS_TEST_100(c))};
static_assert(std::size(names) == 300 && names[0] == "a00" && names[299] == "c99");

TEST(PpForEach, ExpandsEveryElementInOrder)
{
    EXPECT_EQ(names[150], "b50");
}

} // namespace
