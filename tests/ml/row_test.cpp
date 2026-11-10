//
//  row_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/row.hpp>: which ranges are rows, which are mutable, and that the spans
//  cover the caller's elements without a copy.
//

#include <snicholls/core/ml/row.hpp>

#include <gtest/gtest.h>

#include <array>
#include <list>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace ml = snicholls::ml;

static_assert(ml::row<std::vector<float>> && ml::row<std::array<double, 4>> && ml::row<std::span<const float>> && ml::row<float[3]>);
static_assert(ml::row<const std::vector<float>&> && ml::row<long double[2]>);
static_assert(!ml::row<std::list<float>> && !ml::row<std::string> && !ml::row<std::vector<bool>> && !ml::row<float>);
static_assert(ml::mutable_row<std::vector<float>&> && ml::mutable_row<std::span<float>> && ml::mutable_row<double (&)[3]>);
static_assert(!ml::mutable_row<const std::vector<float>&> && !ml::mutable_row<std::span<const float>>);
static_assert(std::is_same_v<ml::row_value_t<std::span<const float>>, float>);
static_assert(noexcept(ml::as_row(std::declval<const std::vector<float>&>())));

constexpr bool spans_are_constexpr()
{
    std::array<double, 3> a{1, 2, 3};
    ml::as_mutable_row(a)[1] = 5;
    return ml::as_row(a).size() == 3 && ml::as_row(a)[1] == 5;
}
static_assert(spans_are_constexpr());

TEST(MlRow, SpansCoverTheCallersElementsWithoutACopy)
{
    std::vector<float> v{1, 2, 3, 4};
    const auto r = ml::as_row(v);
    EXPECT_EQ(r.data(), v.data());
    EXPECT_EQ(r.size(), 4u);
    ml::as_mutable_row(v)[2] = 9;
    EXPECT_EQ(v[2], 9.0f);
    float c[2] = {7, 8};
    EXPECT_EQ(ml::as_row(c).data(), c);
    std::vector<float> empty;
    EXPECT_TRUE(ml::as_row(empty).empty());
}

} // namespace
