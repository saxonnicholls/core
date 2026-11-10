//
//  md_array_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/containers/md_array.hpp>: row-major offsets and strides at compile
//  time, the run-time view, the unrolled index walk, and the layout contract.
//

#include <snicholls/core/containers/md_array.hpp>

#include <gtest/gtest.h>

#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

using snicholls::containers::md_array;
using snicholls::containers::md_span;
using cube = md_array<int, 2, 3, 4>;

static_assert(cube::rank == 3 && cube::size == 24 && cube::strides == std::array<std::size_t, 3>{12, 4, 1});
static_assert(cube::offset(1, 2, 3) == 23);
static_assert(sizeof(cube) == 24 * sizeof(int) && std::is_trivially_copyable_v<cube>);
consteval int filled()
{
    cube c{};
    cube::for_each_index([&](auto i, auto j, auto k) { c(i(), j(), k()) = static_cast<int>(100 * i + 10 * j + k); });
    return c(1, 2, 3);
}
static_assert(filled() == 123);

TEST(MdArray, ViewAndCheckedAccess)
{
    std::vector<double> storage(2 * 3 * 4 * 5);
    const md_span<double, 4> v(storage.data(), {2, 3, 4, 5});
    v(1, 2, 3, 4) = 7.5;
    EXPECT_EQ(storage.back(), 7.5);
    EXPECT_EQ(v.size(), 120u);
    EXPECT_EQ(v.extent(2), 4u);
    cube c{};
    EXPECT_THROW(static_cast<void>(c.at(2, 0, 0)), std::out_of_range);
    c.at(1, 0, 0) = 1;
    EXPECT_EQ(c.flat()[12], 1);
}

} // namespace
