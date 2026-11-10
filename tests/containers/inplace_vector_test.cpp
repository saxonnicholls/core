//
//  inplace_vector_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/containers/inplace_vector.hpp>: compile-time use, capacity limits,
//  insert and erase order, comparison, and the layout contract.
//

#include <snicholls/core/containers/inplace_vector.hpp>

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

using snicholls::containers::inplace_vector;

consteval inplace_vector<int, 4> built()
{
    inplace_vector<int, 4> v{3, 1};
    v.push_back(4);
    v.insert(v.begin() + 1, 9);   // 3 9 1 4
    v.erase(v.begin());           // 9 1 4
    return v;
}
static_assert(built() == inplace_vector<int, 4>{9, 1, 4} && built().size() == 3 && !built().full());
static_assert(inplace_vector<int, 2>{1, 2} < inplace_vector<int, 2>{1, 3});
static_assert(sizeof(inplace_vector<double, 8>) == 8 * sizeof(double) + sizeof(std::size_t));
static_assert(std::is_trivially_copyable_v<inplace_vector<double, 8>>);
static_assert(inplace_vector<int, 3>::capacity() == 3);

TEST(InplaceVector, RefusesPastCapacity)
{
    inplace_vector<std::string, 2> v;
    v.push_back("a");
    v.emplace_back(3, 'b');
    EXPECT_FALSE(v.try_push_back("c"));
    EXPECT_THROW(v.push_back("c"), std::length_error);
    EXPECT_THROW(v.resize(3), std::length_error);
    EXPECT_THROW(static_cast<void>(v.at(2)), std::out_of_range);
    EXPECT_EQ(v.back(), "bbb");
    v.pop_back();
    EXPECT_EQ(v.size(), 1u);
}

TEST(InplaceVector, EraseIfKeepsOrder)
{
    inplace_vector<int, 8> v{1, 2, 3, 4, 5, 6};
    EXPECT_EQ(snicholls::containers::erase_if(v, [](int x) { return x % 2 == 0; }), 3u);
    EXPECT_EQ(v, (inplace_vector<int, 8>{1, 3, 5}));
}

} // namespace
