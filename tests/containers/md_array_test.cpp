//
//  md_array_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/containers/md_array.hpp>: row-major offsets and strides at compile
//  time, the run-time view in each layout, conversions, submdspan, the unrolled index walk, and the
//  layout contract.
//

#include <snicholls/core/containers/md_array.hpp>

#include <gtest/gtest.h>

#include <array>
#include <stdexcept>
#include <utility>
#include <type_traits>
#include <vector>

namespace {

using snicholls::containers::md_array;
using snicholls::containers::full_extent;
using snicholls::containers::layout_left;
using snicholls::containers::layout_right;
using snicholls::containers::layout_stride;
using snicholls::containers::md_span;
using snicholls::containers::submdspan;
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

// Views are a pointer and two arrays, trivially copyable, in every layout; the default layout is row-major,
// so md_span<T, Rank> written before layouts existed is unchanged.
static_assert(std::is_same_v<md_span<double, 2>, md_span<double, 2, layout_right>>);
static_assert(std::is_trivially_copyable_v<md_span<double, 3, layout_right>>);
static_assert(std::is_trivially_copyable_v<md_span<double, 3, layout_left>>);
static_assert(std::is_trivially_copyable_v<md_span<const float, 3, layout_stride>>);
static_assert(sizeof(md_span<double, 2, layout_stride>) == sizeof(double*) + 4 * sizeof(std::size_t));
static_assert(md_span<int, 3>::rank() == 3);

// Conversions follow std::mdspan's: to const, and from row- or column-major to strided; never the other way.
static_assert(std::is_convertible_v<md_span<double, 2>, md_span<const double, 2>>);
static_assert(std::is_convertible_v<md_span<double, 2, layout_left>, md_span<const double, 2, layout_stride>>);
static_assert(!std::is_convertible_v<md_span<const double, 2>, md_span<double, 2>>);
static_assert(!std::is_convertible_v<md_span<double, 2, layout_stride>, md_span<double, 2, layout_right>>);
static_assert(!std::is_convertible_v<md_span<double, 2, layout_left>, md_span<double, 2, layout_right>>);
static_assert(!std::is_constructible_v<md_span<double, 2, layout_right>, double*, std::array<std::size_t, 2>, std::array<std::size_t, 2>>);

consteval bool layouts_agree()
{
    std::array<int, 6> storage{0, 1, 2, 3, 4, 5};
    const md_span<int, 2, layout_right> r(storage.data(), {2, 3});
    const md_span<int, 2, layout_left> l(storage.data(), {3, 2});
    const md_span<int, 2, layout_stride> s(storage.data(), {2, 3}, {3, 1});
    const md_span<int, 2, layout_stride> t(storage.data(), {3, 2}, {1, 3});   // r's transpose
    bool ok = r.stride(0) == 3 && r.stride(1) == 1 && l.stride(0) == 1 && l.stride(1) == 3;
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            ok = ok && r(i, j) == static_cast<int>(3 * i + j) && s(i, j) == r(i, j) && t(j, i) == r(i, j) && l(j, i) == r(i, j);
    return ok && r.mapping().required_span_size() == 6 && s.mapping().required_span_size() == 6 && s.is_exhaustive();
}
static_assert(layouts_agree());

TEST(MdSpanLayouts, ColumnMajorPutsTheFirstIndexContiguous)
{
    std::array<double, 6> storage{};
    const md_span<double, 2, layout_left> a(storage.data(), {2, 3});
    a(1, 0) = 1.0;
    a(0, 1) = 2.0;
    EXPECT_EQ(storage[1], 1.0);
    EXPECT_EQ(storage[2], 2.0);
    EXPECT_EQ(a.size(), 6u);
    EXPECT_TRUE(a.is_exhaustive());
}

TEST(MdSpanLayouts, StridedViewSkipsPaddingAndReportsItsSpan)
{
    // Two 2 x 2 matrices with 3-element padded rows and a gap of 2 between them: 2 * 8 - 2 elements used.
    std::array<double, 16> storage{};
    const md_span<double, 3, layout_stride> batch(storage.data(), {2, 2, 2}, {8, 3, 1});
    batch(1, 1, 1) = 9.0;
    EXPECT_EQ(storage[8 + 3 + 1], 9.0);
    EXPECT_EQ(batch.mapping().required_span_size(), 13u);
    EXPECT_FALSE(batch.is_exhaustive());
    EXPECT_EQ(batch.extents(), (std::array<std::size_t, 3>{2, 2, 2}));
    EXPECT_EQ(batch.mapping().strides(), (std::array<std::size_t, 3>{8, 3, 1}));
}

TEST(MdSpanLayouts, ConversionKeepsTheElementsAndTheOffsets)
{
    std::array<double, 6> storage{0, 1, 2, 3, 4, 5};
    const md_span<double, 2, layout_left> l(storage.data(), {2, 3});
    const md_span<const double, 2, layout_stride> s = l;
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            EXPECT_EQ(&s(i, j), &l(i, j));
    EXPECT_EQ(s.data_handle(), storage.data());
    EXPECT_EQ(s.stride(0), 1u);
    EXPECT_EQ(s.stride(1), 2u);
}

TEST(MdSpanLayouts, DefaultViewIsEmpty)
{
    const md_span<double, 2> a;
    EXPECT_TRUE(a.empty());
    EXPECT_EQ(a.data(), nullptr);
    EXPECT_EQ(a.mapping().required_span_size(), 0u);
    const md_span<double, 2, layout_stride> s;
    EXPECT_EQ(s.mapping().required_span_size(), 0u);
    EXPECT_TRUE(s.mapping() == s.mapping());
    EXPECT_TRUE(layout_left::mapping<2>({2, 3}) == layout_left::mapping<2>({2, 3}));
    EXPECT_FALSE(layout_right::mapping<2>({2, 3}) == layout_right::mapping<2>({3, 2}));
}

TEST(MdSpanSubmdspan, IndexDropsADimensionAndFullExtentKeepsIt)
{
    std::array<int, 24> storage{};
    for (std::size_t i = 0; i < storage.size(); ++i)
        storage[i] = static_cast<int>(i);
    const md_span<int, 3> cube(storage.data(), {2, 3, 4});
    const auto second = submdspan(cube, 1, full_extent, full_extent);   // one system of a batch
    static_assert(decltype(second)::rank() == 2);
    static_assert(std::is_same_v<decltype(second)::layout_type, layout_stride>);
    EXPECT_EQ(second.extent(0), 3u);
    EXPECT_EQ(second(2, 3), 23);
    const auto column = submdspan(cube, 0, full_extent, 2);
    EXPECT_EQ(column.extent(0), 3u);
    EXPECT_EQ(column(1), 6);
    EXPECT_EQ(column.stride(0), 4u);
}

TEST(MdSpanSubmdspan, PairKeepsAHalfOpenRange)
{
    std::array<int, 12> storage{};
    for (std::size_t i = 0; i < storage.size(); ++i)
        storage[i] = static_cast<int>(i);
    const md_span<int, 2, layout_left> a(storage.data(), {3, 4});
    const auto block = submdspan(a, std::pair<std::size_t, std::size_t>{1, 3}, std::pair<std::size_t, std::size_t>{2, 4});
    EXPECT_EQ(block.extent(0), 2u);
    EXPECT_EQ(block.extent(1), 2u);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            EXPECT_EQ(block(i, j), a(i + 1, j + 2));
    EXPECT_FALSE(block.is_exhaustive());
    // Any integral pair, as the documentation writes it: std::pair{1, 3} is a std::pair<int, int>.
    const auto same = submdspan(a, std::pair{1, 3}, std::pair{2u, 4ull});
    EXPECT_EQ(same.extent(0), 2u);
    EXPECT_EQ(&same(1, 1), &block(1, 1));
}

} // namespace
