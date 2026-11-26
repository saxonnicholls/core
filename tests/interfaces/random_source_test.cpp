//
//  random_source_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/random_source.hpp>.
//

#include <snicholls/core/interfaces/random_source.hpp>
#include <snicholls/core/interfaces/random_source_reference.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <memory>
#include <system_error>

namespace {

using namespace snicholls;

std::errc draw(random_source_interface& s, std::array<std::byte, 16>& out) { return s.fill(out.data(), out.size()); }

template <typename S>
std::errc draw_generic(S& s, std::array<std::byte, 16>& out) { return s.fill(out.data(), out.size()); }

TEST(RandomSourceInterface, TheAdapterHoldsByValueOrBorrows)
{
    random_source_adapter<test_random_source> owned{test_random_source{7}};
    test_random_source mine{7};
    random_source_adapter<test_random_source&> borrowed{mine};
    std::array<std::byte, 16> a{}, b{};
    EXPECT_EQ(draw(owned, a), std::errc{});
    EXPECT_EQ(draw(borrowed, b), std::errc{});
    EXPECT_EQ(a, b) << "the same seed through either adapter";
    EXPECT_EQ(mine.blocks(), 2u) << "the borrowed source itself advanced";
}

TEST(RandomSourceInterface, AnyRandomSourceIsTheConceptOverTheInterface)
{
    any_random_source any = any_random_source::of(test_random_source{1});
    ASSERT_TRUE(any);
    std::array<std::byte, 16> out{};
    EXPECT_EQ(draw_generic(any, out), std::errc{});
}

TEST(RandomSourceInterface, ErrorsPassThroughEveryForm)
{
    const auto code = std::errc::resource_unavailable_try_again;
    random_source_adapter<unavailable_random_source> adapted{unavailable_random_source{}};
    std::array<std::byte, 16> out{};
    EXPECT_EQ(draw(adapted, out), code);
    any_random_source any = any_random_source::of(unavailable_random_source{});
    EXPECT_EQ(draw_generic(any, out), code);
}

TEST(RandomSourceInterface, AnEmptyAnyFailsClosed)
{
    any_random_source empty;
    EXPECT_FALSE(empty);
    std::array<std::byte, 16> out{};
    EXPECT_EQ(draw_generic(empty, out), std::errc::operation_not_permitted);
}

} // namespace
