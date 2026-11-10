//
//  struct_tuple_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/reflect/struct_tuple.hpp>.
//

#include <array>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <snicholls/core/reflect/struct_tuple.hpp>

namespace {

// Plain structs.
struct s_flat { int a; double b; std::string c; };
struct s_inner { double b; int c; };
struct s_outer { int a; s_inner inner; };
struct s_move { std::unique_ptr<int> p; std::string s; };
struct s_empty {};

// --- concepts -------------------------------------------------------------------

TEST(StructTupleConcepts, TupleLikeAndPlainStruct)
{
    static_assert(snicholls::tuple_like<std::tuple<int>> && snicholls::tuple_like<std::pair<int, int>> &&
                  snicholls::tuple_like<std::array<int, 2>> && snicholls::tuple_like<const std::tuple<>&>);
    static_assert(!snicholls::tuple_like<int> && !snicholls::tuple_like<s_flat>);
    static_assert(snicholls::plain_struct<s_flat> && snicholls::plain_struct<s_empty>);
    static_assert(!snicholls::plain_struct<int>);
}

// --- tie ------------------------------------------------------------------------

TEST(StructTupleTie, StructGivesReferencesToMembers)
{
    s_flat s{1, 2.0, "x"};
    auto refs = snicholls::tie(s);
    static_assert(std::is_same_v<decltype(refs), std::tuple<int&, double&, std::string&>>);
    std::get<2>(refs) = "y";
    EXPECT_EQ(s.c, "y");
}

TEST(StructTupleTie, EmptyStruct)
{
    s_empty s;
    static_assert(std::is_same_v<decltype(snicholls::tie(s)), std::tuple<>>);
}

TEST(StructTupleTie, AlwaysNoexcept)
{
    static_assert(noexcept(snicholls::tie(std::declval<s_flat&>())));
}

// --- to_tuple -------------------------------------------------------------------

TEST(StructTupleToTuple, StructToValues)
{
    const auto t = snicholls::to_tuple(s_flat{1, 2.0, "x"});
    static_assert(std::is_same_v<std::remove_const_t<decltype(t)>, std::tuple<int, double, std::string>>);
    EXPECT_EQ(t, std::make_tuple(1, 2.0, std::string("x")));
}

TEST(StructTupleToTuple, StructNestedMembersStayAsStructs)
{
    const auto t = snicholls::to_tuple(s_outer{1, {2.0, 3}});
    static_assert(std::is_same_v<std::tuple_element_t<1, std::remove_const_t<decltype(t)>>, s_inner>);
}

TEST(StructTupleToTuple, MoveOnlyStructMembersMove)
{
    s_move s{std::make_unique<int>(5), "abc"};
    auto t = snicholls::to_tuple(std::move(s));
    ASSERT_NE(std::get<0>(t), nullptr);
    EXPECT_EQ(*std::get<0>(t), 5);
    EXPECT_EQ(s.p, nullptr); // NOLINT(bugprone-use-after-move)
}

TEST(StructTupleToTuple, EmptyGivesEmptyTuple)
{
    static_assert(std::is_same_v<decltype(snicholls::to_tuple(s_empty{})), std::tuple<>>);
}

TEST(StructTupleToTuple, IsConstexpr)
{
    constexpr auto st = snicholls::to_tuple(s_inner{1.5, 2});
    static_assert(std::get<1>(st) == 2);
}

TEST(StructTupleToTuple, NoexceptExactlyWhenCopiesCannotThrow)
{
    static_assert(!noexcept(snicholls::to_tuple(std::declval<const s_flat&>())));
    static_assert(noexcept(snicholls::to_tuple(std::declval<s_flat&&>())));
}

// --- from_tuple -------------------------------------------------------------------

TEST(StructTupleFromTuple, AcceptsPairArrayAndNonNarrowingConversions)
{
    const auto s = snicholls::from_tuple<s_inner>(std::make_pair(4.0, 5));
    EXPECT_EQ(s.c, 5);
}

TEST(StructTupleFromTuple, AcceptsStdArray)
{
    struct s_pair { int x; int y; };
    const auto s = snicholls::from_tuple<s_pair>(std::array<int, 2>{3, 4});
    EXPECT_EQ(s.y, 4);
}

TEST(StructTupleFromTuple, TupleToStruct)
{
    const auto s = snicholls::from_tuple<s_flat>(std::make_tuple(1, 2.0, std::string("x")));
    EXPECT_EQ(s.a, 1);
    EXPECT_EQ(s.c, "x");
    const auto m = snicholls::from_tuple<s_move>(std::make_tuple(std::make_unique<int>(3), std::string("y")));
    EXPECT_EQ(*m.p, 3);
}

TEST(StructTupleFromTuple, EmptyTuple)
{
    static_assert(std::is_same_v<decltype(snicholls::from_tuple<s_empty>(std::tuple<>{})), s_empty>);
}

TEST(StructTupleFromTuple, IsConstexpr)
{
    static_assert(snicholls::from_tuple<s_inner>(std::make_tuple(1.5, 2)).c == 2);
}

TEST(StructTupleFromTuple, NoexceptExactlyWhenAssignmentsCannotThrow)
{
    static_assert(noexcept(snicholls::from_tuple<s_inner>(std::declval<std::tuple<double, int>>())));
}

// --- round trips ------------------------------------------------------------------

TEST(StructTupleRoundTrip, StructTupleStruct)
{
    const s_flat s{5, 6.5, "seven"};
    const auto back = snicholls::from_tuple<s_flat>(snicholls::to_tuple(s));
    EXPECT_EQ(snicholls::to_tuple(back), snicholls::to_tuple(s));
}

} // namespace
