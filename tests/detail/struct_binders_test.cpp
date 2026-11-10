//
//  struct_binders_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/detail/struct_binders.hpp>.
//

#include <string>
#include <tuple>
#include <type_traits>

#include <gtest/gtest.h>

#include <snicholls/core/detail/struct_binders.hpp>

namespace {

struct empty_s {};
struct three { int a; double b; std::string c; };
struct big { int m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m16, m17, m18, m19; };

TEST(DetailStructBinders, ZeroMembersGivesEmptyTuple)
{
    empty_s e;
    static_assert(std::is_same_v<decltype(snicholls::detail::binder<0>::tie(e)), std::tuple<>>);
}

TEST(DetailStructBinders, BindsMembersByReference)
{
    three t{1, 2.5, "x"};
    auto refs = snicholls::detail::binder<3>::tie(t);
    static_assert(std::is_same_v<decltype(refs), std::tuple<int&, double&, std::string&>>);
    std::get<0>(refs) = 9;
    std::get<2>(refs) = "y";
    EXPECT_EQ(t.a, 9);
    EXPECT_EQ(t.c, "y");
}

TEST(DetailStructBinders, ConstStructGivesConstReferences)
{
    const three t{1, 2.5, "x"};
    static_assert(std::is_same_v<decltype(snicholls::detail::binder<3>::tie(t)),
                                 std::tuple<const int&, const double&, const std::string&>>);
    EXPECT_EQ(&std::get<1>(snicholls::detail::binder<3>::tie(t)), &t.b);
}

TEST(DetailStructBinders, ManyMembersInDeclarationOrder)
{
    big g{};
    auto refs = snicholls::detail::binder<20>::tie(g);
    std::get<0>(refs) = 1;
    std::get<19>(refs) = 20;
    EXPECT_EQ(g.m0, 1);
    EXPECT_EQ(g.m19, 20);
}

TEST(DetailStructBinders, IsConstexprAndNoexcept)
{
    static_assert([] {
        struct p { int x; int y; } v{3, 4};
        auto [x, y] = snicholls::detail::binder<2>::tie(v);
        return x * 10 + y;
    }() == 34);
    three t;
    static_assert(noexcept(snicholls::detail::binder<3>::tie(t)));
}

TEST(DetailStructBinders, GeneratedUpToMaxMembers)
{
    static_assert(SNICHOLLS_MAX_STRUCT_MEMBERS == 256);
    static_assert(std::is_empty_v<snicholls::detail::binder<SNICHOLLS_MAX_STRUCT_MEMBERS>>); // requires a complete type
}

} // namespace
