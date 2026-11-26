//
//  types_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/functional/types.hpp>: the type-level aliases, at compile time.
//

#include <snicholls/core/functional/types.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace {

namespace ty = snicholls::fn::types;
using L = ty::list<int, char, int, double>;
static_assert(std::is_same_v<ty::map_t<L, std::add_pointer_t>, ty::list<int*, char*, int*, double*>>);
static_assert(std::is_same_v<ty::filter_t<L, std::is_integral>, ty::list<int, char, int>>);
static_assert(std::is_same_v<ty::nub_t<L>, ty::list<int, char, double>>);
static_assert(ty::length_v<L> == 4 && ty::elem_v<L, char> && !ty::elem_v<L, float>);
template <class A, class B> struct bigger { using type = std::conditional_t<(sizeof(B) > sizeof(A)), B, A>; };
template <class A, class B> using bigger_t = typename bigger<A, B>::type;
static_assert(std::is_same_v<ty::foldl_t<L, char, bigger_t>, double>);
static_assert(std::is_same_v<ty::reverse_t<ty::list<int, char>>, ty::list<char, int>>);
static_assert(std::is_same_v<ty::zip_t<ty::list<int>, ty::list<char>>, ty::list<std::pair<int, char>>>);
template <class A, class B>
struct smaller : std::bool_constant<(sizeof(A) < sizeof(B))> {};
static_assert(std::is_same_v<ty::sort_by_t<ty::list<double, char, int>, smaller>, ty::list<char, int, double>>);
static_assert(std::is_same_v<ty::take_t<L, 2>, ty::list<int, char>> && std::is_same_v<ty::drop_t<L, 3>, ty::list<double>>);
static_assert(std::is_same_v<ty::concat_t<ty::list<int>, ty::list<char>>, ty::list<int, char>>);
static_assert(std::is_same_v<ty::zip_with_t<std::pair, ty::list<int, char>, ty::list<char>>, ty::list<std::pair<int, char>>>);
template <class A, class B>
using bigger_r = typename bigger<B, A>::type;
static_assert(std::is_same_v<ty::foldr_t<ty::list<char, int>, char, bigger_r>, int>);

TEST(FunctionalTypes, AreCheckedAtCompileTime)
{
    SUCCEED() << "every check in this file is a static_assert";
}

} // namespace
