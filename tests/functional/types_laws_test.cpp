//
//  types_laws_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Laws for <snicholls/core/functional/types.hpp>: the list laws one level up, on lists of types (map identity and
//  fusion, reverse involution, take ++ drop, zip lengths, nub idempotence, sort stability on equals), and the empty
//  list at every function.
//

#include <snicholls/core/functional/types.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <type_traits>
#include <utility>

namespace {

namespace ty = snicholls::fn::types;

using L = ty::list<std::int8_t, double, std::int16_t, char, double>;
using E = ty::list<>;

template <class T>
using id_t = T;
template <class T>
using ptr_const_t = std::add_pointer_t<std::add_const_t<T>>;

template <class A, class B>
using wider_t = std::conditional_t<(sizeof(B) > sizeof(A)), B, A>;

template <class A, class B>
struct by_size : std::bool_constant<(sizeof(A) < sizeof(B))> {};

// Functor laws on types.
static_assert(std::is_same_v<ty::map_t<L, id_t>, L>);
static_assert(std::is_same_v<ty::map_t<ty::map_t<L, std::add_const_t>, std::add_pointer_t>, ty::map_t<L, ptr_const_t>>);

// reverse is an involution; take ++ drop is the identity; lengths add.
static_assert(std::is_same_v<ty::reverse_t<ty::reverse_t<L>>, L>);
static_assert(std::is_same_v<ty::concat_t<ty::take_t<L, 2>, ty::drop_t<L, 2>>, L>);
static_assert(ty::length_v<ty::concat_t<L, L>> == 2 * ty::length_v<L>);
static_assert(ty::length_v<ty::zip_t<L, ty::take_t<L, 3>>> == 3);   // the shorter wins

// filter: the kept and the dropped partition the list.
template <class T>
struct not_integral : std::bool_constant<!std::is_integral_v<T>> {};
static_assert(ty::length_v<ty::filter_t<L, std::is_integral>> + ty::length_v<ty::filter_t<L, not_integral>> == ty::length_v<L>);

// nub is idempotent; sort by size is stable for equal sizes and idempotent.
static_assert(std::is_same_v<ty::nub_t<ty::nub_t<L>>, ty::nub_t<L>> && ty::length_v<ty::nub_t<L>> == 4);
static_assert(std::is_same_v<ty::sort_by_t<ty::list<char, std::int8_t, double>, by_size>, ty::list<char, std::int8_t, double>>);
static_assert(std::is_same_v<ty::sort_by_t<ty::sort_by_t<L, by_size>, by_size>, ty::sort_by_t<L, by_size>>);

// A fold over types: the widest type, from either end.
static_assert(std::is_same_v<ty::foldl_t<L, char, wider_t>, double> && std::is_same_v<ty::foldr_t<L, char, wider_t>, double>);

// The empty list.
static_assert(std::is_same_v<ty::map_t<E, std::add_pointer_t>, E> && std::is_same_v<ty::filter_t<E, std::is_integral>, E>);
static_assert(std::is_same_v<ty::reverse_t<E>, E> && std::is_same_v<ty::nub_t<E>, E> && ty::length_v<E> == 0 && !ty::elem_v<E, int>);
static_assert(std::is_same_v<ty::foldl_t<E, char, wider_t>, char> && std::is_same_v<ty::zip_t<E, L>, E>);
static_assert(std::is_same_v<ty::take_t<L, 0>, E> && std::is_same_v<ty::drop_t<L, 5>, E>);

TEST(FunctionalTypesLaws, EveryLawIsCheckedByTheCompiler)
{
    SUCCEED() << "every check in this file is a static_assert";
}

} // namespace
