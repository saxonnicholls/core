//
//  type_list_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/meta/type_list.hpp>.
//

#include <concepts>
#include <cstddef>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <snicholls/core/fixed_string.hpp>
#include <snicholls/core/overloaded.hpp>
#include <snicholls/core/meta/type_list.hpp>

namespace {

using namespace snicholls::meta;
using std::is_same_v;

using L = type_list<int, double, char>;
using E = type_list<>;

template <typename A, typename B>
struct smaller : std::bool_constant<(sizeof(A) < sizeof(B))> {};
template <typename A, typename B>
using same_size = std::bool_constant<sizeof(A) == sizeof(B)>;
template <typename Acc, typename T>
using add_size = std::integral_constant<std::size_t, Acc::value + sizeof(T)>;
template <typename Acc, typename T>
using keep_newest = T;

// --- core ------------------------------------------------------------------------------------------------------------

static_assert(size_v<L> == 3 && size_v<E> == 0 && L::size == 3);
static_assert(empty_v<E> && !empty_v<L>);
static_assert(is_type_list_v<L> && !is_type_list_v<std::tuple<int>>);
static_assert(is_same_v<at_t<L, 0>, int> && is_same_v<at_t<L, 1>, double> && is_same_v<at_t<L, 2>, char>);
static_assert(is_same_v<front_t<L>, int> && is_same_v<back_t<L>, char>);
static_assert(is_same_v<at_t<type_list<int, int, const int>, 2>, const int>);   // duplicates are fine

static_assert(is_same_v<push_front_t<L, bool>, type_list<bool, int, double, char>>);
static_assert(is_same_v<push_back_t<E, bool>, type_list<bool>>);
static_assert(is_same_v<pop_front_t<L>, type_list<double, char>>);
static_assert(is_same_v<pop_back_t<L>, type_list<int, double>> && is_same_v<pop_back_t<E>, E>);
static_assert(is_same_v<concat_t<>, E> && is_same_v<concat_t<L>, L>);
static_assert(is_same_v<concat_t<type_list<int>, E, type_list<char, bool>>, type_list<int, char, bool>>);

// take/drop clamp; N == 0 on an empty list was ambiguous in the original
static_assert(is_same_v<take_t<E, 0>, E> && is_same_v<drop_t<E, 0>, E>);
static_assert(is_same_v<take_t<L, 2>, type_list<int, double>> && is_same_v<take_t<L, 9>, L>);
static_assert(is_same_v<drop_t<L, 1>, type_list<double, char>> && is_same_v<drop_t<L, 9>, E>);
static_assert(is_same_v<erase_at_t<L, 1>, type_list<int, char>> && is_same_v<erase_at_t<L, 9>, L>);
static_assert(is_same_v<insert_at_t<L, 1, bool>, type_list<int, bool, double, char>>);
static_assert(is_same_v<insert_at_t<L, 3, bool>, push_back_t<L, bool>>);
static_assert(is_same_v<replace_at_t<L, 2, bool>, type_list<int, double, bool>>);

// --- queries -----------------------------------------------------------------------------------------------------------

static_assert(contains_v<L, double> && !contains_v<L, bool> && !contains_v<E, int>);
static_assert(count_v<type_list<int, char, int>, int> == 2 && count_v<L, bool> == 0);
static_assert(index_of_v<L, char> == 2 && index_of_v<L, bool> == npos && index_of_v<type_list<int, int>, int> == 0);
static_assert(any_of_v<L, std::is_floating_point> && !any_of_v<E, std::is_floating_point>);
static_assert(all_of_v<L, std::is_arithmetic> && all_of_v<E, std::is_pointer> && !all_of_v<L, std::is_integral>);
static_assert(none_of_v<L, std::is_pointer> && !none_of_v<L, std::is_integral>);

// --- element-wise ------------------------------------------------------------------------------------------------------

static_assert(is_same_v<transform_t<L, std::add_pointer_t>, type_list<int*, double*, char*>>);
static_assert(is_same_v<filter_t<L, std::is_integral>, type_list<int, char>>);
static_assert(is_same_v<remove_if_t<L, std::is_integral>, type_list<double>>);
static_assert(is_same_v<remove_t<type_list<int, char, int>, int>, type_list<char>>);
static_assert(is_same_v<replace_t<type_list<int, char, char>, char, bool>, type_list<int, bool, bool>>);
static_assert(is_same_v<replace_if_t<L, std::is_integral, std::add_const_t>, type_list<const int, double, const char>>);
static_assert(is_same_v<reverse_t<L>, type_list<char, double, int>> && is_same_v<reverse_t<E>, E>);
static_assert(is_same_v<flatten_t<type_list<int, type_list<char, bool>, type_list<>, type_list<type_list<long>>>>,
                        type_list<int, char, bool, type_list<long>>>);
static_assert(is_same_v<flatten_t<E>, E>);

// --- folds ---------------------------------------------------------------------------------------------------------------

static_assert(fold_left_t<L, std::integral_constant<std::size_t, 0>, add_size>::value
              == sizeof(int) + sizeof(double) + sizeof(char));
static_assert(is_same_v<fold_left_t<L, void, keep_newest>, char>);   // left fold ends at the last element
static_assert(is_same_v<fold_right_t<L, void, keep_newest>, int>);   // right fold ends at the first
static_assert(is_same_v<fold_left_t<E, bool, keep_newest>, bool>);
static_assert(is_same_v<fold_right_t<L, type_list<>, push_back_t>, reverse_t<L>>);

// --- sets ---------------------------------------------------------------------------------------------------------------

static_assert(is_same_v<unique_t<type_list<int, double, int, char, double>>, L>);   // keeps the first occurrence
static_assert(is_same_v<set_union_t<type_list<int, char>, type_list<char, bool>>, type_list<int, char, bool>>);
static_assert(is_same_v<set_intersection_t<type_list<int, char, int, bool>, type_list<bool, int>>, type_list<int, bool>>);
static_assert(is_same_v<set_difference_t<type_list<int, char, int>, type_list<char>>, type_list<int>>);
static_assert(is_same_v<set_symmetric_difference_t<type_list<int, double, char>, type_list<char, short, int>>,
                        type_list<double, short>>);

// values as types: fixed strings form a set like any other types
using usd = constant<snicholls::fixed_string{"usd"}>;
using aud = constant<snicholls::fixed_string{"aud"}>;
static_assert(is_same_v<usd, constant<snicholls::fixed_string{"usd"}>> && !is_same_v<usd, aud>);
static_assert(is_same_v<set_union_t<type_list<usd>, type_list<aud, usd>>, type_list<usd, aud>>);
static_assert(usd::value.view() == "usd" && constant<42>::value == 42);

// --- zip / unzip / partition -----------------------------------------------------------------------------------------

static_assert(is_same_v<zip_t<type_list<int, char>, type_list<bool, long>>,
                        type_list<std::pair<int, bool>, std::pair<char, long>>>);
static_assert(is_same_v<zip_t<L, type_list<bool>>, type_list<std::pair<int, bool>>>);   // truncates to the shorter
static_assert(is_same_v<zip_t<E, L>, E>);
static_assert(is_same_v<zip_with_t<std::tuple, type_list<int>, type_list<char>>, type_list<std::tuple<int, char>>>);
using zipped = zip_t<type_list<int, char>, type_list<bool, long>>;
static_assert(is_same_v<unzip<zipped>::first, type_list<int, char>> && is_same_v<unzip<zipped>::second, type_list<bool, long>>);
static_assert(is_same_v<partition<L, std::is_integral>::first, type_list<int, char>>);
static_assert(is_same_v<partition<L, std::is_integral>::second, type_list<double>>);

// --- sort (stable) ---------------------------------------------------------------------------------------------------

static_assert(is_same_v<sort_t<type_list<double, char, int>, smaller>, type_list<char, int, double>>);
static_assert(is_same_v<sort_t<E, smaller>, E> && is_same_v<sort_t<type_list<int>, smaller>, type_list<int>>);
// int and unsigned (and float) are the same size: their relative order must survive (the original swapped them)
static_assert(is_same_v<sort_t<type_list<double, int, char, unsigned, float>, smaller>,
                        type_list<char, int, unsigned, float, double>>);
static_assert(is_same_v<sort_t<type_list<unsigned, int>, smaller>, type_list<unsigned, int>>);

// --- grouping ----------------------------------------------------------------------------------------------------------

static_assert(is_same_v<group_by_t<type_list<int, double, unsigned, char, long double, float>, same_size>,
                        type_list<type_list<int, unsigned, float>, type_list<double>, type_list<char>, type_list<long double>>>
              || sizeof(double) == sizeof(long double));
static_assert(is_same_v<group_by_t<E, same_size>, E>);
static_assert(is_same_v<group_adjacent_t<type_list<int, unsigned, char, int>, same_size>,
                        type_list<type_list<int, unsigned>, type_list<char>, type_list<int>>>);
static_assert(is_same_v<group_adjacent_t<E, same_size>, E>);
static_assert(is_same_v<group_adjacent_by_key_t<type_list<int, const int, double, double, char>, std::remove_cv_t>,
                        type_list<type_list<int, const int>, type_list<double, double>, type_list<char>>>);

// --- topological sort -------------------------------------------------------------------------------------------------

struct a {};
struct b {};
struct c {};
struct d {};
template <typename X, typename Y>
struct depends : std::false_type {};
template <>
struct depends<a, b> : std::true_type {};   // a needs b
template <>
struct depends<a, c> : std::true_type {};   // a needs c
template <>
struct depends<c, d> : std::true_type {};   // c needs d
template <>
struct depends<a, a> : std::true_type {};   // self-dependence is ignored

using order = topological_sort_t<type_list<a, b, c, d>, depends>;
static_assert(is_same_v<order, type_list<b, d, c, a>>);
static_assert(index_of_v<order, b> < index_of_v<order, a> && index_of_v<order, d> < index_of_v<order, c>);
static_assert(is_same_v<topological_sort_t<E, depends>, E>);
static_assert(is_same_v<topological_sort_t<type_list<b, d>, depends>, type_list<b, d>>);   // independent: order kept

// --- products ------------------------------------------------------------------------------------------------------------

static_assert(is_same_v<cartesian_product_t<type_list<int, char>, type_list<bool, long>>,
                        type_list<type_list<int, bool>, type_list<int, long>, type_list<char, bool>, type_list<char, long>>>);
static_assert(is_same_v<cartesian_product_t<>, type_list<E>>);
static_assert(is_same_v<cartesian_product_t<L, E>, E>);
static_assert(size_v<cartesian_product_t<L, L, type_list<bool, long>>> == 18);
static_assert(is_same_v<front_t<cartesian_product_t<L, L, type_list<bool>>>, type_list<int, int, bool>>);
static_assert(is_same_v<product_with_t<std::pair, type_list<int>, type_list<char, bool>>,
                        type_list<std::pair<int, char>, std::pair<int, bool>>>);
static_assert(is_same_v<cartesian_power_t<L, 0>, type_list<E>>);
static_assert(is_same_v<cartesian_power_t<type_list<int, char>, 1>, type_list<type_list<int>, type_list<char>>>);
static_assert(size_v<cartesian_power_t<type_list<int, char>, 3>> == 8);
static_assert(is_same_v<back_t<cartesian_power_t<type_list<int, char>, 3>>, type_list<char, char, char>>);

// --- interop ------------------------------------------------------------------------------------------------------------

static_assert(is_same_v<rename_t<L, std::tuple>, std::tuple<int, double, char>>);
static_assert(is_same_v<rename_t<L, std::variant>, std::variant<int, double, char>>);
static_assert(is_same_v<as_type_list_t<std::tuple<int, char>>, type_list<int, char>>);
static_assert(is_same_v<as_type_list_t<std::pair<int, char>>, type_list<int, char>>);

// --- reify (also usable at compile time) -------------------------------------------------------------------------------

constexpr std::size_t total_size()
{
    std::size_t total = 0;
    for_each_type<L>([&]<typename T>() { total += sizeof(T); });
    return total;
}
static_assert(total_size() == sizeof(int) + sizeof(double) + sizeof(char));
static_assert(make_array<L>([]<typename T>() { return sizeof(T); })[1] == sizeof(double));

// --- dot and outer products, Thread, permutations --------------------------------------------------

struct a; struct b; struct c; struct x; struct y; struct z; struct h;
template <typename...> struct f;
template <typename A, typename B> using pair_alias = std::pair<A, B>;

// The outer product is the cartesian product by its other name
static_assert(is_same_v<outer_product_t<type_list<a, b>, type_list<x, y>>, cartesian_product_t<type_list<a, b>, type_list<x, y>>>);
static_assert(is_same_v<outer_with_t<f, type_list<a>, type_list<x, y>>, type_list<f<a, x>, f<a, y>>>);

// The dot product pairs corresponding elements, of any number of lists
static_assert(is_same_v<dot_product_t<type_list<a, b>, type_list<x, y>>, type_list<type_list<a, x>, type_list<b, y>>>);
static_assert(is_same_v<dot_product_t<type_list<a>, type_list<x>, type_list<c>>, type_list<type_list<a, x, c>>>);
static_assert(is_same_v<dot_product_t<type_list<a, b>>, type_list<type_list<a>, type_list<b>>>);
static_assert(is_same_v<dot_product_t<E, E>, E>);
static_assert(is_same_v<dot_with_t<std::pair, type_list<a, b>, type_list<x, y>>, zip_t<type_list<a, b>, type_list<x, y>>>);
static_assert(is_same_v<dot_with_t<pair_alias, type_list<a>, type_list<x>>, type_list<std::pair<a, x>>>);

// Thread, the Wolfram reference's Basic Examples in order
static_assert(is_same_v<thread_t<f<type_list<a, b, c>>>, type_list<f<a>, f<b>, f<c>>>);
static_assert(is_same_v<thread_t<f<type_list<a, b, c>, x>>, type_list<f<a, x>, f<b, x>, f<c, x>>>);
static_assert(is_same_v<thread_t<f<type_list<a, b, c>, type_list<x, y, z>>>, type_list<f<a, x>, f<b, y>, f<c, z>>>);
// "Elements that are not lists are repeated"
static_assert(is_same_v<thread_t<f<type_list<a, b>, h, type_list<x, y>>>, type_list<f<a, h, x>, f<b, h, y>>>);
// Thread[{{a, b, c}, {x, y, z}}] is the transposition; Thread[{{a, b, c}, 0}] pairs with a constant
static_assert(is_same_v<thread_t<type_list<type_list<a, b>, type_list<x, y>>>, type_list<type_list<a, x>, type_list<b, y>>>);
static_assert(is_same_v<thread_t<type_list<type_list<a, b>, h>>, type_list<type_list<a, h>, type_list<b, h>>>);
// Nothing to thread over: the expression itself; an empty list threads to nothing
static_assert(is_same_v<thread_t<f<a, x>>, f<a, x>>);
static_assert(is_same_v<thread_t<f<>>, f<>>);
static_assert(is_same_v<thread_t<f<E, x>>, E>);
// One level only: an inner list reaches f whole
static_assert(is_same_v<thread_t<f<type_list<type_list<a, b>, c>>>, type_list<f<type_list<a, b>>, f<c>>>);
// Heads: by default only type_list; another head threads over it and heads the result
static_assert(is_same_v<thread_t<f<std::tuple<a, b>, std::tuple<x, y>>>, f<std::tuple<a, b>, std::tuple<x, y>>>);
static_assert(is_same_v<thread_t<f<std::tuple<a, b>, std::tuple<x, y>>, std::tuple>, std::tuple<f<a, x>, f<b, y>>>);
static_assert(is_same_v<thread_t<f<std::tuple<a, b>, type_list<x>>, std::tuple>, std::tuple<f<a, type_list<x>>, f<b, type_list<x>>>>);
// Position specifications, on the reference's f[{a, b}, {r, s}, {u, v}, {x, y}]
using l1 = type_list<a, b>; using l2 = type_list<c, h>; using l3 = type_list<x, y>; using l4 = type_list<z, int>;
using four = f<l1, l2, l3, l4>;
static_assert(is_same_v<thread_t<four, type_list, thread_all>, thread_t<four>>);
static_assert(is_same_v<thread_t<four, type_list, thread_none>, four>);
static_assert(is_same_v<thread_t<four, type_list, thread_first<2>>, type_list<f<a, c, l3, l4>, f<b, h, l3, l4>>>);
static_assert(is_same_v<thread_t<four, type_list, thread_first<-2>>, type_list<f<l1, l2, x, z>, f<l1, l2, y, int>>>);
static_assert(is_same_v<thread_t<four, type_list, thread_at<2>>, type_list<f<l1, c, l3, l4>, f<l1, h, l3, l4>>>);
static_assert(is_same_v<thread_t<four, type_list, thread_at<-1>>, type_list<f<l1, l2, l3, z>, f<l1, l2, l3, int>>>);
static_assert(is_same_v<thread_t<four, type_list, thread_range<2, 4>>, type_list<f<l1, c, x, z>, f<l1, h, y, int>>>);
static_assert(is_same_v<thread_t<four, type_list, thread_range<1, -1, 2>>, type_list<f<a, l2, x, l4>, f<b, l2, y, l4>>>);
static_assert(is_same_v<thread_t<four, type_list, thread_range<4, 1, -2>>, type_list<f<l1, c, l3, z>, f<l1, h, l3, int>>>);
// Different lengths are allowed where they are not threaded
static_assert(is_same_v<thread_t<f<l1, type_list<x>>, type_list, thread_at<1>>, type_list<f<a, type_list<x>>, f<b, type_list<x>>>>);
// F apart from its arguments, for an alias, which could not be named with a list as an argument
static_assert(is_same_v<thread_with_t<pair_alias, type_list<a, b>, x>, type_list<std::pair<a, x>, std::pair<b, x>>>);
static_assert(is_same_v<thread_with_t<f, a, x>, f<a, x>>);

// Permutations, in lexicographic order
static_assert(is_same_v<permutations_t<type_list<a, b, c>>,
                        type_list<type_list<a, b, c>, type_list<a, c, b>, type_list<b, a, c>, type_list<b, c, a>, type_list<c, a, b>, type_list<c, b, a>>>);
static_assert(is_same_v<permutations_t<type_list<a, b, c>, 2>,
                        type_list<type_list<a, b>, type_list<a, c>, type_list<b, a>, type_list<b, c>, type_list<c, a>, type_list<c, b>>>);
// A repeated type gives each distinct row once, ranked by first appearance
static_assert(is_same_v<permutations_t<type_list<b, a, b>>, type_list<type_list<b, b, a>, type_list<b, a, b>, type_list<a, b, b>>>);
static_assert(is_same_v<permutations_t<type_list<a, a>>, type_list<type_list<a, a>>>);
// The edges: nothing to permute is one empty row, asking for too many is no rows
static_assert(is_same_v<permutations_t<E>, type_list<E>>);
static_assert(is_same_v<permutations_t<type_list<a, b>, 0>, type_list<E>>);
static_assert(is_same_v<permutations_t<type_list<a, b>, 3>, E>);
static_assert(is_same_v<permutations_t<E, 1>, E>);
static_assert(is_same_v<permutations_t<type_list<a>>, type_list<type_list<a>>>);
static_assert(is_same_v<permutations_with_t<f, type_list<a, b>>, type_list<f<a, b>, f<b, a>>>);
static_assert(size_v<permutations_t<type_list<a, b, c, x, y>>> == 120);
static_assert(size_v<permutations_t<type_list<a, b, c, x, y, z>>> == 720);

TEST(TypeListThread, ForEachTagLetsAnOverloadSetChooseForEachThreadedType)
{
    // A request type threaded over its calls, one handler per result
    using requests = thread_with_t<std::pair, type_list<int, double, char>, std::string>;
    std::string seen;
    for_each_tag<requests>(snicholls::overloaded{
        [&](std::type_identity<std::pair<int, std::string>>) { seen += "int "; },
        [&](std::type_identity<std::pair<double, std::string>>) { seen += "double "; },
        [&](std::type_identity<std::pair<char, std::string>>) { seen += "char "; },
    });
    EXPECT_EQ(seen, "int double char ");
}

TEST(TypeListThread, ForEachTagWorksWithFirstMatchAndConcepts)
{
    int integers = 0, others = 0;
    for_each_tag<type_list<int, double, long>>(snicholls::first_match{
        [&]<std::integral T>(std::type_identity<T>) { ++integers; },
        [&]<typename T>(std::type_identity<T>) { ++others; },
    });
    EXPECT_EQ(integers, 2);
    EXPECT_EQ(others, 1);
}

TEST(TypeList, ForEachTypeVisitsInOrder)
{
    std::vector<std::string> seen;
    for_each_type<type_list<int, double, char>>([&]<typename T>() {
        seen.push_back(std::is_same_v<T, int> ? "int" : std::is_same_v<T, double> ? "double" : "char");
    });
    EXPECT_EQ(seen, (std::vector<std::string>{"int", "double", "char"}));
}

TEST(TypeList, ForEachTypeOnEmptyListCallsNothing)
{
    int calls = 0;
    for_each_type<E>([&]<typename>() { ++calls; });
    EXPECT_EQ(calls, 0);
}

TEST(TypeList, ForEachTypeIsNoexceptWhenTheCallIs)
{
    auto quiet = []<typename>() noexcept {};
    auto loud = []<typename>() {};
    static_assert(noexcept(for_each_type<L>(quiet)));
    static_assert(!noexcept(for_each_type<L>(loud)));
    for_each_type<L>(quiet);
    for_each_type<L>(loud);
    SUCCEED();
}

TEST(TypeList, MakeArrayUsesTheCommonType)
{
    const auto a = make_array<type_list<char, double>>([]<typename T>() { return T{2}; });
    static_assert(std::is_same_v<decltype(a), const std::array<double, 2>>);
    EXPECT_EQ(a[0], 2.0);
    EXPECT_EQ(a[1], 2.0);
}

TEST(TypeList, CartesianProductEnumeratesCurrencyPairsAtRunTime)
{
    using ccys = type_list<usd, aud, constant<snicholls::fixed_string{"eur"}>>;
    std::vector<std::string> pairs;
    for_each_type<cartesian_power_t<ccys, 2>>([&]<typename Pair>() {
        pairs.push_back(std::string(front_t<Pair>::value.view()) + std::string(back_t<Pair>::value.view()));
    });
    ASSERT_EQ(pairs.size(), 9u);
    EXPECT_EQ(pairs.front(), "usdusd");
    EXPECT_EQ(pairs[1], "usdaud");
    EXPECT_EQ(pairs.back(), "eureur");
}

// --- find_if_t and partition_n_t -------------------------------------------------------------------
static_assert(is_same_v<find_if_t<type_list<char, double, int, float>, std::is_floating_point>, double>);
static_assert(find_index_if_v<type_list<char, double, int>, std::is_integral> == 0);
static_assert(find_index_if_v<type_list<double, float>, std::is_integral> == npos);
static_assert(is_same_v<find_if_t<type_list<double, float>, std::is_integral>, void>);   // not an error
static_assert(is_same_v<find_if_t<type_list<double>, std::is_integral, long>, long>);
static_assert(is_same_v<find_if_t<E, std::is_integral>, void>);
// Wolfram's Partition for types
static_assert(is_same_v<partition_n_t<type_list<a, b, c, x>, 2>, type_list<type_list<a, b>, type_list<c, x>>>);
static_assert(is_same_v<partition_n_t<type_list<a, b, c>, 2>, type_list<type_list<a, b>>>);   // incomplete dropped
static_assert(is_same_v<partition_n_t<type_list<a, b, c>, 2, 1>, type_list<type_list<a, b>, type_list<b, c>>>);
static_assert(is_same_v<partition_n_t<type_list<a>, 2>, E>);

// --- more of Wolfram's list functions --------------------------------------------------------------
template <typename A, typename B>
struct g2;
template <typename... Ts>
struct gn;
template <typename X>
struct w;
template <typename T, typename I>
struct at_index;
static_assert(is_same_v<range_t<3>, type_list<constant<std::size_t{0}>, constant<std::size_t{1}>, constant<std::size_t{2}>>>);
static_assert(is_same_v<rotate_left_t<type_list<a, b, c, x>, 1>, type_list<b, c, x, a>>);
static_assert(is_same_v<rotate_left_t<type_list<a, b, c>, 4>, type_list<b, c, a>>);
static_assert(is_same_v<rotate_right_t<type_list<a, b, c, x>, 1>, type_list<x, a, b, c>>);
static_assert(is_same_v<rotate_left_t<E, 2>, E>);
static_assert(is_same_v<riffle_t<type_list<a, b, c>, x>, type_list<a, x, b, x, c>>);
static_assert(is_same_v<riffle_t<E, x>, E> && is_same_v<riffle_t<type_list<a>, x>, type_list<a>>);
static_assert(is_same_v<fold_list_t<type_list<a, b>, x, g2>, type_list<x, g2<x, a>, g2<g2<x, a>, b>>>);
static_assert(is_same_v<nest_list_t<w, x, 2>, type_list<x, w<x>, w<w<x>>>> && is_same_v<nest_t<w, x, 2>, w<w<x>>>);
static_assert(is_same_v<map_indexed_t<type_list<a, b>, at_index>, type_list<at_index<a, constant<std::size_t{0}>>, at_index<b, constant<std::size_t{1}>>>>);
static_assert(is_same_v<tally_t<type_list<a, b, a>>, type_list<type_list<a, constant<std::size_t{2}>>, type_list<b, constant<std::size_t{1}>>>>);
// Subsets[{a, b, c}, {2}] and Subsets[{a, b, c}]
static_assert(is_same_v<subsets_of_size_t<type_list<a, b, c>, 2>, type_list<type_list<a, b>, type_list<a, c>, type_list<b, c>>>);
static_assert(is_same_v<subsets_t<type_list<a, b>>, type_list<E, type_list<a>, type_list<b>, type_list<a, b>>>);
static_assert(size_v<subsets_t<type_list<a, b, c, x, y>>> == 32);
// Inner[f, {a, b}, {x, y}, g] = g[f[a, x], f[b, y]]
static_assert(is_same_v<inner_t<std::pair, type_list<a, b>, type_list<x, y>, gn>, gn<std::pair<a, x>, std::pair<b, y>>>);

} // namespace
