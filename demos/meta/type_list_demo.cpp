//
//  type_list_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/meta/type_list.hpp>.
//

// Demo: meta/type_list.hpp - compile-time type lists: queries, transforms, sets, sorting, grouping, products.

#include <cstdio>
#include <string>
#include <type_traits>
#include <utility>

#include <snicholls/core/fixed_string.hpp>
#include <snicholls/core/overloaded.hpp>
#include <snicholls/core/meta/type_list.hpp>

namespace {

using namespace snicholls::meta;

template <typename A, typename B>
using smaller = std::bool_constant<(sizeof(A) < sizeof(B))>;
template <typename A, typename B>
using same_size = std::bool_constant<sizeof(A) == sizeof(B)>;
template <typename Acc, typename T>
using add_size = std::integral_constant<std::size_t, Acc::value + sizeof(T)>;

struct load_config {};
struct open_feed {};
struct start_ui {};
template <typename X, typename Y>
struct needs : std::false_type {};
template <>
struct needs<open_feed, load_config> : std::true_type {};
template <>
struct needs<start_ui, open_feed> : std::true_type {};

// A type's name: an overload set over std::type_identity, with a catch-all last. Overload resolution picks the exact
// match; no if constexpr chain.
template <typename T>
using tag = std::type_identity<T>;
constexpr auto type_names = snicholls::overloaded{
    [](tag<char>) { return "char"; },
    [](tag<int>) { return "int"; },
    [](tag<unsigned>) { return "unsigned"; },
    [](tag<double>) { return "double"; },
    [](tag<load_config>) { return "load_config"; },
    [](tag<open_feed>) { return "open_feed"; },
    [](tag<start_ui>) { return "start_ui"; },
    [](auto) { return "?"; },
};

template <typename T>
const char* name()
{
    return type_names(tag<T>{});
}

// A list prints as <...>, recursively; anything else by name. first_match: a nested list is tried first.
template <typename List>
std::string show()
{
    std::string out = "<";
    for_each_type<List>([&]<typename T>() {
        out += snicholls::first_match{
            []<typename... Ts>(tag<type_list<Ts...>>) { return show<type_list<Ts...>>(); },
            []<typename U>(tag<U>) { return std::string(name<U>()); },
        }(tag<T>{});
        out += ' ';
    });
    if (out.size() > 1) out.pop_back();
    return out + ">";
}

} // namespace

//! [demo]
int main()
{
    using L = type_list<double, int, char, unsigned, int>;
    std::printf("  list                 %s  size %zu, index_of<char> %zu, count<int> %zu\n", show<L>().c_str(), size_v<L>,
                index_of_v<L, char>, count_v<L, int>);
    std::printf("  unique               %s\n", show<unique_t<L>>().c_str());
    std::printf("  stable sort by size  %s\n", show<sort_t<L, smaller>>().c_str());
    std::printf("  integral only        %s\n", show<filter_t<L, std::is_integral>>().c_str());
    std::printf("  group by size        %s\n", show<group_by_t<unique_t<L>, same_size>>().c_str());
    std::printf("  adjacent runs        %s\n", show<group_adjacent_t<L, same_size>>().c_str());
    std::printf("  sum of sizes         %zu\n", fold_left_t<L, std::integral_constant<std::size_t, 0>, add_size>::value);
    std::printf("  start-up order       %s\n", show<topological_sort_t<type_list<start_ui, open_feed, load_config>, needs>>().c_str());
    std::printf("  <int char> x <double> %s\n", show<cartesian_product_t<type_list<int, char>, type_list<double>>>().c_str());
    std::printf("  <int char>^2         %s\n", show<cartesian_power_t<type_list<int, char>, 2>>().c_str());

    using ccy = type_list<constant<snicholls::fixed_string{"AUD"}>, constant<snicholls::fixed_string{"USD"}>>;
    std::string pairs;
    for_each_type<cartesian_power_t<ccy, 2>>([&]<typename P>() {
        pairs += std::string(front_t<P>::value.view()) + std::string(back_t<P>::value.view()) + ' ';
    });
    std::printf("  currency pairs       %s\n", pairs.c_str());
    // The dot and outer products, permutations, and Mathematica's Thread. A scalar is repeated.
    std::printf("  <int char> . <bool long> %s\n", show<dot_product_t<type_list<int, char>, type_list<bool, long>>>().c_str());
    std::printf("  <int char> o <bool>      %s\n", show<outer_product_t<type_list<int, char>, type_list<bool>>>().c_str());
    std::printf("  permutations <int char>  %s\n", show<permutations_t<type_list<int, char>>>().c_str());
    std::printf("  Thread[f[<int char>, x]] %s\n", show<thread_t<type_list<type_list<int, char>, bool>>>().c_str());

    // One overload per threaded type: a tag gives the overload set something to choose on.
    using requests = thread_with_t<std::pair, type_list<int, char>, bool>;
    std::string handled;
    for_each_tag<requests>(snicholls::overloaded{
        [&](std::type_identity<std::pair<int, bool>>) { handled += "int "; },
        [&](std::type_identity<std::pair<char, bool>>) { handled += "char "; },
    });
    std::printf("  handled                  %s\n", handled.c_str());
    if (handled != "int char ")
        return 1;

    constexpr auto sizes = make_array<L>([]<typename T>() { return sizeof(T); });
    return sizes[0] == sizeof(double) && pairs == "AUDAUD AUDUSD USDAUD USDUSD " ? 0 : 1;
}
//! [demo]
