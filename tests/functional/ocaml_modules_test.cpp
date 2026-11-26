//
//  ocaml_modules_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Every entry of OCaml's Option, Result and List modules in <snicholls/core/functional/ocaml.hpp>, with the empty and
//  error cases, and a pipeline in OCaml's |> style.
//

#include <snicholls/core/functional/ocaml.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace {

namespace fn = snicholls::fn;
namespace option = snicholls::fn::ocaml::option;
namespace list = snicholls::fn::ocaml::list;
namespace res = snicholls::fn::ocaml::result;

enum class e { parse, range };
using R = snicholls::result<int, e>;

constexpr auto succ = [](int x) { return x + 1; };
constexpr std::optional<int> none_int = option::none;

// ---- Option ------------------------------------------------------------------------------------------------------------

static_assert(option::map(succ, none_int) == std::nullopt);
static_assert(option::bind(none_int, [](int x) { return std::optional<int>{x}; }) == std::nullopt);
static_assert(option::bind(std::optional<int>{3}, [](int) { return std::optional<int>{}; }) == std::nullopt);
static_assert(option::join(std::optional<std::optional<int>>{std::optional<int>{}}) == std::nullopt);
static_assert(option::join(std::optional<std::optional<int>>{}) == std::nullopt);
static_assert(option::fold(-1, succ, none_int) == -1);
static_assert(option::value(std::optional<int>{5}, 4) == 5);
static_assert(option::is_some(std::optional<int>{0}) && !option::is_some(none_int) && !option::is_none(std::optional<int>{0}));
static_assert(option::some(std::string_view("x")) == std::optional<std::string_view>("x"));

// ---- Result ------------------------------------------------------------------------------------------------------------

constexpr R parse_digit(char c) { return c >= '0' && c <= '9' ? R(snicholls::ok(c - '0')) : R(snicholls::err(e::parse)); }
constexpr R in_range(int x) { return x < 8 ? R(snicholls::ok(x)) : R(snicholls::err(e::range)); }

static_assert(res::map(succ, parse_digit('4')).unwrap_unchecked() == 5);
static_assert(res::map(succ, parse_digit('x')).unwrap_err_unchecked() == e::parse);
static_assert(res::bind(parse_digit('4'), in_range).unwrap_unchecked() == 4);
static_assert(res::bind(parse_digit('9'), in_range).unwrap_err_unchecked() == e::range);
static_assert(res::bind(parse_digit('x'), in_range).unwrap_err_unchecked() == e::parse);   // the first error
static_assert(res::join(snicholls::result<R, e>(snicholls::ok(R(snicholls::ok(2))))).unwrap_unchecked() == 2);
static_assert(res::value(parse_digit('x'), 0) == 0 && res::value(parse_digit('3'), 0) == 3);
static_assert(res::fold([](int v) { return v; }, [](e) { return -1; }, parse_digit('x')) == -1);
static_assert(!res::to_option(parse_digit('x')).has_value());
static_assert(res::map_error([](e) { return 'p'; }, parse_digit('1')).unwrap_unchecked() == 1);

// ---- List --------------------------------------------------------------------------------------------------------------

constexpr std::array<int, 5> xs{3, 1, 4, 1, 5};
static_assert(list::map(succ, xs) == std::array{4, 2, 5, 2, 6});
static_assert(list::filter([](int x) { return x > 2; }, xs).size() == 3);
static_assert(list::filter([](int x) { return x > 9; }, xs).empty());
static_assert(list::fold_left([](int acc, int x) { return acc * 10 + x; }, 0, xs) == 31415);
static_assert(list::fold_right([](int x, int acc) { return acc * 10 + x; }, xs, 0) == 51413);
static_assert(list::rev(xs) == std::array{5, 1, 4, 1, 3} && list::rev(std::array<int, 0>{}) == std::array<int, 0>{});
static_assert(list::length(xs) == 5 && list::length(std::array<int, 0>{}) == 0);
static_assert(list::mem(4, xs) && !list::mem(9, xs));
static_assert(list::find_opt([](int x) { return x > 3; }, xs) == std::optional<int>{4});
static_assert(list::find_opt([](int x) { return x > 9; }, xs) == std::nullopt);
static_assert(list::partition([](int x) { return x % 2 == 1; }, xs).first.size() == 4);
static_assert(list::partition([](int x) { return x % 2 == 1; }, xs).second[0] == 4);
static_assert(list::combine(std::array{1, 2}, std::array{'a', 'b'}) == std::array{std::pair{1, 'a'}, std::pair{2, 'b'}});
static_assert(list::split(std::array{std::pair{1, 'a'}, std::pair{2, 'b'}}) == std::pair{std::array{1, 2}, std::array{'a', 'b'}});
static_assert(list::concat_map([](int x) { return std::array{x, -x}; }, std::array{1, 2}) == std::array{1, -1, 2, -2});
static_assert(!list::exists([](int x) { return x > 9; }, xs) && list::for_all([](int x) { return x > 9; }, std::array<int, 0>{}));
static_assert(list::assoc_opt(7, std::array{std::pair{2, 'z'}}) == std::nullopt);
static_assert(list::sort([](int a, int b) { return a - b; }, xs) == std::array{1, 1, 3, 4, 5});
static_assert(list::init<0>([](std::size_t i) { return i; }).empty());

// ---- |> --------------------------------------------------------------------------------------------------------------------

// xs |> List.filter odd |> List.length
static_assert((xs | fn::then([](const auto& l) { return list::filter([](int x) { return x % 2 == 1; }, l); }) |
               fn::then(list::length)) == 4);
// "7" |> parse |> Result.bind in_range |> Result.map succ |> Result.value ~default:0
static_assert((parse_digit('7') | fn::then([](const R& r) { return res::bind(r, in_range); }) |
               fn::then([](const R& r) { return res::map(succ, r); }) | fn::then([](const R& r) { return res::value(r, 0); })) == 8);

TEST(FunctionalOcamlModules, GetOfAValueAndResultsWithStrings)
{
    EXPECT_EQ(option::get(std::optional<std::string>("LMAX")), "LMAX");
    using S = snicholls::result<std::string, std::string>;
    EXPECT_EQ(res::get_ok(S(snicholls::ok(std::string("fill")))), "fill");
    EXPECT_EQ(res::get_error(S(snicholls::err(std::string("reject")))), "reject");
}

} // namespace
