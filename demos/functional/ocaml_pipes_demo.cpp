//
//  ocaml_pipes_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Showcase for <snicholls/core/functional/ocaml.hpp>: OCaml programs written with |> pipes over the List, Option and
//  Result modules, each line of OCaml beside its C++ and the answers checked by the compiler. Exits non-zero if a
//  result is not the one documented.
//

#include <snicholls/core/functional/curry.hpp>
#include <snicholls/core/functional/ocaml.hpp>

#include <array>
#include <cstddef>
#include <cstdio>
#include <optional>
#include <string_view>
#include <utility>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

namespace fn = snicholls::fn;
namespace List = snicholls::fn::ocaml::list;
namespace Option = snicholls::fn::ocaml::option;
namespace Result = snicholls::fn::ocaml::result;

constexpr auto odd = [](long x) { return x % 2 == 1; };
constexpr auto square = [](long x) { return x * x; };
constexpr auto plus = [](long a, long b) { return a + b; };

//! [sum_of_squares]
// List.init 10 (fun i -> i + 1) |> List.map square |> List.filter odd |> List.fold_left (+) 0
constexpr long sum_odd_squares = List::init<10>([](std::size_t i) { return static_cast<long>(i) + 1; })
                                 | fn::then(fn::partial(List::map, square))
                                 | fn::then(fn::partial(List::filter, odd))
                                 | fn::then(fn::partial(List::fold_left, plus, 0L));
//! [sum_of_squares]

//! [option]
// let price_in_usd ticker =
//   List.assoc_opt ticker prices |> Option.map (fun p -> p * fx) |> Option.value ~default:0
constexpr std::array prices{std::pair<std::string_view, long>{"VOD", 72}, std::pair<std::string_view, long>{"BP", 480}};
constexpr long price_in_usd(std::string_view ticker)
{
    return List::assoc_opt(ticker, prices) | fn::then(fn::partial(Option::map, [](long p) { return p * 127 / 100; }))
           | fn::then([](const std::optional<long>& o) { return Option::value(o, 0L); });
}
//! [option]

//! [result]
// let settle qty = parse qty |> Result.bind check_lot |> Result.map (fun q -> q * 100) |> Result.fold ~ok:Fun.id ~error:(fun _ -> -1)
enum class why { empty, not_a_lot };
using checked = snicholls::result<long, why>;
constexpr checked parse(std::string_view s)
{
    if (s.empty()) return snicholls::err(why::empty);
    long n = 0;
    for (char c : s) n = n * 10 + (c - '0');
    return snicholls::ok(n);
}
constexpr checked check_lot(long q) { return q % 100 == 0 ? checked(snicholls::ok(q)) : checked(snicholls::err(why::not_a_lot)); }
constexpr long settle(std::string_view qty)
{
    return parse(qty) | fn::then([](const checked& r) { return Result::bind(r, check_lot); })
           | fn::then(fn::partial(Result::map, [](long q) { return q * 100; }))
           | fn::then(fn::partial(Result::fold, [](long v) { return v; }, [](why) { return -1L; }));
}
//! [result]

//! [lists]
// let ranked = List.combine names sizes |> List.sort (fun (_, a) (_, b) -> compare b a) |> List.split |> fst
constexpr std::array<std::string_view, 4> names{"LMAX", "EBS", "CBOE", "HSBC"};
constexpr std::array<long, 4> sizes{5, 9, 1, 7};
constexpr auto ranked = List::combine(names, sizes)
                        | fn::then(fn::partial(List::sort, [](const auto& a, const auto& b) { return static_cast<int>(b.second - a.second); }))
                        | fn::then(List::split);
//! [lists]
} // namespace

int main()
{
    std::printf("snicholls-core functional/ocaml pipes demo\n");

    static_assert(sum_odd_squares == 1 + 9 + 25 + 49 + 81);
    check(sum_odd_squares == 165, "odd squares of 1..10 sum to 165");

    static_assert(price_in_usd("BP") == 609 && price_in_usd("XYZ") == 0);
    check(price_in_usd("VOD") == 91, "VOD 72 pence at 1.27 is 91 cents; an unknown ticker is 0");

    static_assert(settle("300") == 30'000 && settle("250") == -1 && settle("") == -1);
    check(settle("1200") == 120'000, "a round lot settles; an odd lot and an empty field do not");

    static_assert(ranked.first == std::array<std::string_view, 4>{"EBS", "HSBC", "LMAX", "CBOE"});
    static_assert(ranked.second == std::array<long, 4>{9, 7, 5, 1});
    check(ranked.first[0] == "EBS", "the venue with the most size first");

    // List.concat_map: each parent order split into two child slices.
    constexpr auto children = List::concat_map([](long q) { return std::array{q / 2, q - q / 2}; }, std::array{100L, 301L});
    static_assert(children == std::array{50L, 50L, 150L, 151L});
    check(List::fold_left(plus, 0L, children) == 401, "the children add back to the parents");
    check(List::exists([](long q) { return q > 150; }, children) && List::for_all([](long q) { return q >= 50; }, children),
          "List.exists and List.for_all");
    check(List::partition(odd, children).first.size() == 1 && List::rev(children)[0] == 151 && List::mem(150L, children),
          "List.partition, List.rev, List.mem");
    check(Option::bind(List::find_opt([](long q) { return q > 100; }, children), [](long q) { return std::optional<long>(q - 100); }) ==
              std::optional<long>(50),
          "List.find_opt |> Option.bind");
    check(Option::is_some(Result::to_option(parse("1"))) && Result::is_error(parse("")), "Result.to_option and Result.is_error");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
