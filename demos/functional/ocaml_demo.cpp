//
//  ocaml_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/functional/ocaml.hpp>: a |> pipeline, the Option, Result and List modules, labelled and
//  optional arguments as designated initialisers, and an OCaml functor as a class template over a concept.
//  Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/functional/ocaml.hpp>

#include <array>
#include <concepts>
#include <cstdio>
#include <optional>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

//! [labelled]
// OCaml: let quote ~rate ~spread_bp ?(floor = 0.) price = ...
struct quote_args {
    double rate;            // labelled, required: no default
    double spread_bp;       // labelled, required
    double floor = 0.0;     // optional, with its default
};
constexpr double quote(const quote_args& a, double price)
{
    const double q = price * a.rate * (1 + a.spread_bp / 1e4);
    return q < a.floor ? a.floor : q;
}
//! [labelled]

//! [functor]
// module type ORDERED = sig type t val compare : t -> t -> int end
template <class M>
concept ordered = requires(const typename M::t& a) {
    { M::compare(a, a) } -> std::convertible_to<int>;
};
// module Make_best (Ord : ORDERED) = struct let best xs = ... end
template <ordered Ord>
struct make_best {
    template <std::size_t N>
    static constexpr typename Ord::t best(const std::array<typename Ord::t, N>& xs)
    {
        return snicholls::fn::foldl1([](const auto& a, const auto& b) { return Ord::compare(b, a) > 0 ? b : a; }, xs);
    }
};
// Two modules of the module type, and the functor applied to each.
struct by_price {
    struct t { double price; int size; };
    static constexpr int compare(const t& a, const t& b) { return a.price < b.price ? -1 : (b.price < a.price ? 1 : 0); }
};
struct by_size {
    using t = by_price::t;
    static constexpr int compare(const t& a, const t& b) { return a.size - b.size; }
};
using best_price = make_best<by_price>;
using best_size = make_best<by_size>;
//! [functor]
} // namespace

int main()
{
    namespace fn = snicholls::fn;
    namespace oc = snicholls::fn::ocaml;
    std::printf("snicholls-core functional/ocaml demo\n");

    //! [pipeline]
    // OCaml: [1.0850; 1.0852; 1.0849] |> List.map to_bp |> List.filter wide |> List.length
    constexpr auto to_bp = [](double x) { return x * 10'000.0; };
    constexpr auto n = std::array{1.0850, 1.0852, 1.0849}
                       | fn::then(fn::partial(oc::list::map, to_bp))
                       | fn::then(fn::partial(oc::list::filter, [](double bp) { return bp > 10'850.0; }))
                       | fn::then(oc::list::length);
    static_assert(n == 1);
    // f @@ x
    static_assert(fn::call(to_bp, 0.0001) == 1.0);
    //! [pipeline]
    check(n == 1, "one quote above 10850 bp");

    static_assert(quote({.rate = 1.085, .spread_bp = 0}, 100.0) == 108.5);
    static_assert(quote({.rate = 1.0, .spread_bp = 0, .floor = 150.0}, 100.0) == 150.0);
    check(quote({.rate = 1.0, .spread_bp = 100}, 100.0) == 101.0, "labelled arguments, the optional one left out");

    constexpr std::array book{by_price::t{1.0851, 5}, by_price::t{1.0853, 1}, by_price::t{1.0852, 9}};
    static_assert(best_price::best(book).price == 1.0853 && best_size::best(book).size == 9);
    check(best_size::best(book).price == 1.0852, "a functor applied to two modules");

    // Option and Result.
    check(oc::option::value(oc::option::map(to_bp, std::optional<double>{}), -1.0) == -1.0, "Option.map, then Option.value");
    enum class why { stale };
    using px = snicholls::result<double, why>;
    check(oc::result::fold([](double v) { return v; }, [](why) { return 0.0; }, px(snicholls::err(why::stale))) == 0.0,
          "Result.fold over an error");
    check(oc::list::fold_left([](double a, double b) { return a + b; }, 0.0, book | fn::then([](const auto& b) {
              return fn::map([](const by_price::t& t) { return static_cast<double>(t.size); }, b);
          })) == 15.0,
          "List.fold_left over the sizes");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
