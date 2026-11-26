//
//  monad_railway_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Showcase for <snicholls/core/functional/monad.hpp>: railway-oriented error handling. An order ticket is parsed,
//  validated and priced by functions that may each fail; they are joined with Kleisli composition (>=>), run over a
//  batch with traverse (all or nothing) and map_maybe (keep the good ones), checked against a budget with fold_m, and
//  combined independently with lift. The same pipeline written with std::optional (Maybe: no reason) and with
//  snicholls::result (Either: the first reason). Run by the compiler, then checked at run time. Exits non-zero if a
//  result is not the one documented.
//

#include <snicholls/core/functional/monad.hpp>

#include <array>
#include <cstdio>
#include <optional>
#include <string_view>
#include <variant>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

namespace fn = snicholls::fn;
using snicholls::err;
using snicholls::ok;

//! [tracks]
enum class reject { not_a_number, zero, not_a_lot, too_large, no_quote, over_budget };
template <class T>
using checked = snicholls::result<T, reject>;

// Each step is a function a -> Either reject b: a track that may switch to the failure line.
constexpr checked<long> parse(std::string_view s)
{
    if (s.empty()) return err(reject::not_a_number);
    long n = 0;
    for (char c : s) {
        if (c < '0' || c > '9') return err(reject::not_a_number);
        n = n * 10 + (c - '0');
    }
    return ok(n);
}
constexpr checked<long> validate(long qty)
{
    if (qty == 0) return err(reject::zero);
    if (qty % 100 != 0) return err(reject::not_a_lot);
    if (qty > 10'000) return err(reject::too_large);
    return ok(qty);
}
constexpr checked<long> price(long qty) { return ok(qty * 125 / 100); }   // cents, at 1.25

// parse >=> validate >=> price: one function from text to a priced order, or the first reason it is not one.
constexpr auto ticket = fn::kleisli(fn::kleisli(parse, validate), price);
//! [tracks]

// The same steps on the Maybe track: a failure has no reason.
constexpr auto maybe_ticket = [](std::string_view s) {
    return fn::mbind(fn::mbind(parse(s).ok(), [](long q) { return validate(q).ok(); }), [](long q) { return price(q).ok(); });
};
} // namespace

int main()
{
    std::printf("snicholls-core functional/monad railway demo\n");

    //! [railway]
    static_assert(ticket("400").unwrap_unchecked() == 500);
    static_assert(ticket("4x0").unwrap_err_unchecked() == reject::not_a_number);   // stops at parse
    static_assert(ticket("450").unwrap_err_unchecked() == reject::not_a_lot);      // stops at validate
    static_assert(ticket("0").unwrap_err_unchecked() == reject::zero);
    static_assert(maybe_ticket("400") == std::optional<long>(500) && maybe_ticket("450") == std::nullopt);
    //! [railway]
    check(ticket("20000").unwrap_err_unchecked() == reject::too_large, "20000 is too large");
    check(fn::either([](reject) { return -1L; }, [](long c) { return c; }, ticket("100")) == 125, "either: 100 shares cost 125 cents");
    check(fn::maybe(-1L, [](long c) { return c; }, maybe_ticket("")) == -1, "maybe: an empty ticket on the Maybe track");

    //! [batch]
    // A basket is all or nothing (traverse); a blotter keeps what it can (map_maybe).
    constexpr std::array<std::string_view, 3> basket{"100", "200", "300"};
    constexpr std::array<std::string_view, 3> mixed{"100", "15", "300"};
    static_assert(fn::traverse(ticket, basket).unwrap_unchecked() == std::array<long, 3>{125, 250, 375});
    static_assert(fn::traverse(ticket, mixed).unwrap_err_unchecked() == reject::not_a_lot);
    static_assert(fn::map_maybe(maybe_ticket, mixed).size() == 2);
    //! [batch]
    check(fn::sequence(std::array{ticket("100"), ticket("abc")}).is_err(), "sequence: one bad ticket sinks the basket");
    check(fn::cat_maybes(std::array{maybe_ticket("100"), maybe_ticket("x"), maybe_ticket("200")}).size() == 2, "cat_maybes: two good");

    //! [budget]
    // Spend from a budget of 600 cents, ticket by ticket: fold_m stops at the first step that fails.
    constexpr auto spend = [](long left, std::string_view s) {
        return fn::mbind(ticket(s), [left](long cost) {
            return cost <= left ? checked<long>(ok(left - cost)) : checked<long>(err(reject::over_budget));
        });
    };
    static_assert(fn::fold_m(spend, 600L, basket).unwrap_err_unchecked() == reject::over_budget);   // 125 + 250 + 375 > 600
    static_assert(fn::fold_m(spend, 800L, basket).unwrap_unchecked() == 50);
    //! [budget]
    check(fn::fold_m(spend, 600L, std::array<std::string_view, 2>{"100", "200"}).unwrap_unchecked() == 225, "two tickets fit in 600");

    //! [lift]
    // Two independent checks combined: applicative, not sequential. The first error, left to right.
    constexpr auto notional = fn::lift([](long qty, long px) { return qty * px; });
    static_assert(notional(parse("300"), parse("12")).unwrap_unchecked() == 3600);
    static_assert(notional(parse("3x0"), parse("")).unwrap_err_unchecked() == reject::not_a_number);
    //! [lift]
    check(fn::lift([](long a, long b, long c) { return a + b + c; })(ticket("100"), ticket("200"), ticket("300")).unwrap_unchecked() == 750,
          "lift of three: the basket's total");
    check(fn::zip_with_m([](long a, long b) { return checked<long>(ok(a - b)); }, std::array{500L, 900L}, std::array{125L, 250L})
                  .unwrap_unchecked() == std::array{375L, 650L},
          "zip_with_m: change from two payments");
    check(fn::replicate_m<3>(ticket("100")).unwrap_unchecked() == std::array{125L, 125L, 125L}, "replicate_m: the same ticket three times");
    check(fn::value_or(ticket("bad"), 0L) == 0 && fn::from_maybe(0L, maybe_ticket("200")) == 250, "value_or and from_maybe");
    check(fn::when_m(true, checked<std::monostate>(err(reject::zero))).is_err() &&
              fn::unless_m(true, checked<std::monostate>(err(reject::zero))).is_ok(),
          "when_m / unless_m run the action only on their condition");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
