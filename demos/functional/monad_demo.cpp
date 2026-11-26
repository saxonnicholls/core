//
//  monad_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/functional/monad.hpp>: functions on plain values used on optional quotes and on
//  results that may have failed. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/functional/monad.hpp>

#include <array>
#include <cstdio>
#include <optional>
#include <tuple>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

enum class feed_error { stale, crossed };
using price = snicholls::result<double, feed_error>;

constexpr price checked(double bid, double ask)
{
    if (ask < bid) return snicholls::err(feed_error::crossed);
    return snicholls::ok((bid + ask) / 2);
}
} // namespace

int main()
{
    namespace fn = snicholls::fn;
    using snicholls::err;
    using snicholls::ok;
    std::printf("snicholls-core functional/monad demo\n");

    //! [lift]
    constexpr auto spread = [](double bid, double ask) { return ask - bid; };
    // lift: a function of plain values, used on values that may be missing...
    constexpr std::optional<double> bid = 1.0850, ask = 1.0852, none{};
    static_assert(fn::lift(spread)(bid, ask).has_value());
    static_assert(!fn::lift(spread)(bid, none).has_value());
    // ...or may have failed: the first error is kept.
    static_assert(fn::lift(spread)(price(ok(1.0)), price(err(feed_error::stale))).unwrap_err_unchecked() ==
                  feed_error::stale);
    //! [lift]
    check(fn::lift(spread)(bid, ask).value() > 0.0, "a spread of two present prices");

    // fmap, and_then and value_or: one step at a time.
    constexpr auto in_bp = [](double r) { return r * 10'000.0; };
    constexpr auto positive = [](double r) { return r > 0 ? std::optional<double>(r) : std::nullopt; };
    check(fn::value_or(fn::fmap(in_bp, none), -1.0) == -1.0, "fmap keeps an empty optional empty; value_or fills it");
    check(!fn::and_then(positive, std::optional<double>(-1.0)).has_value(), "and_then: a step that may itself fail");
    check(fn::fmap(in_bp, checked(1.0, 1.0002)).is_ok(), "fmap over a result maps the value");
    check(fn::value_or(fn::and_then([](double m) { return checked(m, m); }, checked(2.0, 1.0)), 0.0) == 0.0,
          "and_then over a result keeps the error; value_or replaces it");
    check(fn::maybe(-1.0, in_bp, bid) > 10849.0, "maybe: a default for none, a function for some");
    check(fn::either([](feed_error) { return 0.0; }, in_bp, checked(1.0, 1.0)) == 10000.0,
          "either: one function per side of a result");
    check(fn::fold_m([](double acc, double x) { return checked(acc, acc + x); }, 1.0, std::array{0.5, 0.25}).is_ok(),
          "fold_m: a running check that stops at the first failure");

    //! [sequence]
    // sequence: a tuple of results turned inside out. traverse: map, then sequence.
    constexpr auto book = std::tuple{checked(1.0, 1.2), checked(2.0, 2.2)};
    static_assert(fn::sequence(book).is_ok());
    constexpr auto mids = fn::traverse([](auto quote) { return checked(quote.first, quote.second); },
                                       std::tuple{std::pair{1.0, 1.2}, std::pair{2.2, 2.0}});
    static_assert(mids.unwrap_err_unchecked() == feed_error::crossed);
    //! [sequence]
    check(std::get<1>(fn::sequence(book).unwrap_unchecked()) == 2.1, "both mids present");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
