//
//  curry_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/functional/curry.hpp>: a price-adjustment pipeline built from curried pieces, then
//  partial application and uncurry. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/functional/combinators.hpp>
#include <snicholls/core/functional/curry.hpp>

#include <cmath>
#include <cstdio>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

// The pieces: plain functions of several arguments, written once.
constexpr double convert(double rate, double price) noexcept { return price * rate; }
constexpr double widen(double spread_bp, double price) noexcept { return price * (1.0 + spread_bp / 10'000.0); }
constexpr double clamp(double lo, double hi, double price) noexcept { return price < lo ? lo : (hi < price ? hi : price); }
} // namespace

int main()
{
    namespace fn = snicholls::fn;
    std::printf("snicholls-core functional/curry demo\n");

    //! [curry]
    constexpr auto volume = [](double price, int lots, int lot_size) { return price * lots * lot_size; };
    // One argument at a time, several at a time, or all at once: the same call.
    static_assert(fn::curry(volume)(2.0)(3)(100) == 600.0);
    static_assert(fn::curry(volume)(2.0, 3)(100) == 600.0);
    // A generic lambda has no arity of its own: say it.
    static_assert(fn::curry<2>([](auto a, auto b) { return a * b; })(6)(7) == 42);
    //! [curry]

    //! [price]
    // Morning: fix the parameters of the day. Each is a curried function waiting for a price.
    constexpr double eur_usd = 1.0850;
    const auto to_usd = fn::curry(convert)(eur_usd);
    const auto client_markup = fn::curry(widen)(25.0);                       // 25 bp
    const auto within_limits = fn::curry(clamp)(1.00, 200.00);
    // The pipeline: convert, mark up, clamp. One function, built from pieces, trivially copyable.
    const auto quote = fn::pipe(to_usd, client_markup, within_limits);

    // Per tick: one call. The same as the hand-written lambda, instruction for instruction.
    const auto by_hand = [](double p) { return clamp(1.00, 200.00, widen(25.0, convert(1.0850, p))); };
    //! [price]
    check(quote(100.0) == by_hand(100.0), "the curried pipeline equals the hand-written lambda");
    check(std::fabs(quote(100.0) - 108.77125) < 1e-9, "100 EUR quotes 108.77125 USD");
    check(quote(1000.0) == 200.0, "a large price is clamped to the limit");

    //! [partial]
    // partial binds the leading arguments, partial_right the trailing ones.
    constexpr auto at_100 = fn::partial(convert, 100.0);                   // convert(100, price)
    constexpr auto floor_only = fn::partial_right(clamp, 1e9, 0.0);       // clamp(lo, 1e9, 0) for any lo
    static_assert(at_100(1.5) == 150.0);
    static_assert(floor_only(5.0) == 5.0);
    // uncurry: a hand-curried chain called as one function.
    constexpr auto chained = [](double rate) { return [rate](double price) { return price * rate; }; };
    static_assert(fn::uncurry(chained)(2.0, 3.0) == 6.0);
    //! [partial]
    check(at_100(2.0) == 200.0, "partial: rate fixed at 100");
    check(fn::uncurry(chained)(1.5, 4.0) == 6.0, "uncurry: two arguments, two calls");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
