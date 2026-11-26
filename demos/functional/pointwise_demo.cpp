//
//  pointwise_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/functional/pointwise.hpp>: functions added, scaled, compared and combined point by point.
//  Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/functional/pointwise.hpp>

#include <cstdio>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

struct instrument {
    double price;
    double volume;
    double spread_bp;
};
} // namespace

int main()
{
    namespace fn = snicholls::fn;
    std::printf("snicholls-core functional/pointwise demo\n");

    //! [payoff]
    // A call spread from two legs, as functions of the spot price, added and scaled like numbers.
    constexpr fn::pointwise call_100{[](double s) { return s > 100 ? s - 100 : 0.0; }};
    constexpr fn::pointwise call_110{[](double s) { return s > 110 ? s - 110 : 0.0; }};
    constexpr auto spread = 2.0 * (call_100 - call_110);
    static_assert(spread(90) == 0 && spread(105) == 10 && spread(130) == 20);
    //! [payoff]
    check(spread(110) == 20, "the spread pays 2 x 10 at 110");
    check((-spread)(105) == -10 && (spread / 2.0)(105) == 5 && (spread + 1.0)(90) == 1 && (spread * call_100)(105) == 50,
          "negate, divide, add a constant, multiply two functions");

    //! [predicates]
    // A screen built from predicates: liquid and tight, or simply cheap.
    constexpr fn::pointwise price{[](const instrument& i) { return i.price; }};
    constexpr fn::pointwise volume{[](const instrument& i) { return i.volume; }};
    constexpr fn::pointwise spread_bp{[](const instrument& i) { return i.spread_bp; }};
    constexpr auto screen = ((price * volume >= 1e6) && (spread_bp < 5.0)) || (price <= 1.0);
    static_assert(screen(instrument{50, 30'000, 2}) && !screen(instrument{50, 30'000, 8}) && screen(instrument{0.5, 10, 50}));
    //! [predicates]
    check(!screen(instrument{50, 100, 1}), "an illiquid instrument fails the screen");
    check((price > volume)(instrument{10, 5, 0}) && (spread_bp > 1.0)(instrument{0, 0, 2}), "> between functions and with a constant");
    check((!(price < 1.0))(instrument{2, 0, 0}), "! negates a predicate");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
