//
//  curry_showcase_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Showcase for <snicholls/core/functional/curry.hpp>: a pricing function configured in stages by currying, Haskell's
//  operator sections, partial application from either end, and uncurry flattening a hand-curried chain, all evaluated
//  by the compiler. Exits non-zero on a mismatch.
//

#include <snicholls/core/functional/combinators.hpp>
#include <snicholls/core/functional/curry.hpp>

#include <array>
#include <cstdio>
#include <functional>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

// One function of four arguments, known at different times: the venue at start-up, the fx rate at start of day,
// the client's spread per order, the price per tick.
constexpr double price(double venue_fee, double fx, double spread_bp, double mid)
{
    return mid * fx * (1 + spread_bp / 10'000) + venue_fee;
}
} // namespace

int main()
{
    namespace fn = snicholls::fn;
    std::printf("snicholls-core functional/curry showcase\n");

    //! [stages]
    constexpr auto at_venue = fn::curry(price)(0.25);        // start-up
    constexpr auto today = at_venue(1.10);                    // start of day
    constexpr auto for_client = today(20.0);                  // per order
    static_assert(for_client(100.0) == 100.0 * 1.10 * 1.002 + 0.25);   // per tick
    // Every grouping of the arguments is the same call.
    static_assert(fn::curry(price)(0.25, 1.10)(20.0, 100.0) == for_client(100.0));
    //! [stages]
    check(for_client(200.0) == 200.0 * 1.10 * 1.002 + 0.25, "the same configured function prices another tick");

    //! [sections]
    // Haskell's sections: (2 *), (subtract 1), (100 /), as values to compose.
    constexpr auto twice = fn::lsection(std::multiplies<>{}, 2);
    constexpr auto minus_one = fn::rsection(std::minus<>{}, 1);
    constexpr auto hundred_over = fn::lsection(std::divides<>{}, 100);
    static_assert(fn::pipe(twice, minus_one, hundred_over)(3) == 20);   // 100 / (2 * 3 - 1)
    //! [sections]
    check(fn::compose(minus_one, twice)(10) == 19, "(subtract 1) . (2 *)");

    //! [partial]
    // Bind from the left or the right; an uncurried view of a hand-curried chain.
    constexpr auto discount = [](double rate, double years, double amount) { return amount / (1 + rate * years); };
    constexpr auto quarter = fn::partial(discount, 0.25);
    constexpr auto of_1000 = fn::partial_right(discount, 1000.0);
    static_assert(quarter(2.0, 1500.0) == 1000.0 && of_1000(0.25, 2.0) == quarter(2.0, 1000.0));
    constexpr auto chain = [](int a) { return [a](int b) { return [a, b](int c) { return a * b + c; }; }; };
    static_assert(fn::uncurry(chain)(6, 7, 0) == 42 && fn::uncurry(fn::curry<3>([](int a, int b, int c) { return a * b + c; }))(6, 7, 0) == 42);
    //! [partial]
    check(quarter(1.0, 125.0) == 100.0, "partial: 25% for a year discounts 125 to 100");

    //! [map_with_curried]
    // A curried function mapped over data: one rate applied to a book of amounts.
    constexpr auto scale = fn::curry([](double k, double x) { return k * x; });
    constexpr std::array<double, 3> book{100.0, 250.0, 40.0};
    constexpr auto scaled = [scale](const std::array<double, 3>& amounts) {
        std::array<double, 3> out{};
        const auto by = scale(1.5);
        for (std::size_t i = 0; i < amounts.size(); ++i) out[i] = by(amounts[i]);
        return out;
    }(book);
    static_assert(scaled == std::array{150.0, 375.0, 60.0});
    //! [map_with_curried]
    check(scaled[1] == 375.0, "a curried scale mapped over a book");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
