//
//  instances_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/functional/instances.hpp>: Maybe, Either, products and fixed lists as Functors, Applicatives and Monads. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/functional/instances.hpp>

#include <array>
#include <cstdio>
#include <optional>
#include <tuple>
#include <utility>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}
} // namespace

int main()
{
    namespace fn = snicholls::fn;
    using snicholls::err;
    using snicholls::ok;
    std::printf("snicholls-core functional/instances demo\n");

    //! [maybe]
    // Maybe: a mid needs both sides; the first missing one stops the calculation.
    constexpr auto mid = [](double b, double a) { return (b + a) / 2; };
    constexpr std::optional<double> bid = 1.0850, ask = 1.0852, stale{};
    static_assert(fn::lift_a2(mid, bid, ask).has_value() && !fn::lift_a2(mid, bid, stale).has_value());
    // Either: the error says why.
    enum class why { crossed };
    using quote = snicholls::result<double, why>;
    constexpr auto checked = [](double b) { return b > 1.0 ? quote(ok(b)) : quote(err(why::crossed)); };
    static_assert(fn::mbind(quote(ok(1.2)), checked).is_ok());
    //! [maybe]
    check(fn::lift_a2(mid, bid, ask).value() > 1.0850, "a mid of two present prices");
    check(fn::mbind(quote(ok(0.5)), checked).is_err(), "a failed check is an error, not a value");

    //! [list]
    // The array monad is Haskell's list monad: every scenario through every shock.
    constexpr std::array spot{100.0, 110.0};
    constexpr auto shocked = fn::mbind(spot, [](double s) { return std::array{s * 0.99, s, s * 1.01}; });
    static_assert(shocked.size() == 6);
    // ap: every function applied to every value.
    constexpr auto grid = fn::ap(std::array{+[](double x) { return x + 1; }, +[](double x) { return x * 2; }}, std::array{1.0, 2.0});
    static_assert(grid == std::array{2.0, 3.0, 2.0, 4.0});
    //! [list]
    check(shocked[5] > 111.0, "six shocked spots");
    check(fn::fmap([](double x) { return x * 2; }, std::pair{1.0, 2.0}) == std::pair{2.0, 4.0}, "a pair maps both sides");
    check(fn::bimap([](int x) { return x + 1; }, [](why) { return 0; }, snicholls::result<int, why>(err(why::crossed))).is_err(),
          "bimap over a result maps the error side");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
