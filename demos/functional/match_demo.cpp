//
//  match_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/functional/match.hpp>: order events matched as a sum type with guards, and an order result matched by side. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/functional/match.hpp>

#include <concepts>
#include <cstdio>
#include <utility>
#include <variant>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}
} // namespace

namespace {
struct fill { double price; long quantity; };
struct cancel { long id; };
struct reject { int code; };
enum class order_error { insufficient_margin, market_closed };
} // namespace

int main()
{
    namespace fn = snicholls::fn;
    using snicholls::err;
    using snicholls::ok;
    std::printf("snicholls-core functional/match demo\n");

    //! [variant]
    // An order event is one of three; the match is checked for exhaustiveness by the compiler.
    using event = std::variant<fill, cancel, reject>;
    constexpr auto notional = [](const event& e) {
        return fn::match(e,
                         fn::guard([](const fill& f) { return f.quantity > 1'000'000; },
                                   fn::destructure([](double, long) { return -1.0; })),        // too large: flagged
                         fn::destructure([](double px, long qty) { return px * static_cast<double>(qty); }),
                         [](const cancel&) { return 0.0; },
                         fn::otherwise([](const auto&) { return 0.0; }));                     // anything else
    };
    static_assert(notional(event{fill{1.25, 1000}}) == 1250.0);
    static_assert(notional(event{fill{1.25, 5'000'000}}) == -1.0);
    static_assert(notional(event{reject{42}}) == 0.0);
    //! [variant]
    check(notional(event{cancel{7}}) == 0.0, "a cancel has no notional");

    //! [result]
    // A result is matched by side; a guard can look at the value.
    using placed = snicholls::result<long, order_error>;
    constexpr auto status = [](const placed& r) {
        return fn::match(r, fn::guard([](long id) { return id == 0; }, fn::on_ok([](long) { return 'T'; })),   // a test order
                         fn::on_ok([](long) { return 'P'; }),                                                 // placed
                         fn::guard([](order_error e) { return e == order_error::market_closed; }, fn::on_err([](order_error) { return 'C'; })),
                         fn::on_err([](order_error) { return 'R'; }));                                        // rejected
    };
    static_assert(status(ok(17L)) == 'P' && status(ok(0L)) == 'T');
    static_assert(status(err(order_error::market_closed)) == 'C' && status(err(order_error::insufficient_margin)) == 'R');
    //! [result]
    check(status(ok(5L)) == 'P', "an order placed");
    check(fn::match(std::pair{2, 3}, fn::unpack([](int a, int b) { return a * b; })) == 6, "a pair destructured");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
