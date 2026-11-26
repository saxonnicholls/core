//
//  state_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/functional/state.hpp>: a State monad threading cash through a day of fills, a Writer keeping an audit trail, a Reader passing settings down. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/functional/instances.hpp>
#include <snicholls/core/functional/state.hpp>

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

int main()
{
    namespace fn = snicholls::fn;
    std::printf("snicholls-core functional/state demo\n");

    //! [state]
    // A fill costs cash; it reports whether the cash covered it. The state is the cash balance.
    constexpr auto fill = [](double cost) {
        return fn::mbind(fn::get_state, [cost](double cash) {
            return fn::mthen(fn::put_state(cash - cost), fn::pure<fn::state<>>(cash >= cost));
        });
    };
    // A day: two fills, then a sweep that adds interest; the value is whether every fill was covered.
    constexpr auto day = fn::mbind(fill(40.0), [fill](bool a) {
        return fn::mbind(fill(50.0), [a](bool b) {
            return fn::mthen(fn::modify([](double cash) { return cash * 1.01; }), fn::pure<fn::state<>>(a && b));
        });
    });
    static_assert(fn::eval_state(day, 100.0));          // covered
    static_assert(!fn::eval_state(day, 60.0));          // the second fill was not
    static_assert(fn::exec_state(day, 100.0) == 10.0 * 1.01);
    //! [state]
    check(fn::run_state(day, 100.0).first, "a day of fills, run by the compiler");

    //! [writer]
    // Writer: every step adds to a log; here the log is a count of trades (a sum monoid).
    using trades = fn::sum_of<int>;
    constexpr auto trade = [](double px) {
        return fn::mthen(fn::tell(trades{1}), fn::pure<fn::writer<trades>>(px));
    };
    constexpr auto book = fn::mbind(trade(1.08), [trade](double a) { return fn::fmap([a](double b) { return a + b; }, trade(1.09)); });
    static_assert(fn::exec_writer(book).value == 2);
    //! [writer]
    check(fn::run_writer(book).first > 2.16, "two trades logged");

    // Reader: settings passed down without a parameter on every function.
    struct settings { double spread_bp; double floor; };
    constexpr auto widen = [](double px) { return fn::asks([px](const settings& s) { return px * (1 + s.spread_bp / 1e4); }); };
    constexpr auto floored = [](double px) { return fn::asks([px](const settings& s) { return px < s.floor ? s.floor : px; }); };
    constexpr auto quote = fn::kleisli(widen, floored);
    check(fn::run_reader(quote(100.0), settings{25.0, 0.0}) == 100.25, "a Reader pipeline with the settings given once");
    check(fn::run_reader(fn::local([](settings s) { s.floor = 200.0; return s; }, quote(100.0)), settings{25.0, 0.0}) == 200.0,
          "local: the same pipeline under other settings");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
