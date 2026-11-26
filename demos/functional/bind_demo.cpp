//
//  bind_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/functional/bind.hpp>: arguments reordered, repeated and nested with placeholders.
//  Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/functional/bind.hpp>

#include <cstdio>
#include <functional>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

constexpr double pnl(double entry, double exit, double quantity) noexcept { return (exit - entry) * quantity; }
} // namespace

int main()
{
    namespace fn = snicholls::fn;
    using namespace snicholls::fn::placeholders;
    std::printf("snicholls-core functional/bind demo\n");

    //! [bind]
    // A short position's P&L is the long one with entry and exit swapped: reorder with placeholders.
    constexpr auto long_pnl = fn::bind(pnl, _1, _2, _3);
    constexpr auto short_pnl = fn::bind(pnl, _2, _1, _3);
    static_assert(long_pnl(100.0, 110.0, 2.0) == 20.0);
    static_assert(short_pnl(100.0, 110.0, 2.0) == -20.0);
    // A value in the middle: one lot of 100.
    constexpr auto per_lot = fn::bind(pnl, _1, _2, 100.0);
    static_assert(per_lot(1.10, 1.12) > 1.99 && per_lot(1.10, 1.12) < 2.01);
    //! [bind]
    check(short_pnl(50.0, 40.0, 1.0) == 10.0, "a short gains when the price falls");

    //! [nested]
    // A nested bind is called with the same arguments: the exit is the mid of a bid and an ask.
    constexpr auto mid = fn::bind(std::divides<>{}, fn::bind(std::plus<>{}, _2, _3), 2.0);
    constexpr auto mark_to_mid = fn::bind(pnl, _1, mid, 10.0);
    static_assert(mark_to_mid(100.0, 101.0, 103.0) == 20.0);   // (102 - 100) * 10
    //! [nested]
    check(mark_to_mid(100.0, 99.0, 99.0) == -10.0, "marked to a mid of 99");
    check(fn::bind(pnl, fn::arg<1>, fn::arg<1>, 5.0)(7.0) == 0.0, "arg<N> spelled out; the same argument twice");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
