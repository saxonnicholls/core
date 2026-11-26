//
//  rewrite_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/functional/rewrite.hpp>: rules as values, applied once, to a normal form, and over a tree.
//  Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/functional/rewrite.hpp>

#include <cstdio>
#include <optional>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

// An order quantity normalised by rules: round to the lot, cap at the limit, never below zero.
struct qty {
    int lots;
    int odd;   // shares that are not a whole lot
    friend constexpr bool operator==(const qty&, const qty&) = default;
};

// A tiny pricing language: a number, or a sum or product of terms.
struct term {
    enum class op { num, add, mul } kind = op::num;
    double value = 0;
    std::vector<term> args;
    friend bool operator==(const term&, const term&) = default;
};
term num(double v) { return {term::op::num, v, {}}; }
term add(std::vector<term> a) { return {term::op::add, 0, std::move(a)}; }
term mul(std::vector<term> a) { return {term::op::mul, 0, std::move(a)}; }
} // namespace

int main()
{
    namespace fn = snicholls::fn;
    std::printf("snicholls-core functional/rewrite demo\n");

    //! [rules]
    // Each rule fires or not; `rules` tries them in order; `except` fences one off.
    constexpr auto round_odd = fn::rule([](qty q) { return q.odd >= 50; }, [](qty q) { return qty{q.lots + 1, 0}; });
    constexpr auto drop_odd = fn::rule([](qty q) { return q.odd > 0 && q.odd < 50; }, [](qty q) { return qty{q.lots, 0}; });
    constexpr auto cap = fn::rule([](qty q) { return q.lots > 10; }, [](qty q) { return qty{10, q.odd}; });
    constexpr auto normalise = fn::rules(cap, round_odd, drop_odd);
    static_assert(fn::replace_once(qty{3, 70}, normalise) == qty{4, 0});
    //! [rules]
    check(fn::replace_once(qty{3, 0}, normalise) == qty{3, 0}, "replace_once: nothing fires on a whole number of lots");
    check(fn::except([](qty q) { return q.lots == 0; }, round_odd)(qty{0, 60}) == std::nullopt,
          "except: never round up an order with no lots");
    check(fn::except([](qty q) { return q.lots == 0; }, round_odd)(qty{1, 60}) == std::optional<qty>(qty{2, 0}),
          "except: otherwise the rule fires");

    //! [repeated]
    // Rewrite until no rule fires: 12 lots and 60 shares is first capped, then rounded, then capped again.
    constexpr auto r = fn::replace_repeated(qty{12, 60}, normalise, 10);
    static_assert(r.value == qty{10, 0} && r.iterations == 3 && r.converged);
    //! [repeated]
    check(r.converged, "replace_repeated reaches a normal form in three rewrites");
    const auto looping = fn::replace_repeated(1, fn::rule([](int) { return true; }, [](int x) { return -x; }), 5);
    check(!looping.converged && looping.iterations == 5 && looping.value == -1, "a rule that always fires stops at the limit");

    //! [tree]
    // The tree is the caller's: `children` maps a function over a node's children. Rules fold constants.
    const auto children = [](const auto& g, term t) {
        for (auto& a : t.args) a = g(a);
        return t;
    };
    const auto fold = fn::rule(
        [](const term& t) {
            if (t.kind == term::op::num) return false;
            for (const auto& a : t.args)
                if (a.kind != term::op::num) return false;
            return true;
        },
        [](const term& t) {
            double v = t.kind == term::op::add ? 0 : 1;
            for (const auto& a : t.args) v = t.kind == term::op::add ? v + a.value : v * a.value;
            return num(v);
        });
    const term notional = mul({num(1'000'000), add({num(1), mul({num(0.5), num(0.25)})})});
    const term folded = fn::rewrite_up(children, fold, notional);
    //! [tree]
    check(folded == num(1'125'000), "rewrite_up folds the whole tree to one number");
    check(fn::transform_up(children, [&](const term& t) { return fn::replace_once(t, fold); }, notional) == num(1'125'000),
          "transform_up with replace folds it in one bottom-up pass");
    const auto to_add = [](term t) {
        if (t.kind == term::op::mul) t.kind = term::op::add;
        return t;
    };
    check(fn::rewrite_up(children, fold, fn::transform_down(children, to_add, notional)) == num(1'000'000 + 1 + 0.5 + 0.25),
          "transform_down turns every product into a sum, top first");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
