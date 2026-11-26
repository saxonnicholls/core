//
//  rewrite_simplifier_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Showcase for <snicholls/core/functional/rewrite.hpp>: an algebraic simplifier written as rules, as a computer
//  algebra system writes one. x + 0 -> x, x * 1 -> x, x * 0 -> 0, x - x -> 0, -(-x) -> x and constant folding are
//  each a rule; rewrite_up applies them everywhere until none fires. The tree is a plain struct with std::vector
//  children, simplified by the compiler in a static_assert as well as at run time. Exits non-zero if a result is not
//  the one documented.
//

#include <snicholls/core/functional/rewrite.hpp>

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

namespace fn = snicholls::fn;

//! [tree]
struct expr {
    enum class kind { num, var, add, sub, mul, neg } k = kind::num;
    long value = 0;
    char name = 0;
    std::vector<expr> args;
    friend constexpr bool operator==(const expr&, const expr&) = default;
};
using kind = expr::kind;

constexpr expr num(long v) { return {kind::num, v, 0, {}}; }
constexpr expr var(char c) { return {kind::var, 0, c, {}}; }
constexpr expr node(kind k, expr a, expr b) { return {k, 0, 0, {std::move(a), std::move(b)}}; }
constexpr expr operator+(expr a, expr b) { return node(kind::add, std::move(a), std::move(b)); }
constexpr expr operator-(expr a, expr b) { return node(kind::sub, std::move(a), std::move(b)); }
constexpr expr operator*(expr a, expr b) { return node(kind::mul, std::move(a), std::move(b)); }
constexpr expr operator-(expr a) { return {kind::neg, 0, 0, {std::move(a)}}; }

// descend: apply g to each child (Uniplate's view of the tree).
constexpr auto children = [](const auto& g, expr e) {
    for (auto& a : e.args) a = g(a);
    return e;
};
//! [tree]

//! [rules]
constexpr bool is(const expr& e, kind k) { return e.k == k; }
constexpr bool is_num(const expr& e, long v) { return e.k == kind::num && e.value == v; }
constexpr auto lhs = [](const expr& e) { return e.args[0]; };
constexpr auto rhs = [](const expr& e) { return e.args[1]; };

constexpr auto add_zero = fn::rules(fn::rule([](const expr& e) { return is(e, kind::add) && is_num(e.args[1], 0); }, lhs),
                                    fn::rule([](const expr& e) { return is(e, kind::add) && is_num(e.args[0], 0); }, rhs));
constexpr auto mul_one = fn::rules(fn::rule([](const expr& e) { return is(e, kind::mul) && is_num(e.args[1], 1); }, lhs),
                                   fn::rule([](const expr& e) { return is(e, kind::mul) && is_num(e.args[0], 1); }, rhs));
constexpr auto mul_zero = fn::rule([](const expr& e) { return is(e, kind::mul) && (is_num(e.args[0], 0) || is_num(e.args[1], 0)); },
                                   [](const expr&) { return num(0); });
constexpr auto sub_self = fn::rule([](const expr& e) { return is(e, kind::sub) && e.args[0] == e.args[1]; }, [](const expr&) { return num(0); });
constexpr auto neg_neg = fn::rule([](const expr& e) { return is(e, kind::neg) && is(e.args[0], kind::neg); },
                                  [](const expr& e) { return e.args[0].args[0]; });
constexpr auto fold = fn::rule(
    [](const expr& e) {
        if (is(e, kind::num) || is(e, kind::var)) return false;
        for (const expr& a : e.args)
            if (!is(a, kind::num)) return false;
        return true;
    },
    [](const expr& e) {
        const long a = e.args[0].value;
        const long b = e.args.size() > 1 ? e.args[1].value : 0;
        return num(e.k == kind::add ? a + b : e.k == kind::sub ? a - b : e.k == kind::mul ? a * b : -a);
    });

// The simplifier: the identities first, then folding; rewrite_up applies it until it fires nowhere.
constexpr auto simplify_rule = fn::rules(add_zero, mul_one, mul_zero, sub_self, neg_neg, fold);
constexpr expr simplify(const expr& e) { return fn::rewrite_up(children, simplify_rule, e); }
//! [rules]

std::string show(const expr& e)
{
    switch (e.k) {
    case kind::num: return std::to_string(e.value);
    case kind::var: return std::string(1, e.name);
    case kind::neg: return "-" + show(e.args[0]);
    case kind::add: return "(" + show(e.args[0]) + " + " + show(e.args[1]) + ")";
    case kind::sub: return "(" + show(e.args[0]) + " - " + show(e.args[1]) + ")";
    case kind::mul: return "(" + show(e.args[0]) + " * " + show(e.args[1]) + ")";
    }
    return "?";
}
} // namespace

int main()
{
    std::printf("snicholls-core functional/rewrite simplifier demo\n");

    //! [simplify]
    // ((x * 1) + (0 * y)) * (2 + 3)  ->  x * 5
    static_assert(simplify((var('x') * num(1) + num(0) * var('y')) * (num(2) + num(3))) == var('x') * num(5));
    // -(-(x - x)) + y * (4 - 3)  ->  y
    static_assert(simplify(-(-(var('x') - var('x'))) + var('y') * (num(4) - num(3))) == var('y'));
    //! [simplify]
    const expr e = (var('x') * num(1) + num(0) * var('y')) * (num(2) + num(3));
    std::printf("  %s  ->  %s\n", show(e).c_str(), show(simplify(e)).c_str());
    check(show(simplify(e)) == "(x * 5)", "the identities and folding leave x * 5");
    check(simplify(num(2) * num(3) + num(4)) == num(10), "a closed term folds to a number");
    check(simplify(var('z')) == var('z'), "a variable is already in normal form");
    check(simplify((var('a') - var('a')) * var('b') + var('c')) == var('c'), "x - x is 0, 0 * b is 0, 0 + c is c");

    //! [one_step]
    // replace applies a rule once, at the top only; replace_repeated keeps going there until it stops.
    const expr top = (var('x') + num(0)) + num(0);
    check(fn::replace_once(top, simplify_rule) == var('x') + num(0), "replace_once: one step at the root");
    const auto r = fn::replace_repeated(top, simplify_rule, 10);
    check(r.value == var('x') && r.iterations == 2 && r.converged, "replace_repeated: two steps to x");
    //! [one_step]

    // transform_up runs a function at every node once, children first: one pass of folding.
    check(fn::transform_up(children, [](const expr& n) { return fn::replace_once(n, fold); }, num(1) + num(2) * num(3)) == num(7),
          "one bottom-up pass folds a nested constant");
    // transform_down: the parent first, so a rename at the top reaches the children it creates.
    const auto rename = [](expr n) {
        if (is(n, kind::var) && n.name == 'x') n.name = 'y';
        return n;
    };
    check(fn::transform_down(children, rename, var('x') * var('x')) == var('y') * var('y'), "transform_down renames x to y everywhere");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
