//
//  match_evaluator_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Showcase for <snicholls/core/functional/match.hpp>: an expression language as an algebraic data type (a
//  std::variant of node kinds in an index arena), evaluated by pattern matching with guards, failures carried as a
//  snicholls::result, and everything run by the compiler as well. Exits non-zero if a result is not the one
//  documented.
//

#include <snicholls/core/functional/match.hpp>
#include <snicholls/core/functional/monad.hpp>

#include <algorithm>
#include <array>
#include <cstdio>
#include <functional>
#include <span>
#include <variant>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

namespace fn = snicholls::fn;
using snicholls::err;
using snicholls::ok;

//! [adt]
// data Expr = Lit Int | Var Char | Neg Expr | Bin Op Expr Expr  -- children are indices into one arena.
enum class op { add, sub, mul, div };
struct lit {
    long v;
};
struct var {
    char name;
};
struct neg {
    int arg;
};
struct bin {
    op o;
    int lhs;
    int rhs;
};
using node = std::variant<lit, var, neg, bin>;

struct binding {
    char name;
    long value;
};

enum class eval_error { divide_by_zero, unbound_variable };
using value = snicholls::result<long, eval_error>;

constexpr long apply(op o, long a, long b)
{
    return o == op::add ? a + b : o == op::sub ? a - b : a * b;
}

// eval :: Expr -> Env -> Either Error Int. Every alternative is handled, or this does not compile.
constexpr value eval(std::span<const node> t, int i, std::span<const binding> env)
{
    return fn::match(
        t[static_cast<std::size_t>(i)],
        [](const lit& l) -> value { return ok(l.v); },
        [&](const var& v) -> value {
            for (const binding& b : env)
                if (b.name == v.name) return ok(b.value);
            return err(eval_error::unbound_variable);
        },
        [&](const neg& n) -> value { return fn::fmap(std::negate<>{}, eval(t, n.arg, env)); },
        fn::guard([](const bin& b) { return b.o == op::div; },
                  [&](const bin& b) -> value {   // the railway: divisor first, then a checked division
                      return fn::mbind(eval(t, b.rhs, env), [&](long d) -> value {
                          if (d == 0) return err(eval_error::divide_by_zero);
                          return fn::fmap([d](long n) { return n / d; }, eval(t, b.lhs, env));
                      });
                  }),
        [&](const bin& b) -> value {   // both sides, the first error wins (an applicative lift)
            return fn::lift([o = b.o](long x, long y) { return apply(o, x, y); })(eval(t, b.lhs, env), eval(t, b.rhs, env));
        });
}
//! [adt]

// (x + 3) * -(y / 2), as an arena: the root is the last node.
constexpr std::array<node, 8> program{node{var{'x'}}, node{lit{3}},          node{bin{op::add, 0, 1}}, node{var{'y'}},
                                      node{lit{2}},   node{bin{op::div, 3, 4}}, node{neg{5}},             node{bin{op::mul, 2, 6}}};
constexpr int root = 7;

// The depth of a tree, by the same kind of match: a fold over the ADT.
constexpr int depth(std::span<const node> t, int i)
{
    return fn::match(t[static_cast<std::size_t>(i)], [](const lit&) { return 1; }, [](const var&) { return 1; },
                     [&](const neg& n) { return 1 + depth(t, n.arg); },
                     [&](const bin& b) { return 1 + std::max(depth(t, b.lhs), depth(t, b.rhs)); });
}
} // namespace

int main()
{
    std::printf("snicholls-core functional/match evaluator demo\n");

    //! [run]
    constexpr std::array<binding, 2> env{{{'x', 4}, {'y', 10}}};
    static_assert(eval(program, root, env).unwrap_unchecked() == (4 + 3) * -(10 / 2));
    static_assert(eval(program, root, std::array<binding, 1>{{{'x', 4}}}).unwrap_err_unchecked() == eval_error::unbound_variable);
    static_assert(depth(program, root) == 4);
    //! [run]
    check(eval(program, root, env).unwrap_unchecked() == -35, "(x + 3) * -(y / 2) at x = 4, y = 10 is -35");

    constexpr std::array<binding, 2> zero_y{{{'x', 4}, {'y', 0}}};
    check(eval(program, root, zero_y).unwrap_unchecked() == 0, "y = 0: 0 / 2 is fine");
    constexpr std::array<node, 3> divide{node{lit{1}}, node{lit{0}}, node{bin{op::div, 0, 1}}};
    static_assert(eval(divide, 2, std::span<const binding>()).unwrap_err_unchecked() == eval_error::divide_by_zero);
    check(eval(divide, 2, {}).is_err(), "1 / 0 is a divide_by_zero error, by the guarded case");
    constexpr std::array<node, 3> both_bad{node{var{'q'}}, node{lit{0}}, node{bin{op::div, 0, 1}}};
    check(eval(both_bad, 2, {}).unwrap_err_unchecked() == eval_error::divide_by_zero,
          "q / 0: the divisor is checked first, so the division error wins");
    constexpr std::array<node, 3> sub{node{var{'q'}}, node{var{'r'}}, node{bin{op::sub, 0, 1}}};
    check(eval(sub, 2, {}).unwrap_err_unchecked() == eval_error::unbound_variable, "q - r: the first error, left to right");
    check(depth(divide, 2) == 2, "depth of 1 / 0 is 2");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
