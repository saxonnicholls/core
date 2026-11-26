//
//  typeclasses_parser_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/functional/typeclasses.hpp>: parser combinators over std::string_view. A parser joins
//  the Functor, Applicative and Alternative classes, and fmap, lift_a2 and alt build an order-ticket parser from
//  pieces, run by the compiler. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/functional/curry.hpp>
#include <snicholls/core/functional/typeclasses.hpp>

#include <cstdio>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

//! [parser]
// A parser is a function from the input to a value and the rest, or nothing.
template <class F = void>
struct parser {
    F run;
};
template <class F>
parser(F) -> parser<F>;

template <class T>
using parsed = std::optional<std::pair<T, std::string_view>>;
} // namespace

namespace snicholls::fn {
template <class F>
struct functor_instance<parser<F>> {
    static constexpr bool defined = true;
    template <class G, class P>
    static constexpr auto fmap(G g, P p)
    {
        return parser{[g, p](std::string_view s) {
            auto r = p.run(s);
            using T = std::remove_cvref_t<decltype(g(r->first))>;
            return r ? parsed<T>(std::pair{g(r->first), r->second}) : parsed<T>();
        }};
    }
};
template <class F>
struct applicative_instance<parser<F>> {
    static constexpr bool defined = true;
    template <class X>
    static constexpr auto pure(X x)
    {
        return parser{[x](std::string_view s) { return parsed<X>(std::pair{x, s}); }};
    }
    template <class PF, class PX>
    static constexpr auto ap(PF pf, PX px)   // run the function's parser, then the value's on what is left
    {
        return parser{[pf, px](std::string_view s) {
            auto f = pf.run(s);
            using T = std::remove_cvref_t<decltype(f->first(px.run(s)->first))>;
            if (!f) return parsed<T>();
            auto x = px.run(f->second);
            return x ? parsed<T>(std::pair{f->first(x->first), x->second}) : parsed<T>();
        }};
    }
};
template <class F>
struct alternative_instance<parser<F>> {
    static constexpr bool defined = true;
    template <class A, class B>
    static constexpr auto alt(A a, B b)   // the first that parses
    {
        return parser{[a, b](std::string_view s) {
            auto r = a.run(s);
            return r ? r : b.run(s);
        }};
    }
};
} // namespace snicholls::fn

namespace {
namespace fn = snicholls::fn;

// The pieces: a literal, a three-letter currency, an unsigned integer.
constexpr auto literal(std::string_view word)
{
    return parser{[word](std::string_view s) {
        return s.starts_with(word) ? parsed<std::string_view>(std::pair{word, s.substr(word.size())}) : parsed<std::string_view>();
    }};
}
constexpr auto currency = parser{[](std::string_view s) {
    const bool ok = s.size() >= 3 && s[0] >= 'A' && s[0] <= 'Z' && s[1] >= 'A' && s[1] <= 'Z' && s[2] >= 'A' && s[2] <= 'Z';
    return ok ? parsed<std::string_view>(std::pair{s.substr(0, 3), s.substr(3)}) : parsed<std::string_view>();
}};
constexpr auto quantity = parser{[](std::string_view s) {
    long n = 0;
    std::size_t i = 0;
    for (; i < s.size() && s[i] >= '0' && s[i] <= '9'; ++i) n = n * 10 + (s[i] - '0');
    return i > 0 ? parsed<long>(std::pair{n, s.substr(i)}) : parsed<long>();
}};

// Sequencing that keeps the left or the right value: Haskell's <* and *>.
constexpr auto keep_left = [](auto a, auto b) { return fn::lift_a2([](auto x, auto) { return x; }, a, b); };
constexpr auto keep_right = [](auto a, auto b) { return fn::lift_a2([](auto, auto y) { return y; }, a, b); };

struct ticket {
    bool buy;
    long amount;
    std::string_view base;
    std::string_view quote;
};

// "BUY 1000000 EUR/USD": side, amount, pair. Alternative chooses the side; Applicative strings the rest together.
constexpr auto side = fn::alt(fn::replace(true, literal("BUY ")), fn::replace(false, literal("SELL ")));
constexpr auto pair_p = fn::lift_a2([](std::string_view b, std::string_view q) { return std::pair{b, q}; },
                                    keep_left(currency, literal("/")), currency);
constexpr auto ticket_p = fn::ap(fn::lift_a2([](bool buy, long amount) {
                                     return [buy, amount](std::pair<std::string_view, std::string_view> p) {
                                         return ticket{buy, amount, p.first, p.second};
                                     };
                                 }, side, keep_left(quantity, literal(" "))),
                                 pair_p);
//! [parser]
} // namespace

int main()
{
    std::printf("snicholls-core functional/typeclasses parser demo\n");

    //! [parse]
    constexpr auto t = ticket_p.run("BUY 1000000 EUR/USD");
    static_assert(t && t->first.buy && t->first.amount == 1'000'000 && t->first.base == "EUR" && t->first.quote == "USD");
    static_assert(t->second.empty());
    constexpr auto s = ticket_p.run("SELL 250 GBP/JPY rest");
    static_assert(s && !s->first.buy && s->first.quote == "JPY" && s->second == " rest");
    static_assert(!ticket_p.run("HOLD 5 EUR/USD"));        // neither side parses
    static_assert(!ticket_p.run("BUY lots EUR/USD"));      // no quantity
    //! [parse]
    check(t.has_value() && t->first.amount == 1'000'000, "BUY 1000000 EUR/USD parsed by the compiler");
    check(!ticket_p.run("BUY 5 EURUSD"), "a pair without a slash is refused");
    check(keep_right(literal("x"), quantity).run("x42")->first == 42, "*> keeps the right value");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
