//
//  rewrite.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Rewriting by rules, Wolfram's `/.` and `//.` over values: `rule`, `rules`, `except`, `replace_once`,
///        `replace_repeated`, and over any tree `transform_up`, `transform_down` and `rewrite_up` (Uniplate's
///        `transform` and `rewrite`). All `constexpr`.
/// \ingroup core
///
/// **Overview.** A *rule* is a function that may fire: `T -> std::optional<T>`, the rewritten value or nothing.
/// Rules are built and combined as values, and applied once or until none fires:
///
/// | here                               | what it is                                                     | Wolfram / Haskell                  |
/// | ---------------------------------- | -------------------------------------------------------------- | ---------------------------------- |
/// | `rule(p, f)`                       | fires `f(x)` where `p(x)` holds                                | `x_?p :> f[x]` (`PatternTest`)     |
/// | `rules(r1, r2, ...)`               | the first rule that fires, in the order written                | `{r1, r2, ...}`, `Alternatives`    |
/// | `except(bad, r)`                   | `r`, but never where `bad(x)` holds                            | `Except[bad, r]`                   |
/// | `replace_once(x, r)`               | `r(x)` if it fires, else `x`                                   | `x /. r` (at the top)              |
/// | `replace_repeated(x, r, limit)`    | `r` until it no longer fires, at most `limit` times            | `x //. r`                          |
/// | `transform_up(children, f, x)`     | `f` at every node, children first                              | Uniplate `transform`               |
/// | `transform_down(children, f, x)`   | `f` at every node, the node first                              | Uniplate `descend`-based top-down  |
/// | `rewrite_up(children, r, x)`       | `r` everywhere until it fires nowhere (a normal form)          | Uniplate `rewrite`, Wolfram `//.`  |
///
/// \snippet rewrite_demo.cpp rules
///
/// A tree is any value type whose immediate children can be mapped: `children(g, node)` returns `node` with
/// `g` applied to each child (Uniplate's `descend`). The tree's representation is the caller's: a variant with
/// `std::vector` children, an index arena, a fixed-depth aggregate. Nothing here knows the node types:
///
/// \snippet rewrite_demo.cpp tree
///
/// The idea comes from symbolic-expression-templates (SXT), whose `replace` and `rewrite` do this over expression
/// *types* at compile time (a rule that fires changes the C++ type), with blanks, sequence blanks and
/// commutative matching. Those are algebra-specific and stay in SXT; this is the value-level core of it, for
/// simplifying an order, normalising a schedule, or folding constants in a small language.
///
/// **Relationships.** `snicholls::fixed_point` and `fixed_point_result` (`iterate.hpp`: iteration to `==`, where
/// this stops when no rule fires), `functional/match.hpp` (one match, no iteration), `functional/recursion.hpp`
/// (folds over lists), `fn::fix`. Group: \ref core.
///
/// **Implementations / variants.** `rule`, `rules` and `except` are closures holding their parts (trivially
/// copyable when the parts are). The tree functions recurse once per level of the tree.
///
/// **Guarantees.** `constexpr`. A rule is called with a `const T&`. `replace_repeated` makes at most `limit`
/// rewrites and says whether it reached a normal form. `transform_up` and `rewrite_up` visit every node;
/// `rewrite_up` re-normalises what a rule produces, so the result has no node where a rule fires.
/// \pre For `rewrite_up`, the rules terminate (each firing makes progress by some measure); otherwise it does not
/// return (at compile time, the constant evaluator's step limit stops it).
///
/// **Errors.** None of their own; a rule's or `children`'s exception propagates.
///
/// **Examples.** \snippet rewrite_demo.cpp repeated
///
/// **Performance.** One call per rule tried per node; not benchmarked.
///
/// **Tests and demos.** `tests/functional/rewrite_test.cpp`, `demos/functional/rewrite_demo.cpp`,
/// `demos/functional/rewrite_simplifier_demo.cpp` (an algebraic simplifier to a normal form).
///
/// **Dependencies.** `<cstddef>`, `<functional>`, `<optional>`, `<type_traits>`, `<utility>`,
/// `snicholls/core/iterate.hpp` and `functional/combinators.hpp`.
///
/// **Known issues and gotchas.** The tree functions recurse on the stack, one frame per level: fine for the
/// expression trees they are for, not for a list of a million nodes (fold those with `functional/recursion.hpp`).
/// Rules see values, not patterns: matching a shape is the rule's predicate (often `fn::match` or a variant
/// `std::get_if`).
///
/// **Status.** Experimental, new in 0.27.0.

#ifndef snicholls_core_functional_rewrite_hpp
#define snicholls_core_functional_rewrite_hpp

#include <snicholls/core/functional/combinators.hpp>
#include <snicholls/core/iterate.hpp>

#include <cstddef>
#include <functional>
#include <optional>
#include <type_traits>
#include <utility>

namespace snicholls::fn {

    namespace detail {

        struct rule_op {
            template <std::size_t... I, class S, class X, class T = std::remove_cvref_t<X>>
            static constexpr std::optional<T> call(std::index_sequence<I...>, S&& s, const X& x)
            {
                if (std::invoke(detail::part<0>(s), x)) return T(std::invoke(detail::part<1>(s), x));
                return std::nullopt;
            }
        };

        struct rules_op {
            template <std::size_t... I, class S, class X, class T = std::remove_cvref_t<X>>
            static constexpr std::optional<T> call(std::index_sequence<I...>, S&& s, const X& x)
            {
                std::optional<T> out;
                static_cast<void>(((out = std::invoke(detail::part<I>(s), x)).has_value() || ...));
                return out;
            }
        };

        struct except_op {
            template <std::size_t... I, class S, class X, class T = std::remove_cvref_t<X>>
            static constexpr std::optional<T> call(std::index_sequence<I...>, S&& s, const X& x)
            {
                if (std::invoke(detail::part<0>(s), x)) return std::nullopt;
                return std::invoke(detail::part<1>(s), x);
            }
        };

        struct rules_fn {
            template <class... R>
                requires(sizeof...(R) > 0)
            constexpr auto operator()(R&&... r) const noexcept(nothrow_store_v<R...>)
            {
                return detail::make<rules_op>(std::forward<R>(r)...);
            }
        };

        struct replace_once_fn {
            template <class T, class R>
            constexpr T operator()(const T& x, R&& r) const
            {
                std::optional<T> y = std::invoke(r, x);
                return y.has_value() ? std::move(*y) : x;
            }
        };

        struct replace_repeated_fn {
            template <class T, class R>
            constexpr fixed_point_result<T> operator()(T x, R&& r, std::size_t limit) const
            {
                fixed_point_result<T> out{std::move(x), 0, false};
                for (; out.iterations < limit; ++out.iterations) {
                    std::optional<T> y = std::invoke(r, std::as_const(out.value));
                    if (!y.has_value()) break;
                    out.value = std::move(*y);
                }
                out.converged = out.iterations < limit || !std::invoke(r, std::as_const(out.value)).has_value();
                return out;
            }
        };

        struct transform_up_fn {
            template <class C, class F, class T>
            constexpr T operator()(const C& children, const F& f, const T& x) const
            {
                return std::invoke(f, std::invoke(children, [&](const T& c) -> T { return (*this)(children, f, c); }, x));
            }
        };

        struct transform_down_fn {
            template <class C, class F, class T>
            constexpr T operator()(const C& children, const F& f, const T& x) const
            {
                return std::invoke(children, [&](const T& c) -> T { return (*this)(children, f, c); }, T(std::invoke(f, x)));
            }
        };

        struct rewrite_up_fn {
            template <class C, class R, class T>
            constexpr T operator()(const C& children, const R& r, const T& x) const
            {
                T y = std::invoke(children, [&](const T& c) -> T { return (*this)(children, r, c); }, x);
                std::optional<T> z = std::invoke(r, std::as_const(y));
                return z.has_value() ? (*this)(children, r, *z) : y;
            }
        };

    } // namespace detail

    /// \brief `rule(p, f)`: the rule that rewrites `x` to `f(x)` where `p(x)` holds, and does not fire elsewhere.
    /// \ingroup core
    inline constexpr detail::binary_maker<detail::rule_op> rule{};
    /// \brief `rules(r1, r2, ...)`: the rule that fires as the first of `r1, r2, ...` that fires.
    /// \ingroup core
    inline constexpr detail::rules_fn rules{};
    /// \brief `except(bad, r)`: the rule `r`, except where `bad(x)` holds.
    /// \ingroup core
    inline constexpr detail::binary_maker<detail::except_op> except{};
    /// \brief `replace_once(x, r)`: `r(x)` if the rule fires, else `x` (Wolfram's `/.` at the top level). Not
    ///        `fn::replace`, which is Haskell's `<$`.
    /// \ingroup core
    inline constexpr detail::replace_once_fn replace_once{};
    /// \brief `replace_repeated(x, r, limit)`: `r` applied until it no longer fires, at most `limit` times, as a
    ///        `fixed_point_result` (the value, the rewrites made, whether it reached a normal form).
    /// \ingroup core
    inline constexpr detail::replace_repeated_fn replace_repeated{};
    /// \brief `transform_up(children, f, x)`: `f` applied at every node of a tree, children before parents.
    /// \ingroup core
    inline constexpr detail::transform_up_fn transform_up{};
    /// \brief `transform_down(children, f, x)`: `f` applied at every node of a tree, parents before children
    ///        (the children are those of `f`'s result).
    /// \ingroup core
    inline constexpr detail::transform_down_fn transform_down{};
    /// \brief `rewrite_up(children, r, x)`: a tree's normal form under the rule `r`: `r` applied bottom-up, and again
    ///        to whatever it produces, until it fires at no node.
    /// \ingroup core
    inline constexpr detail::rewrite_up_fn rewrite_up{};

} // namespace snicholls::fn

#endif /* snicholls_core_functional_rewrite_hpp */
