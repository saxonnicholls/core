//
//  pointwise.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::fn::pointwise{f}`: a function that is a value of an algebra, added, scaled, compared and
///        combined with `&&` / `||` point by point. All `constexpr`.
/// \ingroup core
///
/// **Overview.** A C++ function can be called; it cannot be added to another function to make a third. Wrapped
/// in `pointwise`, it can: `(f + g)(x)` is `f(x) + g(x)`, `(2.0 * f)(x)` is `2.0 * f(x)`, `(f < g)(x)` is
/// `f(x) < g(x)`, and `(p && q)(x)` is `p(x) && q(x)`, short-circuited. Any operand that is not a `pointwise` is
/// a constant. The result is a `pointwise` again, so expressions nest, and the arguments may be many:
///
/// \snippet pointwise_demo.cpp payoff
///
/// | here                                 | `(... )(x...)`                         | Haskell                     |
/// | ------------------------------------ | -------------------------------------- | --------------------------- |
/// | `f + g`, `f - g`, `f * g`, `f / g`   | `f(x...) op g(x...)`                   | `liftA2 (+) f g`, `Num (a -> b)` |
/// | `-f`, `!p`                           | `-f(x...)`, `!p(x...)`                 | `negate . f`, `not . p`     |
/// | `f < g`, `f <= g`, `f > g`, `f >= g` | `f(x...) op g(x...)`: a predicate       | `liftA2 (<) f g`            |
/// | `p && q`, `p \|\| q`                 | `p(x...) && q(x...)`, short-circuited  | `liftA2 (&&) p q`           |
/// | `f + 1.0`, `2.0 * f`                 | a constant operand                     | `const 1.0`                 |
///
/// It is SXT's operators-as-types idea (symbolic-expression-templates' `operators.hpp`, an algebra of
/// differential operators) with the symbolic part left behind: here the elements are ordinary callables and
/// the operations are evaluated, not simplified. Composition is `fn::compose`; repeated application is
/// `snicholls::nest`. Use it to assemble a payoff from legs, a score from weighted features, or a filter from
/// predicates; write the lambda when the expression is used once.
///
/// **Relationships.** `fn::compose`, `fn::constant`, `fn::on`, `fn::lift_a2` (the same through the Reader
/// applicative, `functional/state.hpp`), `snicholls::nest`. Group: \ref core.
///
/// **Implementations / variants.** One class, `pointwise<F>`, and one node per operation, a `fn` closure holding its
/// operands by value: trivially copyable when they are, and with no storage for a stateless operand (`f + g` of two
/// stateless lambdas is one byte; `2.5 * f` is one `double`).
///
/// **Guarantees.** `constexpr`. A call's `noexcept` and type are exactly those of the expression it evaluates. Each
/// operand is called once per call with the same arguments, as lvalues (so an argument is never moved from twice);
/// `&&` and `||` call the right operand only when needed.
///
/// **Errors.** None of their own.
///
/// **Examples.** \snippet pointwise_demo.cpp predicates
///
/// **Performance.** The calls inline to the hand-written expression; not benchmarked separately (see
/// `benchmarks/functional/combinators_benchmark.cpp` for the composition of closures).
///
/// **Tests and demos.** `tests/functional/pointwise_test.cpp`, `demos/functional/pointwise_demo.cpp`.
///
/// **Dependencies.** `<concepts>`, `<cstddef>`, `<functional>`, `<type_traits>`, `<utility>` and `functional/combinators.hpp`.
///
/// **Known issues and gotchas.** `==` and `!=` are not lifted: `f == g` stays a question about the functions,
/// never a new one. A `pointwise` holding a lambda that captures by reference refers to what it captured.
/// Inside an fsm table (`using namespace snicholls::fsm`), `&&`, `||` and `!` on two `pointwise` guards are ambiguous
/// with the table's own guard algebra: combine them outside the table and use the result as one guard
/// (`demos/fsm/state_machine_railway_demo.cpp`). And enter it in the table behind a lambda that takes the event
/// (`[](const authorise& a) { return authorisable(a); }`): the table reads a guard's parameters from its signature, and
/// Clang 18 crashes deducing them through `pointwise`'s generic call operator.
///
/// **Status.** Experimental, new in 0.27.0.

#ifndef snicholls_core_functional_pointwise_hpp
#define snicholls_core_functional_pointwise_hpp

#include <snicholls/core/functional/combinators.hpp>

#include <concepts>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

namespace snicholls::fn {

    /// \brief A function as an element of an algebra: arithmetic, comparisons and logic on it act point by point.
    /// \ingroup core
    /// \tparam F The function's type.
    template <class F>
    struct pointwise {
        F f; ///< The function.

        // Constrained with requires rather than by SFINAE in the return type: fsm/state_machine.hpp reads a generic
        // guard's signature as `&F::template operator()<Event>`, and Apple Clang 17 crashes on that address when the
        // substitution fails in a trailing return type (a pointwise guard on a later row of a table).
        /// \brief `f(a...)`.
        /// \param a The arguments.
        /// \return What `f` returns.
        template <class... A>
            requires std::invocable<const F&, A...>
        constexpr decltype(auto) operator()(A&&... a) const noexcept(std::is_nothrow_invocable_v<const F&, A...>)
        {
            return std::invoke(f, std::forward<A>(a)...);
        }
    };

    /// \brief Deduction guide: `pointwise{f}` holds a copy of `f`.
    template <class F>
    pointwise(F) -> pointwise<F>;

    namespace detail {

        template <class T>
        inline constexpr bool is_pointwise_v = false;
        template <class F>
        inline constexpr bool is_pointwise_v<pointwise<F>> = true;

        template <class T>
        concept pointwise_ = is_pointwise_v<std::remove_cvref_t<T>>;

        template <class L, class R>
        concept pointwise_operands = pointwise_<L> || pointwise_<R>;

        // An operand as a function: a pointwise's function, or a constant.
        template <pointwise_ P>
        constexpr auto as_function(P&& p) noexcept(nothrow_store_v<decltype((std::forward<P>(p).f))>)
        {
            return std::forward<P>(p).f;
        }
        template <class S>
            requires(!pointwise_<S>)
        constexpr auto as_function(S&& s) noexcept(nothrow_store_v<S>)
        {
            return fn::constant(std::forward<S>(s));
        }

        template <class Op>
        struct pointwise_binary_op {
            template <std::size_t... I, class S, class... A>
            static constexpr auto call(std::index_sequence<I...>, S&& s, A&&... a)
                SNICHOLLS_FN_RETURNS(Op{}(std::invoke(detail::part<0>(s), a...), std::invoke(detail::part<1>(s), a...)))
        };

        template <class Op>
        struct pointwise_unary_op {
            template <std::size_t... I, class S, class... A>
            static constexpr auto call(std::index_sequence<I...>, S&& s, A&&... a)
                SNICHOLLS_FN_RETURNS(Op{}(std::invoke(detail::part<0>(s), std::forward<A>(a)...)))
        };

        template <bool And>
        struct pointwise_logical_op {
            template <std::size_t... I, class S, class... A>
            static constexpr auto call(std::index_sequence<I...>, S&& s, A&&... a)
                noexcept(noexcept(static_cast<bool>(std::invoke(detail::part<0>(s), a...))) &&
                         noexcept(static_cast<bool>(std::invoke(detail::part<1>(s), a...))))
                    -> decltype(static_cast<bool>(std::invoke(detail::part<0>(s), a...)) &&
                                static_cast<bool>(std::invoke(detail::part<1>(s), a...)))
            {
                if (static_cast<bool>(std::invoke(detail::part<0>(s), a...)) != And) return !And;
                return static_cast<bool>(std::invoke(detail::part<1>(s), a...));
            }
        };

        template <class Op, class... P>
        constexpr auto lift(P&&... p)
        {
            return pointwise{detail::make<Op>(detail::as_function(std::forward<P>(p))...)};
        }

    } // namespace detail

    /// \brief `(l + r)(x...)`: `l(x...) + r(x...)`, a non-`pointwise` operand a constant.
    /// \param l The left operand. \param r The right operand. \return A `pointwise`.
    template <class L, class R>
        requires detail::pointwise_operands<L, R>
    constexpr auto operator+(L&& l, R&& r) { return detail::lift<detail::pointwise_binary_op<std::plus<>>>(std::forward<L>(l), std::forward<R>(r)); }

    /// \brief `(l - r)(x...)`: `l(x...) - r(x...)`.
    /// \param l The left operand. \param r The right operand. \return A `pointwise`.
    template <class L, class R>
        requires detail::pointwise_operands<L, R>
    constexpr auto operator-(L&& l, R&& r) { return detail::lift<detail::pointwise_binary_op<std::minus<>>>(std::forward<L>(l), std::forward<R>(r)); }

    /// \brief `(l * r)(x...)`: `l(x...) * r(x...)`.
    /// \param l The left operand. \param r The right operand. \return A `pointwise`.
    template <class L, class R>
        requires detail::pointwise_operands<L, R>
    constexpr auto operator*(L&& l, R&& r) { return detail::lift<detail::pointwise_binary_op<std::multiplies<>>>(std::forward<L>(l), std::forward<R>(r)); }

    /// \brief `(l / r)(x...)`: `l(x...) / r(x...)`.
    /// \param l The left operand. \param r The right operand. \return A `pointwise`.
    template <class L, class R>
        requires detail::pointwise_operands<L, R>
    constexpr auto operator/(L&& l, R&& r) { return detail::lift<detail::pointwise_binary_op<std::divides<>>>(std::forward<L>(l), std::forward<R>(r)); }

    /// \brief `(l < r)(x...)`: `l(x...) < r(x...)`.
    /// \param l The left operand. \param r The right operand. \return A `pointwise` predicate.
    template <class L, class R>
        requires detail::pointwise_operands<L, R>
    constexpr auto operator<(L&& l, R&& r) { return detail::lift<detail::pointwise_binary_op<std::less<>>>(std::forward<L>(l), std::forward<R>(r)); }

    /// \brief `(l <= r)(x...)`: `l(x...) <= r(x...)`.
    /// \param l The left operand. \param r The right operand. \return A `pointwise` predicate.
    template <class L, class R>
        requires detail::pointwise_operands<L, R>
    constexpr auto operator<=(L&& l, R&& r) { return detail::lift<detail::pointwise_binary_op<std::less_equal<>>>(std::forward<L>(l), std::forward<R>(r)); }

    /// \brief `(l > r)(x...)`: `l(x...) > r(x...)`.
    /// \param l The left operand. \param r The right operand. \return A `pointwise` predicate.
    template <class L, class R>
        requires detail::pointwise_operands<L, R>
    constexpr auto operator>(L&& l, R&& r) { return detail::lift<detail::pointwise_binary_op<std::greater<>>>(std::forward<L>(l), std::forward<R>(r)); }

    /// \brief `(l >= r)(x...)`: `l(x...) >= r(x...)`.
    /// \param l The left operand. \param r The right operand. \return A `pointwise` predicate.
    template <class L, class R>
        requires detail::pointwise_operands<L, R>
    constexpr auto operator>=(L&& l, R&& r) { return detail::lift<detail::pointwise_binary_op<std::greater_equal<>>>(std::forward<L>(l), std::forward<R>(r)); }

    /// \brief `(l && r)(x...)`: `l(x...) && r(x...)`, `r` called only when `l` holds.
    /// \param l The left operand. \param r The right operand. \return A `pointwise` predicate.
    template <class L, class R>
        requires detail::pointwise_operands<L, R>
    constexpr auto operator&&(L&& l, R&& r) { return detail::lift<detail::pointwise_logical_op<true>>(std::forward<L>(l), std::forward<R>(r)); }

    /// \brief `(l || r)(x...)`: `l(x...) || r(x...)`, `r` called only when `l` does not hold.
    /// \param l The left operand. \param r The right operand. \return A `pointwise` predicate.
    template <class L, class R>
        requires detail::pointwise_operands<L, R>
    constexpr auto operator||(L&& l, R&& r) { return detail::lift<detail::pointwise_logical_op<false>>(std::forward<L>(l), std::forward<R>(r)); }

    /// \brief `(-p)(x...)`: `-p(x...)`.
    /// \param p The operand. \return A `pointwise`.
    template <detail::pointwise_ P>
    constexpr auto operator-(P&& p) { return detail::lift<detail::pointwise_unary_op<std::negate<>>>(std::forward<P>(p)); }

    /// \brief `(!p)(x...)`: `!p(x...)`.
    /// \param p The operand. \return A `pointwise` predicate.
    template <detail::pointwise_ P>
    constexpr auto operator!(P&& p) { return detail::lift<detail::pointwise_unary_op<std::logical_not<>>>(std::forward<P>(p)); }

} // namespace snicholls::fn

#endif /* snicholls_core_functional_pointwise_hpp */
