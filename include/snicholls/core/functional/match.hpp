//
//  match.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::fn::match`: pattern matching over values, `std::variant` and `snicholls::result`, with guards,
///        exhaustiveness checked at compile time, and destructuring of tuples and plain aggregates.
/// \ingroup core
///
/// **Overview.** `match(value, cases...)` runs the first case, in the order written, that accepts the
/// value and whose guard (if it has one) holds:
///
/// \snippet match_demo.cpp variant
///
/// | case                       | accepts                                                          | Haskell / OCaml               |
/// | -------------------------- | ---------------------------------------------------------------- | ----------------------------- |
/// | a callable `[](T x) {...}` | what it can be called with (a variant's alternative, the value)  | `T x -> ...`                  |
/// | `guard(p, case)`           | what `case` accepts, when `p` of it holds                        | `x \| p x -> ...`, `when`     |
/// | `on_ok(f)`, `on_err(f)`    | a result's value, a result's error                               | `Right v`, `Left e`, `Ok v`   |
/// | `unpack(f)`                | a tuple, pair or array, `f` of its elements                      | `(a, b) -> ...`               |
/// | `destructure(f)`           | a plain aggregate, `f` of its members                            | `{x; y}`, record patterns     |
/// | `otherwise(f)`             | anything (`[](const auto&) {...}`)                               | `_ -> ...`                    |
///
/// A `std::variant` is matched by its alternative and a `result` by its side; anything else is matched
/// as itself. Exhaustiveness is a compile-time check: every alternative (both sides of a result, or
/// the value) must be accepted by some case *without* a guard, or the match does not compile, with
/// "snicholls::fn::match: not exhaustive". The cases that come after it can never run for that
/// alternative, as in Haskell, and are allowed. The match's result type is that of the first unguarded
/// case for the first alternative; every case that runs must return something convertible to it.
///
/// Sum types are `std::variant` (with `match` in place of `std::visit` plus `overloaded`), product
/// types are plain aggregates (destructured by `destructure` through `snicholls::tie`), Maybe is
/// `std::optional` and Either is `snicholls::result`.
///
/// **Relationships.** `snicholls::first_match` (the same first-accepting rule, without guards or
/// exhaustiveness), `snicholls::overloaded` and `std::visit`, `snicholls::tie` (`reflect/struct_tuple.hpp`),
/// `functional/combinators.hpp`. Group: \ref core.
///
/// **Implementations / variants.** One: the cases are tried in order by overloads, never an
/// `if constexpr` chain.
///
/// **Guarantees.** `constexpr`. The value is passed to the cases as an lvalue (of the constness given);
/// nothing is copied. A case that does not accept an alternative is never instantiated with it.
///
/// **Errors.** "snicholls::fn::match: not exhaustive" when an alternative has no unguarded case. Not
/// `noexcept`: the guards and cases may throw.
///
/// **Examples.** \snippet match_demo.cpp result
///
/// **Performance.** A variant's `std::visit`, a result's branch, then the guards in order; not
/// benchmarked.
///
/// **Tests and demos.** `tests/functional/match_test.cpp`, `tests/functional/match_patterns_test.cpp`,
/// `demos/functional/match_demo.cpp`, `demos/functional/match_evaluator_demo.cpp` (an expression evaluator).
///
/// **Dependencies.** `<cstddef>`, `<functional>`, `<tuple>`, `<type_traits>`, `<utility>`, `<variant>`,
/// `snicholls/core/result.hpp`, `snicholls/core/reflect/struct_tuple.hpp` (heavier than it looks: the
/// generated member binders) and `functional/combinators.hpp`.
///
/// **Known issues and gotchas.** "Accepts" is `std::is_invocable`, so implicit conversions count, as in
/// `first_match`: a `[](double)` case accepts an `int` alternative. Constrain a case
/// (`[](std::same_as<int> auto)`) or order the exact cases first.
///
/// **Status.** Experimental, new in 0.27.0.

#ifndef snicholls_core_functional_match_hpp
#define snicholls_core_functional_match_hpp

#include <snicholls/core/functional/combinators.hpp>
#include <snicholls/core/meta/concepts.hpp>
#include <snicholls/core/reflect/struct_tuple.hpp>
#include <snicholls/core/result.hpp>

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace snicholls::fn {

    namespace detail {

        // A result's two sides as distinct alternatives.
        template <class T>
        struct ok_side {
            T& value;
        };
        template <class E>
        struct err_side {
            E& error;
        };

        template <class F>
        struct plain_case {
            F f;
            static constexpr bool guarded = false;
            template <class X>
            static constexpr bool accepts = std::is_invocable_v<const F&, X&>;
            template <class X>
            constexpr bool test(X&) const noexcept { return true; }
            template <class X>
            constexpr decltype(auto) operator()(X& x) const { return std::invoke(f, x); }
        };

        template <class F, bool Ok, class X>
        inline constexpr bool side_accepts = false;
        template <class F, class T>
        inline constexpr bool side_accepts<F, true, ok_side<T>> = std::is_invocable_v<const F&, T&>;
        template <class F, class E>
        inline constexpr bool side_accepts<F, false, err_side<E>> = std::is_invocable_v<const F&, E&>;

        template <class F, bool Ok>
        struct side_case {
            F f;
            static constexpr bool guarded = false;
            template <class X>
            static constexpr bool accepts = side_accepts<F, Ok, X>;
            template <class X>
            constexpr bool test(X&) const noexcept { return true; }
            template <class T>
            constexpr decltype(auto) operator()(ok_side<T>& x) const { return std::invoke(f, x.value); }
            template <class E>
            constexpr decltype(auto) operator()(err_side<E>& x) const { return std::invoke(f, x.error); }
            template <class T>
            static constexpr T& inner(ok_side<T>& x) noexcept { return x.value; }
            template <class E>
            static constexpr E& inner(err_side<E>& x) noexcept { return x.error; }
        };

        template <class P, class C>
        struct guarded_case {
            P p;
            C c;
            static constexpr bool guarded = true;
            template <class X>
            static constexpr bool accepts = C::template accepts<X>;
            template <class X>
            constexpr bool test(X& x) const { return c.test(x) && static_cast<bool>(std::invoke(p, inner(x))); }
            template <class X>
            constexpr decltype(auto) operator()(X& x) const { return c(x); }

        private:
            // The guard sees what the case's function sees: a result's value or error, else the value.
            template <class X>
            static constexpr auto& inner(X& x) noexcept
            {
                return x;
            }
            template <class T>
            static constexpr T& inner(ok_side<T>& x) noexcept
            {
                return x.value;
            }
            template <class E>
            static constexpr E& inner(err_side<E>& x) noexcept
            {
                return x.error;
            }
        };

        template <class C>
        struct as_case_t {
            using type = plain_case<C>;
        };
        template <class F>
        struct as_case_t<plain_case<F>> {
            using type = plain_case<F>;
        };
        template <class F, bool Ok>
        struct as_case_t<side_case<F, Ok>> {
            using type = side_case<F, Ok>;
        };
        template <class P, class C>
        struct as_case_t<guarded_case<P, C>> {
            using type = guarded_case<P, C>;
        };

        template <class C>
        using as_case = typename as_case_t<std::decay_t<C>>::type;

        template <class C>
        constexpr as_case<C> to_case(C&& c)
        {
            return as_case<C>{std::forward<C>(c)};
        }
        template <class F>
        constexpr plain_case<F> to_case(plain_case<F> c)
        {
            return c;
        }

        // The result type for X: the first unguarded accepting case's.
        template <class X, class... Cs>
        struct match_result {
            static constexpr std::size_t index =
                snicholls::detail::first_true<(Cs::template accepts<X> && !Cs::guarded)...>();
            static constexpr bool exhaustive = index < sizeof...(Cs);
        };

        template <class R, class X>
        constexpr R dispatch(X&)
        {
            throw std::logic_error("snicholls::fn::match: no case matched"); // unreachable: exhaustiveness is checked
        }

        template <class R, class X, class C, class... Cs>
        constexpr R dispatch(X& x, const C& c, const Cs&... cs);

        // Three cases that cannot overlap: the case does not accept X; it does, unguarded; it does, guarded.
        template <class R, class X, class C, class... Cs>
            requires(!C::template accepts<X>)
        constexpr R try_case(X& x, const C&, const Cs&... cs)
        {
            return detail::dispatch<R>(x, cs...);
        }
        template <class R, class X, class C, class... Cs>
            requires(C::template accepts<X> && !C::guarded)
        constexpr R try_case(X& x, const C& c, const Cs&...)
        {
            return c(x);
        }
        template <class R, class X, class C, class... Cs>
            requires(C::template accepts<X> && C::guarded)
        constexpr R try_case(X& x, const C& c, const Cs&... cs)
        {
            if (c.test(x)) return c(x);
            return detail::dispatch<R>(x, cs...);
        }

        template <class R, class X, class C, class... Cs>
        constexpr R dispatch(X& x, const C& c, const Cs&... cs)
        {
            return detail::try_case<R>(x, c, cs...);
        }

        template <class X, class... Cs>
        constexpr decltype(auto) match_one(X& x, const Cs&... cs)
        {
            using info = match_result<X, Cs...>;
            static_assert(info::exhaustive,
                          "snicholls::fn::match: not exhaustive: a value or an alternative has no case without a guard");
            if constexpr (info::exhaustive) {
                using C = meta::at_t<meta::type_list<Cs...>, info::index>;
                using R = std::remove_cvref_t<decltype(std::declval<const C&>()(x))>;
                return detail::dispatch<R>(x, cs...);
            }
        }

        // What is matched: a variant's alternative, a result's side, or the value.
        struct match_fn {
            template <class V, class... C>
                requires specialisation_of<V, std::variant>
            constexpr decltype(auto) operator()(V&& v, C&&... c) const
            {
                return [&](const auto&... cs) -> decltype(auto) {
                    // The first alternative's result type, which every other alternative's converts to (as documented)
                    using R = decltype(detail::match_one(std::get<0>(v), cs...));
                    return std::visit([&](auto& alt) -> R { return detail::match_one(alt, cs...); }, v);
                }(detail::to_case(std::forward<C>(c))...);
            }

            template <class V, class... C>
                requires specialisation_of<V, snicholls::result>
            constexpr decltype(auto) operator()(V&& r, C&&... c) const
            {
                return [&](const auto&... cs) -> decltype(auto) {
                    using T = std::remove_reference_t<decltype(r.unwrap_unchecked())>;
                    using E = std::remove_reference_t<decltype(r.unwrap_err_unchecked())>;
                    using R = std::common_type_t<decltype(detail::match_one(std::declval<ok_side<T>&>(), cs...)),
                                                 decltype(detail::match_one(std::declval<err_side<E>&>(), cs...))>;
                    if (r.is_ok()) {
                        ok_side<T> side{r.unwrap_unchecked()};
                        return static_cast<R>(detail::match_one(side, cs...));
                    }
                    err_side<E> side{r.unwrap_err_unchecked()};
                    return static_cast<R>(detail::match_one(side, cs...));
                }(detail::to_case(std::forward<C>(c))...);
            }

            template <class V, class... C>
                requires(!specialisation_of<V, std::variant> && !specialisation_of<V, snicholls::result>)
            constexpr decltype(auto) operator()(V&& v, C&&... c) const
            {
                return [&](const auto&... cs) -> decltype(auto) { return detail::match_one(v, cs...); }(
                           detail::to_case(std::forward<C>(c))...);
            }
        };

        struct guard_fn {
            template <class P, class C>
            constexpr guarded_case<std::decay_t<P>, as_case<C>> operator()(P&& p, C&& c) const
            {
                return {std::forward<P>(p), detail::to_case(std::forward<C>(c))};
            }
        };

        template <bool Ok>
        struct side_fn {
            template <class F>
            constexpr side_case<std::decay_t<F>, Ok> operator()(F&& f) const
            {
                return {std::forward<F>(f)};
            }
        };

        // Destructuring a plain aggregate: its members as a tuple of references, then f.
        template <class F>
        struct unpack_struct {
            F f;
            template <class S>
                requires(!tuple_like<S>) && plain_struct<std::remove_const_t<S>>
            constexpr auto operator()(S& s) const SNICHOLLS_FN_RETURNS(
                detail::apply_tuple(f, snicholls::tie(s), tuple_indices<decltype(snicholls::tie(s))>{}))
        };

        struct destructure_fn {
            template <class F>
            constexpr unpack_struct<std::decay_t<F>> operator()(F&& f) const
            {
                return {std::forward<F>(f)};
            }
        };

    } // namespace detail

    /// \brief `match(value, cases...)`: the first case that accepts the value (a variant's alternative, a result's side)
    ///        and whose guard holds; exhaustive by a compile-time check.
    /// \ingroup core
    inline constexpr detail::match_fn match{};

    /// \brief `guard(p, case)`: `case`, only when `p` holds of what the case receives (Haskell's `| p`, OCaml's `when`).
    /// \ingroup core
    inline constexpr detail::guard_fn guard{};

    /// \brief `on_ok(f)`: the case for a result's value.
    /// \ingroup core
    inline constexpr detail::side_fn<true> on_ok{};

    /// \brief `on_err(f)`: the case for a result's error.
    /// \ingroup core
    inline constexpr detail::side_fn<false> on_err{};

    /// \brief `unpack(f)`: the case for a tuple, pair or array, `f` called with its elements.
    /// \ingroup core
    inline constexpr detail::unary_maker<detail::apply_op> unpack{};

    /// \brief `destructure(f)`: the case for a plain aggregate, `f` called with its members (through `snicholls::tie`).
    /// \ingroup core
    inline constexpr detail::destructure_fn destructure{};

    /// \brief `otherwise(f)`: a case that accepts anything, Haskell's `_`; `f` takes the value (`[](const auto&) {...}`).
    /// \ingroup core
    inline constexpr detail::identity_fn otherwise{};

} // namespace snicholls::fn

#endif /* snicholls_core_functional_match_hpp */
