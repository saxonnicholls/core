//
//  state.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief The Identity, Const, Reader, Writer and State monads as `constexpr` value types, with their Functor,
///        Applicative and Monad instances.
/// \ingroup core
///
/// **Overview.**
///
/// | type                 | holds                                  | run                                  | operations                                    |
/// | -------------------- | -------------------------------------- | ------------------------------------ | --------------------------------------------- |
/// | `identity_of<A>`     | a value                                | `.value`                             | (the trivial monad)                           |
/// | `const_of<C>`        | a value; `fmap` leaves it alone        | `.value`                             | Applicative when `C` is a monoid              |
/// | `reader<F>`          | `F: env -> a`                          | `run_reader(m, env)`                 | `ask`, `asks(f)`, `local(f, m)`               |
/// | `writer<W, A>`       | a value and a log `W` (a monoid)       | `run_writer(m)`, `exec_writer(m)`    | `tell(w)`                                     |
/// | `state<F>`           | `F: s -> std::pair<a, s>`              | `run_state`, `eval_state`, `exec_state` | `get_state`, `put_state(s)`, `modify(f)`, `gets(f)` |
///
/// \snippet state_demo.cpp state
///
/// Each is a value: a `state` is a function from a state to a result and the next state, built by
/// `mbind` and the operations, and run when given the initial state, by the compiler in a `static_assert`
/// or at run time. Nothing is mutated and nothing is allocated. The carrier keys for `pure` /
/// `mreturn` are the types with their defaults: `pure<state<>>(x)`, `pure<reader<>>(x)`,
/// `pure<writer<W>>(x)`, `pure<identity_of<>>(x)`, `pure<const_of<C>>(x)`.
///
/// Use State for a calculation that threads a value through many steps (a cash balance through a day's
/// fills, a random generator's seed); Reader for configuration passed down a call tree; Writer for an
/// audit trail combined with a monoid. Write the loop when it is one loop.
///
/// **Relationships.** `functional/typeclasses.hpp` (`fmap`, `ap`, `mbind`, `mthen`), `functional/monad.hpp`
/// (`fold_m`, `replicate_m`), `functional/combinators.hpp`. Group: \ref core.
///
/// **Implementations / variants.** One each. `reader` and `state` hold one function; every operation
/// returns a new one wrapping the old (the types grow with the program, as lambdas do).
///
/// **Guarantees.** `constexpr`. The functions are held by value; `mbind(m, k)` holds copies of `m` and `k`.
/// Trivially copyable when the functions are. Running is `noexcept` when the functions are.
///
/// **Errors.** None of their own.
///
/// **Examples.** \snippet state_demo.cpp writer
///
/// **Performance.** After inlining, a `state` program is the code it describes.
/// `benchmarks/functional/lazy_benchmark.cpp`, a day of three fills as a State program against the
/// hand-written loop, fairness-checked: 1.82 ns each (macOS x86-64 Xeon W-3245, Apple Clang 17, Release, 2026-11-26, 5 repetitions).
///
/// **Tests and demos.** `tests/functional/state_test.cpp`, `tests/functional/laws_test.cpp`,
/// `demos/functional/state_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<functional>`, `<type_traits>`, `<utility>`, `<variant>` (for
/// `std::monostate`, the unit value) and `functional/typeclasses.hpp`.
///
/// **Known issues and gotchas.** A long chain of `mbind`s nests one closure per step, which the
/// compiler instantiates; for thousands of steps, use `fold_m` over a range instead of a written chain.
/// The unit value is `std::monostate`.
///
/// **Status.** Experimental, new in 0.27.0.

#ifndef snicholls_core_functional_state_hpp
#define snicholls_core_functional_state_hpp

#include <snicholls/core/functional/typeclasses.hpp>

#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>
#include <variant>

namespace snicholls::fn {

    /// \brief The Identity monad: a value and nothing else.
    /// \ingroup core
    /// \tparam A The value's type; `void` only as the carrier key of `pure`.
    template <class A = void>
    struct identity_of {
        A value; ///< The value.
        /// \brief Equal when the values are.
        friend constexpr bool operator==(const identity_of&, const identity_of&) = default;
    };

    /// \brief The Const functor: a value that `fmap` leaves alone (an Applicative when it is a monoid).
    /// \ingroup core
    /// \tparam C The value's type.
    template <class C>
    struct const_of {
        C value; ///< The value.
        /// \brief Equal when the values are.
        friend constexpr bool operator==(const const_of&, const const_of&) = default;
    };

    /// \brief The Reader monad: a function of an environment.
    /// \ingroup core
    /// \tparam F `env -> a`; `void` only as the carrier key.
    template <class F = void>
    struct reader {
        F run; ///< The function.
    };

    /// \brief Deduction guide: `reader{f}` holds a copy of `f`.
    /// \tparam F The function's type.
    template <class F>
    reader(F) -> reader<F>;

    /// \brief The Writer monad: a value and a log, the logs combined with `W`'s monoid.
    /// \ingroup core
    /// \tparam W The log's type, a monoid.
    /// \tparam A The value's type; `void` only as the carrier key.
    template <class W, class A = void>
    struct writer {
        A value; ///< The value.
        W log;   ///< The log.
        /// \brief Equal when the values and the logs are.
        friend constexpr bool operator==(const writer&, const writer&) = default;
    };

    /// \brief The State monad: a function from a state to a value and the next state, `s -> std::pair<a, s>`.
    /// \ingroup core
    /// \tparam F The function's type; `void` only as the carrier key.
    template <class F = void>
    struct state {
        F run; ///< The function.
    };

    /// \brief Deduction guide: `state{f}` holds a copy of `f`.
    /// \tparam F The function's type.
    template <class F>
    state(F) -> state<F>;

    namespace detail {

        struct run_fn {
            template <class M, class X>
            constexpr auto operator()(M&& m, X&& x) const SNICHOLLS_FN_RETURNS(std::invoke(std::forward<M>(m).run, std::forward<X>(x)))
        };

        struct eval_state_fn {
            template <class M, class S>
            constexpr auto operator()(M&& m, S&& s) const
                SNICHOLLS_FN_RETURNS(std::invoke(std::forward<M>(m).run, std::forward<S>(s)).first)
        };

        struct exec_state_fn {
            template <class M, class S>
            constexpr auto operator()(M&& m, S&& s) const
                SNICHOLLS_FN_RETURNS(std::invoke(std::forward<M>(m).run, std::forward<S>(s)).second)
        };

        struct get_state_run {
            template <class S>
            constexpr std::pair<S, S> operator()(const S& s) const noexcept(std::is_nothrow_copy_constructible_v<S>)
            {
                return {s, s};
            }
        };

        template <class F>
        struct gets_run {
            F f;
            template <class S>
            constexpr auto operator()(const S& s) const
                SNICHOLLS_FN_RETURNS(std::pair<std::remove_cvref_t<std::invoke_result_t<const F&, const S&>>, S>(std::invoke(f, s), s))
        };

        template <class T>
        struct put_run {
            T next;
            template <class S>
            constexpr std::pair<std::monostate, T> operator()(const S&) const noexcept(std::is_nothrow_copy_constructible_v<T>)
            {
                return {std::monostate{}, next};
            }
        };

        template <class F>
        struct modify_run {
            F f;
            template <class S>
            constexpr auto operator()(const S& s) const
                SNICHOLLS_FN_RETURNS(std::pair<std::monostate, S>(std::monostate{}, std::invoke(f, s)))
        };

        template <class M, class G>
        struct state_map_run {
            M m;
            G g;
            template <class S>
            constexpr auto operator()(S&& s) const
            {
                auto r = std::invoke(m.run, std::forward<S>(s));
                return std::pair{std::invoke(g, std::move(r.first)), std::move(r.second)};
            }
        };

        template <class M, class K>
        struct state_bind_run {
            M m;
            K k;
            template <class S>
            constexpr auto operator()(S&& s) const
            {
                auto r = std::invoke(m.run, std::forward<S>(s));
                return std::invoke(std::invoke(k, std::move(r.first)).run, std::move(r.second));
            }
        };

        template <class A>
        struct state_pure_run {
            A a;
            template <class S>
            constexpr std::pair<A, std::remove_cvref_t<S>> operator()(S&& s) const
            {
                return {a, std::forward<S>(s)};
            }
        };

        template <class MF, class MA>
        struct state_ap_run {
            MF mf;
            MA ma;
            template <class S>
            constexpr auto operator()(S&& s) const
            {
                auto f = std::invoke(mf.run, std::forward<S>(s));
                auto a = std::invoke(ma.run, std::move(f.second));
                return std::pair{std::invoke(f.first, std::move(a.first)), std::move(a.second)};
            }
        };

        template <class M, class G>
        struct reader_map_run {
            M m;
            G g;
            template <class E>
            constexpr auto operator()(const E& e) const SNICHOLLS_FN_RETURNS(std::invoke(g, std::invoke(m.run, e)))
        };

        template <class M, class K>
        struct reader_bind_run {
            M m;
            K k;
            template <class E>
            constexpr auto operator()(const E& e) const
                SNICHOLLS_FN_RETURNS(std::invoke(std::invoke(k, std::invoke(m.run, e)).run, e))
        };

        template <class MF, class MA>
        struct reader_ap_run {
            MF mf;
            MA ma;
            template <class E>
            constexpr auto operator()(const E& e) const
                SNICHOLLS_FN_RETURNS(std::invoke(std::invoke(mf.run, e), std::invoke(ma.run, e)))
        };

        template <class F, class M>
        struct local_run {
            F f;
            M m;
            template <class E>
            constexpr auto operator()(const E& e) const SNICHOLLS_FN_RETURNS(std::invoke(m.run, std::invoke(f, e)))
        };

        struct put_state_fn {
            template <class T>
            constexpr state<put_run<std::decay_t<T>>> operator()(T&& s) const
                noexcept(std::is_nothrow_constructible_v<std::decay_t<T>, T>)
            {
                return {{std::forward<T>(s)}};
            }
        };

        struct modify_fn {
            template <class F>
            constexpr state<modify_run<std::decay_t<F>>> operator()(F&& f) const noexcept(nothrow_store_v<F>)
            {
                return {{std::forward<F>(f)}};
            }
        };

        struct gets_fn {
            template <class F>
            constexpr state<gets_run<std::decay_t<F>>> operator()(F&& f) const noexcept(nothrow_store_v<F>)
            {
                return {{std::forward<F>(f)}};
            }
        };

        struct asks_fn {
            template <class F>
            constexpr reader<std::decay_t<F>> operator()(F&& f) const noexcept(nothrow_store_v<F>)
            {
                return {std::forward<F>(f)};
            }
        };

        struct local_fn {
            template <class F, class M>
            constexpr reader<local_run<std::decay_t<F>, std::remove_cvref_t<M>>> operator()(F&& f, M&& m) const
                noexcept(nothrow_store_v<F, M>)
            {
                return {{std::forward<F>(f), std::forward<M>(m)}};
            }
        };

        struct tell_fn {
            template <class W>
            constexpr writer<std::decay_t<W>, std::monostate> operator()(W&& w) const
                noexcept(std::is_nothrow_constructible_v<std::decay_t<W>, W>)
            {
                return {std::monostate{}, std::forward<W>(w)};
            }
        };

        struct run_writer_fn {
            template <class W, class A>
            constexpr std::pair<A, W> operator()(const writer<W, A>& m) const { return {m.value, m.log}; }
        };

        struct exec_writer_fn {
            template <class W, class A>
            constexpr W operator()(const writer<W, A>& m) const { return m.log; }
        };

    } // namespace detail

    /// \brief `run_state(m, s)`: the State `m` run from `s`, `std::pair{value, final state}`. Also `run_reader(m, env)`.
    /// \ingroup core
    inline constexpr detail::run_fn run_state{};

    /// \brief `run_reader(m, env)`: the Reader `m` run in `env`.
    /// \ingroup core
    inline constexpr detail::run_fn run_reader{};

    /// \brief `eval_state(m, s)`: the value of `run_state(m, s)`.
    /// \ingroup core
    inline constexpr detail::eval_state_fn eval_state{};

    /// \brief `exec_state(m, s)`: the final state of `run_state(m, s)`.
    /// \ingroup core
    inline constexpr detail::exec_state_fn exec_state{};

    /// \brief `get_state`: the State whose value is the state (Haskell's `get`).
    /// \ingroup core
    inline constexpr state<detail::get_state_run> get_state{};

    /// \brief `put_state(s)`: the State that replaces the state with `s` (Haskell's `put`); its value is `std::monostate`.
    /// \ingroup core
    inline constexpr detail::put_state_fn put_state{};

    /// \brief `modify(f)`: the State that applies `f` to the state.
    /// \ingroup core
    inline constexpr detail::modify_fn modify{};

    /// \brief `gets(f)`: the State whose value is `f(state)`.
    /// \ingroup core
    inline constexpr detail::gets_fn gets{};

    /// \brief `ask`: the Reader whose value is the environment.
    /// \ingroup core
    inline constexpr reader<detail::identity_fn> ask{};

    /// \brief `asks(f)`: the Reader whose value is `f(environment)`.
    /// \ingroup core
    inline constexpr detail::asks_fn asks{};

    /// \brief `local(f, m)`: the Reader `m` run in the environment `f(env)`.
    /// \ingroup core
    inline constexpr detail::local_fn local{};

    /// \brief `tell(w)`: the Writer that logs `w`; its value is `std::monostate`.
    /// \ingroup core
    inline constexpr detail::tell_fn tell{};

    /// \brief `run_writer(m)`: `std::pair{value, log}`.
    /// \ingroup core
    inline constexpr detail::run_writer_fn run_writer{};

    /// \brief `exec_writer(m)`: the log.
    /// \ingroup core
    inline constexpr detail::exec_writer_fn exec_writer{};

#ifndef SNICHOLLS_DOXYGEN

    // ---- Identity -------------------------------------------------------------------------------------------------------

    template <class A>
    struct functor_instance<identity_of<A>> {
        static constexpr bool defined = true;
        template <class F, class M>
        static constexpr auto fmap(F&& f, M&& m) SNICHOLLS_FN_RETURNS(
            identity_of<std::remove_cvref_t<std::invoke_result_t<F, decltype((std::forward<M>(m).value))>>>{
                std::invoke(std::forward<F>(f), std::forward<M>(m).value)})
    };
    template <class A>
    struct applicative_instance<identity_of<A>> {
        static constexpr bool defined = true;
        template <class X>
        static constexpr identity_of<std::decay_t<X>> pure(X&& x) { return {std::forward<X>(x)}; }
        template <class MF, class MA>
        static constexpr auto ap(MF&& mf, MA&& ma)
            SNICHOLLS_FN_RETURNS(functor_instance<identity_of<A>>::fmap(std::forward<MF>(mf).value, std::forward<MA>(ma)))
    };
    template <class A>
    struct monad_instance<identity_of<A>> {
        static constexpr bool defined = true;
        template <class M, class K>
        static constexpr auto bind(M&& m, K&& k) SNICHOLLS_FN_RETURNS(std::invoke(std::forward<K>(k), std::forward<M>(m).value))
    };

    // ---- Const ------------------------------------------------------------------------------------------------------------

    template <class C>
    struct functor_instance<const_of<C>> {
        static constexpr bool defined = true;
        template <class F, class M>
        static constexpr const_of<C> fmap(F&&, M&& m) { return std::forward<M>(m); }
    };
    template <class C>
        requires monoid<C>
    struct applicative_instance<const_of<C>> {
        static constexpr bool defined = true;
        template <class X>
        static constexpr const_of<C> pure(X&&) { return {mempty<C>}; }
        template <class MF, class MA>
        static constexpr const_of<C> ap(MF&& mf, MA&& ma) { return {fn::combine(mf.value, ma.value)}; }
    };

    // ---- Reader -----------------------------------------------------------------------------------------------------------

    template <class F>
    struct functor_instance<reader<F>> {
        static constexpr bool defined = true;
        template <class G, class M>
        static constexpr auto fmap(G&& g, M&& m)
        {
            return reader<detail::reader_map_run<std::remove_cvref_t<M>, std::decay_t<G>>>{{std::forward<M>(m), std::forward<G>(g)}};
        }
    };
    template <class F>
    struct applicative_instance<reader<F>> {
        static constexpr bool defined = true;
        template <class X>
        static constexpr auto pure(X&& x)
        {
            return reader{detail::constant_fn{}(std::forward<X>(x))};
        }
        template <class MF, class MA>
        static constexpr auto ap(MF&& mf, MA&& ma)
        {
            return reader<detail::reader_ap_run<std::remove_cvref_t<MF>, std::remove_cvref_t<MA>>>{{std::forward<MF>(mf), std::forward<MA>(ma)}};
        }
    };
    template <class F>
    struct monad_instance<reader<F>> {
        static constexpr bool defined = true;
        template <class M, class K>
        static constexpr auto bind(M&& m, K&& k)
        {
            return reader<detail::reader_bind_run<std::remove_cvref_t<M>, std::decay_t<K>>>{{std::forward<M>(m), std::forward<K>(k)}};
        }
    };

    // ---- Writer -----------------------------------------------------------------------------------------------------------

    template <class W, class A>
    struct functor_instance<writer<W, A>> {
        static constexpr bool defined = true;
        template <class F, class M, class R = std::remove_cvref_t<std::invoke_result_t<F, decltype((std::declval<M>().value))>>>
        static constexpr writer<W, R> fmap(F&& f, M&& m)
        {
            return {std::invoke(std::forward<F>(f), std::forward<M>(m).value), std::forward<M>(m).log};
        }
    };
    template <class W, class A>
        requires monoid<W>
    struct applicative_instance<writer<W, A>> {
        static constexpr bool defined = true;
        template <class X>
        static constexpr writer<W, std::decay_t<X>> pure(X&& x) { return {std::forward<X>(x), mempty<W>}; }
        template <class MF, class MA,
                  class R = std::remove_cvref_t<std::invoke_result_t<decltype((std::declval<MF>().value)), decltype((std::declval<MA>().value))>>>
        static constexpr writer<W, R> ap(MF&& mf, MA&& ma)
        {
            return {std::invoke(mf.value, ma.value), fn::combine(mf.log, ma.log)};
        }
    };
    template <class W, class A>
        requires monoid<W>
    struct monad_instance<writer<W, A>> {
        static constexpr bool defined = true;
        template <class M, class K>
        static constexpr auto bind(M&& m, K&& k)
        {
            auto r = std::invoke(std::forward<K>(k), m.value);
            using R = decltype(r.value);
            return writer<W, R>{std::move(r.value), fn::combine(m.log, r.log)};
        }
    };

    // ---- State ------------------------------------------------------------------------------------------------------------

    template <class F>
    struct functor_instance<state<F>> {
        static constexpr bool defined = true;
        template <class G, class M>
        static constexpr auto fmap(G&& g, M&& m)
        {
            return state<detail::state_map_run<std::remove_cvref_t<M>, std::decay_t<G>>>{{std::forward<M>(m), std::forward<G>(g)}};
        }
    };
    template <class F>
    struct applicative_instance<state<F>> {
        static constexpr bool defined = true;
        template <class X>
        static constexpr state<detail::state_pure_run<std::decay_t<X>>> pure(X&& x)
        {
            return {{std::forward<X>(x)}};
        }
        template <class MF, class MA>
        static constexpr auto ap(MF&& mf, MA&& ma)
        {
            return state<detail::state_ap_run<std::remove_cvref_t<MF>, std::remove_cvref_t<MA>>>{{std::forward<MF>(mf), std::forward<MA>(ma)}};
        }
    };
    template <class F>
    struct monad_instance<state<F>> {
        static constexpr bool defined = true;
        template <class M, class K>
        static constexpr auto bind(M&& m, K&& k)
        {
            return state<detail::state_bind_run<std::remove_cvref_t<M>, std::decay_t<K>>>{{std::forward<M>(m), std::forward<K>(k)}};
        }
    };

#endif

} // namespace snicholls::fn

#endif /* snicholls_core_functional_state_hpp */
