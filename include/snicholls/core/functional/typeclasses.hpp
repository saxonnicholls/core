//
//  typeclasses.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Haskell's type classes as C++20 concepts with instance traits: Semigroup, Monoid, Functor, Applicative,
///        Monad, Alternative, Bifunctor and Profunctor, and the functions over them, all `constexpr`.
/// \ingroup core
///
/// **Overview.** A type class is a set of operations a type supports, with laws. Here each class is an
/// *instance trait*, a class template a type specialises to join, and a *concept* that says whether it
/// has; the operations are function objects that dispatch to the instance:
///
/// | class        | instance trait                  | operations here                                            | Haskell                         |
/// | ------------ | ------------------------------- | ---------------------------------------------------------- | ------------------------------- |
/// | Semigroup    | `semigroup_instance<T>`         | `combine(a, b)`, `sconcat(a, b, ...)`                      | `<>`, `sconcat`                 |
/// | Monoid       | `monoid_instance<T>`            | `mempty<T>()`, `mconcat(seq)`                                | `mempty`, `mconcat`             |
/// | Functor      | `functor_instance<M>`           | `fmap(f, m)`, `replace(x, m)`                              | `fmap`, `<$`                    |
/// | Applicative  | `applicative_instance<M>`       | `pure<M>(x)`, `ap(mf, mx)`, `lift_a2(f, ma, mb)`           | `pure`, `<*>`, `liftA2`         |
/// | Monad        | `monad_instance<M>`             | `mbind(m, k)`, `and_then(k, m)`, `mthen(ma, mb)`, `join(mm)`, `kleisli(f, g)`, `kleisli_left(g, f)`, `mreturn<M>(x)` | `>>=`, `=<<`, `>>`, `join`, `>=>`, `<=<`, `return` |
/// | Alternative  | `alternative_instance<M>`       | `alt(a, b)`, `mzero<M>`                                    | `<\|>`, `empty` (MonadPlus: `mplus`, `mzero`) |
/// | Bifunctor    | `bifunctor_instance<P>`         | `bimap(f, g, p)`, `first(f, p)`, `second(g, p)`            | `bimap`, `first`, `second`      |
/// | Profunctor   | (functions)                     | `dimap(f, g, h)`, `lmap(f, h)`, `rmap(g, h)`, `contramap`  | `dimap`, `lmap`, `rmap`         |
///
/// The monoid wrappers `sum_of`, `product_of`, `min_of`, `max_of`, `first_of`, `last_of`, `all_of`,
/// `any_of` and `endo` choose which monoid a number, an optional or a function is meant to be, as
/// Haskell's `Sum`, `Product`, `Min`, `Max`, `First`, `Last`, `All`, `Any` and `Endo` do.
///
/// The instances for the standard library's and core's types (`std::optional`, `snicholls::result`,
/// `std::tuple`, `std::pair`, `std::array`, `std::basic_string`, `std::vector`) are in
/// `functional/instances.hpp`; the Identity, Const, Reader, Writer and State monads in
/// `functional/state.hpp`. A type of your own joins a class by specialising its trait in
/// `namespace snicholls::fn`, with `static constexpr bool defined = true;` and the operations:
///
/// \snippet typeclasses_demo.cpp instance
///
/// Use these when a calculation is generic over the carrier (written once for optionals, results and
/// parsers), or when the laws are what you are relying on (a monoid folded in parallel and merged);
/// call the carrier's own members (`std::optional::value_or`, `result::map`) when the carrier is fixed.
///
/// **Relationships.** `functional/instances.hpp` (the instances), `functional/monad.hpp` (`lift`,
/// `sequence`, `traverse`, the monad helpers), `functional/list.hpp` (Foldable: `fold_map`, `sum`,
/// ...), `functional/state.hpp`, `functional/combinators.hpp`. Group: \ref core.
///
/// **Implementations / variants.** A trait per class, so an instance can be partial (a result has
/// `alt` but no `mzero`; an `endo` combines with an `endo` of another type). `mreturn` is `pure`;
/// MonadPlus is Alternative (`mplus` is `alt`).
///
/// **Guarantees.** `constexpr` wherever the instance is. Every operation is a function object, so it
/// can be passed, composed and curried. `noexcept` follows the instance's operation exactly (each
/// dispatch is `noexcept(noexcept(instance call))`). The laws are the instance's promise; for core's
/// instances they are tested at compile time (`tests/functional/laws_test.cpp`): Functor identity
/// and composition, Monad left and right identity and associativity, Monoid identity and associativity.
///
/// **Errors.** An operation on a type with no instance is not invocable (`std::is_invocable_v` false).
///
/// **Examples.** \snippet typeclasses_demo.cpp monoid
///
/// A parser over `std::string_view` joining Functor, Applicative and Alternative, and an order-ticket
/// parser built from pieces with `fmap`, `lift_a2`, `ap` and `alt`, run by the compiler:
///
/// \snippet typeclasses_parser_demo.cpp parse
///
/// **Performance.** Each operation is the instance's call, inlined; not benchmarked separately.
///
/// **Tests and demos.** `tests/functional/typeclasses_test.cpp`, `tests/functional/laws_test.cpp`,
/// `demos/functional/typeclasses_demo.cpp`.
///
/// **Dependencies.** `<concepts>`, `<cstddef>`, `<functional>`, `<limits>`, `<optional>`, `<type_traits>`,
/// `<utility>`, `functional/combinators.hpp`, `functional/curry.hpp`, `functional/fold.hpp`.
///
/// **Known issues and gotchas.**
/// - `pure<M>` and `mzero<M>` need the carrier named, as Haskell infers it from the context:
///   `pure<std::optional<int>>(3)`. The value type in `M` is only a key; the result holds `decay_t` of the argument.
/// - A `std::variant` is not a Functor: nothing marks one alternative as the value. Use `match`
///   (`functional/match.hpp`) or `std::visit`.
/// - `endo` of different function types combine into a third type, so `endo` is a semigroup only in
///   that looser sense; `mconcat` over a range of `endo` needs one function type.
///
/// **Status.** Experimental, new in 0.27.0.

#ifndef snicholls_core_functional_typeclasses_hpp
#define snicholls_core_functional_typeclasses_hpp

#include <snicholls/core/functional/combinators.hpp>
#include <snicholls/core/functional/curry.hpp>
#include <snicholls/core/functional/fold.hpp>

#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

namespace snicholls::fn {

    // ---- the instance traits --------------------------------------------------------------------------------------

    /// \brief Semigroup instance: specialise with `defined = true` and `static constexpr T combine(const T&, const T&)`.
    /// \ingroup core
    /// \tparam T The type, unqualified.
    template <class T>
    struct semigroup_instance {};

    /// \brief Monoid instance: specialise with `defined = true` and `static constexpr T empty()`; `T` is also a semigroup.
    /// \ingroup core
    /// \tparam T The type, unqualified.
    template <class T>
    struct monoid_instance {};

    /// \brief Functor instance: `defined = true` and `static constexpr auto fmap(F&& f, M&& m)`.
    /// \ingroup core
    /// \tparam M The carrier, unqualified (`std::optional<int>`).
    template <class M>
    struct functor_instance {};

    /// \brief Applicative instance: `defined = true`, `static constexpr auto pure(A&& a)`, `static constexpr auto ap(MF&& mf, MA&& ma)`.
    /// \ingroup core
    /// \tparam M The carrier, unqualified.
    template <class M>
    struct applicative_instance {};

    /// \brief Monad instance: `defined = true` and `static constexpr auto bind(M&& m, K&& k)`, `k` returning a carrier.
    /// \ingroup core
    /// \tparam M The carrier, unqualified.
    template <class M>
    struct monad_instance {};

    /// \brief Alternative instance: `defined = true`, `static constexpr auto alt(A&& a, B&& b)`, and optionally `static constexpr M empty()`.
    /// \ingroup core
    /// \tparam M The carrier, unqualified.
    template <class M>
    struct alternative_instance {};

    /// \brief Bifunctor instance: `defined = true` and `static constexpr auto bimap(F&& f, G&& g, P&& p)`.
    /// \ingroup core
    /// \tparam P The two-sided carrier, unqualified (`std::pair<A, B>`, `result<V, E>`).
    template <class P>
    struct bifunctor_instance {};

    // ---- the concepts -----------------------------------------------------------------------------------------------

    /// \brief `T` has a Semigroup instance.
    /// \ingroup core
    template <class T>
    concept semigroup = requires { requires semigroup_instance<std::remove_cvref_t<T>>::defined; };

    /// \brief `T` has a Monoid instance (and a Semigroup one).
    /// \ingroup core
    template <class T>
    concept monoid = semigroup<T> && requires { requires monoid_instance<std::remove_cvref_t<T>>::defined; };

    /// \brief `M` has a Functor instance.
    /// \ingroup core
    template <class M>
    concept functor = requires { requires functor_instance<std::remove_cvref_t<M>>::defined; };

    /// \brief `M` has an Applicative instance (and a Functor one).
    /// \ingroup core
    template <class M>
    concept applicative = functor<M> && requires { requires applicative_instance<std::remove_cvref_t<M>>::defined; };

    /// \brief `M` has a Monad instance (and an Applicative one).
    /// \ingroup core
    template <class M>
    concept monad = applicative<M> && requires { requires monad_instance<std::remove_cvref_t<M>>::defined; };

    /// \brief `M` has an Alternative instance.
    /// \ingroup core
    template <class M>
    concept alternative = requires { requires alternative_instance<std::remove_cvref_t<M>>::defined; };

    /// \brief `P` has a Bifunctor instance.
    /// \ingroup core
    template <class P>
    concept bifunctor = requires { requires bifunctor_instance<std::remove_cvref_t<P>>::defined; };

    namespace detail {

        template <class T>
        using inst_key = std::remove_cvref_t<T>;

        struct combine_fn {
            template <class A, class B>
            constexpr auto operator()(A&& a, B&& b) const
                SNICHOLLS_FN_RETURNS(semigroup_instance<inst_key<A>>::combine(std::forward<A>(a), std::forward<B>(b)))
        };

        struct sconcat_fn {
            template <class A, class... B>
            constexpr auto operator()(A&& a, B&&... b) const
                SNICHOLLS_FN_RETURNS(foldl_fn{}(combine_fn{}, std::forward<A>(a), std::forward<B>(b)...))
        };

        struct fmap_fn {
            template <class F, class M>
            constexpr auto operator()(F&& f, M&& m) const
                SNICHOLLS_FN_RETURNS(functor_instance<inst_key<M>>::fmap(std::forward<F>(f), std::forward<M>(m)))
        };

        struct replace_fn {
            template <class X, class M>
            constexpr auto operator()(X&& x, M&& m) const
                SNICHOLLS_FN_RETURNS(fmap_fn{}(constant_fn{}(std::forward<X>(x)), std::forward<M>(m)))
        };

        template <class M>
        struct pure_fn {
            template <class A>
            constexpr auto operator()(A&& a) const SNICHOLLS_FN_RETURNS(applicative_instance<M>::pure(std::forward<A>(a)))
        };

        struct ap_fn {
            template <class MF, class MA>
            constexpr auto operator()(MF&& mf, MA&& ma) const
                SNICHOLLS_FN_RETURNS(applicative_instance<inst_key<MF>>::ap(std::forward<MF>(mf), std::forward<MA>(ma)))
        };

        struct lift_a2_fn {
            template <class F, class MA, class MB>
            constexpr auto operator()(F&& f, MA&& ma, MB&& mb) const
                SNICHOLLS_FN_RETURNS(ap_fn{}(fmap_fn{}(fn::curry<2>(std::forward<F>(f)), std::forward<MA>(ma)),
                                             std::forward<MB>(mb)))
        };

        struct mbind_fn {
            template <class M, class K>
            constexpr auto operator()(M&& m, K&& k) const
                SNICHOLLS_FN_RETURNS(monad_instance<inst_key<M>>::bind(std::forward<M>(m), std::forward<K>(k)))
        };

        struct and_then_fn {
            template <class K, class M>
            constexpr auto operator()(K&& k, M&& m) const
                SNICHOLLS_FN_RETURNS(mbind_fn{}(std::forward<M>(m), std::forward<K>(k)))
        };

        struct mthen_fn {
            template <class MA, class MB>
            constexpr auto operator()(MA&& ma, MB&& mb) const
                SNICHOLLS_FN_RETURNS(mbind_fn{}(std::forward<MA>(ma), constant_fn{}(std::forward<MB>(mb))))
        };

        struct join_fn {
            template <class MM>
            constexpr auto operator()(MM&& mm) const SNICHOLLS_FN_RETURNS(mbind_fn{}(std::forward<MM>(mm), identity_fn{}))
        };

        // x -> f(x) >>= g
        struct kleisli_op {
            template <std::size_t... I, class S, class... A>
            static constexpr auto call(std::index_sequence<I...>, S&& s, A&&... a)
                SNICHOLLS_FN_RETURNS(mbind_fn{}(std::invoke(detail::part<0>(std::forward<S>(s)), std::forward<A>(a)...),
                                                detail::part<1>(std::forward<S>(s))))
        };

        struct kleisli_left_fn {
            template <class G, class F>
            constexpr auto operator()(G&& g, F&& f) const noexcept(nothrow_store_v<F, G>)
            {
                return detail::make<kleisli_op>(std::forward<F>(f), std::forward<G>(g));
            }
        };

        struct alt_fn {
            template <class A, class B>
            constexpr auto operator()(A&& a, B&& b) const
                SNICHOLLS_FN_RETURNS(alternative_instance<inst_key<A>>::alt(std::forward<A>(a), std::forward<B>(b)))
        };

        struct bimap_fn {
            template <class F, class G, class P>
            constexpr auto operator()(F&& f, G&& g, P&& p) const
                SNICHOLLS_FN_RETURNS(bifunctor_instance<inst_key<P>>::bimap(std::forward<F>(f), std::forward<G>(g),
                                                                            std::forward<P>(p)))
        };

        struct first_fn {
            template <class F, class P>
            constexpr auto operator()(F&& f, P&& p) const
                SNICHOLLS_FN_RETURNS(bimap_fn{}(std::forward<F>(f), identity_fn{}, std::forward<P>(p)))
        };

        struct second_fn {
            template <class G, class P>
            constexpr auto operator()(G&& g, P&& p) const
                SNICHOLLS_FN_RETURNS(bimap_fn{}(identity_fn{}, std::forward<G>(g), std::forward<P>(p)))
        };

        // dimap f g h = g . h . f
        struct dimap_fn {
            template <class F, class G, class H>
            constexpr auto operator()(F&& f, G&& g, H&& h) const
                SNICHOLLS_FN_RETURNS(compose_fn{}(std::forward<G>(g), std::forward<H>(h), std::forward<F>(f)))
        };

        struct lmap_fn {
            template <class F, class H>
            constexpr auto operator()(F&& f, H&& h) const
                SNICHOLLS_FN_RETURNS(compose_fn{}(std::forward<H>(h), std::forward<F>(f)))
        };

    } // namespace detail

    // ---- Semigroup and Monoid ---------------------------------------------------------------------------------------

    /// \brief `combine(a, b)`: Haskell's `a <> b`, the instance's associative operation.
    /// \ingroup core
    inline constexpr detail::combine_fn combine{};

    /// \brief `sconcat(a, b, c)`: `a <> b <> c`, combined left to right (a semigroup has no empty, so one or more).
    /// \ingroup core
    inline constexpr detail::sconcat_fn sconcat{};

    /// \brief `mempty<T>()`: the identity of `T`'s monoid, `combine(mempty<T>(), x) == x == combine(x, mempty<T>())`.
    /// \ingroup core
    /// \tparam T A monoid.
    /// \return The identity.
    template <monoid T>
    [[nodiscard]] constexpr T mempty() noexcept(noexcept(monoid_instance<T>::empty()))
    {
        return monoid_instance<T>::empty();   // a function, not a variable: a constexpr std::string variable cannot exist
                                              // on libstdc++, while one made and destroyed in a constant evaluation can
    }

    /// \brief The type of `mconcat`: every element of a range or array combined, left to right, from `mempty`.
    /// \ingroup core
    struct mconcat_fn {
        /// \brief Combine the elements of `seq` from `mempty`.
        /// \tparam Seq The sequence's type: an input range whose elements are a monoid.
        /// \tparam T   The monoid, the range's value type.
        /// \param seq The elements.
        /// \return Their combination; `mempty` for none.
        template <class Seq, class T = std::remove_cvref_t<decltype(*std::ranges::begin(std::declval<Seq&>()))>>
            requires monoid<T>
        constexpr T operator()(Seq&& seq) const
            noexcept(noexcept(foldl_of(detail::combine_fn{}, monoid_instance<T>::empty(), std::forward<Seq>(seq))))
        {
            return foldl_of(detail::combine_fn{}, monoid_instance<T>::empty(), std::forward<Seq>(seq));
        }
    };

    /// \brief `mconcat(seq)`: the elements of a range or array combined from `mempty`, left to right.
    /// \ingroup core
    inline constexpr mconcat_fn mconcat{};

    // ---- Functor, Applicative, Monad, Alternative ---------------------------------------------------------------------

    /// \brief `fmap(f, m)`: `f` applied inside the carrier `m` (Functor).
    /// \ingroup core
    ///
    /// The function comes first, so `fmap` curries (`fn::curry<2>(fn::fmap)(f)` is `lift`).
    inline constexpr detail::fmap_fn fmap{};

    /// \brief `replace(x, m)`: Haskell's `x <$ m`, every value in `m` replaced by `x`.
    /// \ingroup core
    inline constexpr detail::replace_fn replace{};

    /// \brief `pure<M>(x)`: `x` in the carrier `M` with no effect (Applicative's `pure`).
    /// \ingroup core
    /// \tparam M The carrier, named because C++ cannot infer it from the context: `pure<std::optional<int>>(3)`.
    template <class M>
    inline constexpr detail::pure_fn<std::remove_cvref_t<M>> pure{};

    /// \brief `mreturn<M>(x)`: Monad's `return`, the same as `pure<M>(x)`.
    /// \ingroup core
    /// \tparam M The carrier.
    template <class M>
    inline constexpr detail::pure_fn<std::remove_cvref_t<M>> mreturn{};

    /// \brief `ap(mf, mx)`: Haskell's `mf <*>  mx`, a function in a carrier applied to a value in one.
    /// \ingroup core
    inline constexpr detail::ap_fn ap{};

    /// \brief `lift_a2(f, ma, mb)`: Haskell's `liftA2`, `f <$> ma <*> mb`.
    /// \ingroup core
    inline constexpr detail::lift_a2_fn lift_a2{};

    /// \brief `mbind(m, k)`: Haskell's `m >>= k`, `k` run on the value in `m`, its carrier the result.
    /// \ingroup core
    inline constexpr detail::mbind_fn mbind{};

    /// \brief `and_then(k, m)`: Haskell's `k =<< m`, `mbind` with the function first (so it curries).
    /// \ingroup core
    inline constexpr detail::and_then_fn and_then{};

    /// \brief `mthen(ma, mb)`: Haskell's `ma >> mb`, `ma`'s effect, then `mb`.
    /// \ingroup core
    inline constexpr detail::mthen_fn mthen{};

    /// \brief `join(mm)`: one carrier layer removed, `mm >>= id`.
    /// \ingroup core
    inline constexpr detail::join_fn join{};

    /// \brief `kleisli(f, g)`: Haskell's `f >=> g`, the function `x -> f(x) >>= g`.
    /// \ingroup core
    inline constexpr detail::binary_maker<detail::kleisli_op> kleisli{};

    /// \brief `kleisli_left(g, f)`: Haskell's `g <=< f`, the same as `kleisli(f, g)`.
    /// \ingroup core
    inline constexpr detail::kleisli_left_fn kleisli_left{};

    /// \brief `alt(a, b)`: Haskell's `a <|> b`, the first that succeeded (Alternative; MonadPlus's `mplus`).
    /// \ingroup core
    inline constexpr detail::alt_fn alt{};

    /// \brief `mzero<M>`: Alternative's `empty` (MonadPlus's `mzero`), the identity of `alt`.
    /// \ingroup core
    /// \tparam M The carrier.
    template <class M>
    inline constexpr std::remove_cvref_t<M> mzero = alternative_instance<std::remove_cvref_t<M>>::empty();

    // ---- Bifunctor and Profunctor ---------------------------------------------------------------------------------------

    /// \brief `bimap(f, g, p)`: `f` on the first side, `g` on the second (a pair's halves, a result's value and error).
    /// \ingroup core
    inline constexpr detail::bimap_fn bimap{};

    /// \brief `first(f, p)`: Bifunctor's `first`, `bimap(f, id, p)`.
    /// \ingroup core
    inline constexpr detail::first_fn first{};

    /// \brief `second(g, p)`: Bifunctor's `second`, `bimap(id, g, p)`.
    /// \ingroup core
    inline constexpr detail::second_fn second{};

    /// \brief `dimap(f, g, h)`: the function `h` with `f` before and `g` after, `g . h . f` (Profunctor on functions).
    /// \ingroup core
    inline constexpr detail::dimap_fn dimap{};

    /// \brief `lmap(f, h)`: `h . f`, the input adapted (Profunctor's `lmap`, Contravariant's `contramap` on the argument).
    /// \ingroup core
    inline constexpr detail::lmap_fn lmap{};

    /// \brief `contramap(f, h)`: `h . f`, the same as `lmap`.
    /// \ingroup core
    inline constexpr detail::lmap_fn contramap{};

    /// \brief `rmap(g, h)`: `g . h`, the output adapted (Profunctor's `rmap`; `fmap` on functions).
    /// \ingroup core
    inline constexpr detail::compose_fn rmap{};

    // ---- the monoid wrappers ---------------------------------------------------------------------------------------------

    /// \brief Haskell's `Sum`: numbers under `+`, from 0.
    /// \ingroup core
    /// \tparam T The number type.
    template <class T>
    struct sum_of {
        T value; ///< The number.
        /// \brief Equal when the numbers are.
        friend constexpr bool operator==(const sum_of&, const sum_of&) = default;
    };

    /// \brief Haskell's `Product`: numbers under `*`, from 1.
    /// \ingroup core
    /// \tparam T The number type.
    template <class T>
    struct product_of {
        T value; ///< The number.
        /// \brief Equal when the numbers are.
        friend constexpr bool operator==(const product_of&, const product_of&) = default;
    };

    /// \brief Haskell's `Min`: the smaller, from the largest value of `T` (`std::numeric_limits<T>::max()`).
    /// \ingroup core
    /// \tparam T An ordered type.
    template <class T>
    struct min_of {
        T value; ///< The value.
        /// \brief Equal when the values are.
        friend constexpr bool operator==(const min_of&, const min_of&) = default;
    };

    /// \brief Haskell's `Max`: the larger, from the lowest value of `T` (`std::numeric_limits<T>::lowest()`).
    /// \ingroup core
    /// \tparam T An ordered type.
    template <class T>
    struct max_of {
        T value; ///< The value.
        /// \brief Equal when the values are.
        friend constexpr bool operator==(const max_of&, const max_of&) = default;
    };

    /// \brief Haskell's `First`: the first value present, from none.
    /// \ingroup core
    /// \tparam T The value type.
    template <class T>
    struct first_of {
        std::optional<T> value; ///< The value, if any.
        /// \brief Equal when the optionals are.
        friend constexpr bool operator==(const first_of&, const first_of&) = default;
    };

    /// \brief Haskell's `Last`: the last value present, from none.
    /// \ingroup core
    /// \tparam T The value type.
    template <class T>
    struct last_of {
        std::optional<T> value; ///< The value, if any.
        /// \brief Equal when the optionals are.
        friend constexpr bool operator==(const last_of&, const last_of&) = default;
    };

    /// \brief Haskell's `All`: booleans under `&&`, from true.
    /// \ingroup core
    struct all_of {
        bool value; ///< The truth value.
        /// \brief Equal when the booleans are.
        friend constexpr bool operator==(const all_of&, const all_of&) = default;
    };

    /// \brief Haskell's `Any`: booleans under `||`, from false.
    /// \ingroup core
    struct any_of {
        bool value; ///< The truth value.
        /// \brief Equal when the booleans are.
        friend constexpr bool operator==(const any_of&, const any_of&) = default;
    };

    /// \brief Haskell's `Endo`: functions from a type to itself under composition, from `identity`.
    /// \ingroup core
    ///
    /// `combine(endo{f}, endo{g})` is `endo{compose(f, g)}`; `endo{f}(x)` calls `f`.
    /// \tparam F The function's type.
    template <class F>
    struct endo {
        F f; ///< The function.

        /// \brief Call the function.
        /// \param x The argument.
        /// \return `f(x)`.
        template <class X>
        constexpr auto operator()(X&& x) const SNICHOLLS_FN_RETURNS(std::invoke(f, std::forward<X>(x)))
    };

    /// \brief Deduction guide: `endo{f}` holds a copy of `f`.
    /// \tparam F The function's type.
    template <class F>
    endo(F) -> endo<F>;

#ifndef SNICHOLLS_DOXYGEN
    template <class T>
    struct semigroup_instance<sum_of<T>> {
        static constexpr bool defined = true;
        static constexpr sum_of<T> combine(const sum_of<T>& a, const sum_of<T>& b) noexcept(noexcept(a.value + b.value))
        {
            return {static_cast<T>(a.value + b.value)};
        }
    };
    template <class T>
    struct monoid_instance<sum_of<T>> {
        static constexpr bool defined = true;
        static constexpr sum_of<T> empty() noexcept { return {T(0)}; }
    };
    template <class T>
    struct semigroup_instance<product_of<T>> {
        static constexpr bool defined = true;
        static constexpr product_of<T> combine(const product_of<T>& a, const product_of<T>& b) noexcept(noexcept(a.value * b.value))
        {
            return {static_cast<T>(a.value * b.value)};
        }
    };
    template <class T>
    struct monoid_instance<product_of<T>> {
        static constexpr bool defined = true;
        static constexpr product_of<T> empty() noexcept { return {T(1)}; }
    };
    template <class T>
    struct semigroup_instance<min_of<T>> {
        static constexpr bool defined = true;
        static constexpr min_of<T> combine(const min_of<T>& a, const min_of<T>& b) noexcept(noexcept(b.value < a.value))
        {
            return b.value < a.value ? b : a;
        }
    };
    template <class T>
        requires std::numeric_limits<T>::is_specialized
    struct monoid_instance<min_of<T>> {
        static constexpr bool defined = true;
        static constexpr min_of<T> empty() noexcept { return {std::numeric_limits<T>::max()}; }
    };
    template <class T>
    struct semigroup_instance<max_of<T>> {
        static constexpr bool defined = true;
        static constexpr max_of<T> combine(const max_of<T>& a, const max_of<T>& b) noexcept(noexcept(a.value < b.value))
        {
            return a.value < b.value ? b : a;
        }
    };
    template <class T>
        requires std::numeric_limits<T>::is_specialized
    struct monoid_instance<max_of<T>> {
        static constexpr bool defined = true;
        static constexpr max_of<T> empty() noexcept { return {std::numeric_limits<T>::lowest()}; }
    };
    template <class T>
    struct semigroup_instance<first_of<T>> {
        static constexpr bool defined = true;
        static constexpr first_of<T> combine(const first_of<T>& a, const first_of<T>& b) { return a.value ? a : b; }
    };
    template <class T>
    struct monoid_instance<first_of<T>> {
        static constexpr bool defined = true;
        static constexpr first_of<T> empty() noexcept { return {}; }
    };
    template <class T>
    struct semigroup_instance<last_of<T>> {
        static constexpr bool defined = true;
        static constexpr last_of<T> combine(const last_of<T>& a, const last_of<T>& b) { return b.value ? b : a; }
    };
    template <class T>
    struct monoid_instance<last_of<T>> {
        static constexpr bool defined = true;
        static constexpr last_of<T> empty() noexcept { return {}; }
    };
    template <>
    struct semigroup_instance<all_of> {
        static constexpr bool defined = true;
        static constexpr all_of combine(all_of a, all_of b) noexcept { return {a.value && b.value}; }
    };
    template <>
    struct monoid_instance<all_of> {
        static constexpr bool defined = true;
        static constexpr all_of empty() noexcept { return {true}; }
    };
    template <>
    struct semigroup_instance<any_of> {
        static constexpr bool defined = true;
        static constexpr any_of combine(any_of a, any_of b) noexcept { return {a.value || b.value}; }
    };
    template <>
    struct monoid_instance<any_of> {
        static constexpr bool defined = true;
        static constexpr any_of empty() noexcept { return {false}; }
    };
    template <class F>
    struct semigroup_instance<endo<F>> {
        static constexpr bool defined = true;
        template <class A, class B>
        static constexpr auto combine(A&& a, B&& b)
            SNICHOLLS_FN_RETURNS(fn::endo{detail::compose_fn{}(std::forward<A>(a).f, std::forward<B>(b).f)})
    };
    template <>
    struct monoid_instance<endo<detail::identity_fn>> {
        static constexpr bool defined = true;
        static constexpr endo<detail::identity_fn> empty() noexcept { return {}; }
    };
#endif

} // namespace snicholls::fn

#endif /* snicholls_core_functional_typeclasses_hpp */
