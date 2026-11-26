//
//  instances.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief The type-class instances of `functional/typeclasses.hpp` for `std::optional` (Maybe), `snicholls::result`
///        (Either), `std::tuple` and `std::pair` (products), `std::array` (fixed lists), `std::basic_string` and
///        `std::vector`, all `constexpr`.
/// \ingroup core
///
/// **Overview.**
///
/// | type                         | Semigroup / Monoid                     | Functor            | Applicative / Monad                     | Alternative          | Bifunctor          |
/// | ---------------------------- | -------------------------------------- | ------------------ | --------------------------------------- | -------------------- | ------------------ |
/// | `std::optional<T>` (Maybe)   | lifts `T`'s semigroup; `nullopt`       | the value          | `pure` = engaged; short-circuits        | first engaged; `nullopt` | -              |
/// | `result<V, E>` (Either)      | the first success (no `mempty`)        | the value          | `pure` = `ok`; the first error wins     | first success        | value and error    |
/// | `std::tuple<Ts...>`          | element by element                     | every element      | -                                       | -                    | `pair`, 2-tuples   |
/// | `std::pair<A, B>`            | element by element                     | both elements      | -                                       | -                    | first and second   |
/// | `std::array<T, N>`           | element by element                     | every element      | Haskell's list: cartesian, `N * M`      | -                    | -                  |
/// | `std::basic_string`          | concatenation; `""`                    | -                  | -                                       | -                    | -                  |
/// | `std::vector<T>`             | concatenation; `{}`                    | every element      | -                                       | -                    | -                  |
///
/// \snippet instances_demo.cpp maybe
///
/// A `std::array` is a list whose length is in its type, so its Monoid cannot be concatenation (the
/// length would change): it is the product monoid, element by element. Its Monad is the list monad,
/// with the length multiplied: `mbind(array<T, N>, T -> array<U, M>)` is an `array<U, N * M>`.
/// A tuple's Functor maps every element (with one generic function), not only the last as Haskell's
/// does, because C++'s tuple is a product of equals; `bimap` over a pair is the Haskell one.
///
/// **Relationships.** `functional/typeclasses.hpp` (the classes), `functional/state.hpp` (Identity,
/// Const, Reader, Writer, State), `functional/monad.hpp`, `snicholls::result`. Group: \ref core.
///
/// **Implementations / variants.** One instance per type, partially specialised on the template.
///
/// **Guarantees.** `constexpr` (C++20's `std::optional`, `std::tuple`, `std::array`, `std::string`
/// and `std::vector`, and core's `result`, are). The carrier's value category is kept (an rvalue
/// optional moves its value into `f`). `noexcept` exactly when `f` and the copies are, except that the
/// standard library's string and vector operations allocate and are never `noexcept`. The laws hold
/// and are tested at compile time (`tests/functional/laws_test.cpp`).
///
/// **Errors.** None of their own; results propagate their first error.
///
/// **Examples.** \snippet instances_demo.cpp list
///
/// **Performance.** A branch and the call for Maybe and Either; the elements' calls for products and
/// arrays. Not benchmarked separately.
///
/// **Tests and demos.** `tests/functional/instances_test.cpp`, `tests/functional/laws_test.cpp`,
/// `demos/functional/instances_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<functional>`, `<optional>`, `<string>`, `<tuple>`,
/// `<type_traits>`, `<utility>`, `<vector>`, `snicholls/core/result.hpp`, `snicholls/core/meta/concepts.hpp`
/// and `functional/typeclasses.hpp`.
///
/// **Known issues and gotchas.** `result<void, E>` has the Functor and Monad of its own `map` /
/// `and_then`; it is not an Applicative here (there is no value to apply). An optional's `fmap` stores
/// `f`'s result by value.
///
/// **Status.** Experimental, new in 0.27.0.

#ifndef snicholls_core_functional_instances_hpp
#define snicholls_core_functional_instances_hpp

#include <snicholls/core/functional/typeclasses.hpp>
#include <snicholls/core/meta/concepts.hpp>
#include <snicholls/core/result.hpp>

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace snicholls::fn {

    namespace detail {

        template <class M>
        using opt_ref_t = decltype(*std::declval<M>());

        template <class F, class M>
        using opt_map_t = std::optional<std::remove_cvref_t<std::invoke_result_t<F, opt_ref_t<M>>>>;

        // The elements of a tuple-like, each through f, as a std::tuple of values.
        template <class F, class T, std::size_t... J>
        constexpr auto map_elements(F& f, T&& t, std::index_sequence<J...>)
            SNICHOLLS_FN_RETURNS(std::tuple<std::remove_cvref_t<std::invoke_result_t<F&, decltype(std::get<J>(std::forward<T>(t)))>>...>(
                std::invoke(f, std::get<J>(std::forward<T>(t)))...))

        template <class F, class T, std::size_t... J>
        constexpr auto zip_elements(F& f, const T& a, const T& b, std::index_sequence<J...>)
            SNICHOLLS_FN_RETURNS(T(std::invoke(f, std::get<J>(a), std::get<J>(b))...))

        template <class T, std::size_t... J>
        constexpr T empty_elements(std::index_sequence<J...>) noexcept(noexcept(T(mempty<std::tuple_element_t<J, T>>...)))
        {
            return T(mempty<std::tuple_element_t<J, T>>...);
        }

    } // namespace detail

#ifndef SNICHOLLS_DOXYGEN

    // ---- std::optional: Maybe ---------------------------------------------------------------------------------------

    template <class T>
        requires semigroup<T>
    struct semigroup_instance<std::optional<T>> {
        static constexpr bool defined = true;
        static constexpr std::optional<T> combine(const std::optional<T>& a, const std::optional<T>& b)
        {
            if (a && b) return fn::combine(*a, *b);
            return a ? a : b;
        }
    };
    template <class T>
        requires semigroup<T>
    struct monoid_instance<std::optional<T>> {
        static constexpr bool defined = true;
        static constexpr std::optional<T> empty() noexcept { return std::nullopt; }
    };

    template <class T>
    struct functor_instance<std::optional<T>> {
        static constexpr bool defined = true;
        template <class F, class M, class R = detail::opt_map_t<F, M>>
        static constexpr R fmap(F&& f, M&& m) noexcept(std::is_nothrow_invocable_v<F, detail::opt_ref_t<M>> &&
                                                       std::is_nothrow_move_constructible_v<typename R::value_type>)
        {
            if (!m.has_value()) return std::nullopt;
            return R(std::in_place, std::invoke(std::forward<F>(f), *std::forward<M>(m)));
        }
    };
    template <class T>
    struct applicative_instance<std::optional<T>> {
        static constexpr bool defined = true;
        template <class A>
        static constexpr std::optional<std::decay_t<A>> pure(A&& a) noexcept(std::is_nothrow_constructible_v<std::decay_t<A>, A>)
        {
            return std::optional<std::decay_t<A>>(std::in_place, std::forward<A>(a));
        }
        template <class MF, class MA, class R = detail::opt_map_t<detail::opt_ref_t<MF>, MA>>
        static constexpr R ap(MF&& mf, MA&& ma) noexcept(std::is_nothrow_invocable_v<detail::opt_ref_t<MF>, detail::opt_ref_t<MA>> &&
                                                         std::is_nothrow_move_constructible_v<typename R::value_type>)
        {
            if (!mf.has_value() || !ma.has_value()) return std::nullopt;
            return R(std::in_place, std::invoke(*std::forward<MF>(mf), *std::forward<MA>(ma)));
        }
    };
    template <class T>
    struct monad_instance<std::optional<T>> {
        static constexpr bool defined = true;
        template <class M, class K, class R = std::remove_cvref_t<std::invoke_result_t<K, detail::opt_ref_t<M>>>>
            requires specialisation_of<R, std::optional>
        static constexpr R bind(M&& m, K&& k) noexcept(std::is_nothrow_invocable_v<K, detail::opt_ref_t<M>> &&
                                                       std::is_nothrow_move_constructible_v<R>)
        {
            if (!m.has_value()) return R{};
            return std::invoke(std::forward<K>(k), *std::forward<M>(m));
        }
    };
    template <class T>
    struct alternative_instance<std::optional<T>> {
        static constexpr bool defined = true;
        static constexpr std::optional<T> empty() noexcept { return std::nullopt; }
        template <class A, class B>
        static constexpr std::optional<T> alt(A&& a, B&& b) noexcept(std::is_nothrow_constructible_v<std::optional<T>, A> &&
                                                                     std::is_nothrow_constructible_v<std::optional<T>, B>)
        {
            return a.has_value() ? std::optional<T>(std::forward<A>(a)) : std::optional<T>(std::forward<B>(b));
        }
    };

    // ---- snicholls::result: Either ------------------------------------------------------------------------------------

    template <class V, class E>
    struct semigroup_instance<result<V, E>> {
        static constexpr bool defined = true;
        static constexpr result<V, E> combine(const result<V, E>& a, const result<V, E>& b) { return a.is_ok() ? a : b; }
    };
    template <class V, class E>
    struct functor_instance<result<V, E>> {
        static constexpr bool defined = true;
        template <class F, class M>
        static constexpr auto fmap(F&& f, M&& m) SNICHOLLS_FN_RETURNS(std::forward<M>(m).map(std::forward<F>(f)))
    };
    template <class V, class E>
        requires(!std::is_void_v<V>)
    struct applicative_instance<result<V, E>> {
        static constexpr bool defined = true;
        template <class A>
        static constexpr result<std::decay_t<A>, E> pure(A&& a) noexcept(std::is_nothrow_constructible_v<std::decay_t<A>, A>)
        {
            return result<std::decay_t<A>, E>(std::in_place, std::forward<A>(a));
        }
        template <class MF, class MA,
                  class R = std::remove_cvref_t<std::invoke_result_t<decltype(std::declval<MF>().unwrap_unchecked()),
                                                                     decltype(std::declval<MA>().unwrap_unchecked())>>>
        static constexpr result<R, E> ap(MF&& mf, MA&& ma)
        {
            if (mf.is_err()) return err(std::forward<MF>(mf).unwrap_err_unchecked());
            if (ma.is_err()) return err(std::forward<MA>(ma).unwrap_err_unchecked());
            return result<R, E>(std::in_place,
                                std::invoke(std::forward<MF>(mf).unwrap_unchecked(), std::forward<MA>(ma).unwrap_unchecked()));
        }
    };
    template <class V, class E>
    struct monad_instance<result<V, E>> {
        static constexpr bool defined = true;
        template <class M, class K>
        static constexpr auto bind(M&& m, K&& k) SNICHOLLS_FN_RETURNS(std::forward<M>(m).and_then(std::forward<K>(k)))
    };
    template <class V, class E>
    struct alternative_instance<result<V, E>> {
        static constexpr bool defined = true;
        template <class A, class B>
        static constexpr result<V, E> alt(A&& a, B&& b)
        {
            return a.is_ok() ? result<V, E>(std::forward<A>(a)) : result<V, E>(std::forward<B>(b));
        }
    };
    template <class V, class E>
    struct bifunctor_instance<result<V, E>> {
        static constexpr bool defined = true;
        template <class F, class G, class P>
        static constexpr auto bimap(F&& f, G&& g, P&& p)
            SNICHOLLS_FN_RETURNS(std::forward<P>(p).map(std::forward<F>(f)).map_err(std::forward<G>(g)))
    };

    // ---- products: std::tuple and std::pair ---------------------------------------------------------------------------

    template <class... Ts>
        requires(semigroup<Ts> && ...)
    struct semigroup_instance<std::tuple<Ts...>> {
        static constexpr bool defined = true;
        static constexpr std::tuple<Ts...> combine(const std::tuple<Ts...>& a, const std::tuple<Ts...>& b)
        {
            detail::combine_fn c{};
            return detail::zip_elements(c, a, b, std::index_sequence_for<Ts...>{});
        }
    };
    template <class... Ts>
        requires(monoid<Ts> && ...)
    struct monoid_instance<std::tuple<Ts...>> {
        static constexpr bool defined = true;
        static constexpr std::tuple<Ts...> empty() { return detail::empty_elements<std::tuple<Ts...>>(std::index_sequence_for<Ts...>{}); }
    };
    template <class A, class B>
        requires semigroup<A> && semigroup<B>
    struct semigroup_instance<std::pair<A, B>> {
        static constexpr bool defined = true;
        static constexpr std::pair<A, B> combine(const std::pair<A, B>& a, const std::pair<A, B>& b)
        {
            return {fn::combine(a.first, b.first), fn::combine(a.second, b.second)};
        }
    };
    template <class A, class B>
        requires monoid<A> && monoid<B>
    struct monoid_instance<std::pair<A, B>> {
        static constexpr bool defined = true;
        static constexpr std::pair<A, B> empty() { return {mempty<A>, mempty<B>}; }
    };
    template <class... Ts>
    struct functor_instance<std::tuple<Ts...>> {
        static constexpr bool defined = true;
        template <class F, class M>
        static constexpr auto fmap(F&& f, M&& m)
            SNICHOLLS_FN_RETURNS(detail::map_elements(f, std::forward<M>(m), std::index_sequence_for<Ts...>{}))
    };
    template <class A, class B>
    struct functor_instance<std::pair<A, B>> {
        static constexpr bool defined = true;
        template <class F, class M, class RA = std::remove_cvref_t<std::invoke_result_t<F&, decltype((std::declval<M>().first))>>,
                  class RB = std::remove_cvref_t<std::invoke_result_t<F&, decltype((std::declval<M>().second))>>>
        static constexpr std::pair<RA, RB> fmap(F&& f, M&& m)
        {
            return {std::invoke(f, std::forward<M>(m).first), std::invoke(f, std::forward<M>(m).second)};
        }
    };
    template <class A, class B>
    struct bifunctor_instance<std::pair<A, B>> {
        static constexpr bool defined = true;
        template <class F, class G, class P,
                  class RA = std::remove_cvref_t<std::invoke_result_t<F, decltype((std::declval<P>().first))>>,
                  class RB = std::remove_cvref_t<std::invoke_result_t<G, decltype((std::declval<P>().second))>>>
        static constexpr std::pair<RA, RB> bimap(F&& f, G&& g, P&& p)
        {
            return {std::invoke(std::forward<F>(f), std::forward<P>(p).first),
                    std::invoke(std::forward<G>(g), std::forward<P>(p).second)};
        }
    };
    template <class A, class B>
    struct bifunctor_instance<std::tuple<A, B>> {
        static constexpr bool defined = true;
        template <class F, class G, class P,
                  class RA = std::remove_cvref_t<std::invoke_result_t<F, decltype(std::get<0>(std::declval<P>()))>>,
                  class RB = std::remove_cvref_t<std::invoke_result_t<G, decltype(std::get<1>(std::declval<P>()))>>>
        static constexpr std::tuple<RA, RB> bimap(F&& f, G&& g, P&& p)
        {
            return {std::invoke(std::forward<F>(f), std::get<0>(std::forward<P>(p))),
                    std::invoke(std::forward<G>(g), std::get<1>(std::forward<P>(p)))};
        }
    };

    // ---- std::array: fixed lists --------------------------------------------------------------------------------------

    template <class T, std::size_t N>
        requires semigroup<T>
    struct semigroup_instance<std::array<T, N>> {
        static constexpr bool defined = true;
        static constexpr std::array<T, N> combine(const std::array<T, N>& a, const std::array<T, N>& b)
        {
            return [&]<std::size_t... J>(std::index_sequence<J...>) {
                return std::array<T, N>{{fn::combine(a[J], b[J])...}};
            }(std::make_index_sequence<N>{});
        }
    };
    template <class T, std::size_t N>
        requires monoid<T>
    struct monoid_instance<std::array<T, N>> {
        static constexpr bool defined = true;
        static constexpr std::array<T, N> empty()
        {
            return [&]<std::size_t... J>(std::index_sequence<J...>) {
                return std::array<T, N>{{(static_cast<void>(J), mempty<T>)...}};
            }(std::make_index_sequence<N>{});
        }
    };
    template <class T, std::size_t N>
    struct functor_instance<std::array<T, N>> {
        static constexpr bool defined = true;
        template <class F, class M, class R = std::remove_cvref_t<std::invoke_result_t<F&, decltype(std::declval<M>()[0])>>>
        static constexpr std::array<R, N> fmap(F&& f, M&& m) noexcept(std::is_nothrow_invocable_v<F&, decltype(std::declval<M>()[0])> &&
                                                                       std::is_nothrow_move_constructible_v<R>)
        {
            return [&]<std::size_t... J>(std::index_sequence<J...>) {
                return std::array<R, N>{{std::invoke(f, std::forward<M>(m)[J])...}};
            }(std::make_index_sequence<N>{});
        }
    };
    template <class T, std::size_t N>
    struct applicative_instance<std::array<T, N>> {
        static constexpr bool defined = true;
        template <class A>
        static constexpr std::array<std::decay_t<A>, 1> pure(A&& a) noexcept(std::is_nothrow_constructible_v<std::decay_t<A>, A>)
        {
            return {{std::forward<A>(a)}};
        }
        // Every function applied to every value, functions outermost: Haskell's list <*>.
        template <class MF, class MA, std::size_t NF = std::tuple_size_v<std::remove_cvref_t<MF>>,
                  std::size_t NA = std::tuple_size_v<std::remove_cvref_t<MA>>,
                  class R = std::remove_cvref_t<std::invoke_result_t<const typename std::remove_cvref_t<MF>::value_type&,
                                                                     const typename std::remove_cvref_t<MA>::value_type&>>>
        static constexpr std::array<R, NF * NA> ap(MF&& mf, MA&& ma)
        {
            return [&]<std::size_t... J>(std::index_sequence<J...>) {
                return std::array<R, NF * NA>{{std::invoke(std::as_const(mf[J / NA]), std::as_const(ma[J % NA]))...}};
            }(std::make_index_sequence<NF * NA>{});
        }
    };
    template <class T, std::size_t N>
    struct monad_instance<std::array<T, N>> {
        static constexpr bool defined = true;
        // Every value through k, the arrays concatenated: Haskell's list >>= (concatMap).
        template <class M, class K, class RA = std::remove_cvref_t<std::invoke_result_t<K&, const T&>>,
                  std::size_t NK = std::tuple_size_v<RA>, class R = typename RA::value_type>
        static constexpr std::array<R, N * NK> bind(M&& m, K&& k)
        {
            std::array<RA, N> parts = functor_instance<std::array<T, N>>::fmap(k, std::as_const(m));
            return [&]<std::size_t... J>(std::index_sequence<J...>) {
                return std::array<R, N * NK>{{std::move(parts[J / NK][J % NK])...}};
            }(std::make_index_sequence<N * NK>{});
        }
    };

    // ---- strings and vectors: concatenation ---------------------------------------------------------------------------

    template <class C, class Tr, class A>
    struct semigroup_instance<std::basic_string<C, Tr, A>> {
        static constexpr bool defined = true;
        static constexpr std::basic_string<C, Tr, A> combine(const std::basic_string<C, Tr, A>& a, const std::basic_string<C, Tr, A>& b)
        {
            return a + b;
        }
    };
    template <class C, class Tr, class A>
    struct monoid_instance<std::basic_string<C, Tr, A>> {
        static constexpr bool defined = true;
        static constexpr std::basic_string<C, Tr, A> empty() { return {}; }
    };
    template <class T, class A>
    struct semigroup_instance<std::vector<T, A>> {
        static constexpr bool defined = true;
        static constexpr std::vector<T, A> combine(const std::vector<T, A>& a, const std::vector<T, A>& b)
        {
            std::vector<T, A> out(a);
            out.insert(out.end(), b.begin(), b.end());
            return out;
        }
    };
    template <class T, class A>
    struct monoid_instance<std::vector<T, A>> {
        static constexpr bool defined = true;
        static constexpr std::vector<T, A> empty() { return {}; }
    };
    template <class T, class A>
    struct functor_instance<std::vector<T, A>> {
        static constexpr bool defined = true;
        template <class F, class M, class R = std::remove_cvref_t<std::invoke_result_t<F&, decltype(*std::declval<M>().begin())>>>
        static constexpr std::vector<R> fmap(F&& f, M&& m)
        {
            std::vector<R> out;
            out.reserve(m.size());
            for (auto&& x : std::forward<M>(m)) out.push_back(std::invoke(f, std::forward<decltype(x)>(x)));
            return out;
        }
    };

#endif

} // namespace snicholls::fn

#endif /* snicholls_core_functional_instances_hpp */
