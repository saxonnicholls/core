//
//  monad.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Maybe and Either helpers over `std::optional` and `snicholls::result`: `lift`, `sequence` / `traverse`
///        (Traversable), `fold_m`, `zip_with_m`, `replicate_m`, `for_m`, `when_m` / `unless_m`, the eliminators `maybe`
///        and `either`, and `value_or`, `from_maybe`, `map_maybe`, `cat_maybes`. All `constexpr`.
/// \ingroup core
///
/// **Overview.** The type classes (`fmap`, `ap`, `mbind`, `kleisli`, `alt`) are in
/// `functional/typeclasses.hpp`; these are the helpers built on them for the two carriers of a value
/// that may be missing (`std::optional`, Maybe) or may have failed (`snicholls::result`, Either):
///
/// | here                                 | `std::optional<T>`                         | `snicholls::result<V, E>`              | Haskell                 |
/// | ------------------------------------ | ------------------------------------------ | -------------------------------------- | ----------------------- |
/// | `lift(f)(m1, m2, ...)`               | `f(*m1, *m2, ...)` if all hold a value     | `f(values...)`, or the first error     | `liftA2`, `liftA3`, ... |
/// | `sequence(xs)`, `sequence_a(xs)`     | `optional<tuple>` / `optional<array>`      | `result<tuple, E>` / `result<array, E>` | `sequenceA`, `sequence` |
/// | `traverse(f, xs)`, `map_m`, `for_m(xs, f)` | `sequence(map f xs)`                 | the same                               | `traverse`, `mapM`, `forM` |
/// | `fold_m(f, z, xs)`                   | a fold that stops at the first empty step  | at the first error                     | `foldM`                 |
/// | `zip_with_m(f, a, b)`                | `sequence(zip_with(f, a, b))`              | the same                               | `zipWithM`              |
/// | `replicate_m<N>(m)`                  | `N` copies of `m`, sequenced               | the same                               | `replicateM`            |
/// | `when_m(b, m)`, `unless_m(b, m)`     | `m` or `pure(std::monostate{})`            | the same                               | `when`, `unless` (Control.Monad) |
/// | `maybe(d, f, m)`, `either(f, g, r)`  | `m ? f(*m) : d`                            | `r ? g(value) : f(error)`              | `maybe`, `either`       |
/// | `value_or(m, x)`, `from_maybe(x, m)` | `m.value_or(x)`                            | `m.unwrap_or(x)`                       | `fromMaybe`             |
/// | `map_maybe(f, xs)`, `cat_maybes(xs)` | the values of the engaged results          | -                                      | `mapMaybe`, `catMaybes` |
///
/// \snippet monad_demo.cpp lift
///
/// Use these when a calculation is a chain of steps that may each fail; use `SNICHOLLS_TRY` when the
/// chain is a function body with early returns.
///
/// **Relationships.** `functional/typeclasses.hpp`, `functional/instances.hpp`, `functional/list.hpp`
/// (`zip_with`, `replicate`), `snicholls::result`, `std::optional`. Group: \ref core.
///
/// **Implementations / variants.** One overload per carrier, constrained with
/// `snicholls::specialisation_of`; the carriers do not overlap.
///
/// **Guarantees.** `constexpr`. `noexcept` exactly when `f` and the copies or moves of values and errors
/// are. The carrier's value category is kept. `lift(f)` evaluates `f` only when every argument holds a
/// value; `traverse` calls `f` on every element, then sequences; `fold_m` stops at the first failure.
///
/// **Errors.** None of their own: an empty optional or an error result propagates. Results combined
/// by `lift`, `sequence` or `traverse` need one error type, and the first error, left to right, is kept.
///
/// **Examples.** \snippet monad_demo.cpp sequence
///
/// **Performance.** A branch on `has_value()` / `is_ok()` and the call; not benchmarked.
///
/// **Tests and demos.** `tests/functional/monad_test.cpp`, `tests/functional/monad_traversable_test.cpp`,
/// `demos/functional/monad_demo.cpp`, `demos/functional/monad_railway_demo.cpp` (railway error handling).
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<functional>`, `<optional>`, `<tuple>`, `<type_traits>`,
/// `<utility>`, `<variant>`, `snicholls/core/result.hpp` (heavier than it looks: `<string_view>` and
/// `<stdexcept>`), `functional/instances.hpp` and `functional/list.hpp`.
///
/// **Known issues and gotchas.**
/// - `result<void, E>` is not lifted or sequenced: there is no value to pass on. Use its own `and_then`.
/// - `sequence` / `traverse` of a `std::tuple` or `std::pair` give a `std::tuple` inside; of a
///   `std::array`, an array. `fold_m` is for Maybe and Either; for State, chain `mbind`.
///
/// **Status.** Experimental, new in 0.27.0.

#ifndef snicholls_core_functional_monad_hpp
#define snicholls_core_functional_monad_hpp

#include <snicholls/core/functional/instances.hpp>
#include <snicholls/core/functional/list.hpp>
#include <snicholls/core/meta/concepts.hpp>
#include <snicholls/core/result.hpp>

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace snicholls::fn {

    namespace detail {

        template <class M>
        concept optional_carrier = specialisation_of<M, std::optional>;

        template <class M>
        concept result_carrier =
            specialisation_of<M, snicholls::result> && !std::is_void_v<typename std::remove_cvref_t<M>::value_type>;

        template <class M>
        using res_ref_t = decltype(std::declval<M>().unwrap_unchecked());

        template <class F, class... M>
        using opt_lift_t = std::optional<std::remove_cvref_t<std::invoke_result_t<F, opt_ref_t<M>...>>>;

        template <class M>
        using error_of_t = typename std::remove_cvref_t<M>::error_type;

        // Whether a carrier holds a value, and the value: overloads, one per carrier.
        template <class T>
        constexpr bool engaged(const std::optional<T>& m) noexcept { return m.has_value(); }
        template <class V, class E>
        constexpr bool engaged(const snicholls::result<V, E>& m) noexcept { return m.is_ok(); }
        template <class M>
            requires specialisation_of<M, std::optional>
        constexpr decltype(auto) value_of(M&& m) noexcept { return *std::forward<M>(m); }
        template <class M>
            requires specialisation_of<M, snicholls::result>
        constexpr decltype(auto) value_of(M&& m) noexcept(noexcept(std::forward<M>(m).unwrap_unchecked()))
        {
            return std::forward<M>(m).unwrap_unchecked();
        }

        // The first error, left to right; one of them is an error.
        template <class E, class M0>
        constexpr E first_error(M0&& m0) noexcept(
            std::is_nothrow_constructible_v<E, decltype(std::forward<M0>(m0).unwrap_err_unchecked())>)
        {
            return std::forward<M0>(m0).unwrap_err_unchecked();
        }

        // The noexcept is spelled out: naming this overload in its own specification fails for three or more.
        template <class E, class M0, class M1, class... M>
        constexpr E first_error(M0&& m0, M1&& m1, M&&... m) noexcept(
            ((std::is_nothrow_constructible_v<E, decltype(std::declval<M0>().unwrap_err_unchecked())> &&
              std::is_nothrow_constructible_v<E, decltype(std::declval<M1>().unwrap_err_unchecked())>) && ... &&
             std::is_nothrow_constructible_v<E, decltype(std::declval<M>().unwrap_err_unchecked())>))
        {
            if (m0.is_err()) return detail::first_error<E>(std::forward<M0>(m0));
            return detail::first_error<E>(std::forward<M1>(m1), std::forward<M>(m)...);
        }

        struct lift_op {
            // Every argument an optional.
            template <std::size_t... I, class S, optional_carrier... M, class R = opt_lift_t<part_t<0, S>, M...>>
                requires(sizeof...(M) > 0) && std::is_invocable_v<part_t<0, S>, opt_ref_t<M>...>
            static constexpr R call(std::index_sequence<I...>, S&& s, M&&... m) noexcept(
                std::is_nothrow_invocable_v<part_t<0, S>, opt_ref_t<M>...> &&
                std::is_nothrow_move_constructible_v<typename R::value_type>)
            {
                if (!(m.has_value() && ...)) return std::nullopt;
                return R(std::in_place, std::invoke(detail::part<0>(std::forward<S>(s)), *std::forward<M>(m)...));
            }

            // Every argument a result with a value type, all with the first one's error type.
            template <std::size_t... I, class S, result_carrier M0, result_carrier... M, class E = error_of_t<M0>,
                      class V = std::remove_cvref_t<std::invoke_result_t<part_t<0, S>, res_ref_t<M0>, res_ref_t<M>...>>>
                requires(std::is_same_v<E, error_of_t<M>> && ...) && (!std::is_void_v<V>)
            static constexpr snicholls::result<V, E> call(std::index_sequence<I...>, S&& s, M0&& m0, M&&... m) noexcept(
                std::is_nothrow_invocable_v<part_t<0, S>, res_ref_t<M0>, res_ref_t<M>...> &&
                std::is_nothrow_move_constructible_v<V> && std::is_nothrow_copy_constructible_v<E> &&
                std::is_nothrow_move_constructible_v<E>)
            {
                if (m0.is_ok() && (m.is_ok() && ...))
                    return snicholls::result<V, E>(std::in_place,
                                                   std::invoke(detail::part<0>(std::forward<S>(s)),
                                                               std::forward<M0>(m0).unwrap_unchecked(),
                                                               std::forward<M>(m).unwrap_unchecked()...));
                return snicholls::err(detail::first_error<E>(std::forward<M0>(m0), std::forward<M>(m)...));
            }
        };

        struct value_tuple_fn {
            template <class... V>
            constexpr auto operator()(V&&... v) const
                SNICHOLLS_FN_RETURNS(std::tuple<std::remove_cvref_t<V>...>(std::forward<V>(v)...))
        };

        struct value_array_fn {
            template <class V0, class... V>
            constexpr auto operator()(V0&& v0, V&&... v) const
                SNICHOLLS_FN_RETURNS(std::array<std::remove_cvref_t<V0>, 1 + sizeof...(V)>{{std::forward<V0>(v0), std::forward<V>(v)...}})
        };

        inline constexpr closure<lift_op, value_tuple_fn> lift_tuple{};
        inline constexpr closure<lift_op, value_array_fn> lift_array{};

        // A std::array is sequenced into an array, any other tuple-like into a std::tuple.
        struct sequence_fn {
            template <std_array T>
                requires(std::tuple_size_v<std::remove_cvref_t<T>> > 0)
            constexpr auto operator()(T&& t) const
                SNICHOLLS_FN_RETURNS(detail::apply_tuple(lift_array, std::forward<T>(t), tuple_indices<T>{}))

            template <tuple_like T>
                requires(!std_array<T>) && (std::tuple_size_v<std::remove_cvref_t<T>> > 0)
            constexpr auto operator()(T&& t) const
                SNICHOLLS_FN_RETURNS(detail::apply_tuple(lift_tuple, std::forward<T>(t), tuple_indices<T>{}))
        };

        struct traverse_fn {
            template <class L, class F, class T, std::size_t... J>
            static constexpr auto each(const L& lifted, F& f, T&& t, std::index_sequence<J...>)
                SNICHOLLS_FN_RETURNS(lifted(std::invoke(f, std::get<J>(std::forward<T>(t)))...))

            template <class F, std_array T>
                requires(std::tuple_size_v<std::remove_cvref_t<T>> > 0)
            constexpr auto operator()(F&& f, T&& t) const
                SNICHOLLS_FN_RETURNS(traverse_fn::each(lift_array, f, std::forward<T>(t), tuple_indices<T>{}))

            template <class F, tuple_like T>
                requires(!std_array<T>) && (std::tuple_size_v<std::remove_cvref_t<T>> > 0)
            constexpr auto operator()(F&& f, T&& t) const
                SNICHOLLS_FN_RETURNS(traverse_fn::each(lift_tuple, f, std::forward<T>(t), tuple_indices<T>{}))
        };

        struct for_m_fn {
            template <class T, class F>
            constexpr auto operator()(T&& t, F&& f) const SNICHOLLS_FN_RETURNS(traverse_fn{}(std::forward<F>(f), std::forward<T>(t)))
        };

        struct value_or_fn {
            template <optional_carrier M, class U>
            constexpr auto operator()(M&& m, U&& fallback) const
                SNICHOLLS_FN_RETURNS(std::forward<M>(m).value_or(std::forward<U>(fallback)))

            template <result_carrier M, class U>
            constexpr auto operator()(M&& m, U&& fallback) const
                SNICHOLLS_FN_RETURNS(std::forward<M>(m).unwrap_or(std::forward<U>(fallback)))
        };

        struct from_maybe_fn {
            template <class U, class M>
            constexpr auto operator()(U&& fallback, M&& m) const
                SNICHOLLS_FN_RETURNS(value_or_fn{}(std::forward<M>(m), std::forward<U>(fallback)))
        };

        struct maybe_fn {
            template <class D, class F, optional_carrier M,
                      class R = std::common_type_t<std::decay_t<D>, std::remove_cvref_t<std::invoke_result_t<F, opt_ref_t<M>>>>>
            constexpr R operator()(D&& d, F&& f, M&& m) const
                noexcept(std::is_nothrow_invocable_v<F, opt_ref_t<M>> && std::is_nothrow_constructible_v<R, D> &&
                         std::is_nothrow_move_constructible_v<R>)
            {
                if (!m.has_value()) return R(std::forward<D>(d));
                return R(std::invoke(std::forward<F>(f), *std::forward<M>(m)));
            }
        };

        struct either_fn {
            template <class F, class G, result_carrier M,
                      class R = std::common_type_t<
                          std::remove_cvref_t<std::invoke_result_t<F, decltype(std::declval<M>().unwrap_err_unchecked())>>,
                          std::remove_cvref_t<std::invoke_result_t<G, res_ref_t<M>>>>>
            constexpr R operator()(F&& on_error, G&& on_value, M&& m) const
            {
                if (m.is_ok()) return R(std::invoke(std::forward<G>(on_value), std::forward<M>(m).unwrap_unchecked()));
                return R(std::invoke(std::forward<F>(on_error), std::forward<M>(m).unwrap_err_unchecked()));
            }
        };

        struct fold_m_fn {
            template <class F, class Z, std::ranges::input_range R, class V = std::remove_cvref_t<Z>,
                      class M = std::remove_cvref_t<std::invoke_result_t<F&, V, std::ranges::range_reference_t<R>>>>
                requires optional_carrier<M> || result_carrier<M>
            constexpr M operator()(F&& f, Z&& z, R&& xs) const
            {
                M acc = pure<M>(V(std::forward<Z>(z)));
                for (auto&& x : xs) {
                    if (!detail::engaged(acc)) break;
                    acc = std::invoke(f, detail::value_of(std::move(acc)), x);
                }
                return acc;
            }
        };

        struct zip_with_m_fn {
            template <class F, std_array A, std_array B>
            constexpr auto operator()(F&& f, const A& a, const B& b) const
                SNICHOLLS_FN_RETURNS(sequence_fn{}(zip_with_fn{}(f, a, b)))
        };

        struct when_m_fn {
            template <class M>
            constexpr std::remove_cvref_t<M> operator()(bool b, M&& m) const
            {
                if (b) return std::forward<M>(m);
                return pure<std::remove_cvref_t<M>>(std::monostate{});
            }
        };

        struct unless_m_fn {
            template <class M>
            constexpr std::remove_cvref_t<M> operator()(bool b, M&& m) const
            {
                return when_m_fn{}(!b, std::forward<M>(m));
            }
        };

        struct map_maybe_fn {
            template <class F, std_array A, class O = std::remove_cvref_t<std::invoke_result_t<F&, const elem_t<A>&>>>
                requires optional_carrier<O>
            constexpr vec<typename O::value_type, size_v<A>> operator()(F&& f, const A& xs) const
            {
                vec<typename O::value_type, size_v<A>> out;
                for (const auto& x : xs)
                    if (O o = std::invoke(f, x); o.has_value()) out.push_back(std::move(*o));
                return out;
            }
        };

        struct cat_maybes_fn {
            template <std_array A>
                requires optional_carrier<elem_t<A>>
            constexpr auto operator()(const A& xs) const
                SNICHOLLS_FN_RETURNS(map_maybe_fn{}(identity_fn{}, xs))
        };

    } // namespace detail

    /// \brief `lift(f)`: `f` on plain values made a function on optionals, or on results, of any number of arguments.
    /// \ingroup core
    ///
    /// `lift(f)(m1, m2)` is `f(*m1, *m2)` in a carrier when both hold values, else empty (optionals) or the first
    /// error (results, which must share their error type). All the arguments are of one kind.
    /// \snippet monad_demo.cpp lift
    inline constexpr detail::unary_maker<detail::lift_op> lift{};

    /// \brief `sequence(xs)`: a tuple, pair or array of carriers turned inside out, a carrier of a tuple (an array for
    ///        an array); empty or the first error if any is.
    /// \ingroup core
    inline constexpr detail::sequence_fn sequence{};

    /// \brief `sequence_a(xs)`: Haskell's `sequenceA`, the same as `sequence`.
    /// \ingroup core
    inline constexpr detail::sequence_fn sequence_a{};

    /// \brief `traverse(f, xs)` is `sequence(map f xs)`: `f` may fail on any element; it is called on every element.
    /// \ingroup core
    inline constexpr detail::traverse_fn traverse{};

    /// \brief `map_m(f, xs)`: Haskell's `mapM`, the same as `traverse`.
    /// \ingroup core
    inline constexpr detail::traverse_fn map_m{};

    /// \brief `for_m(xs, f)`: Haskell's `forM`, `traverse` with the arguments flipped.
    /// \ingroup core
    inline constexpr detail::for_m_fn for_m{};

    /// \brief `fold_m(f, z, xs)`: Haskell's `foldM` over Maybe or Either: `f(acc, x)` returns the carrier, and the fold
    ///        stops at the first empty optional or error.
    /// \ingroup core
    inline constexpr detail::fold_m_fn fold_m{};

    /// \brief `zip_with_m(f, a, b)`: Haskell's `zipWithM`, `sequence(zip_with(f, a, b))` over arrays.
    /// \ingroup core
    inline constexpr detail::zip_with_m_fn zip_with_m{};

    /// \brief `when_m(b, m)`: Control.Monad's `when`, `m` when `b`, else `pure(std::monostate{})`; `m` carries `std::monostate`.
    /// \ingroup core
    inline constexpr detail::when_m_fn when_m{};

    /// \brief `unless_m(b, m)`: Control.Monad's `unless`, `when_m(!b, m)`.
    /// \ingroup core
    inline constexpr detail::unless_m_fn unless_m{};

    /// \brief `value_or(m, fallback)`: the value inside `m`, or `fallback` when it is empty or an error.
    /// \ingroup core
    inline constexpr detail::value_or_fn value_or{};

    /// \brief `from_maybe(fallback, m)`: Haskell's `fromMaybe`, `value_or` with the fallback first.
    /// \ingroup core
    inline constexpr detail::from_maybe_fn from_maybe{};

    /// \brief `maybe(d, f, m)`: Haskell's `maybe`, `f(*m)` if `m` holds a value, else `d`.
    /// \ingroup core
    inline constexpr detail::maybe_fn maybe{};

    /// \brief `either(on_error, on_value, r)`: Haskell's `either`, the case analysis of a result (Left is the error).
    /// \ingroup core
    inline constexpr detail::either_fn either{};

    /// \brief `map_maybe(f, xs)`: the values of the engaged `f(x)`, in order, in an `inplace_vector`.
    /// \ingroup core
    inline constexpr detail::map_maybe_fn map_maybe{};

    /// \brief `cat_maybes(xs)`: the values of the engaged optionals of an array, in order.
    /// \ingroup core
    inline constexpr detail::cat_maybes_fn cat_maybes{};

    /// \brief `replicate_m<N>(m)`: Haskell's `replicateM`, `N` copies of `m` sequenced into a carrier of an array.
    /// \ingroup core
    /// \tparam N How many.
    /// \param m The carrier.
    /// \return `sequence(replicate<N>(m))`.
    template <std::size_t N, class M>
        requires(N > 0)
    [[nodiscard]] constexpr auto replicate_m(const M& m) SNICHOLLS_FN_RETURNS(sequence(fn::replicate<N>(m)))

} // namespace snicholls::fn

#endif /* snicholls_core_functional_monad_hpp */
