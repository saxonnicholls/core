//
//  recursion.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Recursion schemes over list-shaped data, `cata`, `ana`, `hylo` and `para`, and a `trampoline` for deep
///        recursion without the stack, all `constexpr`.
/// \ingroup core
///
/// **Overview.** A recursion scheme separates *what* a recursion computes from *how* it walks:
///
/// | here                                   | what it is                                                         | Haskell (recursion-schemes) |
/// | -------------------------------------- | ------------------------------------------------------------------ | --------------------------- |
/// | `cata(alg, nil, xs)`                   | a list torn down: `alg(x0, alg(x1, ... nil))`                      | `cata` on `ListF`           |
/// | `ana<N>(coalg, seed)`                  | a list built: `coalg(seed)` gives `{x, next seed}` or stops        | `ana`                       |
/// | `hylo(alg, nil, coalg, seed)`          | built and torn down in one pass, no list in between (left-nested)  | `hylo` (fused)              |
/// | `para(alg, nil, xs)`                   | `cata` where `alg` also sees the rest of the list                   | `para`                      |
/// | `trampoline(step, state)`              | `step` returns `more(next)` or `done(result)`; a loop, not a stack | `Trampoline`                |
///
/// \snippet recursion_demo.cpp schemes
///
/// The schemes here are over the list functor (one element and the rest), which is the shape most
/// recursions in core's domain take: a cash-flow schedule built from a state (`ana`) and priced back
/// (`cata`), a convergence loop (`hylo`, `trampoline`). Recursion over trees in constant expressions
/// needs a fixed-point type with indirection, which C++ gives only through allocation or an index
/// arena; it is left out (see the report in the CHANGELOG), and `fix` (`functional/combinators.hpp`)
/// gives recursion over any shape for a function.
///
/// **Relationships.** `fn::foldr_of` (`cata` is it), `fn::unfold` (`ana` with a fixed count),
/// `functional/lazy.hpp` (infinite unfolds), `fn::fix`, `snicholls::nest_while`. Group: \ref core.
///
/// **Implementations / variants.** `hylo` is fused: it never materialises the list, so it needs no
/// bound and folds left (`alg(acc, x)`). `ana` collects into a `containers::inplace_vector<T, N>` and
/// stops at `N`.
///
/// **Guarantees.** `constexpr`, allocation-free, and none of them recurse at run time: `cata` and `para`
/// walk backwards in a loop, `hylo` and `trampoline` are loops. So depth is limited by nothing but time
/// (and the compiler's step limit at compile time).
///
/// **Errors.** None of their own. `hylo` and `trampoline` do not terminate if the coalgebra or the
/// step never stops.
///
/// **Examples.** \snippet recursion_demo.cpp trampoline
///
/// **Performance.** The loops; not benchmarked.
///
/// **Tests and demos.** `tests/functional/recursion_test.cpp`, `demos/functional/recursion_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<functional>`, `<optional>`, `<ranges>`, `<type_traits>`, `<utility>`,
/// `<variant>`, `containers/inplace_vector.hpp` and `functional/fold.hpp`.
///
/// **Known issues and gotchas.** `para` gives `alg` the rest of the list as a `std::span`-like pair of
/// iterators over a contiguous range; it needs a contiguous range.
///
/// **Status.** Experimental, new in 0.27.0.

#ifndef snicholls_core_functional_recursion_hpp
#define snicholls_core_functional_recursion_hpp

#include <snicholls/core/containers/inplace_vector.hpp>
#include <snicholls/core/functional/fold.hpp>

#include <cstddef>
#include <functional>
#include <optional>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <variant>

namespace snicholls::fn {

    /// \brief A trampoline step that continues with a new state.
    /// \ingroup core
    /// \tparam S The state's type.
    template <class S>
    struct more_t {
        S state; ///< The next state.
    };

    /// \brief A trampoline step that stops with a result.
    /// \ingroup core
    /// \tparam R The result's type.
    template <class R>
    struct done_t {
        R result; ///< The result.
    };

    /// \brief What a trampoline step returns: `more(next state)` or `done(result)`.
    /// \ingroup core
    /// \tparam S The state's type.
    /// \tparam R The result's type.
    template <class S, class R>
    using bounce = std::variant<more_t<S>, done_t<R>>;

    namespace detail {

        struct cata_fn {
            template <class Alg, class Z, class R>
            constexpr auto operator()(Alg&& alg, Z&& nil, R&& xs) const
                SNICHOLLS_FN_RETURNS(foldr_of_fn{}(std::forward<Alg>(alg), std::forward<Z>(nil), std::forward<R>(xs)))
        };

        struct hylo_fn {
            template <class Alg, class Z, class Coalg, class S, class V = std::remove_cvref_t<Z>>
            constexpr V operator()(Alg&& alg, Z&& z, Coalg&& coalg, S&& seed) const
            {
                V acc(std::forward<Z>(z));
                std::remove_cvref_t<S> state(std::forward<S>(seed));
                for (auto step = std::invoke(coalg, std::as_const(state)); step.has_value();
                     step = std::invoke(coalg, std::as_const(state))) {
                    acc = std::invoke(alg, std::move(acc), std::move(step->first));
                    state = std::move(step->second);
                }
                return acc;
            }
        };

        struct para_fn {
            template <class Alg, class Z, std::ranges::contiguous_range R, class V = std::remove_cvref_t<Z>>
            constexpr V operator()(Alg&& alg, Z&& nil, R&& xs) const
            {
                using T = std::remove_reference_t<std::ranges::range_reference_t<R>>;
                const std::span<T> all(std::ranges::data(xs), std::ranges::size(xs));
                V acc(std::forward<Z>(nil));
                for (std::size_t i = all.size(); i-- > 0;)
                    acc = std::invoke(alg, all[i], all.subspan(i + 1), std::move(acc));
                return acc;
            }
        };

        struct more_fn {
            template <class S>
            constexpr more_t<std::decay_t<S>> operator()(S&& s) const { return {std::forward<S>(s)}; }
        };

        struct done_fn {
            template <class R>
            constexpr done_t<std::decay_t<R>> operator()(R&& r) const { return {std::forward<R>(r)}; }
        };

        template <class B>
        struct bounce_types;
        template <class S, class R>
        struct bounce_types<std::variant<more_t<S>, done_t<R>>> {
            using state = S;
            using result = R;
        };

        struct trampoline_fn {
            template <class Step, class S, class B = std::remove_cvref_t<std::invoke_result_t<Step&, std::remove_cvref_t<S>>>,
                      class R = typename bounce_types<B>::result>
            constexpr R operator()(Step&& step, S&& start) const
            {
                B b = more_t<typename bounce_types<B>::state>{std::forward<S>(start)};
                while (b.index() == 0) b = std::invoke(step, std::move(std::get<0>(b).state));
                return std::move(std::get<1>(b).result);
            }
        };

    } // namespace detail

    /// \brief Catamorphism: `cata(alg, nil, xs)` is `alg(x0, alg(x1, ... alg(x_last, nil)))` (`foldr_of`).
    /// \ingroup core
    inline constexpr detail::cata_fn cata{};

    /// \brief Hylomorphism, fused: unfold with `coalg` (`state -> std::optional{pair{x, next}}`) and fold each `x` into
    ///        `alg(acc, x)` from `z`, without the list in between.
    /// \ingroup core
    inline constexpr detail::hylo_fn hylo{};

    /// \brief Paramorphism: `cata` where `alg(x, rest, acc)` also sees the elements after `x` (a `std::span`).
    /// \ingroup core
    inline constexpr detail::para_fn para{};

    /// \brief `more(s)`: a trampoline step that continues from `s`.
    /// \ingroup core
    inline constexpr detail::more_fn more{};

    /// \brief `done(r)`: a trampoline step that stops with `r`.
    /// \ingroup core
    inline constexpr detail::done_fn done{};

    /// \brief `trampoline(step, s)`: `step` applied while it returns `more(next)`; the result of the `done(r)` it ends with.
    /// \ingroup core
    ///
    /// `step` returns `fn::bounce<S, R>` (a `std::variant<more_t<S>, done_t<R>>`). A recursion turned into a
    /// trampoline runs in constant stack, at any depth.
    inline constexpr detail::trampoline_fn trampoline{};

    /// \brief Anamorphism: the list built by `coalg` (`state -> std::optional{pair{x, next}}`) from `seed`, at most `N`.
    /// \ingroup core
    /// \tparam N The most elements.
    /// \param coalg The coalgebra; `std::nullopt` ends the list.
    /// \param seed The first state.
    /// \return A `containers::inplace_vector<T, N>`.
    template <std::size_t N, class Coalg, class S>
    [[nodiscard]] constexpr auto ana(Coalg&& coalg, S&& seed)
    {
        using state_t = std::remove_cvref_t<S>;
        using step_t = std::remove_cvref_t<std::invoke_result_t<Coalg&, const state_t&>>;
        using T = std::remove_cvref_t<typename step_t::value_type::first_type>;
        containers::inplace_vector<T, N> out;
        state_t state(std::forward<S>(seed));
        while (out.size() < N) {
            step_t step = std::invoke(coalg, std::as_const(state));
            if (!step.has_value()) break;
            out.push_back(std::move(step->first));
            state = std::move(step->second);
        }
        return out;
    }

} // namespace snicholls::fn

#endif /* snicholls_core_functional_recursion_hpp */
